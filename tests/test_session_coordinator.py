from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest

from recoverybox.core import (
    DEFAULT_CUE_CATALOG,
    CueId,
    GuardianAction,
    GuardianDecision,
    GuardianReason,
    SessionMode,
)
from recoverybox.session import (
    ApprovedCuePlaybackAuthorization,
    CueAuthorizationError,
    GuardianDecisionEffect,
    SessionCompositionError,
    SessionCoordinator,
    session_mode_allows_model_audio,
)


@dataclass
class _CueSpeaker:
    events: list[tuple] = field(default_factory=list)
    fail_playback: bool = False

    def preempt_model_audio(self) -> None:
        self.events.append(("preempt_model",))

    def play_approved_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None:
        self.events.append(("play_cue", authorization))
        if self.fail_playback:
            raise OSError("prompt cue request failed")


@dataclass
class _ModelGate:
    events: list[tuple] = field(default_factory=list)

    def preempt_model_audio(self) -> None:
        self.events.append(("preempt_model",))


class _RacingCueSpeaker:
    """Model the cue lane's lock around final PCM handoff and preemption."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.completion_entered = threading.Event()
        self.allow_completion = threading.Event()
        self.preemption_started = threading.Event()
        self.preemption_finished = threading.Event()
        self.playback_queue: list[str] = []
        self.play_requests: list[ApprovedCuePlaybackAuthorization] = []

    def complete_cue(self) -> None:
        with self._lock:
            self.completion_entered.set()
            assert self.allow_completion.wait(timeout=1)
            self.playback_queue.append("approved-pcm")

    def hold_lane(self) -> None:
        self._lock.acquire()

    def release_lane(self) -> None:
        self._lock.release()

    def preempt_model_audio(self) -> None:
        self.preemption_started.set()
        with self._lock:
            self.playback_queue.clear()
        self.preemption_finished.set()

    def play_approved_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None:
        self.play_requests.append(authorization)


@dataclass
class _FailingPreemptor:
    coordinator: SessionCoordinator | None = None
    modes_seen: list[SessionMode] = field(default_factory=list)

    def preempt_model_audio(self) -> None:
        assert self.coordinator is not None
        self.modes_seen.append(self.coordinator.current_mode)
        raise OSError("speaker boundary failed")

    def play_approved_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None:
        raise AssertionError(f"unexpected cue request: {authorization.cue_id}")


def _decision(
    action: GuardianAction,
    *,
    cue_id: str | None = None,
    reason: GuardianReason = GuardianReason.WITHIN_LIMITS,
) -> GuardianDecision:
    return GuardianDecision(
        action=action,
        cue_id=cue_id,
        reason_codes=(reason,),
        rule_version="guardian-test-v1",
    )


def test_coordinator_owns_mode_and_preempts_before_restricted_phases() -> None:
    speaker = _CueSpeaker()
    model_gate = _ModelGate()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.CHECK_IN,
    )
    coordinator.register_model_audio_preemptor(model_gate)

    previous = coordinator.transition_to(SessionMode.ACTIVE_EXERCISE)

    assert previous is SessionMode.CHECK_IN
    assert coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
    assert speaker.events == [("preempt_model",)]
    assert model_gate.events == [("preempt_model",)]


def test_restricted_mode_is_published_only_after_racing_cue_handoff_is_preempted() -> None:
    speaker = _RacingCueSpeaker()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    completion = threading.Thread(target=speaker.complete_cue, daemon=True)
    completion.start()
    assert speaker.completion_entered.wait(timeout=1)

    transition_errors: list[BaseException] = []

    def pause() -> None:
        try:
            coordinator.transition_to(SessionMode.PAUSED)
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            transition_errors.append(exc)

    transition = threading.Thread(target=pause, daemon=True)
    transition.start()
    assert speaker.preemption_started.wait(timeout=1)

    # Preemption is blocked behind the in-progress final cue handoff.  The
    # restricted mode cannot become visible during that window.
    assert coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
    assert not speaker.preemption_finished.is_set()

    speaker.allow_completion.set()
    completion.join(timeout=1)
    transition.join(timeout=1)

    assert not completion.is_alive()
    assert not transition.is_alive()
    assert transition_errors == []
    assert speaker.preemption_finished.is_set()
    assert speaker.playback_queue == []
    assert coordinator.current_mode is SessionMode.PAUSED


def test_guardian_cue_cannot_enqueue_across_a_restricted_transition() -> None:
    speaker = _RacingCueSpeaker()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )

    # Hold the cue lane so the restricted transition owns the coordinator's
    # operation order but has not yet published PAUSED.
    speaker.hold_lane()
    transition = threading.Thread(
        target=coordinator.transition_to,
        args=(SessionMode.PAUSED,),
        daemon=True,
    )
    transition.start()
    assert speaker.preemption_started.wait(timeout=1)

    cue_errors: list[BaseException] = []

    def request_cue() -> None:
        try:
            coordinator.apply_guardian_decision(
                _decision(GuardianAction.CUE, cue_id=CueId.MOVE_SLOWLY.value)
            )
        except BaseException as exc:
            cue_errors.append(exc)

    cue_request = threading.Thread(target=request_cue, daemon=True)
    cue_request.start()
    assert coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
    assert speaker.play_requests == []

    speaker.release_lane()
    transition.join(timeout=1)
    cue_request.join(timeout=1)

    assert not transition.is_alive()
    assert not cue_request.is_alive()
    assert coordinator.current_mode is SessionMode.PAUSED
    assert speaker.play_requests == []
    assert len(cue_errors) == 1
    assert isinstance(cue_errors[0], CueAuthorizationError)


def test_preemption_failure_still_publishes_fail_closed_mode_after_all_attempts() -> None:
    failing_speaker = _FailingPreemptor()
    model_gate = _ModelGate()
    coordinator = SessionCoordinator(
        cue_playback=failing_speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    failing_speaker.coordinator = coordinator
    coordinator.register_model_audio_preemptor(model_gate)

    with pytest.raises(SessionCompositionError, match="1 local boundary"):
        coordinator.transition_to(SessionMode.STOPPED)

    # Ports may safely read the old mode while they drain.  STOPPED becomes
    # observable only after all boundaries have been attempted, even though
    # one of them failed.
    assert failing_speaker.modes_seen == [SessionMode.ACTIVE_EXERCISE]
    assert model_gate.events == [("preempt_model",)]
    assert coordinator.current_mode is SessionMode.STOPPED


@pytest.mark.parametrize("mode", tuple(SessionMode))
def test_only_check_in_and_complete_post_session_allow_model_audio(mode: SessionMode) -> None:
    assert session_mode_allows_model_audio(mode) is (
        mode in {SessionMode.CHECK_IN, SessionMode.COMPLETE}
    )


def test_every_active_exercise_audible_request_is_a_catalog_approved_cue_id() -> None:
    speaker = _CueSpeaker()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )

    effects: list[GuardianDecisionEffect] = []
    for cue_id in CueId:
        effects.append(
            coordinator.apply_guardian_decision(
                _decision(
                    GuardianAction.CUE,
                    cue_id=cue_id.value,
                    reason=GuardianReason.LOCAL_CUE_ACCEPTED,
                )
            )
        )

    audible_requests = [event[1] for event in speaker.events if event[0] == "play_cue"]
    assert len(audible_requests) == len(CueId)
    assert [effect.cue_authorization for effect in effects] == audible_requests
    assert all(effect.model_audio_preempted is False for effect in effects)
    assert all(event[0] != "preempt_model" for event in speaker.events)
    for request in audible_requests:
        assert isinstance(request, ApprovedCuePlaybackAuthorization)
        assert isinstance(request.cue_id, CueId)
        assert DEFAULT_CUE_CATALOG.is_approved(request.cue_id.value)
        cue = DEFAULT_CUE_CATALOG[request.cue_id.value]
        assert request.cue_kind is cue.kind
        assert request.catalog_version == coordinator.catalog_version
        assert request.guardian_rule_version == "guardian-test-v1"
        assert request.reason_codes == (GuardianReason.LOCAL_CUE_ACCEPTED,)
        assert not hasattr(request, "pcm")
        assert not hasattr(request, "spoken_text")


@pytest.mark.parametrize(
    ("action", "expected_mode"),
    (
        (GuardianAction.PAUSE, SessionMode.PAUSED),
        (GuardianAction.STOP, SessionMode.STOPPED),
        (GuardianAction.ESCALATE, SessionMode.STOPPED),
    ),
)
def test_pause_stop_and_escalate_preempt_every_model_audio_boundary(
    action: GuardianAction,
    expected_mode: SessionMode,
) -> None:
    speaker = _CueSpeaker()
    model_gate = _ModelGate()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    coordinator.register_model_audio_preemptor(model_gate)

    effect = coordinator.apply_guardian_decision(_decision(action))

    assert coordinator.current_mode is expected_mode
    assert effect.current_mode is expected_mode
    assert effect.model_audio_preempted is True
    assert speaker.events == [("preempt_model",)]
    assert model_gate.events == [("preempt_model",)]
    assert all(event[0] != "play_cue" for event in speaker.events)


def test_unapproved_or_non_cue_id_decision_fails_closed_to_pause() -> None:
    speaker = _CueSpeaker()
    model_gate = _ModelGate()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    coordinator.register_model_audio_preemptor(model_gate)

    with pytest.raises(CueAuthorizationError, match="approved catalog"):
        coordinator.apply_guardian_decision(
            _decision(GuardianAction.CUE, cue_id="model-generated-audio")
        )

    assert coordinator.current_mode is SessionMode.PAUSED
    assert speaker.events == [("preempt_model",)]
    assert model_gate.events == [("preempt_model",)]


def test_continue_never_resumes_a_paused_session_or_requests_audio() -> None:
    speaker = _CueSpeaker()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.PAUSED,
    )

    effect = coordinator.apply_guardian_decision(_decision(GuardianAction.CONTINUE))

    assert coordinator.current_mode is SessionMode.PAUSED
    assert effect.current_mode is SessionMode.PAUSED
    assert effect.model_audio_preempted is False
    assert speaker.events == []


def test_prompt_cue_request_failure_pauses_and_preempts_model_audio() -> None:
    speaker = _CueSpeaker(fail_playback=True)
    model_gate = _ModelGate()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    coordinator.register_model_audio_preemptor(model_gate)

    with pytest.raises(OSError, match="prompt cue request failed"):
        coordinator.apply_guardian_decision(
            _decision(
                GuardianAction.CUE,
                cue_id=CueId.MOVE_SLOWLY.value,
                reason=GuardianReason.LOCAL_CUE_ACCEPTED,
            )
        )

    assert coordinator.current_mode is SessionMode.PAUSED
    # Enqueueing one cue does not chop an earlier cue.  The failed request
    # enters PAUSED, which closes the prompt-cue and conversation lanes once.
    assert [event[0] for event in speaker.events] == [
        "play_cue",
        "preempt_model",
    ]
    assert model_gate.events == [("preempt_model",)]


def test_check_in_accepts_only_reviewed_script_cues_and_fails_closed_otherwise() -> None:
    speaker = _CueSpeaker()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.CHECK_IN,
    )

    intro = coordinator.apply_guardian_decision(
        _decision(
            GuardianAction.CUE,
            cue_id=CueId.SQUAT_SET_INTRO.value,
            reason=GuardianReason.LOCAL_CUE_ACCEPTED,
        )
    )

    assert intro.current_mode is SessionMode.CHECK_IN
    assert speaker.events[-1][1].cue_id is CueId.SQUAT_SET_INTRO

    with pytest.raises(CueAuthorizationError, match="reviewed scripted"):
        coordinator.apply_guardian_decision(
            _decision(
                GuardianAction.CUE,
                cue_id=CueId.SQUAT_REP_ONE.value,
                reason=GuardianReason.LOCAL_CUE_ACCEPTED,
            )
        )
    assert coordinator.current_mode is SessionMode.PAUSED


def test_scripted_check_in_to_active_preserves_cue_lane_but_preempts_arbitrary_audio() -> None:
    speaker = _CueSpeaker()
    model_gate = _ModelGate()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.CHECK_IN,
    )
    coordinator.register_model_audio_preemptor(model_gate)
    coordinator.apply_guardian_decision(
        _decision(
            GuardianAction.CUE,
            cue_id=CueId.SQUAT_SET_INTRO.value,
            reason=GuardianReason.LOCAL_CUE_ACCEPTED,
        )
    )

    previous = coordinator.begin_active_exercise_from_check_in()

    assert previous is SessionMode.CHECK_IN
    assert coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
    assert [event[0] for event in speaker.events] == ["play_cue"]
    assert model_gate.events == [("preempt_model",)]


def test_forged_cue_without_guardian_provenance_fails_closed() -> None:
    speaker = _CueSpeaker()
    coordinator = SessionCoordinator(
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )

    with pytest.raises(CueAuthorizationError, match="approved catalog"):
        coordinator.apply_guardian_decision(
            _decision(
                GuardianAction.CUE,
                cue_id=CueId.SQUAT_REP_ONE.value,
                reason=GuardianReason.WITHIN_LIMITS,
            )
        )

    assert coordinator.current_mode is SessionMode.PAUSED
    assert [event[0] for event in speaker.events] == ["preempt_model"]
