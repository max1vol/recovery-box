from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest

from recoverybox.core import (
    DEFAULT_CUE_CATALOG,
    ApprovedCueCatalog,
    CueId,
    ExercisePlan,
    Guardian,
    GuardianAction,
    GuardianDecision,
    GuardianReason,
    GuardianRuntimeFault,
    LocalCueRequest,
    MovementObservation,
    SessionMode,
)
from recoverybox.realtime import (
    RuntimeAbortReason,
    SessionEndController,
    SessionEndSource,
    validate_finish_session_call,
)
from recoverybox.session import (
    ApprovedCuePlaybackAuthorization,
    ApprovedCuePlaybackPort,
    CueAuthorizationError,
    EmergencyEscalationPort,
    GuardianDecisionEffect,
    GuardianEscalationRecord,
    SessionCompositionError,
    SessionCoordinator,
    session_mode_allows_model_audio,
)

_GUARDIAN = Guardian(rule_version="guardian-test-v1")
_PLAN = ExercisePlan(
    exercise_id="squat",
    allowed_cue_ids=frozenset(cue.value for cue in CueId),
    required_camera_views=1,
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
    fail: bool = True

    def preempt_model_audio(self) -> None:
        assert self.coordinator is not None
        self.modes_seen.append(self.coordinator.current_mode)
        if self.fail:
            raise OSError("speaker boundary failed")

    def play_approved_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None:
        raise AssertionError(f"unexpected cue request: {authorization.cue_id}")


@dataclass
class _EscalationPort:
    decisions: list[GuardianDecision] = field(default_factory=list)
    fail: bool = False

    def request_emergency_escalation(
        self,
        decision: GuardianDecision,
    ) -> None:
        self.decisions.append(decision)
        if self.fail:
            raise OSError("escalation port unavailable")


def _decision(
    action: GuardianAction,
    *,
    cue_id: str | None = None,
    reason: GuardianReason = GuardianReason.WITHIN_LIMITS,
    guardian: Guardian = _GUARDIAN,
) -> GuardianDecision:
    del reason
    if action is GuardianAction.CUE:
        assert cue_id is not None
        if cue_id in {
            CueId.SQUAT_SET_INTRO.value,
            CueId.SQUAT_PERSON_DETECTED.value,
        }:
            return guardian.decide_scripted_session_cue(
                LocalCueRequest(cue_id),
                _PLAN,
            )
    observation = MovementObservation(
        exercise_id="squat",
        timestamp_ms=1,
        confidence=0.0 if action is GuardianAction.PAUSE else 1.0,
        camera_disagreement_degrees=None,
        pose_age_ms=0,
        camera_view_count=1,
        pain_reported=action is GuardianAction.STOP,
        emergency_reported=action is GuardianAction.ESCALATE,
    )
    suggestion = None
    request = LocalCueRequest(cue_id) if action is GuardianAction.CUE else None
    return guardian.decide(
        observation,
        _PLAN,
        suggestion,
        local_cue_request=request,
    )


def _coordinator(
    *,
    guardian: Guardian = _GUARDIAN,
    cue_playback: ApprovedCuePlaybackPort,
    escalation_port: EmergencyEscalationPort | None = None,
    cue_catalog: ApprovedCueCatalog | None = None,
    initial_mode: SessionMode = SessionMode.IDLE,
    session_end_authority: SessionEndController | None = None,
) -> SessionCoordinator:
    end_controller = session_end_authority or SessionEndController()
    coordinator = SessionCoordinator(
        guardian=guardian,
        cue_playback=cue_playback,
        session_end_authority=end_controller,
        escalation_port=escalation_port,
        cue_catalog=cue_catalog,
    )
    if isinstance(cue_playback, _FailingPreemptor):
        cue_playback.coordinator = coordinator
    if initial_mode is SessionMode.CHECK_IN:
        coordinator.enter_check_in()
    elif initial_mode in {SessionMode.ACTIVE_EXERCISE, SessionMode.PAUSED}:
        coordinator.enter_check_in()
        coordinator.activate_after_guardian_continue(
            _decision(GuardianAction.CONTINUE, guardian=guardian)
        )
        if initial_mode is SessionMode.PAUSED:
            coordinator.apply_guardian_decision(_decision(GuardianAction.PAUSE, guardian=guardian))
    elif initial_mode is not SessionMode.IDLE:
        raise AssertionError(f"unsupported test setup mode: {initial_mode}")
    events = getattr(cue_playback, "events", None)
    if isinstance(events, list):
        events.clear()
    if isinstance(cue_playback, _FailingPreemptor):
        cue_playback.modes_seen.clear()
    return coordinator


def test_coordinator_owns_mode_and_preempts_before_restricted_phases() -> None:
    speaker = _CueSpeaker()
    model_gate = _ModelGate()
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    coordinator.register_model_audio_preemptor(model_gate)

    effect = coordinator.apply_guardian_decision(_decision(GuardianAction.PAUSE))

    assert effect.previous_mode is SessionMode.ACTIVE_EXERCISE
    assert coordinator.current_mode is SessionMode.PAUSED
    assert speaker.events == [("preempt_model",)]
    assert model_gate.events == [("preempt_model",)]


def test_public_arbitrary_mode_transition_is_not_available() -> None:
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=_CueSpeaker(),
    )

    assert not hasattr(coordinator, "transition_to")


@pytest.mark.parametrize(
    "restricted_mode",
    (
        SessionMode.ACTIVE_EXERCISE,
        SessionMode.PAUSED,
        SessionMode.STOPPED,
        SessionMode.COMPLETE,
    ),
)
def test_constructor_cannot_bootstrap_a_restricted_mode(
    restricted_mode: SessionMode,
) -> None:
    with pytest.raises(TypeError, match="initial_mode"):
        SessionCoordinator(
            guardian=_GUARDIAN,
            cue_playback=_CueSpeaker(),
            session_end_authority=SessionEndController(),
            initial_mode=restricted_mode,  # type: ignore[call-arg]
        )


def test_coordinator_uses_the_bound_guardians_exact_cue_catalog() -> None:
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=_CueSpeaker(),
    )
    equal_but_distinct_catalog = ApprovedCueCatalog(DEFAULT_CUE_CATALOG.values())

    assert coordinator.cue_catalog is _GUARDIAN.cue_catalog
    with pytest.raises(ValueError, match="bound Guardian's catalog"):
        _coordinator(
            guardian=_GUARDIAN,
            cue_playback=_CueSpeaker(),
            cue_catalog=equal_but_distinct_catalog,
        )


@pytest.mark.parametrize("initial_mode", (SessionMode.IDLE, SessionMode.PAUSED))
def test_activation_preemption_failure_never_publishes_active(
    initial_mode: SessionMode,
) -> None:
    failing = _FailingPreemptor(fail=False)
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=failing,
        initial_mode=initial_mode,
    )
    failing.coordinator = coordinator
    failing.fail = True

    with pytest.raises(SessionCompositionError, match="during exercise activation"):
        coordinator.activate_after_guardian_continue(_decision(GuardianAction.CONTINUE))

    assert coordinator.current_mode is SessionMode.PAUSED
    assert coordinator.current_mode is not SessionMode.ACTIVE_EXERCISE
    assert coordinator.last_guardian_action is GuardianAction.PAUSE


def test_stale_continue_cannot_resume_a_later_guardian_pause() -> None:
    guardian = Guardian()
    coordinator = _coordinator(
        guardian=guardian,
        cue_playback=_CueSpeaker(),
    )
    old_continue = _decision(GuardianAction.CONTINUE, guardian=guardian)
    coordinator.activate_after_guardian_continue(old_continue)
    coordinator.apply_guardian_decision(_decision(GuardianAction.PAUSE, guardian=guardian))

    with pytest.raises(SessionCompositionError, match="latest unused"):
        coordinator.activate_after_guardian_continue(old_continue)

    assert coordinator.current_mode is SessionMode.PAUSED
    assert coordinator.last_guardian_action is GuardianAction.PAUSE


def test_restricted_mode_is_published_only_after_racing_cue_handoff_is_preempted() -> None:
    speaker = _RacingCueSpeaker()
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    completion = threading.Thread(target=speaker.complete_cue, daemon=True)
    completion.start()
    assert speaker.completion_entered.wait(timeout=1)

    transition_errors: list[BaseException] = []

    def pause() -> None:
        try:
            coordinator.apply_guardian_decision(_decision(GuardianAction.PAUSE))
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
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )

    # Hold the cue lane so the restricted transition owns the coordinator's
    # operation order but has not yet published PAUSED.
    speaker.hold_lane()
    transition = threading.Thread(
        target=coordinator.apply_guardian_decision,
        args=(_decision(GuardianAction.PAUSE),),
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
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=failing_speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    failing_speaker.coordinator = coordinator
    coordinator.register_model_audio_preemptor(model_gate)

    with pytest.raises(SessionCompositionError, match="1 local boundary"):
        coordinator.apply_guardian_decision(_decision(GuardianAction.STOP))

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
    coordinator = _coordinator(
        guardian=_GUARDIAN,
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
    coordinator = _coordinator(
        guardian=_GUARDIAN,
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


def test_escalate_has_a_distinct_effect_and_port_but_stop_does_not() -> None:
    stop_port = _EscalationPort()
    stop_coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=_CueSpeaker(),
        escalation_port=stop_port,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )

    stop = stop_coordinator.apply_guardian_decision(_decision(GuardianAction.STOP))

    assert stop.action is GuardianAction.STOP
    assert stop.escalation_record is None
    assert stop_port.decisions == []

    escalation_port = _EscalationPort()
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=_CueSpeaker(),
        escalation_port=escalation_port,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )

    decision = _decision(GuardianAction.ESCALATE)
    escalated = coordinator.apply_guardian_decision(decision)

    assert escalated.action is GuardianAction.ESCALATE
    assert isinstance(escalated.escalation_record, GuardianEscalationRecord)
    assert escalated.escalation_record.reason_codes == (GuardianReason.EMERGENCY_REPORTED,)
    assert escalated.escalation_record.guardian_sequence == decision.sequence
    assert escalation_port.decisions == [decision]
    assert coordinator.last_escalation_record is escalated.escalation_record
    assert coordinator.last_guardian_action is GuardianAction.ESCALATE
    assert coordinator.current_mode is SessionMode.STOPPED


def test_escalation_port_failure_stays_stopped_and_audited() -> None:
    escalation_port = _EscalationPort(fail=True)
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=_CueSpeaker(),
        escalation_port=escalation_port,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )

    with pytest.raises(SessionCompositionError, match="after safe stop"):
        coordinator.apply_guardian_decision(_decision(GuardianAction.ESCALATE))

    assert coordinator.current_mode is SessionMode.STOPPED
    assert coordinator.last_guardian_action is GuardianAction.ESCALATE
    assert len(escalation_port.decisions) == 1
    assert coordinator.last_escalation_record is not None


def test_default_production_escalation_audit_retains_sealed_provenance() -> None:
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=_CueSpeaker(),
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    decision = _decision(GuardianAction.ESCALATE)

    effect = coordinator.apply_guardian_decision(decision)

    assert effect.escalation_record is coordinator.last_escalation_record
    assert effect.escalation_record is not None
    assert effect.escalation_record.guardian_sequence == decision.sequence


def test_runtime_fault_is_arbitrated_by_bound_guardian() -> None:
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=_CueSpeaker(),
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )

    effect = coordinator.apply_runtime_fault(GuardianRuntimeFault.REALTIME_UNAVAILABLE)

    assert effect.action is GuardianAction.PAUSE
    assert coordinator.last_guardian_action is GuardianAction.PAUSE
    assert coordinator.current_mode is SessionMode.PAUSED


def test_physical_finish_and_runtime_abort_remain_distinct_terminations() -> None:
    physical_controller = SessionEndController()
    physical = physical_controller.request_physical_stop()
    assert physical is not None
    physical_coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=_CueSpeaker(),
        initial_mode=SessionMode.ACTIVE_EXERCISE,
        session_end_authority=physical_controller,
    )
    physical_coordinator.apply_session_end(physical)

    finish_controller = SessionEndController()
    finish = finish_controller.accept_validated_tool_call(
        validate_finish_session_call(call_id="finish-1", arguments_json="{}")
    )
    assert finish is not None
    finish_coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=_CueSpeaker(),
        initial_mode=SessionMode.ACTIVE_EXERCISE,
        session_end_authority=finish_controller,
    )
    finish_coordinator.apply_session_end(finish)

    abort_controller = SessionEndController()
    abort = abort_controller.request_runtime_abort(RuntimeAbortReason.SERVICE_SHUTDOWN)
    assert abort is not None
    abort_coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=_CueSpeaker(),
        initial_mode=SessionMode.ACTIVE_EXERCISE,
        session_end_authority=abort_controller,
    )
    abort_coordinator.apply_session_end(abort)

    assert physical.source is SessionEndSource.PHYSICAL_STOP
    assert finish.source is SessionEndSource.VALIDATED_TOOL_CALL
    assert abort.source is SessionEndSource.RUNTIME_ABORT
    assert abort.abort_reason is RuntimeAbortReason.SERVICE_SHUTDOWN
    assert physical_coordinator.last_guardian_action is GuardianAction.CONTINUE
    assert finish_coordinator.last_guardian_action is GuardianAction.CONTINUE
    assert abort_coordinator.last_guardian_action is GuardianAction.CONTINUE


def test_session_end_source_is_retained_when_preemption_fails() -> None:
    end_controller = SessionEndController()
    signal = end_controller.request_physical_stop()
    assert signal is not None
    failing = _FailingPreemptor()
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=failing,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
        session_end_authority=end_controller,
    )
    failing.coordinator = coordinator

    with pytest.raises(SessionCompositionError, match="1 local boundary"):
        coordinator.apply_session_end(signal)

    assert coordinator.current_mode is SessionMode.STOPPED
    assert coordinator.termination_signal is signal
    assert coordinator.last_guardian_action is GuardianAction.CONTINUE


def test_foreign_controller_end_signal_cannot_terminate_session() -> None:
    bound_controller = SessionEndController()
    foreign_signal = SessionEndController().request_physical_stop()
    assert foreign_signal is not None
    speaker = _CueSpeaker()
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
        session_end_authority=bound_controller,
    )

    with pytest.raises(TypeError, match="bound SessionEndController"):
        coordinator.apply_session_end(foreign_signal)

    assert coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
    assert coordinator.termination_signal is None
    assert speaker.events == []


def test_decision_from_a_different_guardian_cannot_change_mode_or_audio() -> None:
    speaker = _CueSpeaker()
    model_gate = _ModelGate()
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    coordinator.register_model_audio_preemptor(model_gate)

    other_guardian = Guardian()
    foreign = other_guardian.decide_runtime_fault(GuardianRuntimeFault.RUNTIME_BOUNDARY_FAILURE)

    with pytest.raises(TypeError, match="bound Guardian"):
        coordinator.apply_guardian_decision(foreign)

    assert coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
    assert speaker.events == []
    assert model_gate.events == []


def test_continue_never_resumes_a_paused_session_or_requests_audio() -> None:
    speaker = _CueSpeaker()
    coordinator = _coordinator(
        guardian=_GUARDIAN,
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
    coordinator = _coordinator(
        guardian=_GUARDIAN,
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
    coordinator = _coordinator(
        guardian=_GUARDIAN,
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
    coordinator = _coordinator(
        guardian=_GUARDIAN,
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

    previous = coordinator.activate_after_guardian_continue(_decision(GuardianAction.CONTINUE))

    assert previous is SessionMode.CHECK_IN
    assert coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
    assert [event[0] for event in speaker.events] == ["play_cue"]
    assert model_gate.events == [("preempt_model",)]


def test_unissued_decision_object_is_rejected_before_any_effect() -> None:
    speaker = _CueSpeaker()
    coordinator = _coordinator(
        guardian=_GUARDIAN,
        cue_playback=speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )

    forged = object.__new__(GuardianDecision)
    object.__setattr__(forged, "action", GuardianAction.PAUSE)
    object.__setattr__(forged, "cue_id", None)
    object.__setattr__(forged, "reason_codes", (GuardianReason.LOW_CONFIDENCE,))
    object.__setattr__(forged, "rule_version", "forged")

    with pytest.raises(TypeError, match="bound Guardian"):
        coordinator.apply_guardian_decision(forged)

    assert coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
    assert speaker.events == []
