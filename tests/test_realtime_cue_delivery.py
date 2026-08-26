from __future__ import annotations

import base64
from dataclasses import dataclass, field

import pytest

from recoverybox.core import DEFAULT_CUE_CATALOG, CueId, GuardianReason, SessionMode
from recoverybox.realtime import MemoryTransport, RealtimeSession
from recoverybox.realtime.cue_delivery import (
    CueDeliveryConfig,
    CueDeliveryError,
    CueDeliveryFailure,
    CueDeliveryFailureReason,
    CueQueueDisposition,
    RealtimeCueDelivery,
    ReleasedCueAudio,
)
from recoverybox.session import (
    DEFAULT_CUE_CATALOG_VERSION,
    ApprovedCuePlaybackAuthorization,
)

PCM = b"\x01\x00\x02\x00"


@dataclass
class _ModeProvider:
    current_mode: SessionMode = SessionMode.ACTIVE_EXERCISE


@dataclass
class _Clock:
    value: float = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass
class _Harness:
    mode: _ModeProvider = field(default_factory=_ModeProvider)
    clock: _Clock = field(default_factory=_Clock)
    transport: MemoryTransport = field(default_factory=MemoryTransport)
    released: list[ReleasedCueAudio] = field(default_factory=list)
    failures: list[CueDeliveryFailure] = field(default_factory=list)
    preemptions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.session = RealtimeSession(transport=self.transport)
        self.delivery = RealtimeCueDelivery(
            session=self.session,
            mode_provider=self.mode,
            on_audio=self.released.append,
            on_failure=self.failures.append,
            on_preempt=lambda: self.preemptions.append("preempt"),
            count_cue_ids={CueId.READY, CueId.HOLD_POSITION},
            clock=self.clock,
        )

    def route(self, raw: dict) -> bool:
        return self.delivery.handle_result(self.session.handle_event(raw))


def _authorization(cue_id: CueId) -> ApprovedCuePlaybackAuthorization:
    cue = DEFAULT_CUE_CATALOG[cue_id.value]
    return ApprovedCuePlaybackAuthorization(
        cue_id=cue_id,
        cue_kind=cue.kind,
        catalog_version=DEFAULT_CUE_CATALOG_VERSION,
        guardian_rule_version="guardian-squat-v1",
        reason_codes=(GuardianReason.LOCAL_CUE_ACCEPTED,),
    )


def _created(response_id: str, event_id: str) -> dict:
    return {
        "type": "response.created",
        "event_id": event_id,
        "response": {"id": response_id, "status": "in_progress"},
    }


def _audio(response_id: str, item_id: str, event_id: str) -> dict:
    return {
        "type": "response.output_audio.delta",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "content_index": 0,
        "delta": base64.b64encode(PCM).decode("ascii"),
    }


def _audio_done(response_id: str, item_id: str, event_id: str) -> dict:
    return {
        "type": "response.output_audio.done",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "content_index": 0,
    }


def _transcript_done(
    response_id: str,
    item_id: str,
    transcript: str,
    event_id: str,
) -> dict:
    return {
        "type": "response.output_audio_transcript.done",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    }


def _response_done(response_id: str, event_id: str, status: str = "completed") -> dict:
    return {
        "type": "response.done",
        "event_id": event_id,
        "response": {"id": response_id, "status": status},
    }


def _complete_exact(harness: _Harness, cue_id: CueId, suffix: str) -> None:
    response_id = f"resp-{suffix}"
    item_id = f"item-{suffix}"
    phrase = DEFAULT_CUE_CATALOG[cue_id.value].spoken_text
    assert harness.route(_created(response_id, f"created-{suffix}"))
    assert harness.route(_audio(response_id, item_id, f"audio-{suffix}"))
    assert harness.route(_audio_done(response_id, item_id, f"audio-done-{suffix}"))
    assert harness.route(_transcript_done(response_id, item_id, phrase, f"transcript-{suffix}"))
    assert harness.released == []
    assert harness.route(_response_done(response_id, f"done-{suffix}"))


def test_enqueue_accepts_only_typed_authorization_and_reuses_session() -> None:
    harness = _Harness()

    result = harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))

    assert result.disposition is CueQueueDisposition.STARTED
    assert len(harness.transport.sent) == 1
    request = harness.transport.sent[0]["response"]
    assert request["conversation"] == "none"
    assert request["metadata"]["cue_id"] == CueId.MOVE_SLOWLY.value
    assert request["tools"] == []
    assert not hasattr(result, "spoken_text")
    with pytest.raises(TypeError, match="ApprovedCuePlaybackAuthorization"):
        harness.delivery.enqueue("Say something else")  # type: ignore[arg-type]


def test_pcm_waits_for_terminal_completed_response_and_reports_latency() -> None:
    harness = _Harness()
    harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))
    harness.clock.advance(0.4)

    _complete_exact(harness, CueId.MOVE_SLOWLY, "one")

    assert len(harness.released) == 1
    released = harness.released[0]
    assert released.authorization.cue_id is CueId.MOVE_SLOWLY
    assert released.pcm16_mono_24khz == PCM
    assert released.response_latency_ms == pytest.approx(400.0)
    assert released.total_latency_ms == pytest.approx(400.0)
    assert harness.delivery.snapshot.released_count == 1


def test_latest_generic_count_leapfrogs_but_never_interrupts_active_safety_cue() -> None:
    harness = _Harness()
    harness.delivery.enqueue(_authorization(CueId.CAMERA_PAUSE))
    correction = harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))
    duplicate = harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))
    first_count = harness.delivery.enqueue(_authorization(CueId.READY))
    newest_count = harness.delivery.enqueue(_authorization(CueId.HOLD_POSITION))

    assert correction.disposition is CueQueueDisposition.QUEUED
    assert duplicate.disposition is CueQueueDisposition.COALESCED
    assert first_count.disposition is CueQueueDisposition.QUEUED
    assert newest_count.disposition is CueQueueDisposition.SUPERSEDED_COUNT
    snapshot = harness.delivery.snapshot
    assert snapshot.active_cue_id is CueId.CAMERA_PAUSE
    assert snapshot.pending_cue_ids == (
        CueId.HOLD_POSITION,
        CueId.MOVE_SLOWLY,
    )
    assert snapshot.superseded_count == 1
    assert len(harness.transport.sent) == 1

    _complete_exact(harness, CueId.CAMERA_PAUSE, "safety")

    assert len(harness.transport.sent) == 2
    assert harness.transport.sent[-1]["response"]["metadata"]["cue_id"] == (
        CueId.HOLD_POSITION.value
    )


def test_ordered_script_survives_check_in_to_active_and_releases_fifo() -> None:
    harness = _Harness(mode=_ModeProvider(SessionMode.CHECK_IN))
    harness.delivery = RealtimeCueDelivery(
        session=harness.session,
        mode_provider=harness.mode,
        on_audio=harness.released.append,
        on_failure=harness.failures.append,
        on_preempt=lambda: harness.preemptions.append("preempt"),
        ordered_cue_ids={
            CueId.SQUAT_SET_INTRO,
            CueId.SQUAT_PERSON_DETECTED,
            CueId.SQUAT_REP_ONE,
            CueId.SQUAT_REP_TWO,
            CueId.SQUAT_REP_THREE,
        },
        allowed_modes={SessionMode.CHECK_IN, SessionMode.ACTIVE_EXERCISE},
        check_in_cue_ids={CueId.SQUAT_SET_INTRO, CueId.SQUAT_PERSON_DETECTED},
        clock=harness.clock,
    )

    harness.delivery.enqueue(_authorization(CueId.SQUAT_SET_INTRO))
    harness.delivery.enqueue(_authorization(CueId.SQUAT_PERSON_DETECTED))
    harness.mode.current_mode = SessionMode.ACTIVE_EXERCISE
    harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_ONE))
    harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_TWO))
    harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_THREE))

    assert harness.delivery.snapshot.pending_cue_ids == (
        CueId.SQUAT_PERSON_DETECTED,
        CueId.SQUAT_REP_ONE,
        CueId.SQUAT_REP_TWO,
        CueId.SQUAT_REP_THREE,
    )
    expected = (
        CueId.SQUAT_SET_INTRO,
        CueId.SQUAT_PERSON_DETECTED,
        CueId.SQUAT_REP_ONE,
        CueId.SQUAT_REP_TWO,
        CueId.SQUAT_REP_THREE,
    )
    for index, cue_id in enumerate(expected):
        assert harness.transport.sent[-1]["response"]["metadata"]["cue_id"] == cue_id.value
        harness.released.clear()
        _complete_exact(harness, cue_id, f"script-{index}")
        assert [release.authorization.cue_id for release in harness.released] == [cue_id]

    assert harness.delivery.snapshot.pending_cue_ids == ()
    assert harness.delivery.snapshot.released_count == len(expected)


def test_ordered_rep_cues_never_expire_while_preceding_responses_finish_fifo() -> None:
    harness = _Harness()
    harness.delivery = RealtimeCueDelivery(
        session=harness.session,
        mode_provider=harness.mode,
        on_audio=harness.released.append,
        on_failure=harness.failures.append,
        ordered_cue_ids={
            CueId.SQUAT_REP_ONE,
            CueId.SQUAT_REP_TWO,
            CueId.SQUAT_REP_THREE,
        },
        config=CueDeliveryConfig(response_timeout_seconds=8.0),
        clock=harness.clock,
    )

    harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_ONE))
    harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_TWO))
    harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_THREE))

    assert harness.delivery.snapshot.pending_cue_ids == (
        CueId.SQUAT_REP_TWO,
        CueId.SQUAT_REP_THREE,
    )
    harness.clock.advance(7.9)
    _complete_exact(harness, CueId.SQUAT_REP_ONE, "slow-rep-one")
    assert harness.transport.sent[-1]["response"]["metadata"]["cue_id"] == (
        CueId.SQUAT_REP_TWO.value
    )

    harness.released.clear()
    harness.clock.advance(7.9)
    _complete_exact(harness, CueId.SQUAT_REP_TWO, "slow-rep-two")

    # Rep three has now waited 15.8 seconds, longer than the removed SCRIPT
    # age limit, but remains the next request and was never counted stale.
    assert harness.transport.sent[-1]["response"]["metadata"]["cue_id"] == (
        CueId.SQUAT_REP_THREE.value
    )
    assert harness.delivery.snapshot.active_cue_id is CueId.SQUAT_REP_THREE
    assert harness.delivery.snapshot.stale_drop_count == 0


def test_script_timeout_fails_closed_and_clears_all_pending_steps() -> None:
    harness = _Harness()
    harness.delivery = RealtimeCueDelivery(
        session=harness.session,
        mode_provider=harness.mode,
        on_audio=harness.released.append,
        on_failure=harness.failures.append,
        on_preempt=lambda: harness.preemptions.append("preempt"),
        ordered_cue_ids={
            CueId.SQUAT_REP_ONE,
            CueId.SQUAT_REP_TWO,
            CueId.SQUAT_REP_THREE,
        },
        config=CueDeliveryConfig(response_timeout_seconds=8.0),
        clock=harness.clock,
    )

    harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_ONE))
    harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_TWO))
    harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_THREE))
    harness.route(_created("resp-script-timeout", "created-script-timeout"))
    harness.clock.advance(8.1)

    assert harness.delivery.expire_stale() == 0

    snapshot = harness.delivery.snapshot
    assert snapshot.active_cue_id is None
    assert snapshot.pending_cue_ids == ()
    assert snapshot.draining_stale_response
    assert snapshot.stale_drop_count == 0
    assert harness.preemptions == ["preempt"]
    assert harness.failures == [
        CueDeliveryFailure(
            ticket_id=1,
            cue_id=CueId.SQUAT_REP_ONE,
            reason=CueDeliveryFailureReason.RESPONSE_TIMEOUT,
            response_id="resp-script-timeout",
        )
    ]

    assert harness.route(
        _response_done(
            "resp-script-timeout",
            "done-script-timeout",
            status="cancelled",
        )
    )
    assert not harness.delivery.snapshot.draining_stale_response
    assert len(harness.transport.sent) == 2  # create, then scoped cancellation


def test_repeated_script_id_is_queued_as_distinct_fifo_work() -> None:
    harness = _Harness()
    harness.delivery = RealtimeCueDelivery(
        session=harness.session,
        mode_provider=harness.mode,
        on_audio=harness.released.append,
        ordered_cue_ids={CueId.SQUAT_REP_ONE},
        clock=harness.clock,
    )

    first = harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_ONE))
    second = harness.delivery.enqueue(_authorization(CueId.SQUAT_REP_ONE))

    assert first.disposition is CueQueueDisposition.STARTED
    assert second.disposition is CueQueueDisposition.QUEUED
    assert second.ticket_id != first.ticket_id
    assert harness.delivery.snapshot.pending_cue_ids == (CueId.SQUAT_REP_ONE,)
    assert harness.delivery.snapshot.coalesced_count == 0


def test_check_in_allowance_rejects_non_scripted_catalog_cues() -> None:
    harness = _Harness(mode=_ModeProvider(SessionMode.CHECK_IN))
    delivery = RealtimeCueDelivery(
        session=harness.session,
        mode_provider=harness.mode,
        on_audio=harness.released.append,
        allowed_modes={SessionMode.CHECK_IN, SessionMode.ACTIVE_EXERCISE},
        check_in_cue_ids={CueId.SQUAT_SET_INTRO, CueId.SQUAT_PERSON_DETECTED},
    )

    with pytest.raises(CueDeliveryError, match="allowed session mode"):
        delivery.enqueue(_authorization(CueId.SQUAT_REP_ONE))

    assert harness.transport.sent == []


def test_completed_response_without_exact_quarantined_audio_fails_closed() -> None:
    harness = _Harness()
    harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))
    harness.delivery.enqueue(_authorization(CueId.KNEE_ALIGNMENT))
    harness.route(_created("resp-bad", "created-bad"))
    harness.route(_audio("resp-bad", "item-bad", "audio-bad"))
    harness.route(_audio_done("resp-bad", "item-bad", "audio-done-bad"))
    harness.route(
        _transcript_done(
            "resp-bad",
            "item-bad",
            "Move slowly and with control!",
            "transcript-bad",
        )
    )

    assert harness.route(_response_done("resp-bad", "done-bad"))

    assert harness.released == []
    assert harness.failures[-1].reason is CueDeliveryFailureReason.QUARANTINE_REJECTED
    assert harness.delivery.snapshot.pending_cue_ids == ()
    assert len(harness.transport.sent) == 1


def test_preemption_scrubs_candidate_and_drains_before_new_request() -> None:
    harness = _Harness()
    harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))
    harness.route(_created("resp-old", "created-old"))
    harness.route(_audio("resp-old", "item-old", "audio-old"))
    harness.route(_audio_done("resp-old", "item-old", "audio-done-old"))
    harness.route(
        _transcript_done(
            "resp-old",
            "item-old",
            DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value].spoken_text,
            "transcript-old",
        )
    )

    harness.delivery.preempt_model_audio()
    queued = harness.delivery.enqueue(_authorization(CueId.KNEE_ALIGNMENT))

    assert queued.disposition is CueQueueDisposition.QUEUED
    assert harness.preemptions == ["preempt"]
    assert harness.released == []
    assert harness.delivery.snapshot.draining_stale_response
    assert harness.route(_response_done("resp-old", "done-old", status="cancelled"))
    assert len(harness.transport.sent) == 3  # old create, cancel, then new create
    assert harness.transport.sent[1] == {
        "type": "response.cancel",
        "response_id": "resp-old",
    }
    assert harness.transport.sent[-1]["response"]["metadata"]["cue_id"] == (
        CueId.KNEE_ALIGNMENT.value
    )
    assert harness.released == []


def test_preempt_before_created_revokes_locally_then_cancels_scoped_id() -> None:
    harness = _Harness()
    harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))

    harness.delivery.preempt_model_audio()

    # A bare response.cancel here could cancel ordinary conversation audio on
    # the shared session, so the pending cue is only tombstoned locally.
    assert len(harness.transport.sent) == 1
    assert harness.session.audio_gate.pending_cancellation_tombstones == 1
    queued = harness.delivery.enqueue(_authorization(CueId.KNEE_ALIGNMENT))
    assert queued.disposition is CueQueueDisposition.QUEUED

    assert harness.route(_created("resp-late", "created-late"))
    assert harness.transport.sent[-1] == {
        "type": "response.cancel",
        "response_id": "resp-late",
    }
    assert harness.route(_response_done("resp-late", "done-late", status="cancelled"))
    assert harness.transport.sent[-1]["type"] == "response.create"
    assert harness.transport.sent[-1]["response"]["metadata"]["cue_id"] == (
        CueId.KNEE_ALIGNMENT.value
    )


def test_mode_change_at_terminal_discards_audio_and_queue() -> None:
    harness = _Harness()
    harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))
    harness.route(_created("resp-mode", "created-mode"))
    harness.route(_audio("resp-mode", "item-mode", "audio-mode"))
    harness.route(_audio_done("resp-mode", "item-mode", "audio-done-mode"))
    harness.route(
        _transcript_done(
            "resp-mode",
            "item-mode",
            DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value].spoken_text,
            "transcript-mode",
        )
    )
    harness.mode.current_mode = SessionMode.PAUSED

    assert harness.route(_response_done("resp-mode", "done-mode"))

    assert harness.released == []
    assert harness.failures[-1].reason is CueDeliveryFailureReason.MODE_CHANGED


def test_stale_pending_count_is_dropped_without_interrupting_active_cue() -> None:
    harness = _Harness()
    harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))
    harness.delivery.enqueue(_authorization(CueId.READY))
    harness.clock.advance(2.1)

    assert harness.delivery.expire_stale() == 1

    snapshot = harness.delivery.snapshot
    assert snapshot.active_cue_id is CueId.MOVE_SLOWLY
    assert snapshot.pending_cue_ids == ()
    assert snapshot.stale_drop_count == 1
    assert len(harness.transport.sent) == 1


def test_response_timeout_cancels_and_requires_terminal_drain() -> None:
    harness = _Harness()
    harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))
    harness.route(_created("resp-timeout", "created-timeout"))
    harness.clock.advance(8.1)

    assert harness.delivery.expire_stale() == 0

    assert harness.preemptions == ["preempt"]
    assert harness.failures[-1].reason is CueDeliveryFailureReason.RESPONSE_TIMEOUT
    assert harness.delivery.snapshot.draining_stale_response
    assert harness.released == []


def test_cues_are_rejected_outside_active_exercise() -> None:
    harness = _Harness(mode=_ModeProvider(SessionMode.PAUSED))

    with pytest.raises(CueDeliveryError, match="ACTIVE_EXERCISE"):
        harness.delivery.enqueue(_authorization(CueId.MOVE_SLOWLY))

    assert harness.transport.sent == []


def test_config_rejects_unbounded_or_nonfinite_timing() -> None:
    with pytest.raises(ValueError, match="max_pending_cues"):
        CueDeliveryConfig(max_pending_cues=0)
    with pytest.raises(ValueError, match="response_timeout_seconds"):
        CueDeliveryConfig(response_timeout_seconds=float("nan"))
