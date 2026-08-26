from __future__ import annotations

import base64
import threading
from dataclasses import replace

import pytest

from recoverybox.core import (
    DEFAULT_CUE_CATALOG,
    CueId,
    GuardianAction,
    GuardianReason,
    SessionMode,
)
from recoverybox.exercise import (
    SquatAnalysis,
    SquatAssessmentIssue,
    SquatEvent,
    SquatEventType,
    SquatPhase,
)
from recoverybox.laptop.squat_session import (
    SESSION_END_POLICY_INSTRUCTIONS,
    SQUAT_REP_CUE_IDS,
    LaptopSquatSession,
    build_single_camera_squat_plan,
    local_cue_request_for_squat_event,
    observation_from_squat_analysis,
)
from recoverybox.realtime import BoundedOrderedTransport, MemoryTransport

PCM = b"\x01\x00\x02\x00"


class _FailingCueTransport(MemoryTransport):
    def send_event(self, event: dict) -> None:
        if event.get("type") == "response.create":
            raise OSError("simulated network failure")
        super().send_event(event)


class _FailingReceiveTransport(MemoryTransport):
    def receive_event(self) -> dict:
        raise OSError("simulated receive failure")


class _BlockingReceiveTransport(MemoryTransport):
    def __init__(self) -> None:
        super().__init__()
        self.block_next = False
        self.receive_entered = threading.Event()
        self.release_receive = threading.Event()

    def receive_event(self) -> dict:
        event = dict(super().receive_event())
        if self.block_next:
            self.block_next = False
            self.receive_entered.set()
            if not self.release_receive.wait(1.0):
                raise TimeoutError("test did not release receive")
        return event


class _BlockingCueSendTransport(MemoryTransport):
    def __init__(self) -> None:
        super().__init__()
        self.cue_send_entered = threading.Event()
        self.release_cue_send = threading.Event()
        self.close_entered = threading.Event()
        self.release_close = threading.Event()

    def send_event(self, event: dict) -> None:
        self.sent.append(dict(event))
        if event.get("type") == "response.create":
            self.cue_send_entered.set()
            self.release_cue_send.wait()

    def close(self) -> None:
        self.close_entered.set()
        self.release_close.wait()
        self.closed = True


def analysis(
    *,
    timestamp_ms: int = 100,
    assessable: bool = True,
    phase: SquatPhase = SquatPhase.STANDING,
    rep_count: int = 0,
    events: tuple[SquatEvent, ...] = (),
    confidence: float = 0.95,
    issues: tuple[SquatAssessmentIssue, ...] = (),
    arms_in_t: bool | None = True,
) -> SquatAnalysis:
    return SquatAnalysis(
        timestamp_ms=timestamp_ms,
        assessable=assessable,
        phase=phase,
        rep_count=rep_count,
        events=events,
        issues=issues,
        confidence=confidence,
        knee_angle_degrees=170.0 if assessable else None,
        arms_in_t=arms_in_t if assessable else None,
    )


def missing_analysis(*, timestamp_ms: int = 100) -> SquatAnalysis:
    return analysis(
        timestamp_ms=timestamp_ms,
        assessable=False,
        phase=SquatPhase.UNKNOWN,
        confidence=0.0,
        issues=(SquatAssessmentIssue.NO_POSE,),
        arms_in_t=None,
    )


def build_app(
    transport: MemoryTransport | None = None,
) -> tuple[LaptopSquatSession, MemoryTransport, list, list[str]]:
    selected_transport = transport or MemoryTransport()
    released: list = []
    preemptions: list[str] = []
    app = LaptopSquatSession(
        transport=selected_transport,
        on_cue_audio=released.append,
        on_audio_preempt=lambda: preemptions.append("preempt"),
    )
    app.start(instructions="Keep this exercise session concise.", voice="marin")
    return app, selected_transport, released, preemptions


def activate(app: LaptopSquatSession) -> None:
    assert app.activate_exercise(analysis())
    assert app.coordinator.current_mode is SessionMode.ACTIVE_EXERCISE


def created(response_id: str) -> dict:
    return {
        "type": "response.created",
        "event_id": f"created-{response_id}",
        "response": {"id": response_id, "status": "in_progress"},
    }


def audio_delta(response_id: str, item_id: str) -> dict:
    return {
        "type": "response.output_audio.delta",
        "event_id": f"audio-{response_id}",
        "response_id": response_id,
        "item_id": item_id,
        "content_index": 0,
        "delta": base64.b64encode(PCM).decode("ascii"),
    }


def audio_done(response_id: str, item_id: str) -> dict:
    return {
        "type": "response.output_audio.done",
        "event_id": f"audio-done-{response_id}",
        "response_id": response_id,
        "item_id": item_id,
        "content_index": 0,
    }


def transcript_done(response_id: str, item_id: str, transcript: str) -> dict:
    return {
        "type": "response.output_audio_transcript.done",
        "event_id": f"transcript-{response_id}",
        "response_id": response_id,
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    }


def response_done(response_id: str, *, output: list[dict] | None = None) -> dict:
    response: dict = {
        "id": response_id,
        "status": "completed",
        "output": [] if output is None else output,
    }
    return {
        "type": "response.done",
        "event_id": f"done-{response_id}",
        "response": response,
    }


def error_event(event_id: str = "error-control") -> dict:
    return {
        "type": "error",
        "event_id": event_id,
        "error": {"message": "simulated provider error"},
    }


def test_start_configures_one_persistent_session_with_only_finish_tool() -> None:
    transport = MemoryTransport()
    app = LaptopSquatSession(
        transport=transport,
        on_cue_audio=lambda _: None,
        on_audio_preempt=lambda: None,
    )

    assert transport.sent == []
    app.start(instructions="Short exercise session.", voice="marin")

    assert len(transport.sent) == 1
    update = transport.sent[0]
    assert update["type"] == "session.update"
    assert [tool["name"] for tool in update["session"]["tools"]] == ["finish_session"]
    assert update["session"]["audio"]["input"]["turn_detection"] is None
    assert SESSION_END_POLICY_INSTRUCTIONS.strip() in update["session"]["instructions"]
    assert app.coordinator.current_mode is SessionMode.IDLE
    assert not transport.closed
    with pytest.raises(RuntimeError, match="already started"):
        app.start(instructions="Again.", voice="marin")


def test_composition_requires_preemption_and_the_reviewed_squat_ten_rep_plan() -> None:
    with pytest.raises(TypeError, match="on_audio_preempt"):
        LaptopSquatSession(
            transport=MemoryTransport(),
            on_cue_audio=lambda _: None,
            on_audio_preempt=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exactly 10"):
        build_single_camera_squat_plan(target_reps=5)
    with pytest.raises(ValueError, match="squat exercise plan"):
        LaptopSquatSession(
            transport=MemoryTransport(),
            on_cue_audio=lambda _: None,
            on_audio_preempt=lambda: None,
            plan=replace(build_single_camera_squat_plan(), exercise_id="lunge"),
        )


def test_startup_missing_pose_stays_idle_until_explicit_standing_activation() -> None:
    app, transport, _, _ = build_app()

    update = app.process_analysis(missing_analysis())

    assert update.mode is SessionMode.IDLE
    assert update.decisions == ()
    assert app.activate_exercise(missing_analysis()) is False
    assert app.activate_exercise(analysis(phase=SquatPhase.DOWN)) is False
    assert app.coordinator.current_mode is SessionMode.IDLE
    assert not transport.closed

    assert app.activate_exercise(analysis(phase=SquatPhase.STANDING)) is True
    assert app.coordinator.current_mode is SessionMode.ACTIVE_EXERCISE


def test_single_camera_observation_is_explicit_and_unassessable_is_not_safe() -> None:
    plan = build_single_camera_squat_plan()
    good = observation_from_squat_analysis(analysis(), plan=plan, pose_age_ms=25)
    missing = observation_from_squat_analysis(missing_analysis(), plan=plan)

    assert good.camera_view_count == 1
    assert good.camera_disagreement_degrees is None
    assert good.pose_age_ms == 25
    assert good.out_of_distribution is False
    assert missing.camera_view_count == 1
    assert missing.out_of_distribution is True


@pytest.mark.parametrize(
    ("rep_count", "cue_id"),
    tuple(enumerate(SQUAT_REP_CUE_IDS, start=1)),
)
def test_rep_events_map_to_fixed_catalog_cue_ids(rep_count: int, cue_id: CueId) -> None:
    request = local_cue_request_for_squat_event(
        SquatEvent(SquatEventType.REP_COMPLETED, rep_count=rep_count)
    )

    assert request is not None
    assert request.cue_id == cue_id.value
    assert DEFAULT_CUE_CATALOG.is_approved(request.cue_id)


def test_rep_and_form_events_reach_guardian_as_typed_cues_not_phrases() -> None:
    app, transport, _, _ = build_app()
    activate(app)
    rep = analysis(
        rep_count=1,
        events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
    )

    rep_result = app.process_analysis(rep)
    arms_result = app.process_analysis(
        analysis(
            timestamp_ms=133,
            rep_count=1,
            events=(SquatEvent(SquatEventType.ARMS_NOT_IN_T),),
            arms_in_t=False,
        )
    )

    assert rep_result.decisions[-1].action is GuardianAction.CUE
    assert rep_result.decisions[-1].cue_id == CueId.SQUAT_REP_ONE.value
    assert GuardianReason.LOCAL_CUE_ACCEPTED in rep_result.decisions[-1].reason_codes
    assert arms_result.decisions[-1].action is GuardianAction.CUE
    assert arms_result.decisions[-1].cue_id == CueId.ARMS_T_SHAPE.value
    assert transport.sent[-1]["type"] == "response.create"
    assert transport.sent[-1]["response"]["metadata"]["cue_id"] == (CueId.SQUAT_REP_ONE.value)
    assert "spoken_text" not in transport.sent[-1]["response"]["metadata"]
    assert app.ended is False
    assert not transport.closed


def test_unassessable_active_pose_pauses_without_auto_resume_or_session_end() -> None:
    app, transport, _, _ = build_app()
    activate(app)

    paused = app.process_analysis(missing_analysis(timestamp_ms=200))
    still_paused = app.process_analysis(analysis(timestamp_ms=233))

    assert paused.decisions[0].action is GuardianAction.PAUSE
    assert GuardianReason.OUT_OF_DISTRIBUTION in paused.decisions[0].reason_codes
    assert paused.mode is SessionMode.PAUSED
    assert still_paused.decisions[0].action is GuardianAction.CONTINUE
    assert still_paused.mode is SessionMode.PAUSED
    assert app.ended is False
    assert not transport.closed

    assert app.resume_after_assessable_pose(analysis(timestamp_ms=266)) is True
    assert app.coordinator.current_mode is SessionMode.ACTIVE_EXERCISE


def test_cue_request_network_failure_pauses_but_local_tracking_and_session_continue() -> None:
    transport = _FailingCueTransport()
    app, _, _, _ = build_app(transport)
    activate(app)
    rep = analysis(
        rep_count=1,
        events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
    )

    failed = app.process_analysis(rep)
    sent_after_failure = len(transport.sent)
    resume = app.resume_after_assessable_pose(analysis(timestamp_ms=120, rep_count=1))
    later = app.process_analysis(analysis(timestamp_ms=133, rep_count=1))

    assert failed.cue_delivery_failed
    assert failed.mode is SessionMode.PAUSED
    assert later.analysis.timestamp_ms == 133
    assert later.mode is SessionMode.PAUSED
    assert app.last_cue_failure is not None
    assert app.realtime_available is False
    assert app.realtime_failure_kind == "CueDeliveryUnavailable"
    assert resume is False
    assert len(transport.sent) == sent_after_failure
    assert app.submit_user_audio_turn(PCM).failure_kind == "RealtimeUnavailable"
    assert app.ended is False
    assert not transport.closed


def test_inconsistent_or_replayed_rep_events_pause_without_speaking_wrong_count() -> None:
    app, transport, _, _ = build_app()
    activate(app)
    sent_before = len(transport.sent)

    inconsistent = app.process_analysis(
        analysis(
            rep_count=1,
            events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=10),),
        )
    )

    assert inconsistent.decisions[0].action is GuardianAction.PAUSE
    assert GuardianReason.OUT_OF_DISTRIBUTION in inconsistent.decisions[0].reason_codes
    assert inconsistent.mode is SessionMode.PAUSED
    assert len(transport.sent) == sent_before

    replay_app, replay_transport, _, _ = build_app()
    activate(replay_app)
    completed = analysis(
        rep_count=1,
        events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
    )
    replay_app.process_analysis(completed)
    create_count = sum(event["type"] == "response.create" for event in replay_transport.sent)

    replay = replay_app.process_analysis(completed)

    assert replay.decisions[0].action is GuardianAction.PAUSE
    assert replay.mode is SessionMode.PAUSED
    assert sum(event["type"] == "response.create" for event in replay_transport.sent) == (
        create_count
    )


def test_single_dispatcher_releases_exact_cue_only_after_response_done() -> None:
    app, transport, released, _ = build_app()
    activate(app)
    app.process_analysis(
        analysis(
            rep_count=1,
            events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
        )
    )
    phrase = DEFAULT_CUE_CATALOG[CueId.SQUAT_REP_ONE.value].spoken_text
    transport.incoming.extend(
        (
            created("cue-1"),
            audio_delta("cue-1", "item-1"),
            audio_done("cue-1", "item-1"),
            transcript_done("cue-1", "item-1", phrase),
            response_done("cue-1"),
        )
    )

    for _ in range(4):
        dispatched = app.pump_once()
        assert dispatched.cue_event_consumed
        assert released == []
    terminal = app.pump_once()

    assert terminal.cue_event_consumed
    assert terminal.failure_kind is None
    assert len(released) == 1
    assert released[0].pcm16_mono_24khz == PCM
    assert app.ended is False
    assert not transport.closed


@pytest.mark.parametrize("include_transcript", (False, True))
def test_incomplete_or_mismatched_cue_pauses_without_pcm_or_closing_session(
    include_transcript: bool,
) -> None:
    app, transport, released, _ = build_app()
    activate(app)
    app.process_analysis(
        analysis(
            rep_count=1,
            events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
        )
    )
    events = [
        created("cue-bad"),
        audio_delta("cue-bad", "item-bad"),
        audio_done("cue-bad", "item-bad"),
    ]
    if include_transcript:
        events.append(transcript_done("cue-bad", "item-bad", "That was one!"))
    events.append(response_done("cue-bad"))
    transport.incoming.extend(events)

    for _ in events:
        app.pump_once()

    assert released == []
    assert app.coordinator.current_mode is SessionMode.PAUSED
    assert app.ended is False
    assert not transport.closed
    later = app.process_analysis(analysis(timestamp_ms=200, rep_count=1))
    assert later.mode is SessionMode.PAUSED


def test_tenth_rep_does_not_end_or_close_the_long_lived_session() -> None:
    app, transport, _, _ = build_app()
    activate(app)

    for rep_count in range(1, 11):
        result = app.process_analysis(
            analysis(
                timestamp_ms=100 + rep_count * 33,
                rep_count=rep_count,
                events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=rep_count),),
            )
        )

    assert result.decisions[-1].cue_id == CueId.SQUAT_REP_TEN.value
    assert app.coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
    assert app.ended is False
    assert not transport.closed


def test_user_audio_turn_requests_tool_only_response_and_never_ordinary_speech() -> None:
    app, transport, _, _ = build_app()
    activate(app)

    submitted = app.submit_user_audio_turn(PCM)

    assert submitted.submitted
    assert [event["type"] for event in transport.sent[-3:]] == [
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "response.create",
    ]
    assert transport.sent[-1]["response"]["output_modalities"] == ["audio"]
    # The local NO_AUDIO authorization, rather than server wording, is the
    # authority that blocks any ordinary response bytes during exercise.
    assert app.realtime_session.audio_gate.open_authorizations == 1
    assert app.ended is False


def test_user_audio_turn_is_rejected_before_append_while_a_cue_is_open() -> None:
    app, transport, _, _ = build_app()
    activate(app)
    app.process_analysis(
        analysis(
            rep_count=1,
            events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
        )
    )
    sent_before = len(transport.sent)

    busy = app.submit_user_audio_turn(PCM)

    assert not busy.submitted
    assert busy.failure_kind == "RealtimeResponseBusy"
    assert len(transport.sent) == sent_before
    assert app.coordinator.current_mode is SessionMode.ACTIVE_EXERCISE


def test_control_turn_defers_pose_cue_until_its_tool_only_response_is_terminal() -> None:
    app, transport, _, _ = build_app()
    activate(app)
    assert app.submit_user_audio_turn(PCM).submitted
    control_create_count = sum(event["type"] == "response.create" for event in transport.sent)

    during_control = app.process_analysis(
        analysis(
            rep_count=1,
            events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
        )
    )

    assert during_control.decisions[-1].action is GuardianAction.CUE
    assert during_control.mode is SessionMode.ACTIVE_EXERCISE
    assert not during_control.cue_delivery_failed
    assert sum(event["type"] == "response.create" for event in transport.sent) == (
        control_create_count
    )

    transport.incoming.extend((created("control-1"), response_done("control-1")))
    app.pump_once()
    app.pump_once()

    assert transport.sent[-1]["type"] == "response.create"
    assert transport.sent[-1]["response"]["metadata"]["cue_id"] == (CueId.SQUAT_REP_ONE.value)
    assert app.coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
    assert sum(event["type"] == "session.update" for event in transport.sent) == 1
    assert not transport.closed


def test_control_error_clears_deferred_cues_and_pauses_without_ending_session() -> None:
    app, transport, _, _ = build_app()
    activate(app)
    assert app.submit_user_audio_turn(PCM).submitted
    app.process_analysis(
        analysis(
            rep_count=1,
            events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
        )
    )
    create_count = sum(event["type"] == "response.create" for event in transport.sent)
    transport.incoming.append(error_event())

    dispatched = app.pump_once()

    assert dispatched.failure_kind == "RealtimeServerError"
    assert app.coordinator.current_mode is SessionMode.PAUSED
    assert app.ended is False
    assert not transport.closed
    assert app.realtime_available is False
    assert app.realtime_failure_kind == "RealtimeServerError"
    assert sum(event["type"] == "response.create" for event in transport.sent) == create_count
    sent_before_retry = len(transport.sent)
    unavailable = app.submit_user_audio_turn(PCM)
    assert not unavailable.submitted
    assert unavailable.failure_kind == "RealtimeUnavailable"
    assert len(transport.sent) == sent_before_retry
    assert app.resume_after_assessable_pose(analysis(timestamp_ms=200, rep_count=1)) is False


def test_response_done_before_control_created_makes_socket_permanently_unavailable() -> None:
    app, transport, _, _ = build_app()
    activate(app)
    assert app.submit_user_audio_turn(PCM).submitted
    transport.incoming.append(response_done("control-out-of-order"))

    dispatched = app.pump_once()

    assert dispatched.failure_kind == "RealtimeProtocolOrderError"
    assert app.realtime_available is False
    assert app.coordinator.current_mode is SessionMode.PAUSED
    assert app.ended is False
    assert not transport.closed


def test_blocking_receive_does_not_hold_request_lock_or_race_terminal_cue() -> None:
    transport = _BlockingReceiveTransport()
    app, _, _, _ = build_app(transport)
    activate(app)
    app.process_analysis(
        analysis(
            rep_count=1,
            events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
        )
    )
    phrase = DEFAULT_CUE_CATALOG[CueId.SQUAT_REP_ONE.value].spoken_text
    transport.incoming.extend(
        (
            created("cue-race"),
            audio_delta("cue-race", "item-race"),
            audio_done("cue-race", "item-race"),
            transcript_done("cue-race", "item-race", phrase),
        )
    )
    for _ in range(4):
        app.pump_once()

    transport.incoming.append(response_done("cue-race"))
    transport.block_next = True
    pump_results: list = []
    pump_thread = threading.Thread(target=lambda: pump_results.append(app.pump_once()))
    pump_thread.start()
    assert transport.receive_entered.wait(0.5)

    turn_results: list = []
    turn_done = threading.Event()

    def submit_turn() -> None:
        turn_results.append(app.submit_user_audio_turn(PCM))
        turn_done.set()

    turn_thread = threading.Thread(target=submit_turn)
    turn_thread.start()
    try:
        # The cue is not terminal until handle_event runs, so this finishes
        # promptly as busy rather than racing a new authorization onto FIFO.
        assert turn_done.wait(0.5)
        assert not turn_results[0].submitted
        assert turn_results[0].failure_kind == "RealtimeResponseBusy"
    finally:
        transport.release_receive.set()
        pump_thread.join(1.0)
        turn_thread.join(1.0)

    assert not pump_thread.is_alive()
    assert pump_results[0].failure_kind is None
    assert app.submit_user_audio_turn(PCM).submitted


def test_ordered_writer_keeps_guardian_pause_nonblocking_and_cancellation_ordered() -> None:
    delegate = _BlockingCueSendTransport()
    transport = BoundedOrderedTransport(delegate, max_pending_events=4)
    app = LaptopSquatSession(
        transport=transport,
        on_cue_audio=lambda _: None,
        on_audio_preempt=lambda: None,
    )
    app.start(instructions="Keep this exercise session concise.", voice="marin")
    activate(app)
    app.process_analysis(
        analysis(
            rep_count=1,
            events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
        )
    )
    assert delegate.cue_send_entered.wait(0.5)
    delegate.incoming.append(created("cue-blocked"))
    assert app.pump_once().cue_event_consumed

    pause_results: list = []
    pause_returned = threading.Event()

    def pause_from_missing_pose() -> None:
        pause_results.append(app.process_analysis(missing_analysis(timestamp_ms=200)))
        pause_returned.set()

    threading.Thread(target=pause_from_missing_pose, daemon=True).start()
    assert pause_returned.wait(0.5)
    assert pause_results[0].mode is SessionMode.PAUSED

    delegate.release_cue_send.set()
    for _ in range(100):
        if [event["type"] for event in delegate.sent][-1:] == ["response.cancel"]:
            break
        threading.Event().wait(0.005)
    assert [event["type"] for event in delegate.sent] == [
        "session.update",
        "response.create",
        "response.cancel",
    ]
    transport.close()
    delegate.release_close.set()


def test_ordered_writer_keeps_physical_stop_nonblocking_during_blocked_send_and_close() -> None:
    delegate = _BlockingCueSendTransport()
    transport = BoundedOrderedTransport(delegate, max_pending_events=4)
    app = LaptopSquatSession(
        transport=transport,
        on_cue_audio=lambda _: None,
        on_audio_preempt=lambda: None,
    )
    app.start(instructions="Keep this exercise session concise.", voice="marin")
    activate(app)
    app.process_analysis(
        analysis(
            rep_count=1,
            events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
        )
    )
    assert delegate.cue_send_entered.wait(0.5)
    delegate.incoming.append(created("cue-stop"))
    assert app.pump_once().cue_event_consumed

    stop_returned = threading.Event()
    threading.Thread(
        target=lambda: (app.request_physical_stop(), stop_returned.set()),
        daemon=True,
    ).start()

    assert stop_returned.wait(0.5)
    assert app.ended
    assert app.coordinator.current_mode is SessionMode.STOPPED
    assert delegate.close_entered.wait(0.5)
    delegate.release_cue_send.set()
    delegate.release_close.set()


def test_async_speaker_failure_and_physical_stop_preempt_without_killing_tracking() -> None:
    app, transport, _, preemptions = build_app()
    activate(app)
    preemptions.clear()

    app.report_speaker_failure()

    assert app.coordinator.current_mode is SessionMode.PAUSED
    assert preemptions
    assert app.realtime_available is False
    assert app.realtime_failure_kind == "SpeakerPlaybackError"
    sent_after_failure = len(transport.sent)
    assert app.resume_after_assessable_pose(analysis(timestamp_ms=180)) is False
    assert app.process_analysis(analysis(timestamp_ms=200)).mode is SessionMode.PAUSED
    assert app.submit_user_audio_turn(PCM).failure_kind == "RealtimeUnavailable"
    assert len(transport.sent) == sent_after_failure
    assert not transport.closed

    app.request_physical_stop()
    assert app.ended
    assert transport.closed
    assert len(preemptions) >= 2


def test_validated_finish_tool_and_physical_stop_are_only_end_paths() -> None:
    app, transport, _, _ = build_app()
    activate(app)
    assert app.submit_user_audio_turn(PCM).submitted
    transport.incoming.extend(
        (
            created("control-1"),
            {
                "type": "response.function_call_arguments.done",
                "event_id": "tool-finish",
                "response_id": "control-1",
                "item_id": "tool-item-1",
                "name": "finish_session",
                "call_id": "finish-call-1",
                "arguments": "{}",
            },
        )
    )

    assert app.pump_once().end_signal is None
    dispatched = app.pump_once()

    assert dispatched.end_signal is not None
    assert dispatched.end_signal.tool_call_id == "finish-call-1"
    assert app.ended
    assert app.coordinator.current_mode is SessionMode.STOPPED
    assert transport.closed

    second_app, second_transport, _, _ = build_app()
    signal = second_app.request_physical_stop()
    assert signal is not None
    assert second_app.ended
    assert second_transport.closed


def test_unsolicited_finish_tool_cannot_end_the_session() -> None:
    app, transport, _, _ = build_app()
    activate(app)
    transport.incoming.append(
        {
            "type": "response.function_call_arguments.done",
            "event_id": "tool-unsolicited",
            "response_id": "unsolicited-response",
            "item_id": "tool-item-unsolicited",
            "name": "finish_session",
            "call_id": "finish-call-unsolicited",
            "arguments": "{}",
        }
    )

    dispatched = app.pump_once()

    assert dispatched.end_signal is None
    assert app.ended is False
    assert not transport.closed


def test_ordinary_events_and_receive_failure_do_not_end_controller() -> None:
    app, transport, _, _ = build_app()
    transport.incoming.append(response_done("ordinary-1"))

    ordinary = app.pump_once()

    assert ordinary.failure_kind is None
    assert ordinary.end_signal is None
    assert app.ended is False
    assert not transport.closed

    failing_transport = _FailingReceiveTransport()
    failing_app, _, _, _ = build_app(failing_transport)
    activate(failing_app)
    failed = failing_app.pump_once()

    assert failed.realtime_result is None
    assert failed.failure_kind == "OSError"
    assert failing_app.coordinator.current_mode is SessionMode.PAUSED
    assert failing_app.ended is False
    assert not failing_transport.closed
    assert failing_app.realtime_available is False
    sent_before = len(failing_transport.sent)
    assert failing_app.submit_user_audio_turn(PCM).failure_kind == "RealtimeUnavailable"
    assert len(failing_transport.sent) == sent_before
    assert failing_app.resume_after_assessable_pose(analysis(timestamp_ms=200)) is False
    assert failing_app.process_analysis(analysis(timestamp_ms=233)).mode is SessionMode.PAUSED


def test_cue_delivery_can_be_intentionally_disabled_without_rewriting_pose_events() -> None:
    transport = MemoryTransport()
    app = LaptopSquatSession(
        transport=transport,
        on_cue_audio=lambda _: None,
        on_audio_preempt=lambda: None,
        cue_delivery_enabled=False,
    )
    app.start(instructions="Camera-only squat session.", voice="marin")

    assert transport.sent == []
    assert app.cue_delivery_enabled is False
    assert app.realtime_available is False
    assert app.realtime_failure_kind is None
    assert app.activate_exercise(analysis())
    result = app.process_analysis(
        analysis(
            rep_count=1,
            events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
        )
    )

    assert result.decisions[-1].action is GuardianAction.CUE
    assert result.decisions[-1].cue_id == CueId.SQUAT_REP_ONE.value
    assert result.mode is SessionMode.ACTIVE_EXERCISE
    assert not result.cue_delivery_failed
    assert transport.sent == []
    assert app.submit_user_audio_turn(PCM).failure_kind == "VoiceDisabled"
    assert app.pump_once().failure_kind == "VoiceDisabled"
