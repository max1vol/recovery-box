from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass, field

import pytest

from recoverybox.core import (
    DEFAULT_CUE_CATALOG,
    CueId,
    CueKind,
    GuardianReason,
    SessionMode,
)
from recoverybox.device import PCM_S16LE_24K_MONO, AudioFormat
from recoverybox.realtime import (
    REALTIME_MODEL,
    REALTIME_WEBSOCKET_URL,
    AudioGateError,
    ConversationMode,
    FunctionTool,
    MemoryTransport,
    ModelAudioGate,
    ModelAudioPolicy,
    RealtimeConversationAdapter,
    RealtimeProtocolError,
    RealtimeSession,
    ToolRegistry,
    ToolValidationError,
    build_assistant_item_truncate,
    build_audio_append,
    build_audio_commit,
    build_response_cancel,
    build_session_update,
    normalize_authorized_text,
    parse_server_event,
    select_realtime_turn_policy,
)
from recoverybox.session import (
    DEFAULT_CUE_CATALOG_VERSION,
    ApprovedCuePlaybackAuthorization,
)

PCM_A = b"\x01\x00\x02\x00"
PCM_B = b"\x03\x00\x04\x00"


@dataclass
class _ModeProvider:
    current_mode: SessionMode = SessionMode.CHECK_IN


class _BarrierModeProvider:
    """Pause pump_once after receive_once and before its adapter lock."""

    def __init__(self, mode: SessionMode) -> None:
        self.mode = mode
        self.after_receive = threading.Barrier(2)
        self.resume_pump = threading.Barrier(2)
        self._armed = False

    @property
    def current_mode(self) -> SessionMode:
        if self._armed:
            self._armed = False
            self.after_receive.wait(timeout=5)
            self.resume_pump.wait(timeout=5)
        return self.mode

    def arm(self) -> None:
        self._armed = True


def _created(response_id: str = "resp-1", *, event_id: str = "evt-created") -> dict:
    return {
        "type": "response.created",
        "event_id": event_id,
        "response": {"id": response_id, "status": "in_progress"},
    }


def _audio(
    pcm: bytes = PCM_A,
    *,
    response_id: str = "resp-1",
    item_id: str = "item-1",
    event_id: str = "evt-audio",
) -> dict:
    return {
        "type": "response.output_audio.delta",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": base64.b64encode(pcm).decode("ascii"),
    }


def _audio_done(
    *, response_id: str = "resp-1", item_id: str = "item-1", event_id: str = "evt-audio-done"
) -> dict:
    return {
        "type": "response.output_audio.done",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
    }


def _transcript_done(
    transcript: str,
    *,
    response_id: str = "resp-1",
    item_id: str = "item-1",
    event_id: str = "evt-transcript-done",
) -> dict:
    return {
        "type": "response.output_audio_transcript.done",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "transcript": transcript,
    }


def _response_done(
    *,
    response_id: str = "resp-1",
    status: str = "completed",
    event_id: str = "evt-response-done",
    output: list | None = None,
) -> dict:
    response = {"id": response_id, "status": status}
    if output is not None:
        response["output"] = output
    return {
        "type": "response.done",
        "event_id": event_id,
        "response": response,
    }


def _registry() -> ToolRegistry:
    return ToolRegistry(
        (
            FunctionTool(
                name="report_symptom",
                description="Report a patient symptom for deterministic review.",
                parameters={
                    "type": "object",
                    "properties": {
                        "symptom": {
                            "type": "string",
                            "enum": ["pain", "dizziness"],
                        },
                        "severity": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10,
                        },
                    },
                    "required": ["symptom", "severity"],
                    "additionalProperties": False,
                },
            ),
        )
    )


def test_response_status_is_allowlisted_before_crossing_protocol_boundary() -> None:
    event = parse_server_event(_response_done(status="provider prose must not escape"))

    assert event.response_status is None


def _prompt_cue_authorization(
    cue_id: CueId = CueId.MOVE_SLOWLY,
    *,
    cue_kind: CueKind | None = None,
    catalog_version: str = DEFAULT_CUE_CATALOG_VERSION,
) -> ApprovedCuePlaybackAuthorization:
    cue = DEFAULT_CUE_CATALOG[cue_id.value]
    return ApprovedCuePlaybackAuthorization(
        cue_id=cue_id,
        cue_kind=cue.kind if cue_kind is None else cue_kind,
        catalog_version=catalog_version,
        guardian_rule_version="guardian-test-v1",
        reason_codes=(GuardianReason.LOCAL_CUE_ACCEPTED,),
    )


def test_session_update_keeps_model_on_url_and_pins_pcm_manual_turns() -> None:
    tool = _registry().wire_tools[0]
    event = build_session_update(
        instructions="Keep the check-in concise.",
        voice="marin",
        tools=(tool,),
    )

    session = event["session"]
    assert REALTIME_MODEL == "gpt-realtime-2.1"
    assert REALTIME_WEBSOCKET_URL.endswith("model=gpt-realtime-2.1")
    # The model is selected when the WebSocket is opened and is immutable in
    # session.update. Sending it here is rejected by the GA Realtime API.
    assert "model" not in session
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["input"] == {
        "format": {"type": "audio/pcm", "rate": 24_000},
        "turn_detection": None,
    }
    assert session["audio"]["output"]["format"] == {
        "type": "audio/pcm",
        "rate": 24_000,
    }
    assert session["tools"] == [tool]


def test_realtime_session_configure_appends_every_catalog_prompt_phrase() -> None:
    transport = MemoryTransport()
    session = RealtimeSession(transport=transport)

    session.configure(instructions="Keep the check-in concise.", voice="marin")

    wire_instructions = transport.sent[0]["session"]["instructions"]
    assert wire_instructions.startswith("Keep the check-in concise.\n\n")
    assert "ACTIVE_EXERCISE PROMPT CUES" in wire_instructions
    for cue in DEFAULT_CUE_CATALOG.values():
        cue_line = f"- {cue.cue_id}: {json.dumps(cue.spoken_text, ensure_ascii=False)}"
        assert wire_instructions.count(cue_line) == 1


def test_manual_turn_event_builders_and_pcm_validation() -> None:
    append = build_audio_append(PCM_A)
    assert append["type"] == "input_audio_buffer.append"
    assert base64.b64decode(append["audio"]) == PCM_A
    assert build_audio_commit() == {"type": "input_audio_buffer.commit"}
    assert build_response_cancel() == {"type": "response.cancel"}
    assert build_response_cancel(response_id="resp-cue") == {
        "type": "response.cancel",
        "response_id": "resp-cue",
    }
    with pytest.raises(RealtimeProtocolError, match="response id must not be blank"):
        build_response_cancel(response_id=" ")
    assert build_assistant_item_truncate(item_id="assistant-item", audio_end_ms=275) == {
        "type": "conversation.item.truncate",
        "item_id": "assistant-item",
        "content_index": 0,
        "audio_end_ms": 275,
    }

    with pytest.raises(RealtimeProtocolError):
        build_audio_append(b"")
    with pytest.raises(RealtimeProtocolError):
        build_audio_append(b"\x00")


@pytest.mark.parametrize("mode", [ConversationMode.CHECK_IN, ConversationMode.POST_SESSION])
def test_explicit_conversational_stream_releases_low_latency_audio(mode) -> None:
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=mode,
        policy=ModelAudioPolicy.CONVERSATIONAL_STREAM,
    )
    assert gate.ingest_raw(_created()) == ()

    released = gate.ingest_raw(_audio())
    assert len(released) == 1
    assert released[0].pcm16_mono_24khz == PCM_A
    assert released[0].complete is False
    assert released[0].policy is ModelAudioPolicy.CONVERSATIONAL_STREAM

    # Server event IDs make a replay idempotent in the low-latency lane.
    assert gate.ingest_raw(_audio()) == ()


def test_conversational_stream_cannot_be_selected_for_active_exercise() -> None:
    gate = ModelAudioGate()
    with pytest.raises(AudioGateError, match="ACTIVE_EXERCISE"):
        gate.authorize_next_response(
            mode=ConversationMode.ACTIVE_EXERCISE,
            policy=ModelAudioPolicy.CONVERSATIONAL_STREAM,
        )


def test_active_exercise_no_audio_policy_blocks_every_model_audio_event() -> None:
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.ACTIVE_EXERCISE,
        policy=ModelAudioPolicy.NO_AUDIO,
    )
    gate.ingest_raw(_created())

    assert gate.ingest_raw(_audio()) == ()
    assert gate.ingest_raw(_audio_done()) == ()
    assert gate.ingest_raw(_transcript_done("A model-generated sentence.")) == ()
    assert gate.blocked_audio_bytes == len(PCM_A)

    with pytest.raises(AudioGateError, match="NO_AUDIO cannot authorize model text"):
        gate.authorize_next_response(
            mode=ConversationMode.ACTIVE_EXERCISE,
            policy=ModelAudioPolicy.NO_AUDIO,
            authorized_text="A model sentence must never become a local cue.",
        )


def test_active_prompt_cue_quarantine_releases_only_after_exact_transcript() -> None:
    cue = DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value]
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.ACTIVE_EXERCISE,
        policy=ModelAudioPolicy.PROMPT_CUE_QUARANTINE,
        authorized_text=cue.spoken_text,
    )
    assert gate.ingest_raw(_created()) == ()
    assert gate.ingest_raw(_audio(PCM_A, event_id="evt-prompt-a1")) == ()
    assert gate.ingest_raw(_audio(PCM_B, event_id="evt-prompt-a2")) == ()
    assert gate.ingest_raw(_audio_done(event_id="evt-prompt-audio-done")) == ()

    assert (
        gate.ingest_raw(
            _transcript_done(
                cue.spoken_text,
                event_id="evt-prompt-transcript-done",
            )
        )
        == ()
    )
    assert gate.quarantined_audio_bytes == len(PCM_A + PCM_B)

    # Content completion is insufficient for an active-exercise cue.  The
    # entire response must complete before the single matching content is
    # released atomically.
    released = gate.ingest_raw(_response_done())

    assert len(released) == 1
    assert released[0].pcm16_mono_24khz == PCM_A + PCM_B
    assert released[0].policy is ModelAudioPolicy.PROMPT_CUE_QUARANTINE
    assert released[0].complete is True
    assert gate.quarantined_audio_bytes == 0

    # Both an identical replay and a fresh terminal event are fail-closed after
    # the atomic release.
    assert gate.ingest_raw(_response_done()) == ()
    assert gate.ingest_raw(_response_done(event_id="evt-response-done-replay")) == ()


@pytest.mark.parametrize("status", ["failed", "cancelled", "incomplete"])
def test_active_prompt_cue_quarantine_discards_noncompleted_response(status: str) -> None:
    cue = DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value]
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.ACTIVE_EXERCISE,
        policy=ModelAudioPolicy.PROMPT_CUE_QUARANTINE,
        authorized_text=cue.spoken_text,
    )
    gate.ingest_raw(_created())
    gate.ingest_raw(_audio())
    gate.ingest_raw(_audio_done())
    assert gate.ingest_raw(_transcript_done(cue.spoken_text)) == ()

    assert gate.ingest_raw(_response_done(status=status)) == ()
    assert gate.blocked_audio_bytes == len(PCM_A)
    assert gate.quarantined_audio_bytes == 0
    assert (
        gate.ingest_raw(_response_done(status="completed", event_id="evt-late-completed-response"))
        == ()
    )


def test_active_prompt_cue_quarantine_rejects_multiple_audio_contents() -> None:
    cue = DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value]
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.ACTIVE_EXERCISE,
        policy=ModelAudioPolicy.PROMPT_CUE_QUARANTINE,
        authorized_text=cue.spoken_text,
    )
    gate.ingest_raw(_created())

    gate.ingest_raw(_audio(PCM_A, item_id="item-1", event_id="evt-content-1-audio"))
    gate.ingest_raw(_audio_done(item_id="item-1", event_id="evt-content-1-audio-done"))
    gate.ingest_raw(
        _transcript_done(
            cue.spoken_text,
            item_id="item-1",
            event_id="evt-content-1-transcript",
        )
    )
    gate.ingest_raw(_audio(PCM_B, item_id="item-2", event_id="evt-content-2-audio"))
    gate.ingest_raw(_audio_done(item_id="item-2", event_id="evt-content-2-audio-done"))
    gate.ingest_raw(
        _transcript_done(
            cue.spoken_text,
            item_id="item-2",
            event_id="evt-content-2-transcript",
        )
    )

    assert gate.ingest_raw(_response_done()) == ()
    assert gate.blocked_audio_bytes == len(PCM_A + PCM_B)
    assert gate.quarantined_audio_bytes == 0


def test_active_prompt_cue_quarantine_rejects_audio_after_content_done() -> None:
    cue = DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value]
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.ACTIVE_EXERCISE,
        policy=ModelAudioPolicy.PROMPT_CUE_QUARANTINE,
        authorized_text=cue.spoken_text,
    )
    gate.ingest_raw(_created())
    gate.ingest_raw(_audio(PCM_A))
    gate.ingest_raw(_audio_done())
    gate.ingest_raw(_transcript_done(cue.spoken_text))

    assert gate.ingest_raw(_audio(PCM_B, event_id="evt-audio-after-done")) == ()
    assert gate.ingest_raw(_response_done()) == ()
    assert gate.blocked_audio_bytes == len(PCM_A + PCM_B)


@pytest.mark.parametrize("missing", ["audio_done", "transcript_done"])
def test_active_prompt_cue_quarantine_discards_incomplete_content(missing: str) -> None:
    cue = DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value]
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.ACTIVE_EXERCISE,
        policy=ModelAudioPolicy.PROMPT_CUE_QUARANTINE,
        authorized_text=cue.spoken_text,
    )
    gate.ingest_raw(_created())
    gate.ingest_raw(_audio())
    if missing != "audio_done":
        gate.ingest_raw(_audio_done())
    if missing != "transcript_done":
        gate.ingest_raw(_transcript_done(cue.spoken_text))

    assert gate.ingest_raw(_response_done()) == ()
    assert gate.blocked_audio_bytes == len(PCM_A)
    assert gate.quarantined_audio_bytes == 0


@pytest.mark.parametrize(
    "near_match",
    (
        " Move slowly and with control.",
        "Move  slowly and with control.",
        "Move slowly and with control.\n",
        "move slowly and with control.",
        "Move slowly and with control!",
        "Move very slowly and with control.",
        "Move slowly, and with control.",
    ),
)
def test_active_prompt_cue_quarantine_near_matches_never_release(
    near_match: str,
) -> None:
    cue = DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value]
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.ACTIVE_EXERCISE,
        policy=ModelAudioPolicy.PROMPT_CUE_QUARANTINE,
        authorized_text=cue.spoken_text,
    )
    gate.ingest_raw(_created())
    gate.ingest_raw(_audio())
    gate.ingest_raw(_audio_done())

    assert gate.ingest_raw(_transcript_done(near_match)) == ()
    assert gate.blocked_audio_bytes == len(PCM_A)
    assert (
        gate.ingest_raw(_transcript_done(cue.spoken_text, event_id="evt-prompt-late-exact")) == ()
    )


def test_strict_quarantine_is_complete_and_exact_before_release() -> None:
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.CHECK_IN,
        authorized_text="You're ready for today.",
    )
    gate.ingest_raw(_created())

    assert gate.ingest_raw(_audio(PCM_A, event_id="evt-a1")) == ()
    assert gate.ingest_raw(_audio(PCM_B, event_id="evt-a2")) == ()
    assert gate.quarantined_audio_bytes == len(PCM_A + PCM_B)
    assert gate.ingest_raw(_audio_done()) == ()

    released = gate.ingest_raw(_transcript_done("You're ready for today."))
    assert len(released) == 1
    assert released[0].pcm16_mono_24khz == PCM_A + PCM_B
    assert released[0].complete is True
    assert released[0].policy is ModelAudioPolicy.TRANSCRIPT_QUARANTINE
    assert gate.quarantined_audio_bytes == 0


@pytest.mark.parametrize(
    "actual",
    [
        "you are ready.",
        "You are ready!",
        "You are nearly ready.",
        "“You are ready.”",
    ],
)
def test_strict_quarantine_rejects_near_matches_and_cannot_be_replayed(actual) -> None:
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.POST_SESSION,
        authorized_text="You are ready.",
    )
    gate.ingest_raw(_created())
    gate.ingest_raw(_audio())
    gate.ingest_raw(_audio_done())

    assert gate.ingest_raw(_transcript_done(actual)) == ()
    assert gate.blocked_audio_bytes == len(PCM_A)
    # A later exact transcript with a fresh event ID cannot reopen denied audio.
    assert gate.ingest_raw(_transcript_done("You are ready.", event_id="evt-late-exact")) == ()


def test_terminal_response_closes_incomplete_quarantine_against_late_events() -> None:
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.CHECK_IN,
        authorized_text="Approved.",
    )
    gate.ingest_raw(_created())
    gate.ingest_raw(_audio())
    gate.ingest_raw(_response_done())

    assert gate.blocked_audio_bytes == len(PCM_A)
    assert gate.ingest_raw(_audio_done(event_id="evt-late-audio-done")) == ()
    assert gate.ingest_raw(_transcript_done("Approved.", event_id="evt-late-transcript")) == ()


def test_unbound_response_audio_fails_closed() -> None:
    gate = ModelAudioGate()
    assert gate.ingest_raw(_created(response_id="unknown")) == ()
    assert gate.ingest_raw(_audio(response_id="unknown", event_id="evt-unknown-audio")) == ()
    assert gate.blocked_audio_bytes == len(PCM_A)


def test_cancelled_pending_response_cannot_steal_next_turn_authorization() -> None:
    """Reproduce canceled turn 1 / new turn 2 / late old response attack."""

    transport = MemoryTransport()
    session = RealtimeSession(transport=transport)
    session.request_response(
        mode=ConversationMode.CHECK_IN,
        policy=ModelAudioPolicy.CONVERSATIONAL_STREAM,
    )
    session.cancel_response()
    assert session.audio_gate.pending_authorizations == 0
    assert session.audio_gate.pending_cancellation_tombstones == 1

    session.request_response(
        mode=ConversationMode.CHECK_IN,
        policy=ModelAudioPolicy.CONVERSATIONAL_STREAM,
    )
    assert session.audio_gate.pending_authorizations == 1

    old_created = session.handle_event(_created(response_id="resp-old", event_id="evt-old-created"))
    assert old_created.response_authorized is False
    assert session.audio_gate.pending_cancellation_tombstones == 0
    assert session.audio_gate.pending_authorizations == 1
    assert session.current_response_id is None

    old_audio = session.handle_event(
        _audio(
            PCM_A,
            response_id="resp-old",
            item_id="item-old",
            event_id="evt-old-audio",
        )
    )
    assert old_audio.released_audio == ()
    assert session.audio_gate.pending_authorizations == 1

    new_created = session.handle_event(_created(response_id="resp-new", event_id="evt-new-created"))
    assert new_created.response_authorized is True
    new_audio = session.handle_event(
        _audio(
            PCM_B,
            response_id="resp-new",
            item_id="item-new",
            event_id="evt-new-audio",
        )
    )
    assert [released.pcm16_mono_24khz for released in new_audio.released_audio] == [PCM_B]


def test_two_live_response_authorizations_cannot_be_queued() -> None:
    transport = MemoryTransport()
    session = RealtimeSession(transport=transport)
    session.request_response(
        mode=ConversationMode.CHECK_IN,
        policy=ModelAudioPolicy.CONVERSATIONAL_STREAM,
    )

    with pytest.raises(AudioGateError, match="already awaiting"):
        session.request_response(
            mode=ConversationMode.POST_SESSION,
            policy=ModelAudioPolicy.CONVERSATIONAL_STREAM,
        )
    assert [event["type"] for event in transport.sent] == ["response.create"]


@pytest.mark.parametrize("invalid_authorization", (None, "move_slowly", CueId.MOVE_SLOWLY))
def test_prompt_cue_request_accepts_only_typed_guardian_authorization(
    invalid_authorization: object,
) -> None:
    transport = MemoryTransport()
    session = RealtimeSession(transport=transport)

    with pytest.raises(
        TypeError,
        match="authorization must be ApprovedCuePlaybackAuthorization",
    ):
        session.request_approved_prompt_cue(invalid_authorization)  # type: ignore[arg-type]

    assert transport.sent == []
    assert session.audio_gate.open_authorizations == 0


def test_prompt_cue_request_is_isolated_tool_free_and_exactly_instructed() -> None:
    transport = MemoryTransport()
    session = RealtimeSession(transport=transport, tools=_registry())
    authorization = _prompt_cue_authorization()

    session.request_approved_prompt_cue(authorization)

    assert transport.sent == [
        {
            "type": "response.create",
            "response": {
                "output_modalities": ["audio"],
                "instructions": (
                    'The local Guardian selected cue ID "move_slowly". '
                    "Speak exactly the following approved phrase and nothing else. "
                    "Do not add, remove, combine, or paraphrase any words: "
                    '"Move slowly and with control."'
                ),
                "conversation": "none",
                "input": [],
                "tools": [],
                "tool_choice": "none",
                "reasoning": {"effort": "minimal"},
                "max_output_tokens": 256,
                "metadata": {
                    "recoverybox_lane": "prompt_cue",
                    "cue_id": "move_slowly",
                    "catalog_version": DEFAULT_CUE_CATALOG_VERSION,
                },
            },
        }
    ]
    assert session.audio_gate.pending_authorizations == 1


def test_realtime_error_parser_discards_server_prose_and_keeps_safe_codes() -> None:
    secret = "sk-live-must-not-escape"

    event = parse_server_event(
        {
            "type": "error",
            "event_id": "evt-error",
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_api_key",
                "message": f"Incorrect API key provided: {secret}",
            },
        }
    )

    assert event.kind.value == "error"
    assert event.error_type == "invalid_request_error"
    assert event.error_code == "invalid_api_key"
    assert event.error_message is None
    assert secret not in repr(event)


def test_realtime_error_parser_drops_unrecognized_identifiers() -> None:
    secret = "sk-live-must-not-escape"

    event = parse_server_event(
        {
            "type": "error",
            "event_id": "evt-error",
            "error": {
                "type": secret,
                "code": secret,
                "message": secret,
            },
        }
    )

    assert event.error_type is None
    assert event.error_code is None
    assert secret not in repr(event)


def test_generic_response_request_cannot_acquire_prompt_cue_lane() -> None:
    transport = MemoryTransport()
    session = RealtimeSession(transport=transport)
    phrase = DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value].spoken_text

    with pytest.raises(
        AudioGateError,
        match="prompt cue responses require request_approved_prompt_cue",
    ):
        session.request_response(
            mode=ConversationMode.ACTIVE_EXERCISE,
            policy=ModelAudioPolicy.PROMPT_CUE_QUARANTINE,
            authorized_text=phrase,
            instructions=f"Say only: {phrase}",
        )

    assert transport.sent == []
    assert session.audio_gate.open_authorizations == 0


def test_prompt_cue_request_rejects_catalog_version_mismatch() -> None:
    transport = MemoryTransport()
    session = RealtimeSession(transport=transport)

    with pytest.raises(AudioGateError, match="catalog version does not match"):
        session.request_approved_prompt_cue(
            _prompt_cue_authorization(catalog_version="prompt-cues-stale")
        )

    assert transport.sent == []
    assert session.audio_gate.open_authorizations == 0


def test_prompt_cue_request_rejects_catalog_kind_mismatch() -> None:
    transport = MemoryTransport()
    session = RealtimeSession(transport=transport)

    with pytest.raises(AudioGateError, match="kind does not match"):
        session.request_approved_prompt_cue(_prompt_cue_authorization(cue_kind=CueKind.INSTRUCTION))

    assert transport.sent == []
    assert session.audio_gate.open_authorizations == 0


def test_authorized_text_compatibility_helper_preserves_every_code_point() -> None:
    text = " Cafe\u0301\n  ready. "
    assert normalize_authorized_text(text) == text
    assert normalize_authorized_text("Cafe\u0301") != normalize_authorized_text("Café")


@pytest.mark.parametrize(
    ("authorized", "transcript"),
    (
        ("Café.", "Cafe\u0301."),
        ("Move slowly.", " Move slowly."),
        ("Move slowly.", "Move  slowly."),
        ("Move slowly.", "Move slowly.\n"),
        ("Move slowly.", "move slowly."),
        ("Move slowly.", "Move slowly!"),
    ),
)
def test_quarantine_rejects_every_nonliteral_transcript(
    authorized: str,
    transcript: str,
) -> None:
    gate = ModelAudioGate()
    gate.authorize_next_response(
        mode=ConversationMode.ACTIVE_EXERCISE,
        policy=ModelAudioPolicy.PROMPT_CUE_QUARANTINE,
        authorized_text=authorized,
    )
    gate.ingest_raw(_created())
    gate.ingest_raw(_audio())
    gate.ingest_raw(_audio_done())

    assert gate.ingest_raw(_transcript_done(transcript)) == ()
    assert gate.ingest_raw(_response_done()) == ()
    assert gate.blocked_audio_bytes == len(PCM_A)


def test_tool_calls_are_exposed_only_after_local_schema_validation() -> None:
    session = RealtimeSession(transport=MemoryTransport(), tools=_registry())
    valid = {
        "type": "response.function_call_arguments.done",
        "event_id": "evt-tool",
        "response_id": "resp-tools",
        "item_id": "item-tools",
        "output_index": 0,
        "call_id": "call-1",
        "name": "report_symptom",
        "arguments": '{"symptom":"pain","severity":7}',
    }
    result = session.handle_event(valid)
    assert result.rejected_tool_call is None
    assert result.validated_tool_calls[0].arguments == {
        "symptom": "pain",
        "severity": 7,
    }

    invalid = dict(valid)
    invalid.update(
        event_id="evt-tool-invalid",
        call_id="call-2",
        arguments='{"symptom":"pain","severity":7,"command":"unsafe-value"}',
    )
    rejected = session.handle_event(invalid)
    assert rejected.validated_tool_calls == ()
    assert rejected.rejected_tool_call is not None
    assert "unsafe-value" not in rejected.rejected_tool_call.reason


@pytest.mark.parametrize(
    "arguments",
    [
        '{"symptom":"pain","severity":true}',
        '{"symptom":"pain","severity":7,"severity":2}',
        '{"symptom":"unknown","severity":2}',
        "[]",
        "not-json",
    ],
)
def test_malformed_tool_arguments_never_become_calls(arguments) -> None:
    session = RealtimeSession(transport=MemoryTransport(), tools=_registry())
    event = {
        "type": "response.function_call_arguments.done",
        "event_id": f"evt-{len(arguments)}-{arguments[:1]}",
        "response_id": "resp-tools",
        "item_id": "item-tools",
        "call_id": "call-invalid",
        "name": "report_symptom",
        "arguments": arguments,
    }
    result = session.handle_event(event)
    assert result.validated_tool_calls == ()
    assert result.rejected_tool_call is not None


def test_completed_response_tool_output_uses_the_same_validator() -> None:
    session = RealtimeSession(transport=MemoryTransport(), tools=_registry())
    item = {
        "type": "function_call",
        "call_id": "call-response-done",
        "name": "report_symptom",
        "arguments": '{"symptom":"dizziness","severity":4}',
    }
    result = session.handle_event(_response_done(output=[item]))
    assert result.validated_tool_calls[0].arguments["symptom"] == "dizziness"

    cancelled = session.handle_event(
        _response_done(
            response_id="resp-cancelled",
            status="cancelled",
            event_id="evt-cancelled-done",
            output=[item],
        )
    )
    assert cancelled.validated_tool_calls == ()


def test_tool_call_id_is_deduplicated_across_completion_event_shapes() -> None:
    session = RealtimeSession(transport=MemoryTransport(), tools=_registry())
    arguments_done = {
        "type": "response.function_call_arguments.done",
        "event_id": "evt-arguments-done",
        "response_id": "resp-tools",
        "item_id": "item-tools",
        "call_id": "call-shared",
        "name": "report_symptom",
        "arguments": '{"symptom":"pain","severity":6}',
    }
    assert [call.call_id for call in session.handle_event(arguments_done).validated_tool_calls] == [
        "call-shared"
    ]

    repeated_item = {
        "type": "function_call",
        "call_id": "call-shared",
        "name": "report_symptom",
        "arguments": '{"symptom":"pain","severity":6}',
    }
    repeated = session.handle_event(
        _response_done(
            response_id="resp-tools",
            event_id="evt-tools-response-done",
            output=[repeated_item],
        )
    )
    assert repeated.validated_tool_calls == ()


def test_tool_output_followup_is_validated_authorized_and_ordered() -> None:
    transport = MemoryTransport()
    session = RealtimeSession(transport=transport, tools=_registry())
    completed_call = {
        "type": "response.function_call_arguments.done",
        "event_id": "evt-tool-for-output",
        "response_id": "resp-tools",
        "item_id": "item-tools",
        "call_id": "call-output",
        "name": "report_symptom",
        "arguments": '{"symptom":"dizziness","severity":3}',
    }
    session.handle_event(completed_call)

    session.submit_tool_output_and_request(
        call_id="call-output",
        output={"recorded": True},
        mode=ConversationMode.CHECK_IN,
        policy=ModelAudioPolicy.TRANSCRIPT_QUARANTINE,
        authorized_text="I recorded that symptom.",
    )

    assert [event["type"] for event in transport.sent] == [
        "conversation.item.create",
        "response.create",
    ]
    assert transport.sent[0]["item"] == {
        "type": "function_call_output",
        "call_id": "call-output",
        "output": '{"recorded":true}',
    }
    assert session.audio_gate.pending_authorizations == 1

    created = session.handle_event(
        _created(response_id="resp-followup", event_id="evt-followup-created")
    )
    assert created.response_authorized is True
    assert (
        session.handle_event(
            _audio(
                response_id="resp-followup",
                item_id="item-followup",
                event_id="evt-followup-audio",
            )
        ).released_audio
        == ()
    )
    session.handle_event(
        _audio_done(
            response_id="resp-followup",
            item_id="item-followup",
            event_id="evt-followup-audio-done",
        )
    )
    released = session.handle_event(
        _transcript_done(
            "I recorded that symptom.",
            response_id="resp-followup",
            item_id="item-followup",
            event_id="evt-followup-transcript",
        )
    )
    assert [audio.pcm16_mono_24khz for audio in released.released_audio] == [PCM_A]

    with pytest.raises(ToolValidationError, match="already submitted"):
        session.submit_tool_output(call_id="call-output", output={"recorded": True})


def test_tool_output_rejects_call_id_that_was_not_locally_validated() -> None:
    transport = MemoryTransport()
    session = RealtimeSession(transport=transport, tools=_registry())

    with pytest.raises(ToolValidationError, match="not locally validated"):
        session.submit_tool_output_and_request(
            call_id="invented-call",
            output={"recorded": True},
            mode=ConversationMode.ACTIVE_EXERCISE,
            policy=ModelAudioPolicy.NO_AUDIO,
        )
    assert transport.sent == []


def test_cancel_then_truncate_closes_gate_before_stale_audio() -> None:
    transport = MemoryTransport()
    session = RealtimeSession(transport=transport)
    session.request_response(
        mode=ConversationMode.CHECK_IN,
        policy=ModelAudioPolicy.CONVERSATIONAL_STREAM,
    )
    session.handle_event(_created())
    session.handle_event(_audio())

    session.cancel_and_truncate(played_audio_ms=125)
    assert transport.sent[-2:] == [
        {"type": "response.cancel", "response_id": "resp-1"},
        {
            "type": "conversation.item.truncate",
            "item_id": "item-1",
            "content_index": 0,
            "audio_end_ms": 125,
        },
    ]
    assert (
        session.handle_event(_audio(PCM_B, event_id="evt-stale-after-cancel")).released_audio == ()
    )


def test_malformed_output_audio_never_reaches_the_gate() -> None:
    event = _audio()
    event["delta"] = "not base64!"
    with pytest.raises(RealtimeProtocolError):
        parse_server_event(event)


@dataclass
class _Sink:
    events: list[tuple] = field(default_factory=list)

    def on_response_started(self, *, turn_id: str, response_id: str) -> None:
        self.events.append(("started", turn_id, response_id))

    def on_response_audio(
        self,
        *,
        turn_id: str,
        response_id: str,
        item_id: str,
        pcm: bytes,
    ) -> None:
        self.events.append(("audio", turn_id, response_id, item_id, pcm))

    def on_response_done(self, *, turn_id: str, response_id: str | None) -> None:
        self.events.append(("done", turn_id, response_id))

    def on_response_error(
        self,
        *,
        turn_id: str,
        response_id: str | None,
        error: Exception,
    ) -> None:
        self.events.append(("error", turn_id, response_id, str(error)))


def test_device_adapter_composes_button_turn_and_stream_callbacks() -> None:
    transport = MemoryTransport(
        incoming=(
            _created(),
            _audio(),
            _response_done(),
        )
    )
    session = RealtimeSession(transport=transport)
    sink = _Sink()
    adapter = RealtimeConversationAdapter(
        session=session,
        sink=sink,
        mode_provider=_ModeProvider(),
    )

    turn_id = adapter.send_audio_turn(PCM_A, audio_format=PCM_S16LE_24K_MONO)
    assert turn_id == "realtime-turn-1"
    assert [event["type"] for event in transport.sent] == [
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "response.create",
    ]
    assert transport.sent[-1]["response"]["output_modalities"] == ["audio"]

    adapter.pump_once()
    adapter.pump_once()
    adapter.pump_once()
    assert sink.events == [
        ("started", turn_id, "resp-1"),
        ("audio", turn_id, "resp-1", "item-1", PCM_A),
        ("done", turn_id, "resp-1"),
    ]


def test_device_adapter_cancel_and_item_truncation_order() -> None:
    transport = MemoryTransport(incoming=(_created(), _audio()))
    session = RealtimeSession(transport=transport)
    adapter = RealtimeConversationAdapter(
        session=session,
        sink=_Sink(),
        mode_provider=_ModeProvider(),
    )
    turn_id = adapter.send_audio_turn(PCM_A, audio_format=PCM_S16LE_24K_MONO)
    adapter.pump_once()
    adapter.pump_once()

    adapter.cancel_response(turn_id, "resp-1")
    adapter.truncate_assistant("item-1", 80)
    assert transport.sent[-2:] == [
        {"type": "response.cancel", "response_id": "resp-1"},
        {
            "type": "conversation.item.truncate",
            "item_id": "item-1",
            "content_index": 0,
            "audio_end_ms": 80,
        },
    ]


def test_device_adapter_can_dispatch_results_from_one_shared_socket_reader() -> None:
    transport = MemoryTransport(incoming=(_created(), _audio(), _response_done()))
    session = RealtimeSession(transport=transport)
    sink = _Sink()
    adapter = RealtimeConversationAdapter(
        session=session,
        sink=sink,
        mode_provider=_ModeProvider(),
    )
    turn_id = adapter.send_audio_turn(PCM_A, audio_format=PCM_S16LE_24K_MONO)

    for _ in range(3):
        adapter.handle_result(session.receive_once())

    assert sink.events == [
        ("started", turn_id, "resp-1"),
        ("audio", turn_id, "resp-1", "item-1", PCM_A),
        ("done", turn_id, "resp-1"),
    ]


def test_device_adapter_cancel_while_waiting_revokes_pending_audio_gate() -> None:
    transport = MemoryTransport(incoming=(_created(), _audio()))
    session = RealtimeSession(transport=transport)
    sink = _Sink()
    adapter = RealtimeConversationAdapter(
        session=session,
        sink=sink,
        mode_provider=_ModeProvider(),
    )
    turn_id = adapter.send_audio_turn(PCM_A, audio_format=PCM_S16LE_24K_MONO)

    adapter.cancel_response(turn_id, None)
    created_result = adapter.pump_once()
    audio_result = adapter.pump_once()

    assert created_result.released_audio == ()
    assert audio_result.released_audio == ()
    assert sink.events == []
    assert transport.sent[-1] == {"type": "response.cancel"}


def test_device_adapter_never_opens_speaker_for_unauthorized_late_response() -> None:
    transport = MemoryTransport(
        incoming=(
            _created(response_id="resp-old", event_id="evt-adapter-old-created"),
            _created(response_id="resp-new", event_id="evt-adapter-new-created"),
            _audio(
                PCM_B,
                response_id="resp-new",
                item_id="item-new",
                event_id="evt-adapter-new-audio",
            ),
        )
    )
    sink = _Sink()
    adapter = RealtimeConversationAdapter(
        session=RealtimeSession(transport=transport),
        sink=sink,
        mode_provider=_ModeProvider(),
    )

    old_turn = adapter.send_audio_turn(PCM_A, audio_format=PCM_S16LE_24K_MONO)
    adapter.cancel_response(old_turn, None)
    new_turn = adapter.send_audio_turn(PCM_A, audio_format=PCM_S16LE_24K_MONO)

    old_created = adapter.pump_once()
    new_created = adapter.pump_once()
    adapter.pump_once()

    assert old_created.response_authorized is False
    assert new_created.response_authorized is True
    assert sink.events == [
        ("started", new_turn, "resp-new"),
        ("audio", new_turn, "resp-new", "item-new", PCM_B),
    ]


def test_device_adapter_rejects_wrong_audio_format_and_ends_cleanly() -> None:
    transport = MemoryTransport()
    adapter = RealtimeConversationAdapter(
        session=RealtimeSession(transport=transport),
        sink=_Sink(),
        mode_provider=_ModeProvider(),
    )
    with pytest.raises(ValueError, match="24 kHz"):
        adapter.send_audio_turn(
            PCM_A,
            audio_format=AudioFormat(sample_rate_hz=16_000),
        )

    adapter.clear_and_end()
    assert transport.sent[-1] == {"type": "input_audio_buffer.clear"}
    assert transport.closed is True
    with pytest.raises(RuntimeError, match="ended"):
        adapter.send_audio_turn(PCM_A, audio_format=PCM_S16LE_24K_MONO)


@pytest.mark.parametrize(
    ("mode", "conversation_mode", "audio_policy"),
    (
        (
            SessionMode.CHECK_IN,
            ConversationMode.CHECK_IN,
            ModelAudioPolicy.CONVERSATIONAL_STREAM,
        ),
        (
            SessionMode.COMPLETE,
            ConversationMode.POST_SESSION,
            ModelAudioPolicy.CONVERSATIONAL_STREAM,
        ),
        (
            SessionMode.IDLE,
            ConversationMode.ACTIVE_EXERCISE,
            ModelAudioPolicy.NO_AUDIO,
        ),
        (
            SessionMode.ACTIVE_EXERCISE,
            ConversationMode.ACTIVE_EXERCISE,
            ModelAudioPolicy.NO_AUDIO,
        ),
        (
            SessionMode.PAUSED,
            ConversationMode.ACTIVE_EXERCISE,
            ModelAudioPolicy.NO_AUDIO,
        ),
        (
            SessionMode.STOPPED,
            ConversationMode.ACTIVE_EXERCISE,
            ModelAudioPolicy.NO_AUDIO,
        ),
    ),
)
def test_adapter_policy_is_streaming_only_for_explicit_conversation_phases(
    mode: SessionMode,
    conversation_mode: ConversationMode,
    audio_policy: ModelAudioPolicy,
) -> None:
    selected = select_realtime_turn_policy(mode)
    assert selected.conversation_mode is conversation_mode
    assert selected.audio_policy is audio_policy


def test_active_exercise_turn_never_opens_model_speaker_callbacks() -> None:
    transport = MemoryTransport(incoming=(_created(), _audio(), _response_done()))
    provider = _ModeProvider(SessionMode.ACTIVE_EXERCISE)
    sink = _Sink()
    adapter = RealtimeConversationAdapter(
        session=RealtimeSession(transport=transport),
        sink=sink,
        mode_provider=provider,
    )

    turn_id = adapter.send_audio_turn(PCM_A, audio_format=PCM_S16LE_24K_MONO)
    adapter.pump_once()
    adapter.pump_once()
    adapter.pump_once()

    assert sink.events == [("done", turn_id, "resp-1")]


def test_mode_transition_during_response_stops_all_future_model_bytes() -> None:
    transport = MemoryTransport(
        incoming=(
            _created(),
            _audio(PCM_A, event_id="evt-first-audio"),
            _audio(PCM_B, event_id="evt-forbidden-audio"),
        )
    )
    provider = _ModeProvider(SessionMode.CHECK_IN)
    sink = _Sink()
    adapter = RealtimeConversationAdapter(
        session=RealtimeSession(transport=transport),
        sink=sink,
        mode_provider=provider,
    )

    turn_id = adapter.send_audio_turn(PCM_A, audio_format=PCM_S16LE_24K_MONO)
    adapter.pump_once()
    adapter.pump_once()
    provider.current_mode = SessionMode.ACTIVE_EXERCISE
    released_but_not_forwarded = adapter.pump_once()

    assert sink.events == [
        ("started", turn_id, "resp-1"),
        ("audio", turn_id, "resp-1", "item-1", PCM_A),
    ]
    assert released_but_not_forwarded.released_audio == ()
    assert transport.sent[-1] == {"type": "response.cancel", "response_id": "resp-1"}
    assert adapter.active_turn_id is None


def test_preemption_between_receive_and_adapter_lock_scrubs_released_audio() -> None:
    transport = MemoryTransport(incoming=(_created(), _audio(PCM_B)))
    provider = _BarrierModeProvider(SessionMode.CHECK_IN)
    sink = _Sink()
    adapter = RealtimeConversationAdapter(
        session=RealtimeSession(transport=transport),
        sink=sink,
        mode_provider=provider,
    )
    turn_id = adapter.send_audio_turn(PCM_A, audio_format=PCM_S16LE_24K_MONO)
    adapter.pump_once()
    assert sink.events == [("started", turn_id, "resp-1")]

    results = []
    failures: list[BaseException] = []

    def pump_audio() -> None:
        try:
            results.append(adapter.pump_once())
        except BaseException as exc:  # surfaced in the parent test thread
            failures.append(exc)

    provider.arm()
    pump_thread = threading.Thread(target=pump_audio, name="realtime-pump-race")
    pump_thread.start()
    provider.after_receive.wait(timeout=5)
    try:
        # receive_once has already released PCM through ModelAudioGate, while
        # pump_once is deterministically paused before its adapter lock.
        adapter.preempt_model_audio()
    finally:
        provider.resume_pump.wait(timeout=5)
    pump_thread.join(timeout=5)

    assert not pump_thread.is_alive()
    assert failures == []
    assert len(results) == 1
    assert results[0].released_audio == ()
    assert sink.events == [("started", turn_id, "resp-1")]
    assert transport.sent[-1] == {"type": "response.cancel", "response_id": "resp-1"}
