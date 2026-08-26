"""Pure builders and parsers for the OpenAI Realtime WebSocket protocol.

The device layer deliberately talks to this module in dictionaries.  Keeping
event construction and parsing free of sockets makes the protocol replayable
in tests and prevents partially validated server data from reaching hardware.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

REALTIME_MODEL = "gpt-realtime-2.1"
REALTIME_WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
PCM_SAMPLE_RATE_HZ = 24_000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH_BYTES = 2
PCM_FORMAT = {"type": "audio/pcm", "rate": PCM_SAMPLE_RATE_HZ}

# Error messages are untrusted server text and can echo request values (including
# credentials).  Only documented, low-cardinality identifiers cross the parser
# boundary.  Unknown identifiers are deliberately dropped instead of being
# copied into exceptions, logs, or verification artifacts.
_SAFE_SERVER_ERROR_TYPES = frozenset(
    {
        "api_error",
        "authentication_error",
        "conflict_error",
        "invalid_request_error",
        "not_found_error",
        "permission_error",
        "rate_limit_error",
        "server_error",
    }
)
_SAFE_SERVER_ERROR_CODES = frozenset(
    {
        "conversation_already_has_active_response",
        "input_audio_buffer_commit_empty",
        "input_audio_buffer_commit_too_small",
        "insufficient_quota",
        "invalid_api_key",
        "invalid_event",
        "invalid_request_error",
        "invalid_value",
        "missing_required_parameter",
        "model_not_found",
        "permission_denied",
        "rate_limit_exceeded",
        "response_cancel_not_active",
        "server_error",
        "session_expired",
        "unknown_parameter",
        "unsupported_value",
    }
)
_SAFE_RESPONSE_STATUSES = frozenset(
    {"in_progress", "completed", "cancelled", "failed", "incomplete"}
)


class RealtimeProtocolError(ValueError):
    """Raised when a known event does not satisfy the expected wire shape."""


class ServerEventKind(StrEnum):
    RESPONSE_CREATED = "response_created"
    AUDIO_DELTA = "audio_delta"
    AUDIO_DONE = "audio_done"
    TRANSCRIPT_DONE = "transcript_done"
    TOOL_ARGUMENTS_DONE = "tool_arguments_done"
    RESPONSE_DONE = "response_done"
    ERROR = "error"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ParsedServerEvent:
    """A narrow, typed view of a Realtime server event.

    Only fields that have passed local shape/encoding checks are exposed.  In
    particular, audio deltas are base64-decoded before they enter the safety
    gate, and function arguments remain an opaque JSON string until a local
    tool schema validates them.
    """

    kind: ServerEventKind
    event_type: str
    event_id: str | None = None
    response_id: str | None = None
    item_id: str | None = None
    content_index: int = 0
    audio: bytes | None = None
    transcript: str | None = None
    response_status: str | None = None
    tool_name: str | None = None
    call_id: str | None = None
    arguments_json: str | None = None
    error_type: str | None = None
    error_code: str | None = None

    @property
    def error_message(self) -> None:
        """Compatibility shim that never exposes untrusted server prose."""

        return None


def build_session_update(
    *,
    instructions: str,
    voice: str,
    tools: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a manual-turn, audio-in/audio-out Realtime session.

    ``turn_detection`` is explicitly null so the physical button owns turn
    boundaries.  PCM is signed 16-bit little-endian mono at 24 kHz on both
    sides; the wire schema denotes this as ``audio/pcm`` at 24 kHz.
    """

    if not instructions.strip():
        raise RealtimeProtocolError("session instructions must not be blank")
    if not voice.strip():
        raise RealtimeProtocolError("voice must not be blank")

    normalized_tools: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            raise RealtimeProtocolError("every Realtime tool must be a function")
        if not isinstance(tool.get("name"), str) or not tool["name"].strip():
            raise RealtimeProtocolError("every Realtime tool needs a name")
        if not isinstance(tool.get("parameters"), Mapping):
            raise RealtimeProtocolError("every Realtime tool needs a parameter schema")
        normalized_tools.append(dict(tool))

    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": instructions,
            "audio": {
                "input": {
                    "format": dict(PCM_FORMAT),
                    "turn_detection": None,
                },
                "output": {
                    "format": dict(PCM_FORMAT),
                    "voice": voice,
                },
            },
            "tools": normalized_tools,
            "tool_choice": "auto",
        },
    }


def build_audio_append(pcm16_mono_24khz: bytes) -> dict[str, str]:
    """Build ``input_audio_buffer.append`` for complete PCM16 samples."""

    if not pcm16_mono_24khz:
        raise RealtimeProtocolError("audio append must contain at least one sample")
    if len(pcm16_mono_24khz) % PCM_SAMPLE_WIDTH_BYTES:
        raise RealtimeProtocolError("PCM16 audio must contain complete 2-byte samples")
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm16_mono_24khz).decode("ascii"),
    }


def build_audio_commit() -> dict[str, str]:
    return {"type": "input_audio_buffer.commit"}


def build_audio_clear() -> dict[str, str]:
    return {"type": "input_audio_buffer.clear"}


def build_response_create(*, instructions: str | None = None) -> dict[str, Any]:
    """Request one audio response after the caller has committed user audio."""

    response: dict[str, Any] = {"output_modalities": ["audio"]}
    if instructions is not None:
        if not instructions.strip():
            raise RealtimeProtocolError("response instructions must not be blank")
        response["instructions"] = instructions
    return {"type": "response.create", "response": response}


def build_prompt_cue_response_create(
    *,
    cue_id: str,
    catalog_version: str,
    instructions: str,
) -> dict[str, Any]:
    """Request one isolated, tool-free audio rendition of an approved cue.

    The response is out-of-band so a short exercise cue does not mutate the
    conversational history.  An empty input array and response-level
    instructions make the request self-contained, while clearing tools keeps
    this lane speech-only.  Local transcript quarantine remains the authority
    for whether returned PCM may be released.
    """

    normalized_cue_id = cue_id.strip()
    normalized_catalog_version = catalog_version.strip()
    normalized_instructions = instructions.strip()
    if not normalized_cue_id:
        raise RealtimeProtocolError("prompt cue id must not be blank")
    if not normalized_catalog_version:
        raise RealtimeProtocolError("prompt cue catalog version must not be blank")
    if not normalized_instructions:
        raise RealtimeProtocolError("prompt cue instructions must not be blank")

    return {
        "type": "response.create",
        "response": {
            "output_modalities": ["audio"],
            "instructions": normalized_instructions,
            "conversation": "none",
            "input": [],
            "tools": [],
            "tool_choice": "none",
            # Exact, short phrases do not benefit from additional reasoning.
            # Keeping both bounds response-local minimizes cue latency without
            # weakening the transcript/response-completion release gate.
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": 256,
            "metadata": {
                "recoverybox_lane": "prompt_cue",
                "cue_id": normalized_cue_id,
                "catalog_version": normalized_catalog_version,
            },
        },
    }


def build_response_cancel(*, response_id: str | None = None) -> dict[str, str]:
    """Cancel one response, scoped when the server response ID is known.

    Realtime permits multiple out-of-band responses.  Supplying the ID keeps a
    prompt-cue cancellation from accidentally targeting a response in the
    default conversation.  Before ``response.created`` there is no ID, so the
    caller may still emit the protocol's unscoped form for a default response.
    """

    event = {"type": "response.cancel"}
    if response_id is not None:
        normalized_response_id = response_id.strip()
        if not normalized_response_id:
            raise RealtimeProtocolError("response id must not be blank")
        event["response_id"] = normalized_response_id
    return event


def build_assistant_item_truncate(
    *, item_id: str, audio_end_ms: int, content_index: int = 0
) -> dict[str, Any]:
    """Keep server conversation state aligned with locally played audio."""

    if not item_id.strip():
        raise RealtimeProtocolError("assistant item id must not be blank")
    if content_index < 0:
        raise RealtimeProtocolError("content index must be non-negative")
    if audio_end_ms < 0:
        raise RealtimeProtocolError("audio end must be non-negative")
    return {
        "type": "conversation.item.truncate",
        "item_id": item_id,
        "content_index": content_index,
        "audio_end_ms": audio_end_ms,
    }


def build_function_call_output(*, call_id: str, output_json: str) -> dict[str, Any]:
    """Return locally produced function output to the conversation."""

    if not call_id.strip():
        raise RealtimeProtocolError("function call id must not be blank")
    if not output_json.strip():
        raise RealtimeProtocolError("function output must not be blank")
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_json,
        },
    }


def parse_server_event(raw: Mapping[str, Any]) -> ParsedServerEvent:
    """Parse a server event into the small trusted surface used by the app."""

    event_type = raw.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise RealtimeProtocolError("server event needs a non-empty type")

    if event_type == "response.created":
        response = _mapping(raw.get("response"), "response.created.response")
        return ParsedServerEvent(
            kind=ServerEventKind.RESPONSE_CREATED,
            event_type=event_type,
            event_id=_required_string(raw, "event_id"),
            response_id=_required_string(response, "id"),
            response_status=_safe_response_status(response, required=False),
        )

    if event_type == "response.output_audio.delta":
        encoded = _required_string(raw, "delta")
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RealtimeProtocolError("audio delta is not valid base64") from exc
        if not audio or len(audio) % PCM_SAMPLE_WIDTH_BYTES:
            raise RealtimeProtocolError("audio delta is not complete PCM16 data")
        return ParsedServerEvent(
            kind=ServerEventKind.AUDIO_DELTA,
            event_type=event_type,
            event_id=_required_string(raw, "event_id"),
            response_id=_required_string(raw, "response_id"),
            item_id=_required_string(raw, "item_id"),
            content_index=_non_negative_int(raw, "content_index", default=0),
            audio=audio,
        )

    if event_type == "response.output_audio.done":
        return ParsedServerEvent(
            kind=ServerEventKind.AUDIO_DONE,
            event_type=event_type,
            event_id=_required_string(raw, "event_id"),
            response_id=_required_string(raw, "response_id"),
            item_id=_required_string(raw, "item_id"),
            content_index=_non_negative_int(raw, "content_index", default=0),
        )

    if event_type == "response.output_audio_transcript.done":
        return ParsedServerEvent(
            kind=ServerEventKind.TRANSCRIPT_DONE,
            event_type=event_type,
            event_id=_required_string(raw, "event_id"),
            response_id=_required_string(raw, "response_id"),
            item_id=_required_string(raw, "item_id"),
            content_index=_non_negative_int(raw, "content_index", default=0),
            transcript=_required_string(raw, "transcript", allow_blank=True),
        )

    if event_type == "response.function_call_arguments.done":
        return ParsedServerEvent(
            kind=ServerEventKind.TOOL_ARGUMENTS_DONE,
            event_type=event_type,
            event_id=_required_string(raw, "event_id"),
            response_id=_required_string(raw, "response_id"),
            item_id=_required_string(raw, "item_id"),
            tool_name=_required_string(raw, "name"),
            call_id=_required_string(raw, "call_id"),
            arguments_json=_required_string(raw, "arguments", allow_blank=True),
        )

    if event_type == "response.done":
        response = _mapping(raw.get("response"), "response.done.response")
        return ParsedServerEvent(
            kind=ServerEventKind.RESPONSE_DONE,
            event_type=event_type,
            event_id=_required_string(raw, "event_id"),
            response_id=_required_string(response, "id"),
            response_status=_safe_response_status(response, required=True),
        )

    if event_type == "error":
        error = _mapping(raw.get("error"), "error.error")
        # Validate that the documented message is present, but never retain its
        # value.  OpenAI error prose can echo a rejected credential or request
        # field; only allowlisted stable identifiers leave this trust boundary.
        _required_string(error, "message")
        return ParsedServerEvent(
            kind=ServerEventKind.ERROR,
            event_type=event_type,
            event_id=_required_string(raw, "event_id"),
            error_type=_safe_server_error_identifier(
                error,
                "type",
                allowed=_SAFE_SERVER_ERROR_TYPES,
            ),
            error_code=_safe_server_error_identifier(
                error,
                "code",
                allowed=_SAFE_SERVER_ERROR_CODES,
            ),
        )

    return ParsedServerEvent(
        kind=ServerEventKind.OTHER,
        event_type=event_type,
        event_id=_optional_string(raw, "event_id"),
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RealtimeProtocolError(f"{label} must be an object")
    return value


def _required_string(value: Mapping[str, Any], key: str, *, allow_blank: bool = False) -> str:
    found = value.get(key)
    if not isinstance(found, str) or (not allow_blank and not found.strip()):
        raise RealtimeProtocolError(f"{key} must be a string")
    return found


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    found = value.get(key)
    if found is None:
        return None
    if not isinstance(found, str):
        raise RealtimeProtocolError(f"{key} must be a string when present")
    return found


def _safe_server_error_identifier(
    value: Mapping[str, Any],
    key: str,
    *,
    allowed: frozenset[str],
) -> str | None:
    found = _optional_string(value, key)
    if found is None:
        return None
    normalized = found.strip().lower()
    return normalized if normalized in allowed else None


def _safe_response_status(value: Mapping[str, Any], *, required: bool) -> str | None:
    found = _required_string(value, "status") if required else _optional_string(value, "status")
    if found is None:
        return None
    normalized = found.strip().lower()
    return normalized if normalized in _SAFE_RESPONSE_STATUSES else None


def _non_negative_int(value: Mapping[str, Any], key: str, *, default: int | None = None) -> int:
    found = value.get(key, default)
    if isinstance(found, bool) or not isinstance(found, int) or found < 0:
        raise RealtimeProtocolError(f"{key} must be a non-negative integer")
    return found
