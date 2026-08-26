"""Deterministic end boundary for a long-lived Realtime session.

The Realtime model may *request* the single tool below, but only a call that has
already passed :class:`~recoverybox.realtime.tools.ToolRegistry` validation may
enter :class:`SessionEndController`.  The controller emits a local signal; the
composition root remains responsible for stopping audio, closing the socket,
and releasing devices.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from .tools import FunctionTool, ToolRegistry, ValidatedToolCall

FINISH_SESSION_TOOL_NAME = "finish_session"

FINISH_SESSION_TOOL = FunctionTool(
    name=FINISH_SESSION_TOOL_NAME,
    description=(
        "End the audio session only after the user explicitly asks to finish, "
        "leave, or says goodbye."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
)

# This registry deliberately contains no other model-controlled session action.
SESSION_CONTROL_TOOL_REGISTRY = ToolRegistry((FINISH_SESSION_TOOL,))


class SessionControlError(ValueError):
    """An input was not an authorized request to end the session."""


class SessionEndSource(StrEnum):
    """Locally distinguish the two allowed ways a session can end."""

    VALIDATED_TOOL_CALL = "validated_tool_call"
    PHYSICAL_STOP = "physical_stop"


@dataclass(frozen=True, slots=True)
class SessionEndSignal:
    """One-shot signal for the composition root to perform device cleanup."""

    source: SessionEndSource
    tool_call_id: str | None = None


class SessionEndController:
    """Turn an explicit, locally authorized request into one end signal.

    ``accept_validated_tool_call`` intentionally does not accept raw Realtime
    events, JSON, transcripts, or free-form model text.  Callers must route raw
    function output through ``ToolRegistry.validate_call`` first.  Defensive
    checks here preserve that boundary if a ``ValidatedToolCall`` is assembled
    incorrectly by application code.

    The first valid end request wins.  Later valid requests are idempotent and
    return ``None``; the callback runs at most once.
    """

    def __init__(
        self,
        on_end: Callable[[SessionEndSignal], None] | None = None,
    ) -> None:
        self._on_end = on_end
        self._lock = threading.Lock()
        self._end_signal: SessionEndSignal | None = None

    @property
    def ended(self) -> bool:
        with self._lock:
            return self._end_signal is not None

    @property
    def end_signal(self) -> SessionEndSignal | None:
        with self._lock:
            return self._end_signal

    def accept_validated_tool_call(
        self,
        call: ValidatedToolCall,
    ) -> SessionEndSignal | None:
        """Accept only the exact, argument-free ``finish_session`` capability."""

        if not isinstance(call, ValidatedToolCall):
            raise TypeError("call must be a locally validated ValidatedToolCall")
        if call.name != FINISH_SESSION_TOOL_NAME:
            raise SessionControlError("validated tool call cannot end the session")
        if not isinstance(call.arguments, Mapping) or call.arguments:
            raise SessionControlError("finish_session arguments must be an empty object")
        if not call.call_id.strip():
            raise SessionControlError("finish_session call id must not be blank")

        return self._end(
            SessionEndSignal(
                source=SessionEndSource.VALIDATED_TOOL_CALL,
                tool_call_id=call.call_id,
            )
        )

    def request_physical_stop(self) -> SessionEndSignal | None:
        """End from a locally observed physical stop through the same path."""

        return self._end(SessionEndSignal(source=SessionEndSource.PHYSICAL_STOP))

    def _end(self, signal: SessionEndSignal) -> SessionEndSignal | None:
        callback: Callable[[SessionEndSignal], None] | None
        with self._lock:
            if self._end_signal is not None:
                return None
            self._end_signal = signal
            callback = self._on_end

        # Cleanup belongs to the composition root.  Run its notification after
        # committing local state and outside the lock so callback code cannot
        # deadlock or make the end transition non-idempotent.
        if callback is not None:
            callback(signal)
        return signal


def validate_finish_session_call(*, call_id: str, arguments_json: str) -> ValidatedToolCall:
    """Validate raw function arguments through the one-tool local registry."""

    return SESSION_CONTROL_TOOL_REGISTRY.validate_call(
        name=FINISH_SESSION_TOOL_NAME,
        call_id=call_id,
        arguments_json=arguments_json,
    )
