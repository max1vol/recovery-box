"""Sealed, typed capabilities for ending one local RecoveryBox session."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SessionEndSource(StrEnum):
    """Distinguish user-authorized termination from runtime containment."""

    VALIDATED_TOOL_CALL = "validated_tool_call"
    PHYSICAL_STOP = "physical_stop"
    RUNTIME_ABORT = "runtime_abort"


class RuntimeAbortReason(StrEnum):
    """Closed non-user reasons for abandoning a runtime session."""

    EXPLICIT_CLOSE = "explicit_close"
    LAUNCHER_CLEANUP = "launcher_cleanup"
    MAX_FRAMES_REACHED = "max_frames_reached"
    SERVICE_SHUTDOWN = "service_shutdown"
    SESSION_REPLACED = "session_replaced"
    STARTUP_FAILURE = "startup_failure"
    REMOTE_STOP = "remote_stop"
    SAFETY_ENFORCEMENT_FAILURE = "safety_enforcement_failure"
    LOCAL_INPUT_UNAVAILABLE = "local_input_unavailable"


@dataclass(frozen=True, slots=True, init=False)
class SessionEndSignal:
    """One-shot capability minted only by :class:`SessionEndController`."""

    source: SessionEndSource
    tool_call_id: str | None
    abort_reason: RuntimeAbortReason | None
    _issuer: object

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("SessionEndSignal can only be issued by SessionEndController")

    @classmethod
    def _issue(
        cls,
        *,
        source: SessionEndSource,
        tool_call_id: str | None = None,
        abort_reason: RuntimeAbortReason | None = None,
        _issuer: object,
    ) -> SessionEndSignal:
        if _issuer is None:
            raise TypeError("SessionEndSignal can only be issued by SessionEndController")
        signal = object.__new__(cls)
        object.__setattr__(signal, "source", source)
        object.__setattr__(signal, "tool_call_id", tool_call_id)
        object.__setattr__(signal, "abort_reason", abort_reason)
        object.__setattr__(signal, "_issuer", _issuer)
        signal.__post_init__()
        return signal

    def __post_init__(self) -> None:
        if not isinstance(self.source, SessionEndSource):
            raise TypeError("source must be a SessionEndSource")
        if self.source is SessionEndSource.VALIDATED_TOOL_CALL:
            if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
                raise ValueError("validated finish requires a tool_call_id")
            if self.abort_reason is not None:
                raise ValueError("validated finish cannot include an abort reason")
            object.__setattr__(self, "tool_call_id", self.tool_call_id.strip())
            return
        if self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for validated finish")
        if self.source is SessionEndSource.RUNTIME_ABORT:
            if not isinstance(self.abort_reason, RuntimeAbortReason):
                raise ValueError("runtime abort requires a RuntimeAbortReason")
            return
        if self.abort_reason is not None:
            raise ValueError("physical stop cannot include an abort reason")


def _issue_session_end_signal(
    *,
    source: SessionEndSource,
    tool_call_id: str | None = None,
    abort_reason: RuntimeAbortReason | None = None,
    _issuer: object,
) -> SessionEndSignal:
    """Issue one sealed termination capability for one controller instance."""

    return SessionEndSignal._issue(
        source=source,
        tool_call_id=tool_call_id,
        abort_reason=abort_reason,
        _issuer=_issuer,
    )


def _is_session_end_signal_issued_by(value: object, issuer: object) -> bool:
    """Return whether ``value`` carries one controller instance's authority."""

    return isinstance(value, SessionEndSignal) and getattr(value, "_issuer", None) is issuer


__all__ = [
    "RuntimeAbortReason",
    "SessionEndSignal",
    "SessionEndSource",
]
