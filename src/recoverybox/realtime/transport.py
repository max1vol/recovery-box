"""JSON transport boundary for the server-to-server Realtime WebSocket."""

from __future__ import annotations

import json
import queue
import threading
from collections import deque
from collections.abc import Mapping
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

from .protocol import REALTIME_WEBSOCKET_URL, RealtimeProtocolError


@runtime_checkable
class RealtimeTransport(Protocol):
    """Minimal transport required by :class:`RealtimeSession`."""

    def send_event(self, event: Mapping[str, Any]) -> None: ...

    def receive_event(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class RealtimeTransportUnavailableError(RuntimeError):
    """A transport can no longer accept ordered Realtime client events.

    The message is deliberately fixed and content-free.  In particular, an
    exception raised by a WebSocket implementation is never retained or
    surfaced through this boundary because it may contain provider data.
    """


class RealtimeTransportBackpressureError(RealtimeTransportUnavailableError):
    """The bounded outbound queue filled and was permanently failed closed."""


class _OrderedTransportState(Enum):
    OPEN = auto()
    FAILED = auto()
    CLOSED = auto()


_STOP_WRITER = object()


class BoundedOrderedTransport:
    """Non-blocking, single-writer wrapper for a synchronous transport.

    ``send_event`` only snapshots and queues an event.  One daemon writer owns
    the delegate's synchronous send method, preserving request/cancellation
    ordering without allowing a slow socket to stall the camera, Guardian, or
    physical-stop thread.  Queue exhaustion and asynchronous send failure are
    terminal: pending work is discarded, later writes are rejected, and an
    asynchronous delegate abort wakes the sole receiver.

    Shutdown deliberately does not join either helper thread.  Even a broken
    delegate whose send and close methods both block cannot delay a local
    pause/stop boundary.
    """

    def __init__(
        self,
        delegate: RealtimeTransport,
        *,
        max_pending_events: int = 64,
    ) -> None:
        if not isinstance(delegate, RealtimeTransport):
            raise TypeError("delegate must implement RealtimeTransport")
        if (
            isinstance(max_pending_events, bool)
            or not isinstance(max_pending_events, int)
            or max_pending_events <= 0
        ):
            raise ValueError("max_pending_events must be a positive integer")

        self._delegate = delegate
        self._outgoing: queue.Queue[dict[str, Any] | object] = queue.Queue(
            maxsize=max_pending_events
        )
        self._state_lock = threading.Lock()
        self._state = _OrderedTransportState.OPEN
        self._shutdown_started = False
        self._writer = threading.Thread(
            target=self._write_loop,
            name="recoverybox-realtime-writer",
            daemon=True,
        )
        self._writer.start()

    def send_event(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            raise TypeError("event must be a mapping")
        # Snapshot the top-level protocol object before returning to its
        # caller. Protocol builders do not mutate nested values after send.
        queued_event = dict(event)
        start_shutdown = False
        with self._state_lock:
            self._require_open_locked()
            try:
                self._outgoing.put_nowait(queued_event)
                return
            except queue.Full:
                self._state = _OrderedTransportState.FAILED
                self._discard_pending_locked()
                start_shutdown = self._claim_shutdown_locked()

        if start_shutdown:
            self._start_delegate_shutdown()
        raise RealtimeTransportBackpressureError(
            "Realtime transport send queue reached its safe bound"
        )

    def receive_event(self) -> Mapping[str, Any]:
        with self._state_lock:
            self._require_open_locked()
        # The launcher owns exactly one receiver.  Delegate shutdown is what
        # interrupts this potentially blocking call after an outbound failure.
        return self._delegate.receive_event()

    def close(self) -> None:
        """Permanently close without waiting for writer or socket shutdown."""

        self.abort()

    def abort(self) -> None:
        """Discard pending writes and asynchronously abort the delegate."""

        start_shutdown = False
        with self._state_lock:
            if self._state is not _OrderedTransportState.CLOSED:
                self._state = _OrderedTransportState.CLOSED
                self._discard_pending_locked()
            start_shutdown = self._claim_shutdown_locked()
        if start_shutdown:
            self._start_delegate_shutdown()

    def _write_loop(self) -> None:
        while True:
            event = self._outgoing.get()
            if event is _STOP_WRITER:
                return
            with self._state_lock:
                if self._state is not _OrderedTransportState.OPEN:
                    continue
            try:
                assert isinstance(event, dict)
                self._delegate.send_event(event)
            except Exception:
                start_shutdown = False
                with self._state_lock:
                    if self._state is _OrderedTransportState.OPEN:
                        self._state = _OrderedTransportState.FAILED
                        self._discard_pending_locked()
                    start_shutdown = self._claim_shutdown_locked()
                if start_shutdown:
                    self._start_delegate_shutdown()
                return

    def _require_open_locked(self) -> None:
        if self._state is _OrderedTransportState.OPEN:
            return
        if self._state is _OrderedTransportState.CLOSED:
            raise RealtimeTransportUnavailableError("Realtime transport is closed")
        raise RealtimeTransportUnavailableError("Realtime transport is unavailable")

    def _discard_pending_locked(self) -> None:
        while True:
            try:
                self._outgoing.get_nowait()
            except queue.Empty:
                break
        # Wake a writer that is waiting for work. If it is currently blocked in
        # the delegate, it will consume this terminal marker after abort wakes
        # the send call.
        self._outgoing.put_nowait(_STOP_WRITER)

    def _claim_shutdown_locked(self) -> bool:
        if self._shutdown_started:
            return False
        self._shutdown_started = True
        return True

    def _start_delegate_shutdown(self) -> None:
        threading.Thread(
            target=self._shutdown_delegate,
            name="recoverybox-realtime-shutdown",
            daemon=True,
        ).start()

    def _shutdown_delegate(self) -> None:
        try:
            abort = getattr(self._delegate, "abort", None)
            if callable(abort):
                abort()
            else:
                self._delegate.close()
        except Exception:
            # Shutdown diagnostics remain content-free at this boundary.
            pass


class WebSocketJsonTransport:
    """Synchronous production transport using ``websocket-client``.

    The API key is accepted only at connection creation and is never retained
    on the Python object or included in event dictionaries.
    """

    def __init__(self, websocket_connection: Any) -> None:
        self._websocket = websocket_connection
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False

    @classmethod
    def connect(
        cls, *, api_key: str, timeout_seconds: float | None = None
    ) -> WebSocketJsonTransport:
        if not api_key.strip():
            raise RealtimeProtocolError("an API key is required to connect")
        try:
            import websocket
        except ImportError as exc:  # pragma: no cover - exercised on device install
            raise RuntimeError("websocket-client is required for the Realtime transport") from exc
        try:
            import certifi
        except ImportError as exc:  # pragma: no cover - fixed project dependency
            raise RuntimeError("certifi is required for verified Realtime TLS") from exc

        options: dict[str, Any] = {
            "header": [f"Authorization: Bearer {api_key}"],
            "enable_multithread": True,
            "sslopt": {"ca_certs": certifi.where()},
        }
        if timeout_seconds is not None:
            if timeout_seconds <= 0:
                raise RealtimeProtocolError("timeout must be positive")
            options["timeout"] = timeout_seconds
        connection = websocket.create_connection(REALTIME_WEBSOCKET_URL, **options)
        return cls(connection)

    def send_event(self, event: Mapping[str, Any]) -> None:
        payload = json.dumps(event, separators=(",", ":"), allow_nan=False)
        self._websocket.send(payload)

    def receive_event(self) -> Mapping[str, Any]:
        payload = self._websocket.recv()
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if not isinstance(payload, str):
            raise RealtimeProtocolError("WebSocket event must be text JSON")
        if not payload:
            raise EOFError("Realtime WebSocket closed without another event")
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RealtimeProtocolError("WebSocket event is not valid JSON") from exc
        if not isinstance(event, Mapping):
            raise RealtimeProtocolError("WebSocket event JSON must be an object")
        return event

    def close(self) -> None:
        self.abort()

    def abort(self) -> None:
        """Interrupt socket I/O without waiting for a close handshake."""

        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
        for method_name in ("shutdown", "abort"):
            method = getattr(self._websocket, method_name, None)
            if callable(method):
                method()
                return
        # Compatibility fallback for an injected/older connection. Production
        # websocket-client exposes ``shutdown`` and therefore does not use its
        # potentially blocking close-handshake path.
        self._websocket.close()


class MemoryTransport:
    """Deterministic transport for unit tests, replays, and offline demos."""

    def __init__(self, incoming: tuple[Mapping[str, Any], ...] = ()) -> None:
        self.sent: list[dict[str, Any]] = []
        self.incoming = deque(dict(event) for event in incoming)
        self.closed = False

    def send_event(self, event: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("transport is closed")
        self.sent.append(dict(event))

    def receive_event(self) -> Mapping[str, Any]:
        if self.closed:
            raise RuntimeError("transport is closed")
        if not self.incoming:
            raise EOFError("no incoming Realtime event is available")
        return self.incoming.popleft()

    def close(self) -> None:
        self.closed = True
