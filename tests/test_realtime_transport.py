from __future__ import annotations

import sys
import threading
from collections import deque
from types import SimpleNamespace

import pytest

from recoverybox.realtime.protocol import REALTIME_WEBSOCKET_URL
from recoverybox.realtime.transport import (
    BoundedOrderedTransport,
    RealtimeTransportBackpressureError,
    RealtimeTransportUnavailableError,
    WebSocketJsonTransport,
)


class _Connection:
    def close(self) -> None:
        return


class _ImmediateShutdownConnection:
    def __init__(self) -> None:
        self.shutdown_called = False
        self.close_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True

    def close(self) -> None:
        self.close_called = True


class _BlockingDelegate:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.fail_send = fail_send
        self.send_entered = threading.Event()
        self.release_send = threading.Event()
        self.close_entered = threading.Event()
        self.release_close = threading.Event()
        self.sent: list[dict] = []
        self.incoming: deque[dict] = deque()

    def send_event(self, event) -> None:
        self.send_entered.set()
        self.release_send.wait()
        if self.fail_send:
            raise OSError("secret provider send details")
        self.sent.append(dict(event))

    def receive_event(self):
        if not self.incoming:
            raise EOFError("no event")
        return self.incoming.popleft()

    def close(self) -> None:
        self.close_entered.set()
        self.release_close.wait()


def test_realtime_transport_uses_verified_ca_bundle_without_retaining_key(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def create_connection(url: str, **options: object) -> _Connection:
        captured.update({"url": url, **options})
        return _Connection()

    monkeypatch.setitem(
        sys.modules,
        "websocket",
        SimpleNamespace(create_connection=create_connection),
    )
    monkeypatch.setitem(
        sys.modules,
        "certifi",
        SimpleNamespace(where=lambda: "/trusted/cacert.pem"),
    )

    transport = WebSocketJsonTransport.connect(
        api_key="temporary-secret",
        timeout_seconds=12.0,
    )

    assert captured == {
        "url": REALTIME_WEBSOCKET_URL,
        "header": ["Authorization: Bearer temporary-secret"],
        "enable_multithread": True,
        "sslopt": {"ca_certs": "/trusted/cacert.pem"},
        "timeout": 12.0,
    }
    assert "temporary-secret" not in repr(transport)


def test_websocket_transport_prefers_immediate_shutdown_over_close_handshake() -> None:
    connection = _ImmediateShutdownConnection()
    transport = WebSocketJsonTransport(connection)

    transport.close()
    transport.close()

    assert connection.shutdown_called
    assert not connection.close_called


def test_bounded_writer_fails_closed_on_backpressure_and_never_blocks_close() -> None:
    delegate = _BlockingDelegate()
    transport = BoundedOrderedTransport(delegate, max_pending_events=1)
    transport.send_event({"type": "session.update"})
    assert delegate.send_entered.wait(0.5)
    transport.send_event({"type": "response.create"})

    with pytest.raises(
        RealtimeTransportBackpressureError,
        match="safe bound",
    ):
        transport.send_event({"type": "response.cancel"})
    assert delegate.close_entered.wait(0.5)
    with pytest.raises(RealtimeTransportUnavailableError, match="unavailable"):
        transport.send_event({"type": "response.create", "secret": "must not surface"})

    close_returned = threading.Event()
    threading.Thread(
        target=lambda: (transport.close(), close_returned.set()),
        daemon=True,
    ).start()
    assert close_returned.wait(0.5)
    delegate.release_send.set()
    delegate.release_close.set()


def test_async_send_failure_permanently_rejects_later_events_with_redacted_error() -> None:
    delegate = _BlockingDelegate(fail_send=True)
    transport = BoundedOrderedTransport(delegate)
    transport.send_event({"type": "session.update"})
    assert delegate.send_entered.wait(0.5)
    delegate.release_send.set()
    assert delegate.close_entered.wait(0.5)

    with pytest.raises(RealtimeTransportUnavailableError) as raised:
        transport.send_event({"type": "response.create"})

    assert "provider" not in str(raised.value)
    assert "secret" not in str(raised.value)
    delegate.release_close.set()
