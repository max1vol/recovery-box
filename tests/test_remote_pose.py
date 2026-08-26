from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from recoverybox.exercise import (
    SquatAnalysis,
    SquatAssessmentIssue,
    SquatEvent,
    SquatEventType,
    SquatPhase,
)
from recoverybox.remote_pose import (
    MAX_REMOTE_POSE_CHALLENGE_BYTES,
    MAX_REMOTE_POSE_EVIDENCE_AGE_MS,
    MAX_REMOTE_POSE_PAYLOAD_BYTES,
    MAX_REMOTE_POSE_SEQUENCE,
    REMOTE_POSE_VERSION,
    RemotePoseChallenge,
    RemotePoseKind,
    RemotePoseMessage,
    RemotePoseProtocolError,
    RemotePosePublisher,
    RemotePoseRequest,
    decode_remote_pose_challenge,
    decode_remote_pose_message,
    decode_remote_pose_request,
    encode_remote_pose_challenge,
    encode_remote_pose_message,
    encode_remote_pose_request,
    load_remote_pose_token,
    new_remote_pose_request_nonce,
    new_remote_pose_server_nonce,
    new_remote_pose_service_epoch,
)

TOKEN = bytes(range(32))
SESSION_ID = "0123456789abcdef0123456789abcdef"
SERVICE_EPOCH = "a" * 64
SERVER_NONCE = "b" * 64
REQUEST_NONCE = "c" * 64
PEER = "100.106.237.106:45873"


def analysis(
    *,
    timestamp_ms: int = 100,
    assessable: bool = True,
    events: tuple[SquatEvent, ...] = (),
) -> SquatAnalysis:
    return SquatAnalysis(
        timestamp_ms=timestamp_ms,
        assessable=assessable,
        phase=SquatPhase.STANDING if assessable else SquatPhase.UNKNOWN,
        rep_count=events[-1].rep_count if events else 0,
        events=events,
        issues=() if assessable else (SquatAssessmentIssue.NO_POSE,),
        confidence=0.95 if assessable else 0.0,
        knee_angle_degrees=170.25 if assessable else None,
        arms_in_t=True if assessable else None,
    )


def challenge(
    *,
    service_epoch: str = SERVICE_EPOCH,
    server_nonce: str = SERVER_NONCE,
) -> RemotePoseChallenge:
    return RemotePoseChallenge(
        service_epoch=service_epoch,
        server_nonce=server_nonce,
    )


def request(
    sequence: int = 1,
    *,
    session_id: str = SESSION_ID,
    service_epoch: str = SERVICE_EPOCH,
    server_nonce: str = SERVER_NONCE,
    request_nonce: str = REQUEST_NONCE,
) -> RemotePoseRequest:
    return RemotePoseRequest(
        session_id=session_id,
        service_epoch=service_epoch,
        server_nonce=server_nonce,
        request_sequence=sequence,
        request_nonce=request_nonce,
    )


def signed_line(envelope: dict[str, Any], token: bytes = TOKEN) -> bytes:
    unsigned = {key: value for key, value in envelope.items() if key != "hmac"}
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    signed = {
        **unsigned,
        "hmac": hmac.new(token, canonical, hashlib.sha256).hexdigest(),
    }
    return (
        json.dumps(
            signed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("timed out waiting for remote pose state")
        time.sleep(0.005)


def decoded_messages(connection: FakeSocket) -> list[RemotePoseMessage]:
    return [decode_remote_pose_message(line, TOKEN) for line in connection.sent]


def test_challenge_and_request_round_trip_are_canonical_authenticated_and_closed() -> None:
    original_challenge = challenge()
    challenge_line = encode_remote_pose_challenge(original_challenge, TOKEN)
    assert len(challenge_line) <= MAX_REMOTE_POSE_CHALLENGE_BYTES
    assert challenge_line == signed_line(json.loads(challenge_line))
    assert decode_remote_pose_challenge(challenge_line, TOKEN) == original_challenge
    assert set(json.loads(challenge_line)) == {
        "version",
        "kind",
        "service_epoch",
        "server_nonce",
        "hmac",
    }

    original_request = request(7)
    request_line = encode_remote_pose_request(original_request, TOKEN)
    assert request_line == signed_line(json.loads(request_line))
    assert decode_remote_pose_request(request_line, TOKEN) == original_request
    assert set(json.loads(request_line)) == {
        "version",
        "kind",
        "session_id",
        "service_epoch",
        "server_nonce",
        "request_sequence",
        "request_nonce",
        "hmac",
    }


def test_analysis_round_trip_binds_request_connection_and_numeric_age() -> None:
    original = RemotePoseMessage(
        kind=RemotePoseKind.ANALYSIS,
        session_id=SESSION_ID,
        sequence=9,
        analysis=analysis(events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),)),
        server_nonce=SERVER_NONCE,
        service_epoch=SERVICE_EPOCH,
        evidence_age_ms=42,
        request_sequence=7,
        request_nonce=REQUEST_NONCE,
    )

    encoded = encode_remote_pose_message(original, TOKEN)
    envelope = json.loads(encoded)

    assert encoded == signed_line(envelope)
    assert len(encoded) <= MAX_REMOTE_POSE_PAYLOAD_BYTES
    assert set(envelope) == {
        "analysis",
        "evidence_age_ms",
        "hmac",
        "kind",
        "request_nonce",
        "request_sequence",
        "sequence",
        "server_nonce",
        "service_epoch",
        "session_id",
        "version",
    }
    assert envelope["evidence_age_ms"] == 42
    assert not any(
        forbidden in encoded for forbidden in (b"landmark", b"image", b"audio", b"transcript")
    )
    assert decode_remote_pose_message(encoded, TOKEN) == original


@pytest.mark.parametrize(
    ("kind", "sequence"),
    [
        (RemotePoseKind.START, None),
        (RemotePoseKind.RESUME, 1),
        (RemotePoseKind.STOP, MAX_REMOTE_POSE_SEQUENCE),
    ],
)
def test_control_messages_round_trip_with_epoch_and_connection_binding(
    kind: RemotePoseKind,
    sequence: int | None,
) -> None:
    message = RemotePoseMessage(
        kind,
        SESSION_ID,
        sequence,
        server_nonce=SERVER_NONCE,
        service_epoch=SERVICE_EPOCH,
    )
    assert decode_remote_pose_message(encode_remote_pose_message(message, TOKEN), TOKEN) == message


@pytest.mark.parametrize("age", [-1, True, 500, 501, None])
def test_analysis_message_rejects_age_outside_strict_fresh_range(age: object) -> None:
    with pytest.raises((TypeError, ValueError), match="evidence_age_ms"):
        RemotePoseMessage(
            RemotePoseKind.ANALYSIS,
            SESSION_ID,
            1,
            analysis(),
            SERVER_NONCE,
            SERVICE_EPOCH,
            age,  # type: ignore[arg-type]
            1,
            REQUEST_NONCE,
        )


def test_message_and_request_reject_missing_or_malformed_bindings() -> None:
    with pytest.raises(ValueError, match="server_nonce"):
        RemotePoseMessage(
            RemotePoseKind.START,
            SESSION_ID,
            service_epoch=SERVICE_EPOCH,
        )
    with pytest.raises(ValueError, match="service_epoch"):
        RemotePoseMessage(
            RemotePoseKind.START,
            SESSION_ID,
            server_nonce=SERVER_NONCE,
        )
    with pytest.raises(ValueError, match="request_sequence"):
        RemotePoseRequest(SESSION_ID, SERVICE_EPOCH, SERVER_NONCE, 0, REQUEST_NONCE)
    with pytest.raises(ValueError, match="request_nonce"):
        RemotePoseRequest(SESSION_ID, SERVICE_EPOCH, SERVER_NONCE, 1, "A" * 64)


def test_decoders_reject_tampering_noncanonical_json_and_extra_fields() -> None:
    line = bytearray(encode_remote_pose_request(request(), TOKEN))
    line[line.index(b'"request_sequence":1') + len(b'"request_sequence":')] = ord("2")
    with pytest.raises(RemotePoseProtocolError, match="authentication"):
        decode_remote_pose_request(bytes(line), TOKEN)

    envelope = json.loads(encode_remote_pose_challenge(challenge(), TOKEN))
    envelope["extra"] = 1
    with pytest.raises(RemotePoseProtocolError, match="outside its closed schema"):
        decode_remote_pose_challenge(signed_line(envelope), TOKEN)

    valid = encode_remote_pose_message(
        RemotePoseMessage(
            RemotePoseKind.START,
            SESSION_ID,
            server_nonce=SERVER_NONCE,
            service_epoch=SERVICE_EPOCH,
        ),
        TOKEN,
    )
    with pytest.raises(RemotePoseProtocolError, match="canonical"):
        decode_remote_pose_message(valid.replace(b'"kind"', b'"kind" '), TOKEN)


def test_fresh_identifier_helpers_have_expected_entropy_shape() -> None:
    values = {
        new_remote_pose_service_epoch(),
        new_remote_pose_server_nonce(),
        new_remote_pose_request_nonce(),
    }
    assert len(values) == 3
    assert all(len(value) == 64 and value == value.lower() for value in values)
    assert all(bytes.fromhex(value) for value in values)


def test_token_loader_requires_owner_only_regular_exact_hex_file(tmp_path: Path) -> None:
    token_path = tmp_path / "pose.token"
    token_path.write_text(TOKEN.hex() + "\n", encoding="ascii")
    token_path.chmod(0o600)
    assert load_remote_pose_token(token_path) == TOKEN

    token_path.chmod(0o644)
    with pytest.raises(PermissionError, match="owner-only"):
        load_remote_pose_token(token_path)
    token_path.chmod(0o600)
    token_path.write_text("z" * 64, encoding="ascii")
    with pytest.raises(ValueError, match="64 hexadecimal"):
        load_remote_pose_token(token_path)


class FakeSocket:
    def __init__(
        self,
        *,
        server_challenge: RemotePoseChallenge | None = None,
        send_behavior: Callable[[bytes, int], None] | None = None,
    ) -> None:
        selected_challenge = server_challenge or challenge()
        self._incoming: deque[bytes | BaseException] = deque(
            [encode_remote_pose_challenge(selected_challenge, TOKEN)]
        )
        self._condition = threading.Condition()
        self.sent: list[bytes] = []
        self.timeout: float | None = None
        self.closed = False
        self.send_behavior = send_behavior
        self.send_calls = 0
        self.shutdown_called = threading.Event()

    def feed(self, line: bytes | BaseException) -> None:
        with self._condition:
            self._incoming.append(line)
            self._condition.notify_all()

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def recv(self, size: int) -> bytes:
        with self._condition:
            deadline = time.monotonic() + (self.timeout or 0.1)
            while not self._incoming and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                self._condition.wait(remaining)
            if self.closed:
                return b""
            item = self._incoming.popleft()
        if isinstance(item, BaseException):
            raise item
        if len(item) > size:
            head, tail = item[:size], item[size:]
            with self._condition:
                self._incoming.appendleft(tail)
            return head
        return item

    def sendall(self, line: bytes) -> None:
        self.send_calls += 1
        if self.send_behavior is not None:
            self.send_behavior(line, self.send_calls)
        self.sent.append(line)

    def shutdown(self, _: int) -> None:
        self.shutdown_called.set()
        with self._condition:
            self._condition.notify_all()

    def close(self) -> None:
        self.closed = True
        with self._condition:
            self._condition.notify_all()


def test_publisher_is_request_gated_binds_response_and_authorizes_epoch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeSocket()
    monkeypatch.setattr(
        "recoverybox.remote_pose.socket.create_connection",
        lambda *_args, **_kwargs: connection,
    )
    publisher = RemotePosePublisher(
        PEER,
        TOKEN,
        authorize_initial_epoch=True,
        request_poll_timeout_seconds=0.01,
        close_timeout_seconds=0.5,
        clock=lambda: 1.0,
    )

    publisher.start()
    wait_until(lambda: len(connection.sent) >= 2)
    initial = decoded_messages(connection)
    assert [item.kind for item in initial] == [
        RemotePoseKind.START,
        RemotePoseKind.RESUME,
    ]
    assert initial[0].server_nonce == SERVER_NONCE
    assert initial[0].service_epoch == SERVICE_EPOCH

    issued = request(session_id=publisher.session_id)
    connection.feed(encode_remote_pose_request(issued, TOKEN))
    claimed = publisher.wait_for_request(0.2)
    assert claimed == issued

    publisher.submit(analysis(), request=claimed, evidence_age_ms=37)
    wait_until(lambda: len(connection.sent) >= 3)
    response = decoded_messages(connection)[2]
    assert response.kind is RemotePoseKind.ANALYSIS
    assert response.sequence == 2
    assert response.request_sequence == issued.request_sequence
    assert response.request_nonce == issued.request_nonce
    assert response.evidence_age_ms == 37

    publisher.request_resume()
    wait_until(lambda: len(connection.sent) >= 4)
    publisher.close()
    messages = decoded_messages(connection)
    assert [item.kind for item in messages] == [
        RemotePoseKind.START,
        RemotePoseKind.RESUME,
        RemotePoseKind.ANALYSIS,
        RemotePoseKind.RESUME,
        RemotePoseKind.STOP,
    ]
    assert [item.sequence for item in messages] == [None, 1, 2, 3, 4]
    assert all(item.service_epoch == SERVICE_EPOCH for item in messages)
    assert all(item.server_nonce == SERVER_NONCE for item in messages)
    assert publisher.messages_sent == 5
    assert connection.closed


def test_publisher_withholds_500ms_and_sends_499ms_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeSocket()
    monkeypatch.setattr(
        "recoverybox.remote_pose.socket.create_connection",
        lambda *_args, **_kwargs: connection,
    )
    publisher = RemotePosePublisher(
        PEER,
        TOKEN,
        authorize_initial_epoch=True,
        request_poll_timeout_seconds=0.01,
        close_timeout_seconds=0.5,
        clock=lambda: 1.0,
    )
    publisher.start()
    wait_until(lambda: publisher.connected and len(connection.sent) >= 2)

    stale_request = request(1, session_id=publisher.session_id)
    connection.feed(encode_remote_pose_request(stale_request, TOKEN))
    claimed = publisher.wait_for_request(0.2)
    assert claimed is not None
    publisher.submit(
        analysis(timestamp_ms=1),
        request=claimed,
        evidence_age_ms=MAX_REMOTE_POSE_EVIDENCE_AGE_MS,
    )
    time.sleep(0.04)
    assert [item.kind for item in decoded_messages(connection)].count(RemotePoseKind.ANALYSIS) == 0

    fresh_request = request(
        2,
        session_id=publisher.session_id,
        request_nonce="d" * 64,
    )
    connection.feed(encode_remote_pose_request(fresh_request, TOKEN))
    claimed = publisher.wait_for_request(0.2)
    assert claimed is not None
    publisher.submit(
        analysis(timestamp_ms=2),
        request=claimed,
        evidence_age_ms=MAX_REMOTE_POSE_EVIDENCE_AGE_MS - 1,
    )
    wait_until(
        lambda: (
            [item.kind for item in decoded_messages(connection)].count(RemotePoseKind.ANALYSIS) == 1
        )
    )
    delivered = [
        item for item in decoded_messages(connection) if item.kind is RemotePoseKind.ANALYSIS
    ]
    assert delivered[0].evidence_age_ms == 499
    publisher.close()


def test_publisher_drops_response_when_queue_dwell_reaches_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1.0]
    connection = FakeSocket()
    monkeypatch.setattr(
        "recoverybox.remote_pose.socket.create_connection",
        lambda *_args, **_kwargs: connection,
    )
    publisher = RemotePosePublisher(
        PEER,
        TOKEN,
        authorize_initial_epoch=True,
        request_poll_timeout_seconds=0.05,
        close_timeout_seconds=0.5,
        clock=lambda: now[0],
    )
    publisher.start()
    wait_until(lambda: publisher.connected and len(connection.sent) >= 2)
    issued = request(1, session_id=publisher.session_id)
    connection.feed(encode_remote_pose_request(issued, TOKEN))
    claimed = publisher.wait_for_request(0.2)
    assert claimed is not None
    publisher.submit(analysis(), request=claimed, evidence_age_ms=499)
    now[0] = 1.001
    time.sleep(0.08)
    assert [item.kind for item in decoded_messages(connection)].count(RemotePoseKind.ANALYSIS) == 0
    publisher.close()


def test_duplicate_or_second_outstanding_request_disconnects_and_invalidates_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeSocket()
    attempts = 0

    def connect(*_args: object, **_kwargs: object) -> FakeSocket:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return connection
        raise ConnectionRefusedError()

    monkeypatch.setattr("recoverybox.remote_pose.socket.create_connection", connect)
    publisher = RemotePosePublisher(
        PEER,
        TOKEN,
        authorize_initial_epoch=True,
        request_poll_timeout_seconds=0.01,
        reconnect_backoff_seconds=0.01,
        close_timeout_seconds=0.2,
    )
    publisher.start()
    wait_until(lambda: publisher.connected and len(connection.sent) >= 2)
    issued = request(1, session_id=publisher.session_id)
    encoded = encode_remote_pose_request(issued, TOKEN)
    connection.feed(encoded)
    claimed = publisher.wait_for_request(0.2)
    assert claimed == issued
    connection.feed(encoded)
    wait_until(lambda: connection.closed)
    with pytest.raises(RuntimeError, match="claimed request"):
        publisher.submit(analysis(), request=issued, evidence_age_ms=1)
    publisher.close()


def test_request_reader_retains_partial_line_across_poll_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeSocket()
    monkeypatch.setattr(
        "recoverybox.remote_pose.socket.create_connection",
        lambda *_args, **_kwargs: connection,
    )
    publisher = RemotePosePublisher(
        PEER,
        TOKEN,
        request_poll_timeout_seconds=0.01,
        close_timeout_seconds=0.2,
    )
    publisher.start()
    wait_until(lambda: publisher.connected)
    wire = encode_remote_pose_request(request(1, session_id=publisher.session_id), TOKEN)

    connection.feed(wire[:17])
    connection.feed(TimeoutError())
    connection.feed(wire[17:])

    assert publisher.wait_for_request(0.2) is not None
    publisher.close()


def test_request_reader_rejects_partial_line_past_absolute_deadline() -> None:
    publisher = RemotePosePublisher(PEER, TOKEN)
    publisher._request_buffer.extend(b"{")
    publisher._request_line_started_at = time.monotonic() - 0.5

    with pytest.raises(RemotePoseProtocolError, match="deadline"):
        publisher._receive_request_line(FakeSocket())


def test_challenge_reader_has_absolute_deadline() -> None:
    class SlowSocket:
        def settimeout(self, timeout: float) -> None:
            assert timeout > 0

        def recv(self, size: int) -> bytes:
            assert size > 0
            time.sleep(0.02)
            return b"{"

    publisher = RemotePosePublisher(PEER, TOKEN, handshake_timeout_seconds=0.01)

    with pytest.raises(RemotePoseProtocolError, match="deadline"):
        publisher._receive_challenge_line(SlowSocket())  # type: ignore[arg-type]


def test_service_epoch_change_is_terminal_and_never_reauthorizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakeSocket()
    second = FakeSocket(
        server_challenge=challenge(
            service_epoch="e" * 64,
            server_nonce="f" * 64,
        )
    )
    connections = iter((first, second))
    monkeypatch.setattr(
        "recoverybox.remote_pose.socket.create_connection",
        lambda *_args, **_kwargs: next(connections),
    )
    publisher = RemotePosePublisher(
        PEER,
        TOKEN,
        authorize_initial_epoch=True,
        request_poll_timeout_seconds=0.01,
        reconnect_backoff_seconds=0.01,
        close_timeout_seconds=0.2,
    )
    publisher.start()
    wait_until(lambda: publisher.connected and len(first.sent) >= 2)
    first.feed(ConnectionResetError())
    wait_until(lambda: publisher.failure_kind == "ServiceEpochChanged")

    assert [item.kind for item in decoded_messages(first)] == [
        RemotePoseKind.START,
        RemotePoseKind.RESUME,
    ]
    assert second.sent == []
    assert not publisher.connected
    publisher.close()


def test_plain_start_never_authorizes_until_explicit_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeSocket()
    monkeypatch.setattr(
        "recoverybox.remote_pose.socket.create_connection",
        lambda *_args, **_kwargs: connection,
    )
    publisher = RemotePosePublisher(
        PEER,
        TOKEN,
        request_poll_timeout_seconds=0.01,
        close_timeout_seconds=0.2,
    )
    publisher.start()
    wait_until(lambda: publisher.connected)
    time.sleep(0.03)
    assert [item.kind for item in decoded_messages(connection)] == [RemotePoseKind.START]

    publisher.request_resume()
    wait_until(lambda: len(connection.sent) == 2)
    assert decoded_messages(connection)[1].kind is RemotePoseKind.RESUME
    publisher.close()


def test_same_epoch_socket_reconnect_rotates_nonce_without_reauthorizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakeSocket()
    second = FakeSocket(server_challenge=challenge(server_nonce="d" * 64))
    connections = iter((first, second))
    monkeypatch.setattr(
        "recoverybox.remote_pose.socket.create_connection",
        lambda *_args, **_kwargs: next(connections),
    )
    publisher = RemotePosePublisher(
        PEER,
        TOKEN,
        authorize_initial_epoch=True,
        request_poll_timeout_seconds=0.01,
        reconnect_backoff_seconds=0.01,
        close_timeout_seconds=0.2,
    )
    publisher.start()
    wait_until(lambda: publisher.connected and len(first.sent) >= 2)
    first.feed(ConnectionResetError())
    wait_until(lambda: publisher.connected and len(second.sent) >= 1)
    time.sleep(0.03)

    assert [item.kind for item in decoded_messages(first)] == [
        RemotePoseKind.START,
        RemotePoseKind.RESUME,
    ]
    assert [item.kind for item in decoded_messages(second)] == [RemotePoseKind.START]
    assert decoded_messages(second)[0].server_nonce == "d" * 64
    publisher.close()
    assert [item.kind for item in decoded_messages(second)] == [
        RemotePoseKind.START,
        RemotePoseKind.STOP,
    ]


def test_stop_preempts_blocked_analysis_and_reconnects_without_replaying_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_send_entered = threading.Event()
    first: FakeSocket

    def fail_when_preempted(line: bytes, _: int) -> None:
        if json.loads(line)["kind"] == "analysis":
            analysis_send_entered.set()
            assert first.shutdown_called.wait(1.0)
            raise ConnectionResetError()

    first = FakeSocket(send_behavior=fail_when_preempted)
    second = FakeSocket(server_challenge=challenge(server_nonce="d" * 64))
    connections = iter((first, second))
    monkeypatch.setattr(
        "recoverybox.remote_pose.socket.create_connection",
        lambda *_args, **_kwargs: next(connections),
    )
    publisher = RemotePosePublisher(
        PEER,
        TOKEN,
        authorize_initial_epoch=True,
        request_poll_timeout_seconds=0.01,
        reconnect_backoff_seconds=0.01,
        close_timeout_seconds=0.5,
        clock=lambda: 1.0,
    )
    publisher.start()
    wait_until(lambda: publisher.connected and len(first.sent) >= 2)
    issued = request(1, session_id=publisher.session_id)
    first.feed(encode_remote_pose_request(issued, TOKEN))
    claimed = publisher.wait_for_request(0.2)
    assert claimed is not None
    publisher.submit(analysis(), request=claimed, evidence_age_ms=1)
    assert analysis_send_entered.wait(0.5)

    publisher.close()

    assert [item.kind for item in decoded_messages(first)] == [
        RemotePoseKind.START,
        RemotePoseKind.RESUME,
    ]
    assert [item.kind for item in decoded_messages(second)] == [
        RemotePoseKind.START,
        RemotePoseKind.STOP,
    ]
    assert all(item.server_nonce == "d" * 64 for item in decoded_messages(second))


@pytest.mark.parametrize(
    "peer",
    [
        "",
        "pi3:45873",
        "100.63.255.255:45873",
        "100.128.0.0:45873",
        "127.0.0.1:45873",
        "100.106.237.106:0",
        "100.106.237.106:65536",
        "[fd7a:115c:a1e0::1]:45873",
        "http://100.106.237.106:45873",
    ],
)
def test_publisher_rejects_nonliteral_or_non_tailnet_peer(peer: str) -> None:
    with pytest.raises(ValueError):
        RemotePosePublisher(peer, TOKEN)


def test_publisher_accepts_canonical_tailscale_ipv4(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = FakeSocket()
    captured: list[tuple[str, int]] = []

    def connect(peer: tuple[str, int], **_: object) -> FakeSocket:
        captured.append(peer)
        return connection

    monkeypatch.setattr("recoverybox.remote_pose.socket.create_connection", connect)
    publisher = RemotePosePublisher(PEER, TOKEN, request_poll_timeout_seconds=0.01)
    publisher.start()
    wait_until(lambda: publisher.connected)
    publisher.close()
    assert captured == [("100.106.237.106", 45873)]


def test_protocol_version_is_bumped_for_challenge_request_response() -> None:
    assert REMOTE_POSE_VERSION == 2
    assert MAX_REMOTE_POSE_EVIDENCE_AGE_MS == 500
