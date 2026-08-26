"""Authenticated, numeric-only transport for remote squat analysis.

The camera and landmark model remain process-local on the laptop.  This module
can carry only the closed :class:`~recoverybox.exercise.SquatAnalysis` schema
over one outbound TCP connection; it has no representation for images,
landmarks, audio, transcripts, or arbitrary metadata.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import stat
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Final

from recoverybox.exercise import (
    SquatAnalysis,
    SquatAssessmentIssue,
    SquatEvent,
    SquatEventType,
    SquatPhase,
)

REMOTE_POSE_VERSION: Final = 2
MAX_REMOTE_POSE_PAYLOAD_BYTES: Final = 8192
MAX_REMOTE_POSE_CHALLENGE_BYTES: Final = 512
MAX_REMOTE_POSE_SEQUENCE: Final = (1 << 63) - 1
REMOTE_POSE_TOKEN_BYTES: Final = 32
REMOTE_POSE_SERVER_NONCE_BYTES: Final = 32
DEFAULT_REMOTE_POSE_ANALYSIS_QUEUE_SECONDS: Final = 0.4
MAX_REMOTE_POSE_EVIDENCE_AGE_MS: Final = 500
_MAX_GUARDIAN_POSE_AGE_SECONDS: Final = MAX_REMOTE_POSE_EVIDENCE_AGE_MS / 1000
_TAILSCALE_IPV4_NETWORK: Final = ipaddress.IPv4Network("100.64.0.0/10")

_SESSION_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_HMAC_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SERVER_NONCE_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_FILE_PATTERN = re.compile(rb"[0-9a-fA-F]{64}\n?\Z")

_COMMON_KEYS = frozenset({"version", "kind", "session_id", "hmac"})
_CHALLENGE_KEYS = frozenset({"version", "kind", "service_epoch", "server_nonce", "hmac"})
_REQUEST_KEYS = frozenset(
    {
        "version",
        "kind",
        "session_id",
        "service_epoch",
        "server_nonce",
        "request_sequence",
        "request_nonce",
        "hmac",
    }
)
_ANALYSIS_KEYS = frozenset(
    {
        "timestamp_ms",
        "assessable",
        "phase",
        "rep_count",
        "events",
        "issues",
        "confidence",
        "knee_angle_degrees",
        "arms_in_t",
    }
)
_EVENT_KEYS = frozenset({"event_type", "rep_count"})


class RemotePoseProtocolError(ValueError):
    """An authenticated remote-pose line failed its closed protocol schema."""


class RemotePoseKind(StrEnum):
    """Closed message kinds accepted by the Pi pose receiver."""

    START = "start"
    ANALYSIS = "analysis"
    RESUME = "resume"
    STOP = "stop"


def _is_sequence(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and 1 <= value <= MAX_REMOTE_POSE_SEQUENCE
    )


@dataclass(frozen=True, slots=True)
class RemotePoseMessage:
    """One validated protocol message.

    ``start`` is deliberately unsequenced.  All messages after it have a
    strictly positive signed-64-bit sequence number.  The receiver owns
    session-order and duplicate checks across reconnects.
    """

    kind: RemotePoseKind
    session_id: str
    sequence: int | None = None
    analysis: SquatAnalysis | None = None
    server_nonce: str | None = None
    service_epoch: str | None = None
    evidence_age_ms: int | None = None
    request_sequence: int | None = None
    request_nonce: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RemotePoseKind):
            raise TypeError("kind must be a RemotePoseKind")
        if not isinstance(self.session_id, str) or not _SESSION_ID_PATTERN.fullmatch(
            self.session_id
        ):
            raise ValueError("session_id must be exactly 32 lowercase hexadecimal characters")

        _require_server_nonce(self.server_nonce)
        _require_service_epoch(self.service_epoch)

        if self.kind is RemotePoseKind.START:
            if self.sequence is not None:
                raise ValueError("start messages must not carry a sequence")
            if self.analysis is not None:
                raise ValueError("start messages must not carry an analysis")
            if self.evidence_age_ms is not None:
                raise ValueError("start messages must not carry evidence age")
            if self.request_sequence is not None or self.request_nonce is not None:
                raise ValueError("start messages must not carry a request binding")
            return

        if not _is_sequence(self.sequence):
            raise ValueError("messages after start require a bounded positive sequence")
        if self.kind is RemotePoseKind.ANALYSIS:
            if not isinstance(self.analysis, SquatAnalysis):
                raise TypeError("analysis messages require a SquatAnalysis")
            _require_fresh_evidence_age_ms(self.evidence_age_ms)
            if not _is_sequence(self.request_sequence):
                raise ValueError("analysis messages require a bounded request_sequence")
            _require_request_nonce(self.request_nonce)
        elif self.analysis is not None:
            raise ValueError("control messages must not carry an analysis")
        elif self.evidence_age_ms is not None:
            raise ValueError("control messages must not carry evidence age")
        elif self.request_sequence is not None or self.request_nonce is not None:
            raise ValueError("control messages must not carry a request binding")


@dataclass(frozen=True, slots=True)
class RemotePoseChallenge:
    """Fresh authenticated Pi identity for one TCP connection and service epoch."""

    service_epoch: str
    server_nonce: str

    def __post_init__(self) -> None:
        _require_service_epoch(self.service_epoch)
        _require_server_nonce(self.server_nonce)


@dataclass(frozen=True, slots=True)
class RemotePoseRequest:
    """One authenticated Pi request that gates exactly one fresh capture."""

    session_id: str
    service_epoch: str
    server_nonce: str
    request_sequence: int
    request_nonce: str

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not _SESSION_ID_PATTERN.fullmatch(
            self.session_id
        ):
            raise ValueError("session_id must be exactly 32 lowercase hexadecimal characters")
        _require_service_epoch(self.service_epoch)
        _require_server_nonce(self.server_nonce)
        if not _is_sequence(self.request_sequence):
            raise ValueError("request_sequence must be a bounded positive integer")
        _require_request_nonce(self.request_nonce)


def _require_server_nonce(value: object) -> str:
    if not isinstance(value, str) or not _SERVER_NONCE_PATTERN.fullmatch(value):
        raise ValueError("server_nonce must be exactly 64 lowercase hexadecimal characters")
    return value


def _require_service_epoch(value: object) -> str:
    if not isinstance(value, str) or not _SERVER_NONCE_PATTERN.fullmatch(value):
        raise ValueError("service_epoch must be exactly 64 lowercase hexadecimal characters")
    return value


def _require_request_nonce(value: object) -> str:
    if not isinstance(value, str) or not _SERVER_NONCE_PATTERN.fullmatch(value):
        raise ValueError("request_nonce must be exactly 64 lowercase hexadecimal characters")
    return value


def _require_fresh_evidence_age_ms(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < MAX_REMOTE_POSE_EVIDENCE_AGE_MS
    ):
        raise ValueError("evidence_age_ms must be an integer from 0 through 499")
    return value


def _require_non_negative_evidence_age_ms(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_REMOTE_POSE_SEQUENCE
    ):
        raise ValueError("evidence_age_ms must be a bounded non-negative integer")
    return value


def new_remote_pose_server_nonce() -> str:
    """Return one cryptographically random per-connection server challenge."""

    return secrets.token_hex(REMOTE_POSE_SERVER_NONCE_BYTES)


def new_remote_pose_service_epoch() -> str:
    """Return one fresh identifier for the lifetime of one Pi service process."""

    return secrets.token_hex(REMOTE_POSE_SERVER_NONCE_BYTES)


def new_remote_pose_request_nonce() -> str:
    """Return one fresh identifier for a single Pi-to-laptop pose request."""

    return secrets.token_hex(REMOTE_POSE_SERVER_NONCE_BYTES)


def _require_token(token: object) -> bytes:
    if type(token) is not bytes:
        raise TypeError("token must be immutable bytes")
    if len(token) != REMOTE_POSE_TOKEN_BYTES:
        raise ValueError("token must contain exactly 32 bytes")
    return token


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise RemotePoseProtocolError("remote pose message is not canonical JSON") from exc


def _analysis_to_wire(analysis: SquatAnalysis) -> dict[str, object]:
    return {
        "timestamp_ms": analysis.timestamp_ms,
        "assessable": analysis.assessable,
        "phase": analysis.phase.value,
        "rep_count": analysis.rep_count,
        "events": [
            {"event_type": event.event_type.value, "rep_count": event.rep_count}
            for event in analysis.events
        ],
        "issues": [issue.value for issue in analysis.issues],
        "confidence": analysis.confidence,
        "knee_angle_degrees": analysis.knee_angle_degrees,
        "arms_in_t": analysis.arms_in_t,
    }


def _unsigned_envelope(message: RemotePoseMessage) -> dict[str, object]:
    envelope: dict[str, object] = {
        "version": REMOTE_POSE_VERSION,
        "kind": message.kind.value,
        "session_id": message.session_id,
    }
    if message.sequence is not None:
        envelope["sequence"] = message.sequence
    if message.analysis is not None:
        envelope["analysis"] = _analysis_to_wire(message.analysis)
    if message.server_nonce is not None:
        envelope["server_nonce"] = message.server_nonce
    if message.service_epoch is not None:
        envelope["service_epoch"] = message.service_epoch
    if message.evidence_age_ms is not None:
        envelope["evidence_age_ms"] = message.evidence_age_ms
    if message.request_sequence is not None:
        envelope["request_sequence"] = message.request_sequence
    if message.request_nonce is not None:
        envelope["request_nonce"] = message.request_nonce
    return envelope


def encode_remote_pose_message(message: RemotePoseMessage, token: bytes) -> bytes:
    """Return one canonical, HMAC-authenticated, LF-terminated JSON line."""

    if not isinstance(message, RemotePoseMessage):
        raise TypeError("message must be a RemotePoseMessage")
    key = _require_token(token)
    unsigned = _unsigned_envelope(message)
    signature = hmac.new(key, _canonical_json(unsigned), hashlib.sha256).hexdigest()
    signed = {**unsigned, "hmac": signature}
    line = _canonical_json(signed) + b"\n"
    if len(line) > MAX_REMOTE_POSE_PAYLOAD_BYTES:
        raise RemotePoseProtocolError("remote pose message exceeds the payload limit")
    return line


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RemotePoseProtocolError("remote pose message contains a duplicate key")
        value[key] = item
    return value


def _reject_json_constant(_: str) -> None:
    raise RemotePoseProtocolError("remote pose message contains a non-finite number")


def _decode_json_line(line: bytes) -> dict[str, Any]:
    if not line or len(line) > MAX_REMOTE_POSE_PAYLOAD_BYTES:
        raise RemotePoseProtocolError("remote pose message has an invalid payload size")
    if not line.endswith(b"\n") or b"\n" in line[:-1] or b"\r" in line:
        raise RemotePoseProtocolError("remote pose message must be one LF-terminated line")
    payload = line[:-1]
    try:
        text = payload.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except RemotePoseProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise RemotePoseProtocolError("remote pose message is not valid JSON") from None
    if type(parsed) is not dict:
        raise RemotePoseProtocolError("remote pose message must be a JSON object")
    try:
        canonical = _canonical_json(parsed)
    except RemotePoseProtocolError:
        raise RemotePoseProtocolError("remote pose message contains an invalid value") from None
    if payload != canonical:
        raise RemotePoseProtocolError("remote pose message is not canonical JSON")
    return parsed


def encode_remote_pose_challenge(challenge: RemotePoseChallenge, token: bytes) -> bytes:
    """Encode one canonical authenticated server challenge.

    The challenge authenticates the Pi to the publisher before the publisher
    reveals a signed session start.  Its fresh nonce is then bound into that
    START, so a recorded connection transcript cannot be replayed after a Pi
    restart.
    """

    if not isinstance(challenge, RemotePoseChallenge):
        raise TypeError("challenge must be a RemotePoseChallenge")
    key = _require_token(token)
    unsigned: dict[str, object] = {
        "version": REMOTE_POSE_VERSION,
        "kind": "challenge",
        "service_epoch": challenge.service_epoch,
        "server_nonce": challenge.server_nonce,
    }
    signature = hmac.new(key, _canonical_json(unsigned), hashlib.sha256).hexdigest()
    line = _canonical_json({**unsigned, "hmac": signature}) + b"\n"
    if len(line) > MAX_REMOTE_POSE_CHALLENGE_BYTES:
        raise RemotePoseProtocolError("remote pose challenge exceeds the payload limit")
    return line


def decode_remote_pose_challenge(line: bytes, token: bytes) -> RemotePoseChallenge:
    """Authenticate and reconstruct one canonical server challenge."""

    if type(line) is not bytes:
        raise TypeError("line must be immutable bytes")
    if len(line) > MAX_REMOTE_POSE_CHALLENGE_BYTES:
        raise RemotePoseProtocolError("remote pose challenge has an invalid payload size")
    key = _require_token(token)
    envelope = _decode_json_line(line)
    _require_exact_keys(envelope, _CHALLENGE_KEYS)
    if type(envelope["version"]) is not int or envelope["version"] != REMOTE_POSE_VERSION:
        raise RemotePoseProtocolError("remote pose challenge has an unsupported version")
    if envelope["kind"] != "challenge":
        raise RemotePoseProtocolError("remote pose challenge has an invalid kind")
    try:
        nonce = _require_server_nonce(envelope["server_nonce"])
        service_epoch = _require_service_epoch(envelope["service_epoch"])
    except ValueError:
        raise RemotePoseProtocolError("remote pose challenge has an invalid identifier") from None
    signature = envelope["hmac"]
    if not isinstance(signature, str) or not _HMAC_PATTERN.fullmatch(signature):
        raise RemotePoseProtocolError("remote pose challenge has an invalid authenticator")
    unsigned = {field: value for field, value in envelope.items() if field != "hmac"}
    expected = hmac.new(key, _canonical_json(unsigned), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise RemotePoseProtocolError("remote pose challenge authentication failed")
    return RemotePoseChallenge(service_epoch=service_epoch, server_nonce=nonce)


def encode_remote_pose_request(request: RemotePoseRequest, token: bytes) -> bytes:
    """Encode one canonical, authenticated Pi-to-laptop capture request."""

    if not isinstance(request, RemotePoseRequest):
        raise TypeError("request must be a RemotePoseRequest")
    key = _require_token(token)
    unsigned: dict[str, object] = {
        "version": REMOTE_POSE_VERSION,
        "kind": "request",
        "session_id": request.session_id,
        "service_epoch": request.service_epoch,
        "server_nonce": request.server_nonce,
        "request_sequence": request.request_sequence,
        "request_nonce": request.request_nonce,
    }
    signature = hmac.new(key, _canonical_json(unsigned), hashlib.sha256).hexdigest()
    line = _canonical_json({**unsigned, "hmac": signature}) + b"\n"
    if len(line) > MAX_REMOTE_POSE_PAYLOAD_BYTES:
        raise RemotePoseProtocolError("remote pose request exceeds the payload limit")
    return line


def decode_remote_pose_request(line: bytes, token: bytes) -> RemotePoseRequest:
    """Authenticate and reconstruct one strict Pi-to-laptop capture request."""

    if type(line) is not bytes:
        raise TypeError("line must be immutable bytes")
    key = _require_token(token)
    envelope = _decode_json_line(line)
    _require_exact_keys(envelope, _REQUEST_KEYS)
    if type(envelope["version"]) is not int or envelope["version"] != REMOTE_POSE_VERSION:
        raise RemotePoseProtocolError("remote pose request has an unsupported version")
    if envelope["kind"] != "request":
        raise RemotePoseProtocolError("remote pose request has an invalid kind")
    signature = envelope["hmac"]
    if not isinstance(signature, str) or not _HMAC_PATTERN.fullmatch(signature):
        raise RemotePoseProtocolError("remote pose request has an invalid authenticator")
    unsigned = {field: value for field, value in envelope.items() if field != "hmac"}
    expected = hmac.new(key, _canonical_json(unsigned), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise RemotePoseProtocolError("remote pose request authentication failed")
    try:
        return RemotePoseRequest(
            session_id=envelope["session_id"],
            service_epoch=envelope["service_epoch"],
            server_nonce=envelope["server_nonce"],
            request_sequence=envelope["request_sequence"],
            request_nonce=envelope["request_nonce"],
        )
    except (TypeError, ValueError):
        raise RemotePoseProtocolError("remote pose request violates its schema") from None


def _require_exact_keys(value: object, expected: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        raise RemotePoseProtocolError("remote pose message has fields outside its closed schema")
    return value


def _require_int(value: object, *, non_negative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RemotePoseProtocolError("remote pose message has an invalid integer")
    if non_negative and value < 0:
        raise RemotePoseProtocolError("remote pose message has an invalid integer")
    return value


def _require_finite_real(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RemotePoseProtocolError("remote pose message has an invalid number")
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        raise RemotePoseProtocolError("remote pose message has an invalid number") from None
    if not isfinite(converted):
        raise RemotePoseProtocolError("remote pose message has a non-finite number")
    return converted


def _analysis_from_wire(value: object) -> SquatAnalysis:
    analysis = _require_exact_keys(value, _ANALYSIS_KEYS)
    if not isinstance(analysis["assessable"], bool):
        raise RemotePoseProtocolError("remote pose message has an invalid boolean")
    if analysis["arms_in_t"] is not None and not isinstance(analysis["arms_in_t"], bool):
        raise RemotePoseProtocolError("remote pose message has an invalid optional boolean")
    if analysis["knee_angle_degrees"] is not None:
        knee_angle = _require_finite_real(analysis["knee_angle_degrees"])
    else:
        knee_angle = None
    confidence = _require_finite_real(analysis["confidence"])

    events_value = analysis["events"]
    if type(events_value) is not list:
        raise RemotePoseProtocolError("remote pose events must be an array")
    events: list[SquatEvent] = []
    for event_value in events_value:
        event = _require_exact_keys(event_value, _EVENT_KEYS)
        if not isinstance(event["event_type"], str):
            raise RemotePoseProtocolError("remote pose event type must be a string")
        rep_count = event["rep_count"]
        if rep_count is not None:
            rep_count = _require_int(rep_count)
        try:
            events.append(
                SquatEvent(
                    event_type=SquatEventType(event["event_type"]),
                    rep_count=rep_count,
                )
            )
        except (TypeError, ValueError):
            raise RemotePoseProtocolError("remote pose message has an invalid event") from None

    issues_value = analysis["issues"]
    if type(issues_value) is not list or not all(isinstance(issue, str) for issue in issues_value):
        raise RemotePoseProtocolError("remote pose issues must be an array of strings")
    try:
        phase = SquatPhase(analysis["phase"])
        issues = tuple(SquatAssessmentIssue(issue) for issue in issues_value)
    except (TypeError, ValueError):
        raise RemotePoseProtocolError("remote pose message has an invalid enum") from None

    try:
        return SquatAnalysis(
            timestamp_ms=_require_int(analysis["timestamp_ms"]),
            assessable=analysis["assessable"],
            phase=phase,
            rep_count=_require_int(analysis["rep_count"]),
            events=tuple(events),
            issues=issues,
            confidence=confidence,
            knee_angle_degrees=knee_angle,
            arms_in_t=analysis["arms_in_t"],
        )
    except (TypeError, ValueError):
        raise RemotePoseProtocolError("remote pose analysis violates its schema") from None


def decode_remote_pose_message(line: bytes, token: bytes) -> RemotePoseMessage:
    """Authenticate and reconstruct one strict remote-pose JSON line."""

    if type(line) is not bytes:
        raise TypeError("line must be immutable bytes")
    key = _require_token(token)
    envelope = _decode_json_line(line)

    signature = envelope.get("hmac")
    if not isinstance(signature, str) or not _HMAC_PATTERN.fullmatch(signature):
        raise RemotePoseProtocolError("remote pose message has an invalid authenticator")
    unsigned = {field: value for field, value in envelope.items() if field != "hmac"}
    expected = hmac.new(key, _canonical_json(unsigned), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise RemotePoseProtocolError("remote pose message authentication failed")

    if type(envelope.get("version")) is not int or envelope["version"] != REMOTE_POSE_VERSION:
        raise RemotePoseProtocolError("remote pose message has an unsupported version")
    kind_value = envelope.get("kind")
    if not isinstance(kind_value, str):
        raise RemotePoseProtocolError("remote pose kind must be a string")
    try:
        kind = RemotePoseKind(kind_value)
    except ValueError:
        raise RemotePoseProtocolError("remote pose message has an invalid kind") from None

    expected_keys = _COMMON_KEYS | {"service_epoch", "server_nonce"}
    if kind is not RemotePoseKind.START:
        expected_keys = expected_keys | {"sequence"}
    if kind is RemotePoseKind.ANALYSIS:
        expected_keys = expected_keys | {
            "analysis",
            "evidence_age_ms",
            "request_sequence",
            "request_nonce",
        }
    _require_exact_keys(envelope, frozenset(expected_keys))

    session_id = envelope["session_id"]
    sequence: int | None = None
    analysis: SquatAnalysis | None = None
    server_nonce: str | None = None
    service_epoch: str | None = None
    evidence_age_ms: int | None = None
    request_sequence: int | None = None
    request_nonce: str | None = None
    try:
        server_nonce = _require_server_nonce(envelope["server_nonce"])
        service_epoch = _require_service_epoch(envelope["service_epoch"])
    except ValueError:
        raise RemotePoseProtocolError("remote pose message has an invalid server binding") from None
    if kind is not RemotePoseKind.START:
        sequence = envelope["sequence"]
        if not _is_sequence(sequence):
            raise RemotePoseProtocolError("remote pose message has an invalid sequence")
    if kind is RemotePoseKind.ANALYSIS:
        analysis = _analysis_from_wire(envelope["analysis"])
        try:
            evidence_age_ms = _require_fresh_evidence_age_ms(envelope["evidence_age_ms"])
            request_sequence = envelope["request_sequence"]
            if not _is_sequence(request_sequence):
                raise ValueError
            request_nonce = _require_request_nonce(envelope["request_nonce"])
        except ValueError:
            raise RemotePoseProtocolError(
                "remote pose message has an invalid evidence or request binding"
            ) from None
    try:
        return RemotePoseMessage(
            kind=kind,
            session_id=session_id,
            sequence=sequence,
            analysis=analysis,
            server_nonce=server_nonce,
            service_epoch=service_epoch,
            evidence_age_ms=evidence_age_ms,
            request_sequence=request_sequence,
            request_nonce=request_nonce,
        )
    except (TypeError, ValueError):
        raise RemotePoseProtocolError("remote pose message violates its envelope schema") from None


def load_remote_pose_token(path: str | Path) -> bytes:
    """Load a 32-byte token from an owner-only regular 64-hex file.

    A single trailing LF is accepted so a securely provisioned text file can
    remain POSIX-friendly.  Symlinks and any group/other permission bits are
    rejected before the contents are interpreted.
    """

    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("remote pose token file could not be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("remote pose token path must be a regular file")
        permissions = stat.S_IMODE(metadata.st_mode)
        if (
            metadata.st_uid != os.geteuid()
            or not permissions & stat.S_IRUSR
            or permissions & (stat.S_IXUSR | stat.S_IRWXG | stat.S_IRWXO)
        ):
            raise PermissionError("remote pose token file must be owner-only")
        encoded = os.read(descriptor, 66)
    finally:
        os.close(descriptor)
    if not _TOKEN_FILE_PATTERN.fullmatch(encoded):
        raise ValueError("remote pose token file must contain exactly 64 hexadecimal characters")
    if encoded.endswith(b"\n"):
        encoded = encoded[:-1]
    token = bytes.fromhex(encoded.decode("ascii"))
    return _require_token(token)


@dataclass(frozen=True, slots=True)
class _PendingMessage:
    kind: RemotePoseKind
    analysis: SquatAnalysis | None = None
    enqueued_at_seconds: float | None = None
    evidence_age_ms: int | None = None
    request: RemotePoseRequest | None = None

    @property
    def is_priority(self) -> bool:
        if self.kind is not RemotePoseKind.ANALYSIS:
            return True
        assert self.analysis is not None
        return not self.analysis.assessable or bool(self.analysis.events)


def _positive_timeout(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} must be between 0 and 60 seconds") from exc
    if not isfinite(converted) or not 0.0 < converted <= 60.0:
        raise ValueError(f"{field_name} must be between 0 and 60 seconds")
    return converted


def _parse_peer(peer: str) -> tuple[str, int]:
    if not isinstance(peer, str):
        raise TypeError("peer must be a literal Tailscale IPv4:port string")
    host, separator, port_text = peer.rpartition(":")
    if not separator or ":" in host:
        raise ValueError("peer must be a literal Tailscale IPv4:port string")
    if (
        not host
        or host != host.strip()
        or any(character.isspace() or ord(character) < 32 for character in host)
        or any(character in host for character in "/?#@")
        or not port_text.isascii()
        or not port_text.isdecimal()
    ):
        raise ValueError("peer must be a literal Tailscale IPv4:port string")
    try:
        address = ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError:
        raise ValueError("peer must use a literal Tailscale IPv4 address") from None
    if str(address) != host or address not in _TAILSCALE_IPV4_NETWORK:
        raise ValueError("peer must use a literal Tailscale IPv4 address")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError("peer port must be between 1 and 65535")
    return host, port


class RemotePosePublisher:
    """Non-blocking, reconnecting publisher for the laptop camera loop.

    Only the daemon worker performs DNS, connect, or socket writes.  The
    caller-facing methods take a lock briefly to update a bounded deque.
    Routine assessable frames are coalesced under load; exercise events,
    withheld assessments, and controls evict routine frames first.
    """

    def __init__(
        self,
        peer: str,
        token: bytes,
        *,
        max_pending_messages: int = 16,
        authorize_initial_epoch: bool = False,
        connect_timeout_seconds: float = 1.0,
        handshake_timeout_seconds: float = 0.25,
        request_poll_timeout_seconds: float = 0.05,
        send_timeout_seconds: float = 0.25,
        reconnect_backoff_seconds: float = 0.25,
        close_timeout_seconds: float = 2.0,
        max_analysis_queue_seconds: float = DEFAULT_REMOTE_POSE_ANALYSIS_QUEUE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(max_pending_messages, bool)
            or not isinstance(max_pending_messages, int)
            or max_pending_messages < 1
        ):
            raise ValueError("max_pending_messages must be a positive integer")
        if type(authorize_initial_epoch) is not bool:
            raise TypeError("authorize_initial_epoch must be a boolean")
        self._peer = _parse_peer(peer)
        self._token = _require_token(token)
        self._max_pending_messages = max_pending_messages
        self._authorize_initial_epoch = authorize_initial_epoch
        self._connect_timeout_seconds = _positive_timeout(
            connect_timeout_seconds,
            field_name="connect_timeout_seconds",
        )
        self._handshake_timeout_seconds = _positive_timeout(
            handshake_timeout_seconds,
            field_name="handshake_timeout_seconds",
        )
        if self._handshake_timeout_seconds > _MAX_GUARDIAN_POSE_AGE_SECONDS:
            raise ValueError("handshake_timeout_seconds cannot exceed 0.5 seconds")
        self._request_poll_timeout_seconds = _positive_timeout(
            request_poll_timeout_seconds,
            field_name="request_poll_timeout_seconds",
        )
        if self._request_poll_timeout_seconds > _MAX_GUARDIAN_POSE_AGE_SECONDS:
            raise ValueError("request_poll_timeout_seconds cannot exceed 0.5 seconds")
        self._send_timeout_seconds = _positive_timeout(
            send_timeout_seconds,
            field_name="send_timeout_seconds",
        )
        if self._send_timeout_seconds > _MAX_GUARDIAN_POSE_AGE_SECONDS:
            raise ValueError("send_timeout_seconds cannot exceed 0.5 seconds")
        self._reconnect_backoff_seconds = _positive_timeout(
            reconnect_backoff_seconds,
            field_name="reconnect_backoff_seconds",
        )
        self._close_timeout_seconds = _positive_timeout(
            close_timeout_seconds,
            field_name="close_timeout_seconds",
        )
        self._max_analysis_queue_seconds = _positive_timeout(
            max_analysis_queue_seconds,
            field_name="max_analysis_queue_seconds",
        )
        if self._max_analysis_queue_seconds > _MAX_GUARDIAN_POSE_AGE_SECONDS:
            raise ValueError("max_analysis_queue_seconds cannot exceed 0.5 seconds")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._session_id = secrets.token_hex(16)
        self._condition = threading.Condition()
        self._pending: deque[_PendingMessage] = deque()
        self._started = False
        self._closing = False
        self._abort_requested = False
        self._connected = False
        self._failure_kind: str | None = None
        self._messages_sent = 0
        self._next_sequence = 1
        self._socket: socket.socket | None = None
        self._sending_kind: RemotePoseKind | None = None
        self._active_challenge: RemotePoseChallenge | None = None
        self._service_epoch: str | None = None
        self._service_epoch_changed = False
        self._available_request: RemotePoseRequest | None = None
        self._claimed_request: RemotePoseRequest | None = None
        self._last_request_sequence = 0
        self._request_buffer = bytearray()
        self._request_line_started_at: float | None = None
        self._initial_authorization_queued = False
        self._worker: threading.Thread | None = None

    @property
    def connected(self) -> bool:
        with self._condition:
            return self._connected

    @property
    def failure_kind(self) -> str | None:
        with self._condition:
            return self._failure_kind

    @property
    def messages_sent(self) -> int:
        with self._condition:
            return self._messages_sent

    @property
    def session_id(self) -> str:
        return self._session_id

    def start(self) -> None:
        """Start the socket worker without waiting for a connection."""

        with self._condition:
            if self._closing:
                raise RuntimeError("remote pose publisher is closed")
            if self._started:
                return
            self._started = True
            self._worker = threading.Thread(
                target=self._run,
                name="recoverybox-remote-pose-publisher",
                daemon=True,
            )
            self._worker.start()

    def wait_for_request(self, timeout_seconds: float | None = None) -> RemotePoseRequest | None:
        """Wait for and claim one authenticated Pi request.

        Returning a request is the only authorization to acquire the next raw
        frame.  No camera read or pose inference should happen before it.
        """

        if timeout_seconds is not None:
            timeout = _positive_timeout(timeout_seconds, field_name="timeout_seconds")
        else:
            timeout = None
        with self._condition:
            self._require_running_locked()
            deadline = None if timeout is None else self._clock_now() + timeout
            while self._available_request is None:
                if self._closing or self._service_epoch_changed:
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - self._clock_now()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            request = self._available_request
            self._available_request = None
            self._claimed_request = request
            return request

    def submit(
        self,
        analysis: SquatAnalysis,
        *,
        request: RemotePoseRequest,
        evidence_age_ms: int,
    ) -> None:
        """Queue fresh numeric evidence without doing network work here.

        ``evidence_age_ms`` is the launcher's conservative, rounded-up age
        from camera capture through tracker completion.  The worker adds its
        own queue residence before signing the message.  Evidence already at
        the Guardian's 500 ms limit is deliberately withheld; the Pi watchdog
        then fails closed.
        """

        if not isinstance(analysis, SquatAnalysis):
            raise TypeError("analysis must be a SquatAnalysis")
        if not isinstance(request, RemotePoseRequest):
            raise TypeError("request must be a RemotePoseRequest")
        initial_age_ms = _require_non_negative_evidence_age_ms(evidence_age_ms)
        with self._condition:
            self._require_running_locked()
            if request != self._claimed_request:
                raise RuntimeError("remote pose response does not match the claimed request")
            self._claimed_request = None
            if initial_age_ms >= MAX_REMOTE_POSE_EVIDENCE_AGE_MS:
                self._condition.notify_all()
                return
            pending = _PendingMessage(
                RemotePoseKind.ANALYSIS,
                analysis,
                self._clock_now(),
                initial_age_ms,
                request,
            )
            self._enqueue_locked(pending)
            self._condition.notify()

    def request_resume(self) -> None:
        """Queue an authenticated resume control without waiting for the peer."""

        pending = _PendingMessage(RemotePoseKind.RESUME)
        with self._condition:
            self._require_running_locked()
            if any(item.kind is RemotePoseKind.RESUME for item in self._pending):
                return
            self._enqueue_locked(pending)
            self._condition.notify()

    def close(self) -> None:
        """Attempt a bounded STOP flush, then asynchronously abort if needed."""

        worker: threading.Thread | None
        preempt_connection: socket.socket | None
        with self._condition:
            if self._closing:
                return
            self._closing = True
            if not self._started:
                return
            # Local stop has already won.  No queued cue-producing evidence or
            # resume request may overtake its remote STOP.
            self._pending.clear()
            self._pending.append(_PendingMessage(RemotePoseKind.STOP))
            worker = self._worker
            preempt_connection = (
                self._socket if self._sending_kind not in {None, RemotePoseKind.STOP} else None
            )
            self._condition.notify_all()
        if preempt_connection is not None:
            threading.Thread(
                target=self._abort_socket,
                args=(preempt_connection,),
                name="recoverybox-remote-pose-stop-preempt",
                daemon=True,
            ).start()
        if worker is None or worker is threading.current_thread():
            return
        worker.join(self._close_timeout_seconds)
        if not worker.is_alive():
            return
        with self._condition:
            connection = self._socket
            self._connected = False
            self._failure_kind = "CloseTimeout"
            self._abort_requested = True
            self._condition.notify_all()
        if connection is not None:
            threading.Thread(
                target=self._abort_socket,
                args=(connection,),
                name="recoverybox-remote-pose-abort",
                daemon=True,
            ).start()

    def _require_running_locked(self) -> None:
        if not self._started:
            raise RuntimeError("remote pose publisher has not been started")
        if self._closing:
            raise RuntimeError("remote pose publisher is closed")

    def _enqueue_locked(self, incoming: _PendingMessage) -> None:
        if not incoming.is_priority:
            for index, item in enumerate(self._pending):
                if not item.is_priority:
                    del self._pending[index]
                    self._pending.append(incoming)
                    return
            if len(self._pending) < self._max_pending_messages:
                self._pending.append(incoming)
            return

        # A priority item makes every older routine frame obsolete.  Removing
        # it also prevents a lower timestamp from being sent after the newer
        # priority evidence.
        self._pending = deque(item for item in self._pending if item.is_priority)
        if len(self._pending) < self._max_pending_messages:
            self._pending.append(incoming)
            return

        if incoming.kind is RemotePoseKind.STOP:
            self._pending.popleft()
            self._pending.append(incoming)
            return
        if incoming.kind is RemotePoseKind.RESUME:
            for index, item in enumerate(self._pending):
                if item.kind is RemotePoseKind.ANALYSIS:
                    del self._pending[index]
                    self._pending.append(incoming)
                    return
            return
        for index, item in enumerate(self._pending):
            if item.kind is RemotePoseKind.ANALYSIS:
                del self._pending[index]
                self._pending.append(incoming)
                return

    def _pop_next_locked(self) -> _PendingMessage | None:
        while self._pending:
            pending = self._pending.popleft()
            if pending.kind is RemotePoseKind.ANALYSIS:
                enqueued_at = pending.enqueued_at_seconds
                initial_age_ms = pending.evidence_age_ms
                now = self._clock_now()
                if enqueued_at is None or initial_age_ms is None or now < enqueued_at:
                    continue
                queue_seconds = now - enqueued_at
                total_age_ms = initial_age_ms + math.ceil(queue_seconds * 1000.0)
                if (
                    total_age_ms >= MAX_REMOTE_POSE_EVIDENCE_AGE_MS
                    or queue_seconds >= self._max_analysis_queue_seconds
                ):
                    continue
                pending = _PendingMessage(
                    kind=pending.kind,
                    analysis=pending.analysis,
                    enqueued_at_seconds=enqueued_at,
                    evidence_age_ms=total_age_ms,
                    request=pending.request,
                )
            return pending
        return None

    def _clock_now(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError("remote pose clock returned an invalid value")
        converted = float(value)
        if not isfinite(converted):
            raise RuntimeError("remote pose clock returned an invalid value")
        return converted

    def _message_for_pending_locked(self, pending: _PendingMessage) -> RemotePoseMessage:
        if self._next_sequence > MAX_REMOTE_POSE_SEQUENCE:
            raise OverflowError("remote pose sequence exhausted")
        challenge = self._active_challenge
        if challenge is None:
            raise RuntimeError("remote pose connection has no active challenge")
        sequence = self._next_sequence
        self._next_sequence += 1
        request = pending.request
        if pending.kind is RemotePoseKind.ANALYSIS:
            if request is None:
                raise RuntimeError("remote pose analysis has no request binding")
            if (
                request.session_id != self._session_id
                or request.service_epoch != challenge.service_epoch
                or request.server_nonce != challenge.server_nonce
            ):
                raise RuntimeError("remote pose request is no longer current")
        return RemotePoseMessage(
            kind=pending.kind,
            session_id=self._session_id,
            sequence=sequence,
            analysis=pending.analysis,
            server_nonce=challenge.server_nonce,
            service_epoch=challenge.service_epoch,
            evidence_age_ms=pending.evidence_age_ms,
            request_sequence=request.request_sequence if request is not None else None,
            request_nonce=request.request_nonce if request is not None else None,
        )

    def _run(self) -> None:
        current: RemotePoseMessage | None = None
        connection: socket.socket | None = None
        try:
            while True:
                with self._condition:
                    if self._abort_requested:
                        return
                if connection is None:
                    connection = self._connect_and_start()
                    if connection is None:
                        with self._condition:
                            if self._closing or self._service_epoch_changed:
                                return
                            self._condition.wait(self._reconnect_backoff_seconds)
                        continue
                    with self._condition:
                        abort_connected_socket = self._abort_requested
                    if abort_connected_socket:
                        self._close_socket(connection)
                        return

                if current is None:
                    with self._condition:
                        pending = self._pop_next_locked()
                        if pending is not None:
                            try:
                                current = self._message_for_pending_locked(pending)
                            except OverflowError:
                                self._failure_kind = "SequenceExhausted"
                                return
                            except RuntimeError as exc:
                                self._failure_kind = type(exc).__name__
                                continue

                if current is not None:
                    try:
                        with self._condition:
                            if self._closing and current.kind is not RemotePoseKind.STOP:
                                current = None
                                continue
                            self._sending_kind = current.kind
                        connection.settimeout(self._send_timeout_seconds)
                        connection.sendall(encode_remote_pose_message(current, self._token))
                    except Exception as exc:
                        self._record_disconnect(connection, type(exc).__name__)
                        connection = None
                        if current.kind is not RemotePoseKind.STOP:
                            # A response or resume has ambiguous delivery and is
                            # bound to the old per-connection challenge.  Never
                            # replay it after reconnect.
                            current = None
                        with self._condition:
                            if not self._closing:
                                self._condition.wait(self._reconnect_backoff_seconds)
                        continue

                    kind = current.kind
                    with self._condition:
                        self._sending_kind = None
                        self._messages_sent += 1
                        self._failure_kind = None
                    current = None
                    if kind is RemotePoseKind.STOP:
                        return
                    continue

                try:
                    connection.settimeout(self._request_poll_timeout_seconds)
                    request_line = self._receive_request_line(connection)
                    if request_line is None:
                        continue
                    request = decode_remote_pose_request(request_line, self._token)
                    with self._condition:
                        challenge = self._active_challenge
                        if (
                            challenge is None
                            or request.session_id != self._session_id
                            or request.service_epoch != challenge.service_epoch
                            or request.server_nonce != challenge.server_nonce
                            or request.request_sequence <= self._last_request_sequence
                            or self._available_request is not None
                            or self._claimed_request is not None
                        ):
                            raise RemotePoseProtocolError(
                                "remote pose request is stale, replayed, or not current"
                            )
                        self._last_request_sequence = request.request_sequence
                        self._available_request = request
                        self._condition.notify_all()
                except TimeoutError:
                    continue
                except Exception as exc:
                    self._record_disconnect(connection, type(exc).__name__)
                    connection = None
                    with self._condition:
                        if not self._closing:
                            self._condition.wait(self._reconnect_backoff_seconds)
                    continue
        finally:
            if connection is not None:
                self._close_socket(connection)
            with self._condition:
                self._socket = None
                self._sending_kind = None
                self._connected = False
                self._condition.notify_all()

    def _connect_and_start(self) -> socket.socket | None:
        connection: socket.socket | None = None
        try:
            connection = socket.create_connection(
                self._peer,
                timeout=self._connect_timeout_seconds,
            )
            with self._condition:
                abort_connection = self._abort_requested
                if not abort_connection:
                    self._socket = connection
            if abort_connection:
                self._close_socket(connection)
                return None
            connection.settimeout(self._handshake_timeout_seconds)
            challenge_line = self._receive_challenge_line(connection)
            challenge = decode_remote_pose_challenge(challenge_line, self._token)
            with self._condition:
                self._discard_disconnected_messages_locked()
                if self._service_epoch is None:
                    self._service_epoch = challenge.service_epoch
                elif self._service_epoch != challenge.service_epoch:
                    self._service_epoch_changed = True
                    self._failure_kind = "ServiceEpochChanged"
                    self._condition.notify_all()
                    raise RemotePoseProtocolError("remote pose service epoch changed")
                self._active_challenge = challenge
                self._last_request_sequence = 0
            connection.settimeout(self._send_timeout_seconds)
            start = RemotePoseMessage(
                RemotePoseKind.START,
                self._session_id,
                server_nonce=challenge.server_nonce,
                service_epoch=challenge.service_epoch,
            )
            connection.sendall(encode_remote_pose_message(start, self._token))
        except Exception as exc:
            if connection is not None:
                self._close_socket(connection)
            with self._condition:
                self._socket = None
                self._connected = False
                if not self._service_epoch_changed:
                    self._failure_kind = type(exc).__name__
                self._discard_disconnected_messages_locked()
            return None
        with self._condition:
            self._socket = connection
            self._connected = True
            self._failure_kind = None
            self._messages_sent += 1
            if (
                self._authorize_initial_epoch
                and not self._initial_authorization_queued
                and not self._closing
            ):
                # A freshly launched laptop client is the explicit live-user
                # authorization for this pinned Pi service epoch. Queue it
                # exactly once; never reproduce it on reconnect or epoch
                # change, even when delivery later becomes ambiguous.
                self._pending.appendleft(_PendingMessage(RemotePoseKind.RESUME))
                self._initial_authorization_queued = True
                self._condition.notify_all()
        return connection

    def _receive_challenge_line(self, connection: socket.socket) -> bytes:
        payload = bytearray()
        deadline = time.monotonic() + self._handshake_timeout_seconds
        while len(payload) < MAX_REMOTE_POSE_CHALLENGE_BYTES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemotePoseProtocolError("remote pose challenge exceeded its deadline")
            connection.settimeout(min(self._handshake_timeout_seconds, remaining))
            try:
                chunk = connection.recv(MAX_REMOTE_POSE_CHALLENGE_BYTES - len(payload))
            except TimeoutError:
                raise RemotePoseProtocolError(
                    "remote pose challenge exceeded its deadline"
                ) from None
            if not chunk:
                raise RemotePoseProtocolError("remote pose challenge ended before one line")
            if time.monotonic() >= deadline:
                raise RemotePoseProtocolError("remote pose challenge exceeded its deadline")
            newline = chunk.find(b"\n")
            if newline >= 0:
                payload.extend(chunk[: newline + 1])
                if newline + 1 != len(chunk):
                    raise RemotePoseProtocolError("remote pose challenge has trailing bytes")
                return bytes(payload)
            payload.extend(chunk)
        raise RemotePoseProtocolError("remote pose challenge exceeds the payload limit")

    def _receive_request_line(self, connection: socket.socket) -> bytes | None:
        now = time.monotonic()
        started = self._request_line_started_at
        if started is not None and now - started >= _MAX_GUARDIAN_POSE_AGE_SECONDS:
            raise RemotePoseProtocolError("remote pose request exceeded its deadline")
        try:
            chunk = connection.recv(MAX_REMOTE_POSE_PAYLOAD_BYTES - len(self._request_buffer))
        except TimeoutError:
            now = time.monotonic()
            started = self._request_line_started_at
            if started is not None and now - started >= _MAX_GUARDIAN_POSE_AGE_SECONDS:
                raise RemotePoseProtocolError("remote pose request exceeded its deadline") from None
            return None
        if not chunk:
            raise RemotePoseProtocolError("remote pose request ended before one line")
        now = time.monotonic()
        if self._request_line_started_at is None:
            self._request_line_started_at = now
        elif now - self._request_line_started_at >= _MAX_GUARDIAN_POSE_AGE_SECONDS:
            raise RemotePoseProtocolError("remote pose request exceeded its deadline")
        newline = chunk.find(b"\n")
        if newline >= 0:
            self._request_buffer.extend(chunk[: newline + 1])
            if newline + 1 != len(chunk):
                raise RemotePoseProtocolError("remote pose request has trailing bytes")
            line = bytes(self._request_buffer)
            self._request_buffer.clear()
            self._request_line_started_at = None
            return line
        self._request_buffer.extend(chunk)
        if len(self._request_buffer) >= MAX_REMOTE_POSE_PAYLOAD_BYTES:
            raise RemotePoseProtocolError("remote pose request exceeds the payload limit")
        return None

    def _record_disconnect(self, connection: socket.socket, failure_kind: str) -> None:
        self._close_socket(connection)
        with self._condition:
            if self._socket is connection:
                self._socket = None
            self._connected = False
            self._sending_kind = None
            self._failure_kind = failure_kind
            self._discard_disconnected_messages_locked()

    def _discard_disconnected_messages_locked(self) -> None:
        # STOP may survive long enough for the bounded close flush.  All pose
        # evidence and resume requests require a fresh post-handshake submit.
        self._pending = deque(item for item in self._pending if item.kind is RemotePoseKind.STOP)
        self._active_challenge = None
        self._available_request = None
        self._claimed_request = None
        self._last_request_sequence = 0
        self._request_buffer.clear()
        self._request_line_started_at = None
        self._condition.notify_all()

    @staticmethod
    def _abort_socket(connection: socket.socket) -> None:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        RemotePosePublisher._close_socket(connection)

    @staticmethod
    def _close_socket(connection: socket.socket) -> None:
        try:
            connection.close()
        except Exception:
            pass


__all__ = [
    "DEFAULT_REMOTE_POSE_ANALYSIS_QUEUE_SECONDS",
    "MAX_REMOTE_POSE_CHALLENGE_BYTES",
    "MAX_REMOTE_POSE_EVIDENCE_AGE_MS",
    "MAX_REMOTE_POSE_PAYLOAD_BYTES",
    "MAX_REMOTE_POSE_SEQUENCE",
    "REMOTE_POSE_VERSION",
    "RemotePoseChallenge",
    "RemotePoseKind",
    "RemotePoseMessage",
    "RemotePoseProtocolError",
    "RemotePosePublisher",
    "RemotePoseRequest",
    "decode_remote_pose_challenge",
    "decode_remote_pose_message",
    "decode_remote_pose_request",
    "encode_remote_pose_challenge",
    "encode_remote_pose_message",
    "encode_remote_pose_request",
    "load_remote_pose_token",
    "new_remote_pose_request_nonce",
    "new_remote_pose_server_nonce",
    "new_remote_pose_service_epoch",
]
