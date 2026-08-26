"""Tailnet-only, read-only status endpoint for RecoveryBox."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import stat
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final

_DEFAULT_STATUS_PATH: Final = Path("/run/recoverybox/status.json")
_MAX_STATUS_BYTES: Final = 64 * 1024
_DEFAULT_MAX_STATUS_AGE_SECONDS: Final = 2.0
_MAX_ASSESSABLE_POSE_AGE_MS: Final = 499
_ALLOWED_STATUS_FIELDS: Final = frozenset(
    {
        "service",
        "peer",
        "session",
        "mode",
        "rep",
        "age",
        "voice",
        "button",
        "failure",
    }
)
_NULLABLE_TEXT_FIELDS: Final = frozenset({"peer", "session", "mode", "failure"})
_SERVICE_VALUES: Final = frozenset(
    {"starting", "listening", "connected", "local", "failed", "stopped"}
)
_MODE_VALUES: Final = frozenset(
    {"idle", "check_in", "active_exercise", "paused", "stopped", "complete"}
)
_VOICE_VALUES: Final = frozenset({"connected", "silent", "failed"})
_BUTTON_VALUES: Final = frozenset({"starting", "available", "pressed", "unavailable", "closed"})
_SAFE_SESSION: Final = re.compile(r"[0-9a-f]{32}\Z")
_FAILURE_VALUES: Final = frozenset(
    {
        "AcceptError",
        "ChallengeMismatch",
        "ChallengeSendError",
        "ClockError",
        "ConnectionIdle",
        "ConnectionLost",
        "ConnectionReadError",
        "CredentialProviderError",
        "CueTickError",
        "GPIOFactoryError",
        "GPIOInputClosed",
        "GPIOInputUnavailable",
        "GPIOMonitorError",
        "GPIOOpenError",
        "GPIOReadError",
        "GPIOStartError",
        "GPIOStartupTimeout",
        "HandshakeTimeout",
        "LocalPoseCloseError",
        "LocalPoseContractError",
        "LocalPoseFactoryError",
        "LocalPoseOpenError",
        "LocalPoseReadError",
        "LocalPoseShutdownTimeout",
        "LocalPoseStale",
        "LocalPoseStartupTimeout",
        "LocalPoseStopped",
        "LocalPoseTimeout",
        "LocalSessionProcessingError",
        "LocalSessionStartError",
        "PeerBusy",
        "PeerRejected",
        "PhysicalStop",
        "PhysicalStopUnavailable",
        "PoseRequestMismatch",
        "PoseRequestRequired",
        "PoseRequestSendError",
        "PoseResponseBeforeRequest",
        "PoseTimeout",
        "RealtimeConnectError",
        "RealtimeProviderError",
        "RealtimeReceiveError",
        "RealtimeSessionStartError",
        "RemotePoseProtocolError",
        "RetiredSession",
        "SequenceConflict",
        "SequenceReplay",
        "ServiceProcessingError",
        "ServiceStopped",
        "SessionMismatch",
        "SessionPauseError",
        "SessionProcessingError",
        "SessionStartError",
        "SessionStopError",
        "SessionTerminated",
        "SessionUnavailable",
        "SocketSetupError",
        "SpeakerPlaybackError",
        "StartRequired",
        "StopCallbackError",
        "UnexpectedStart",
    }
)
_TAILSCALE_V4: Final = ipaddress.ip_network("100.64.0.0/10")
_TAILSCALE_V6: Final = ipaddress.ip_network("fd7a:115c:a1e0::/48")


class StatusUnavailable(RuntimeError):
    """The sanitized status file cannot safely be served."""


def _safe_value(field: str, value: Any) -> str | int | None:
    if value is None:
        if field in _NULLABLE_TEXT_FIELDS or field == "age":
            return None
        raise StatusUnavailable("status omits a required value")
    if field in {"rep", "age"}:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StatusUnavailable("status contains an invalid integer")
        return value
    if not isinstance(value, str):
        raise StatusUnavailable("status contains an invalid text value")
    if field == "peer":
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise StatusUnavailable("status contains an invalid peer") from exc
        if address not in (_TAILSCALE_V4 if address.version == 4 else _TAILSCALE_V6):
            raise StatusUnavailable("status peer is outside the Tailnet")
        return str(address)
    if field == "session":
        if not _SAFE_SESSION.fullmatch(value):
            raise StatusUnavailable("status contains an invalid session")
        return value
    if field == "service" and value in _SERVICE_VALUES:
        return value
    if field == "mode" and value in _MODE_VALUES:
        return value
    if field == "voice" and value in _VOICE_VALUES:
        return value
    if field == "button" and value in _BUTTON_VALUES:
        return value
    if field == "failure" and value in _FAILURE_VALUES:
        return value
    raise StatusUnavailable("status contains unsafe text")


def _read_status(
    path: Path,
    *,
    now_seconds: float | None = None,
    max_age_seconds: float = _DEFAULT_MAX_STATUS_AGE_SECONDS,
) -> dict[str, Any]:
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be positive")
    now = time.time() if now_seconds is None else now_seconds
    if not isinstance(now, (int, float)) or isinstance(now, bool):
        raise ValueError("now_seconds must be numeric")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StatusUnavailable("status file is unavailable") from exc

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise StatusUnavailable("status path is not a regular file")
        if metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
            raise StatusUnavailable("status file ownership is unsafe")
        if metadata.st_mode & 0o022:
            raise StatusUnavailable("status file is writable by another account")
        if metadata.st_size > _MAX_STATUS_BYTES:
            raise StatusUnavailable("status file is too large")
        age_seconds = float(now) - metadata.st_mtime
        if age_seconds < -1.0 or age_seconds > max_age_seconds:
            raise StatusUnavailable("status file is stale")
        with os.fdopen(descriptor, "rb", closefd=False) as status_file:
            raw = status_file.read(_MAX_STATUS_BYTES + 1)
    except OSError as exc:
        raise StatusUnavailable("status file could not be read") from exc
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise StatusUnavailable("status file could not be closed") from exc

    if len(raw) > _MAX_STATUS_BYTES:
        raise StatusUnavailable("status file is too large")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise StatusUnavailable("status file is invalid") from exc
    if not isinstance(parsed, Mapping):
        raise StatusUnavailable("status root must be an object")
    if frozenset(parsed) != _ALLOWED_STATUS_FIELDS:
        raise StatusUnavailable("status fields do not match the closed schema")

    # Only the clinician-reviewed, content-free status surface can leave the Pi.
    return {key: _safe_value(key, parsed[key]) for key in _ALLOWED_STATUS_FIELDS}


def _healthy_status(status: Mapping[str, Any]) -> bool:
    if (
        status.get("service") not in {"listening", "connected", "local"}
        or status.get("button") != "available"
        or status.get("voice") not in {"silent", "connected"}
        or status.get("failure") is not None
        or status.get("mode") in {"paused", "stopped"}
    ):
        return False
    if status.get("mode") == "active_exercise":
        age = status.get("age")
        return (
            isinstance(age, int)
            and not isinstance(age, bool)
            and age <= _MAX_ASSESSABLE_POSE_AGE_MS
        )
    return True


def _unavailable_status() -> dict[str, Any]:
    return {
        "service": "recoverybox",
        "mode": "unavailable",
        "failure": "status_unavailable",
    }


class _StatusHandler(BaseHTTPRequestHandler):
    server_version = "RecoveryBoxStatus/1"
    sys_version = ""

    def do_GET(self) -> None:
        peer = self.client_address[0] if self.client_address else ""
        if peer != self.server.allowed_peer:  # type: ignore[attr-defined]
            self._send_json(403, {"service": "recoverybox", "mode": "request_rejected"})
            return
        if self.path not in {"/", "/healthz"}:
            self._send_json(404, {"service": "recoverybox", "mode": "not_found"})
            return

        try:
            payload = _read_status(  # type: ignore[attr-defined]
                self.server.status_path,
                max_age_seconds=self.server.max_status_age_seconds,  # type: ignore[attr-defined]
            )
        except StatusUnavailable:
            self._send_json(503, _unavailable_status())
            return
        self._send_json(200 if _healthy_status(payload) else 503, payload)

    def _send_json(self, status_code: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        del message, explain
        self._send_json(code, {"service": "recoverybox", "mode": "request_rejected"})

    def log_message(self, _format: str, *args: object) -> None:
        # Request paths and client-provided bytes do not enter service logs.
        return


class _StatusServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        status_path: Path,
        *,
        allowed_peer: str = "127.0.0.1",
        max_status_age_seconds: float = _DEFAULT_MAX_STATUS_AGE_SECONDS,
    ) -> None:
        self.status_path = status_path
        try:
            peer_address = ipaddress.ip_address(allowed_peer)
        except ValueError as exc:
            raise ValueError("allowed_peer must be an IP address") from exc
        if peer_address.version != 4:
            raise ValueError("allowed_peer must be an IPv4 address")
        self.allowed_peer = str(peer_address)
        if max_status_age_seconds <= 0:
            raise ValueError("max_status_age_seconds must be positive")
        self.max_status_age_seconds = float(max_status_age_seconds)
        super().__init__(address, _StatusHandler)

    def verify_request(self, request: object, client_address: object) -> bool:
        """Reject an unconfigured peer before allocating a handler thread."""

        del request
        return (
            isinstance(client_address, tuple)
            and bool(client_address)
            and client_address[0] == self.allowed_peer
        )

    def handle_error(self, request: object, client_address: object) -> None:
        # socketserver's default traceback includes the client address. A
        # disconnect must not put request or peer data in service logs.
        del request, client_address


def _tailscale_bind_host(raw: str) -> str:
    if not raw:
        raise ValueError("RECOVERYBOX_POSE_BIND_HOST is required")
    try:
        address = ipaddress.ip_address(raw)
    except ValueError as exc:
        raise ValueError("RECOVERYBOX_POSE_BIND_HOST must be a Tailscale IP address") from exc
    if address.version != 4 or address not in _TAILSCALE_V4:
        raise ValueError("RECOVERYBOX_POSE_BIND_HOST must be a Tailscale IPv4 address")
    return str(address)


def _tailscale_peer(raw: str) -> str:
    if not raw:
        raise ValueError("RECOVERYBOX_STATUS_ALLOWED_PEER is required")
    try:
        address = ipaddress.ip_address(raw)
    except ValueError as exc:
        raise ValueError("RECOVERYBOX_STATUS_ALLOWED_PEER must be a Tailscale IP") from exc
    if address.version != 4 or address not in _TAILSCALE_V4:
        raise ValueError("RECOVERYBOX_STATUS_ALLOWED_PEER must be a Tailscale IPv4 address")
    return str(address)


def _port(raw: str) -> int:
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError("RECOVERYBOX_DEBUG_PORT must be an integer") from exc
    if not 1024 <= port <= 65_535:
        raise ValueError("RECOVERYBOX_DEBUG_PORT must be between 1024 and 65535")
    return port


def _positive_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("RECOVERYBOX_STATUS_MAX_AGE_SECONDS must be numeric") from exc
    if not 0 < value <= 10:
        raise ValueError("RECOVERYBOX_STATUS_MAX_AGE_SECONDS must be in (0, 10]")
    return value


def main() -> int:
    """Serve the one configured status file on the Pi's Tailnet address."""
    bind_host = _tailscale_bind_host(os.environ.get("RECOVERYBOX_POSE_BIND_HOST", ""))
    allowed_peer = _tailscale_peer(os.environ.get("RECOVERYBOX_STATUS_ALLOWED_PEER", ""))
    port = _port(os.environ.get("RECOVERYBOX_DEBUG_PORT", "45874"))
    max_status_age_seconds = _positive_seconds(
        os.environ.get("RECOVERYBOX_STATUS_MAX_AGE_SECONDS", "2.0")
    )
    status_path = Path(os.environ.get("RECOVERYBOX_STATUS_PATH", str(_DEFAULT_STATUS_PATH)))
    if not status_path.is_absolute():
        raise ValueError("RECOVERYBOX_STATUS_PATH must be absolute")

    server = _StatusServer(
        (bind_host, port),
        status_path,
        allowed_peer=allowed_peer,
        max_status_age_seconds=max_status_age_seconds,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
