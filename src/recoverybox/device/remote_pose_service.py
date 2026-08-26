"""Headless Pi service for process-local or authenticated remote pose analysis.

In ``remote`` mode the laptop owns camera capture and pose estimation, and this
service accepts only the closed :class:`~recoverybox.exercise.SquatAnalysis`
wire schema.  In explicit ``local`` mode a Pi pose adapter is loaded lazily and
only its numeric ``SquatAnalysis`` result crosses into this module.  Both modes
keep the deterministic Guardian, session lifecycle, exact-cue gate, and
physical-stop boundary on the Pi.

Every accepted TCP connection belongs to the one configured tailnet peer and
must begin with an authenticated ``start`` line.  Sequence and liveness state
survive reconnects for the same workout, so a reconnect can never clear a
Guardian pause or replay a cue-producing analysis.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Protocol, TextIO

from recoverybox.core import CueId, SessionMode
from recoverybox.device.audio_process import AlsaCommandConfig, SubprocessPlayback
from recoverybox.device.gpio_stop import (
    GpioStopConfig,
    PhysicalStopMonitor,
    StopInputSnapshot,
    StopInputState,
    StopInputTrigger,
)
from recoverybox.exercise import (
    SquatAnalysis,
    SquatAssessmentIssue,
    SquatPhase,
)
from recoverybox.laptop.squat_session import LaptopSquatSession
from recoverybox.realtime import (
    BoundedOrderedTransport,
    ReleasedCueAudio,
    WebSocketJsonTransport,
)
from recoverybox.remote_pose import (
    MAX_REMOTE_POSE_PAYLOAD_BYTES,
    RemotePoseChallenge,
    RemotePoseKind,
    RemotePoseMessage,
    RemotePoseProtocolError,
    RemotePoseRequest,
    decode_remote_pose_message,
    encode_remote_pose_challenge,
    encode_remote_pose_request,
    load_remote_pose_token,
    new_remote_pose_request_nonce,
    new_remote_pose_server_nonce,
    new_remote_pose_service_epoch,
)

DEFAULT_REMOTE_POSE_PORT = 45_873
DEFAULT_REMOTE_POSE_WATCHDOG_SECONDS = 0.5
DEFAULT_REMOTE_POSE_STATUS_PATH = Path("/run/recoverybox/status.json")
OPENAI_API_CREDENTIAL_FILE_ENV = "RECOVERYBOX_OPENAI_CREDENTIAL_FILE"
MAX_OPENAI_API_CREDENTIAL_BYTES = 4_096
_TAILSCALE_IPV4_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_REALTIME_CONNECT_TIMEOUT_SECONDS = 5.0
_GPIO_START_TIMEOUT_SECONDS = 1.0
_LOCAL_POSE_START_TIMEOUT_SECONDS = 30.0
_LOCAL_POSE_CLEANUP_TIMEOUT_SECONDS = 2.0
_STATUS_HEARTBEAT_SECONDS = 1.0
_LOCAL_SESSION_ID = "0" * 32

REMOTE_POSE_SESSION_INSTRUCTIONS = """\
You are the voice interface for one locally supervised squat session.
The deterministic local Guardian, never the model, decides exercise cues.
During exercise, produce no ordinary speech. Only fulfill an isolated exact
cue request selected from the reviewed catalog. A network interruption,
silence, pause, or completed set must not weaken or end local safety.
"""

_STATUS_FIELDS = frozenset(
    {
        "service",
        "peer",
        "session",
        "mode",
        "rep",
        "age",
        "voice",
        "failure",
        "button",
    }
)
_GPIO_FAILURE_KINDS = frozenset(
    {
        "GPIOFactoryError",
        "GPIOStartError",
        "GPIOStartupTimeout",
        "GPIOInputUnavailable",
        "GPIOInputClosed",
        "GPIOOpenError",
        "GPIOReadError",
        "GPIOMonitorError",
        "StopCallbackError",
    }
)


class RemotePoseServiceConfigurationError(ValueError):
    """The tailnet listener cannot be started from the supplied settings."""


class PoseSourceMode(StrEnum):
    """Explicitly selected numeric pose acquisition lane."""

    REMOTE = "remote"
    LOCAL = "local"


def _environment_pose_source_mode(
    environment: Mapping[str, str],
) -> PoseSourceMode:
    raw = environment.get("RECOVERYBOX_POSE_SOURCE", PoseSourceMode.REMOTE.value)
    if not isinstance(raw, str):
        raise RemotePoseServiceConfigurationError("RECOVERYBOX_POSE_SOURCE must be remote or local")
    try:
        return PoseSourceMode(raw.strip().lower())
    except ValueError as exc:
        raise RemotePoseServiceConfigurationError(
            "RECOVERYBOX_POSE_SOURCE must be remote or local"
        ) from exc


def _clean_environment_text(
    environment: Mapping[str, str],
    name: str,
    *,
    default: str | None = None,
) -> str:
    raw = environment.get(name, default)
    if not isinstance(raw, str) or not raw.strip():
        raise RemotePoseServiceConfigurationError(f"{name} must be configured")
    value = raw.strip()
    if any(ord(character) < 32 for character in value):
        raise RemotePoseServiceConfigurationError(f"{name} contains invalid characters")
    return value


def _environment_integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw = environment.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RemotePoseServiceConfigurationError(f"{name} must be an integer") from exc
    if isinstance(raw, bool):
        raise RemotePoseServiceConfigurationError(f"{name} must be an integer")
    return value


def _environment_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw = environment.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RemotePoseServiceConfigurationError(f"{name} must be a number") from exc
    return value


def _environment_boolean(
    environment: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    raw = environment.get(name, "1" if default else "0")
    if not isinstance(raw, str):
        raise RemotePoseServiceConfigurationError(f"{name} must be true or false")
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RemotePoseServiceConfigurationError(f"{name} must be true or false")


@dataclass(frozen=True, slots=True)
class RemotePoseServiceConfig:
    """Validated listener, watchdog, Realtime, and status-file settings."""

    bind_host: str
    allowed_peer: str
    token_file: Path
    pose_source: PoseSourceMode = PoseSourceMode.REMOTE
    port: int = DEFAULT_REMOTE_POSE_PORT
    watchdog_seconds: float = DEFAULT_REMOTE_POSE_WATCHDOG_SECONDS
    status_path: Path = DEFAULT_REMOTE_POSE_STATUS_PATH
    playback_device: str = "default"
    realtime_voice: str = "marin"
    audio_enabled: bool = False
    button_gpio: int = 23
    socket_poll_seconds: float = 0.1
    authenticated_idle_seconds: float = 2.0

    def __post_init__(self) -> None:
        try:
            pose_source = PoseSourceMode(self.pose_source)
        except (TypeError, ValueError) as exc:
            raise RemotePoseServiceConfigurationError(
                "pose_source must be remote or local"
            ) from exc
        object.__setattr__(self, "pose_source", pose_source)
        for field_name in ("bind_host", "allowed_peer", "playback_device", "realtime_voice"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise RemotePoseServiceConfigurationError(f"{field_name} must not be blank")
            if any(ord(character) < 32 for character in value):
                raise RemotePoseServiceConfigurationError(
                    f"{field_name} contains invalid characters"
                )
            object.__setattr__(self, field_name, value.strip())
        if self.bind_host in {"0.0.0.0", "::"}:
            raise RemotePoseServiceConfigurationError(
                "bind_host must name one explicit tailnet interface"
            )
        for field_name in ("bind_host", "allowed_peer"):
            try:
                address = ipaddress.ip_address(getattr(self, field_name))
            except ValueError as exc:
                raise RemotePoseServiceConfigurationError(
                    f"{field_name} must be a literal Tailscale IPv4 address"
                ) from exc
            if address.version != 4 or address not in _TAILSCALE_IPV4_NETWORK:
                raise RemotePoseServiceConfigurationError(
                    f"{field_name} must be a literal Tailscale IPv4 address"
                )
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise RemotePoseServiceConfigurationError("port must be an integer")
        if not 1 <= self.port <= 65_535:
            raise RemotePoseServiceConfigurationError("port must be between 1 and 65535")
        if not isinstance(self.audio_enabled, bool):
            raise RemotePoseServiceConfigurationError("audio_enabled must be a boolean")
        if isinstance(self.button_gpio, bool) or not isinstance(self.button_gpio, int):
            raise RemotePoseServiceConfigurationError("button_gpio must be an integer")
        if not 0 <= self.button_gpio <= 1_023:
            raise RemotePoseServiceConfigurationError("button_gpio must be between 0 and 1023")
        for field_name in (
            "watchdog_seconds",
            "socket_poll_seconds",
            "authenticated_idle_seconds",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise RemotePoseServiceConfigurationError(
                    f"{field_name} must be a positive finite number"
                )
        if self.watchdog_seconds > DEFAULT_REMOTE_POSE_WATCHDOG_SECONDS:
            raise RemotePoseServiceConfigurationError(
                "watchdog_seconds cannot exceed the Guardian pose-age limit"
            )
        if self.socket_poll_seconds > self.watchdog_seconds:
            raise RemotePoseServiceConfigurationError(
                "socket_poll_seconds cannot exceed watchdog_seconds"
            )
        if self.authenticated_idle_seconds <= self.watchdog_seconds:
            raise RemotePoseServiceConfigurationError(
                "authenticated_idle_seconds must exceed watchdog_seconds"
            )
        try:
            token_file = Path(self.token_file).expanduser()
            status_path = Path(self.status_path).expanduser()
        except (TypeError, RuntimeError) as exc:
            raise RemotePoseServiceConfigurationError(
                "token_file and status_path must be filesystem paths"
            ) from exc
        if not str(token_file).strip() or not str(status_path).strip():
            raise RemotePoseServiceConfigurationError(
                "token_file and status_path must not be blank"
            )
        if not token_file.is_absolute() or not status_path.is_absolute():
            raise RemotePoseServiceConfigurationError(
                "token_file and status_path must be absolute paths"
            )
        try:
            paths_alias = token_file.resolve(strict=False) == status_path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise RemotePoseServiceConfigurationError(
                "token_file and status_path could not be normalized"
            ) from exc
        if paths_alias:
            raise RemotePoseServiceConfigurationError(
                "token_file and status_path must be different paths"
            )
        object.__setattr__(self, "token_file", token_file)
        object.__setattr__(self, "status_path", status_path)
        object.__setattr__(self, "watchdog_seconds", float(self.watchdog_seconds))
        object.__setattr__(self, "socket_poll_seconds", float(self.socket_poll_seconds))
        object.__setattr__(
            self,
            "authenticated_idle_seconds",
            float(self.authenticated_idle_seconds),
        )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> RemotePoseServiceConfig:
        """Load the closed service environment; the token value is file-only."""

        env = os.environ if environment is None else environment
        pose_source = _environment_pose_source_mode(env)
        return cls(
            bind_host=_clean_environment_text(env, "RECOVERYBOX_POSE_BIND_HOST"),
            allowed_peer=_clean_environment_text(env, "RECOVERYBOX_POSE_ALLOWED_PEER"),
            token_file=Path(_clean_environment_text(env, "RECOVERYBOX_POSE_TOKEN_FILE")),
            pose_source=pose_source,
            port=_environment_integer(
                env,
                "RECOVERYBOX_POSE_PORT",
                DEFAULT_REMOTE_POSE_PORT,
            ),
            watchdog_seconds=_environment_float(
                env,
                "RECOVERYBOX_POSE_WATCHDOG_SECONDS",
                DEFAULT_REMOTE_POSE_WATCHDOG_SECONDS,
            ),
            status_path=Path(
                _clean_environment_text(
                    env,
                    "RECOVERYBOX_STATUS_PATH",
                    default=str(DEFAULT_REMOTE_POSE_STATUS_PATH),
                )
            ),
            playback_device=_clean_environment_text(
                env,
                "RECOVERYBOX_PLAYBACK_DEVICE",
                default="default",
            ),
            realtime_voice=_clean_environment_text(
                env,
                "RECOVERYBOX_REALTIME_VOICE",
                default="marin",
            ),
            audio_enabled=_environment_boolean(
                env,
                "RECOVERYBOX_AUDIO_ENABLED",
                False,
            ),
            button_gpio=_environment_integer(
                env,
                "RECOVERYBOX_BUTTON_GPIO",
                23,
            ),
        )


class _Coordinator(Protocol):
    @property
    def current_mode(self) -> SessionMode: ...


class _Plan(Protocol):
    @property
    def max_pose_age_ms(self) -> int: ...


class _RemoteSquatSession(Protocol):
    @property
    def coordinator(self) -> _Coordinator: ...

    @property
    def plan(self) -> _Plan: ...

    @property
    def ended(self) -> bool: ...

    def start(self, *, instructions: str, voice: str) -> None: ...

    def activate_exercise(
        self,
        analysis: SquatAnalysis,
        *,
        pose_age_ms: int = 0,
    ) -> bool: ...

    def notify_cue_playback_succeeded(self, cue_id: CueId) -> bool: ...

    def resume_after_assessable_pose(
        self,
        analysis: SquatAnalysis,
        *,
        pose_age_ms: int = 0,
    ) -> bool: ...

    def process_analysis(
        self,
        analysis: SquatAnalysis,
        *,
        pose_age_ms: int = 0,
    ) -> object: ...

    def tick(self) -> int: ...

    def request_physical_stop(self) -> object: ...


class _CueSpeaker(Protocol):
    def enqueue(self, clip: ReleasedCueAudio) -> None: ...

    def quiesce(self) -> None: ...

    def preempt(self) -> None: ...

    def close(self) -> None: ...


class _Listener(Protocol):
    def setsockopt(self, level: int, option: int, value: int) -> None: ...

    def bind(self, address: tuple[str, int]) -> None: ...

    def listen(self, backlog: int) -> None: ...

    def settimeout(self, timeout: float) -> None: ...

    def accept(self) -> tuple[_Connection, object]: ...

    def close(self) -> None: ...


class _Connection(Protocol):
    def settimeout(self, timeout: float) -> None: ...

    def sendall(self, data: bytes) -> None: ...

    def recv(self, size: int) -> bytes: ...

    def close(self) -> None: ...


class _StopMonitor(Protocol):
    @property
    def snapshot(self) -> StopInputSnapshot: ...

    def start(self) -> None: ...

    def close(self, *, timeout_seconds: float = 1.0) -> None: ...


class _LocalPoseObservation(Protocol):
    @property
    def analysis(self) -> SquatAnalysis: ...


class _LocalPoseSource(Protocol):
    def open(self) -> None: ...

    def read(self) -> _LocalPoseObservation: ...

    def close(self) -> None: ...


SessionFactory = Callable[..., _RemoteSquatSession]
SpeakerFactory = Callable[[AlsaCommandConfig], _CueSpeaker]
ListenerFactory = Callable[[], _Listener]
CredentialProvider = Callable[[], str | None]
StatusFileWriter = Callable[[Path, Mapping[str, object]], None]
StopMonitorFactory = Callable[..., _StopMonitor]
LocalPoseSourceFactory = Callable[[], _LocalPoseSource]
IdentifierFactory = Callable[[], str]


class _LocalOnlyTransport:
    """No-network transport for the silent Guardian-only fallback lane."""

    def __init__(self) -> None:
        self._closed = False

    def send_event(self, event: Mapping[str, object]) -> None:
        del event
        if self._closed:
            raise RuntimeError("local-only transport is closed")

    def receive_event(self) -> Mapping[str, object]:
        raise EOFError("local-only transport has no receiver")

    def close(self) -> None:
        self._closed = True


def _default_listener_factory() -> _Listener:
    return socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def load_openai_api_key_credential(path: str | Path) -> str:
    """Read one private file credential without accepting links or whitespace."""

    credential_path = Path(path)
    if not credential_path.is_absolute():
        raise RemotePoseServiceConfigurationError(
            f"{OPENAI_API_CREDENTIAL_FILE_ENV} must be an absolute path"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(credential_path, flags)
    except OSError as exc:
        raise RemotePoseServiceConfigurationError(
            "OpenAI credential could not be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RemotePoseServiceConfigurationError("OpenAI credential must be one regular file")
        if stat.S_IMODE(before.st_mode) & 0o077:
            raise RemotePoseServiceConfigurationError(
                "OpenAI credential must not be accessible by group or others"
            )
        raw = os.read(descriptor, MAX_OPENAI_API_CREDENTIAL_BYTES + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or after.st_size != len(
            raw
        ):
            raise RemotePoseServiceConfigurationError(
                "OpenAI credential changed while it was being read"
            )
    finally:
        os.close(descriptor)
    if not raw or len(raw) > MAX_OPENAI_API_CREDENTIAL_BYTES:
        raise RemotePoseServiceConfigurationError("OpenAI credential has an invalid size")
    try:
        api_key = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RemotePoseServiceConfigurationError(
            "OpenAI credential must contain printable ASCII"
        ) from exc
    if any(not 33 <= ord(character) <= 126 for character in api_key):
        raise RemotePoseServiceConfigurationError(
            "OpenAI credential must contain one non-blank token"
        )
    return api_key


def _credential_provider_from_environment(
    environment: Mapping[str, str],
) -> str | None:
    raw_path = environment.get(OPENAI_API_CREDENTIAL_FILE_ENV)
    if raw_path is None:
        return None
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RemotePoseServiceConfigurationError(
            f"{OPENAI_API_CREDENTIAL_FILE_ENV} must be configured"
        )
    if raw_path != raw_path.strip() or any(ord(character) < 32 for character in raw_path):
        raise RemotePoseServiceConfigurationError(
            f"{OPENAI_API_CREDENTIAL_FILE_ENV} contains invalid characters"
        )
    return load_openai_api_key_credential(raw_path)


def _default_credential_provider() -> str | None:
    return _credential_provider_from_environment(os.environ)


def _local_pose_source_from_environment(
    environment: Mapping[str, str] | None = None,
) -> _LocalPoseSource:
    """Load the bounded V4L2/NCNN worker only after ``local`` is selected."""

    from recoverybox.device.pi_pose_v4l2 import V4L2NcnnPoseSource

    return V4L2NcnnPoseSource.from_environment(environment)


def _default_local_pose_source_factory() -> _LocalPoseSource:
    return _local_pose_source_from_environment()


def _default_session_factory(
    *,
    api_key: str | None,
    on_cue_audio: Callable[[ReleasedCueAudio], None],
    on_audio_preempt: Callable[[], None],
) -> LaptopSquatSession:
    if api_key is None:
        return LaptopSquatSession(
            transport=_LocalOnlyTransport(),
            on_cue_audio=on_cue_audio,
            on_audio_preempt=on_audio_preempt,
            cue_delivery_enabled=False,
        )
    connection = WebSocketJsonTransport.connect(
        api_key=api_key,
        timeout_seconds=_REALTIME_CONNECT_TIMEOUT_SECONDS,
    )
    transport: BoundedOrderedTransport | None = None
    try:
        connection.set_receive_timeout(None)
        transport = BoundedOrderedTransport(connection)
        return LaptopSquatSession(
            transport=transport,
            on_cue_audio=on_cue_audio,
            on_audio_preempt=on_audio_preempt,
        )
    except Exception:
        if transport is not None:
            transport.close()
        else:
            connection.close()
        raise


class _SilentCueSpeaker:
    """Capability-free speaker used when Realtime is not available."""

    def enqueue(self, clip: ReleasedCueAudio) -> None:
        del clip

    def quiesce(self) -> None:
        return

    def preempt(self) -> None:
        return

    def close(self) -> None:
        return


class _SubprocessCueSpeaker:
    """Non-blocking complete-clip handoff to one preemptible ALSA worker."""

    def __init__(self, playback: SubprocessPlayback) -> None:
        self._playback = playback
        self._condition = threading.Condition(threading.RLock())
        # Serialize the last cancellation check with process creation. This is
        # deliberately separate from the queue condition so ordinary enqueue
        # calls never wait for ``aplay`` to start.
        self._start_gate = threading.Lock()
        self._queue: deque[ReleasedCueAudio] = deque()
        self._generation = 0
        self._accepting = True
        self._closed = False
        self._failure_callback: Callable[[], None] | None = None
        self._playback_succeeded_callback: Callable[[CueId], None] | None = None
        self._worker = threading.Thread(
            target=self._run,
            name="recoverybox-pi-cue-playback",
            daemon=True,
        )
        self._worker.start()

    def bind_failure_callback(self, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._condition:
            if self._failure_callback is not None:
                raise RuntimeError("speaker failure callback is already bound")
            self._failure_callback = callback

    def bind_playback_succeeded_callback(
        self,
        callback: Callable[[CueId], None],
    ) -> None:
        """Bind the post-``aplay`` completion edge used by check-in gating."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._condition:
            if self._playback_succeeded_callback is not None:
                raise RuntimeError("speaker success callback is already bound")
            self._playback_succeeded_callback = callback

    def enqueue(self, clip: ReleasedCueAudio) -> None:
        if not isinstance(clip, ReleasedCueAudio):
            raise TypeError("clip must be ReleasedCueAudio")
        with self._condition:
            if not self._accepting:
                raise RuntimeError("cue speaker is quiesced")
            self._queue.append(clip)
            self._condition.notify()

    def quiesce(self) -> None:
        """Permanently close admission without waiting for playback cleanup.

        The start gate makes this linearizable with the worker's final check:
        once this method returns, no not-yet-started clip can create an ALSA
        subprocess and callbacks retaining ``enqueue`` can never add work.
        """

        with self._start_gate:
            with self._condition:
                if not self._accepting:
                    return
                self._accepting = False
                self._generation += 1
                self._queue.clear()
                self._condition.notify_all()

    def preempt(self) -> None:
        with self._condition:
            self._generation += 1
            self._queue.clear()
            self._condition.notify_all()
        # SubprocessPlayback.stop deliberately interrupts a blocked write.
        self._playback.stop()

    def close(self) -> None:
        if threading.current_thread() is self._worker:
            raise RuntimeError("cue worker cannot close itself")
        self.quiesce()
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        try:
            self._playback.stop()
        finally:
            self._worker.join(timeout=3.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                clip = self._queue.popleft()
                generation = self._generation
            stream_id = f"approved-cue-{clip.ticket_id}-{generation}"
            failed = False
            try:
                with self._start_gate:
                    with self._condition:
                        cancelled = (
                            self._closed or not self._accepting or generation != self._generation
                        )
                    if cancelled:
                        continue
                    self._playback.start(stream_id)
                with self._condition:
                    cancelled = self._closed or generation != self._generation
                if cancelled:
                    self._playback.stop()
                    continue
                self._playback.write(stream_id, clip.pcm16_mono_24khz)
                with self._condition:
                    cancelled = self._closed or generation != self._generation
                if cancelled:
                    self._playback.stop()
                    continue
                self._playback.finish()
                with self._condition:
                    cancelled = (
                        self._closed or not self._accepting or generation != self._generation
                    )
                    success_callback = self._playback_succeeded_callback
                if not cancelled and success_callback is not None:
                    success_callback(clip.authorization.cue_id)
            except Exception:
                with self._condition:
                    failed = not self._closed and generation == self._generation
                    callback = self._failure_callback
                if failed and callback is not None:
                    try:
                        callback()
                    except Exception:
                        pass


def _quiesce_speaker(speaker: _CueSpeaker | None) -> None:
    """Irreversibly close cue admission while keeping cleanup fail-safe."""

    if speaker is None:
        return
    try:
        speaker.quiesce()
    except Exception:
        # The independent preempt/close path must still run after a faulty
        # adapter. The production implementations make quiescence infallible.
        pass


def _default_speaker_factory(config: AlsaCommandConfig) -> _CueSpeaker:
    return _SubprocessCueSpeaker(SubprocessPlayback(config))


def _default_status_file_writer(path: Path, status: Mapping[str, object]) -> None:
    _atomic_write_status(path, status)


def _default_stop_monitor_factory(
    *,
    config: GpioStopConfig,
    on_stop: Callable[[StopInputTrigger], None],
    on_status: Callable[[StopInputSnapshot], None],
) -> _StopMonitor:
    return PhysicalStopMonitor(
        config,
        on_stop=on_stop,
        on_status=on_status,
    )


@dataclass(frozen=True, slots=True)
class RemotePoseServiceDependencies:
    """Replaceable socket, clock, GPIO, session, and speaker boundaries."""

    listener_factory: ListenerFactory = _default_listener_factory
    clock: Callable[[], float] = time.monotonic
    status_clock: Callable[[], float] = time.monotonic
    session_factory: SessionFactory = _default_session_factory
    speaker_factory: SpeakerFactory = _default_speaker_factory
    credential_provider: CredentialProvider = _default_credential_provider
    token_loader: Callable[[str | Path], bytes] = load_remote_pose_token
    status_file_writer: StatusFileWriter = _default_status_file_writer
    stop_monitor_factory: StopMonitorFactory = _default_stop_monitor_factory
    local_pose_source_factory: LocalPoseSourceFactory = _default_local_pose_source_factory
    service_epoch_factory: IdentifierFactory = new_remote_pose_service_epoch
    server_nonce_factory: IdentifierFactory = new_remote_pose_server_nonce
    request_nonce_factory: IdentifierFactory = new_remote_pose_request_nonce


class _ConnectionRejected(RuntimeError):
    def __init__(self, failure_kind: str) -> None:
        super().__init__(failure_kind)
        self.failure_kind = failure_kind


@dataclass(frozen=True, slots=True)
class _LocalPoseSample:
    """One numeric result plus its conservative acquisition start time."""

    analysis: SquatAnalysis
    acquisition_started: float


class _LocalPoseWorker:
    """Keep native capture/inference off the Guardian/watchdog thread.

    The one-slot handoff is overwritten, so the service can consume only the
    latest numeric result and can never build a stale frame or result queue.
    The native source owns its own open/read/close lifecycle on this worker.
    """

    def __init__(
        self,
        factory: LocalPoseSourceFactory,
        *,
        clock: Callable[[], float],
    ) -> None:
        self._factory = factory
        self._clock = clock
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._done = threading.Event()
        self._latest: _LocalPoseSample | None = None
        self._failure_kind: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="recoverybox-pi-local-pose",
            daemon=True,
        )

    @property
    def failure_kind(self) -> str | None:
        with self._condition:
            return self._failure_kind

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout_seconds: float) -> bool:
        return self._ready.wait(timeout_seconds)

    def take_latest(self, timeout_seconds: float) -> _LocalPoseSample | None:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._latest is None and not self._done.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            sample = self._latest
            self._latest = None
            return sample

    def stop(self, timeout_seconds: float) -> bool:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout_seconds)
        return not self._thread.is_alive()

    def _read_clock(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("local pose clock is invalid")
        return float(value)

    def _fail(self, failure_kind: str) -> None:
        with self._condition:
            self._failure_kind = failure_kind
            self._condition.notify_all()
        self._ready.set()

    def _run(self) -> None:
        source: _LocalPoseSource | None = None
        try:
            try:
                source = self._factory()
            except Exception:
                self._fail("LocalPoseFactoryError")
                return
            if source is None:
                self._fail("LocalPoseFactoryError")
                return
            try:
                source.open()
            except Exception:
                self._fail("LocalPoseOpenError")
                return
            self._ready.set()
            while not self._stop.is_set():
                try:
                    acquisition_started = self._read_clock()
                    observation = source.read()
                    self._read_clock()
                except Exception:
                    self._fail("LocalPoseReadError")
                    return
                analysis = getattr(observation, "analysis", None)
                if not isinstance(analysis, SquatAnalysis):
                    self._fail("LocalPoseContractError")
                    return
                sample = _LocalPoseSample(
                    analysis=analysis,
                    acquisition_started=acquisition_started,
                )
                with self._condition:
                    if self._stop.is_set():
                        return
                    self._latest = sample
                    self._condition.notify_all()
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    if self.failure_kind is None:
                        self._fail("LocalPoseCloseError")
            self._done.set()
            with self._condition:
                self._condition.notify_all()
            self._ready.set()


class _CoalescingStatusWriter:
    """Persist only the newest status without blocking pose safety work."""

    def __init__(
        self,
        path: Path,
        writer: StatusFileWriter,
    ) -> None:
        self._path = path
        self._writer = writer
        self._condition = threading.Condition()
        self._pending: dict[str, object] | None = None
        self._last_offered: dict[str, object] | None = None
        self._busy = False
        self._closed = False
        self._failed = False
        self._worker = threading.Thread(
            target=self._run,
            name="recoverybox-status-writer",
            daemon=True,
        )
        self._worker.start()

    def publish(self, status: Mapping[str, object], *, force: bool = False) -> None:
        if type(force) is not bool:
            raise TypeError("force must be a boolean")
        snapshot = dict(status)
        with self._condition:
            if self._closed or (snapshot == self._last_offered and not force):
                return
            self._last_offered = snapshot
            self._pending = snapshot
            self._condition.notify()

    def wait_idle(self, timeout: float | None = None) -> bool:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be a finite non-negative number or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._busy or self._pending is not None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout: float = 0.25) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._condition:
            if not self._closed:
                self._closed = True
                self._condition.notify_all()
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._pending is None and self._closed:
                    return
                status = self._pending
                self._pending = None
                self._busy = True
            assert status is not None
            try:
                self._writer(self._path, status)
            except Exception:
                with self._condition:
                    self._failed = True
                    if self._last_offered == status:
                        self._last_offered = None
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()


class RemotePoseService:
    """One-peer, one-workout-at-a-time remote pose service."""

    def __init__(
        self,
        config: RemotePoseServiceConfig,
        *,
        token: bytes | None,
        dependencies: RemotePoseServiceDependencies | None = None,
    ) -> None:
        if not isinstance(config, RemotePoseServiceConfig):
            raise TypeError("config must be a RemotePoseServiceConfig")
        if config.pose_source is PoseSourceMode.REMOTE:
            if type(token) is not bytes:
                raise TypeError("remote mode requires an immutable token")
            if len(token) != 32:
                raise ValueError("token must contain exactly 32 bytes")
        elif token is not None:
            if type(token) is not bytes:
                raise TypeError("token must be immutable bytes or None")
            if len(token) != 32:
                raise ValueError("token must contain exactly 32 bytes")
        self.config = config
        self._token = token
        self._dependencies = dependencies or RemotePoseServiceDependencies()
        if config.pose_source is PoseSourceMode.REMOTE:
            try:
                service_epoch = self._dependencies.service_epoch_factory()
                RemotePoseChallenge(
                    service_epoch=service_epoch,
                    server_nonce="0" * 64,
                )
            except Exception as exc:
                raise RemotePoseServiceConfigurationError(
                    "service epoch factory returned an invalid identifier"
                ) from exc
            self._service_epoch = service_epoch
        else:
            self._service_epoch = "0" * 64
        self._next_request_sequence = 0
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._shutdown_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._shutdown_complete = threading.Event()
        self._listener: _Listener | None = None
        self._connection: _Connection | None = None
        self._stop_monitor: _StopMonitor | None = None
        self._stop_monitor_started = False
        self._stop_input_ready = threading.Event()
        self._stop_input_state = StopInputState.STARTING
        self._stop_input_failure_kind: str | None = None
        self._input_terminal = False
        self._stop_request_epoch = 0
        self._session: _RemoteSquatSession | None = None
        self._speaker: _CueSpeaker | None = None
        self._session_id: str | None = None
        self._session_terminal = False
        self._session_stop_requested = False
        self._retired_session_ids: set[str] = set()
        self._last_sequence = 0
        self._last_message_digest: bytes | None = None
        self._last_analysis: SquatAnalysis | None = None
        self._last_analysis_receipt: float | None = None
        self._last_pose_age_ms: int | None = None
        self._resume_armed = False
        self._loss_injected = False
        self._exercise_activated = False
        self._service_state = "starting"
        self._peer: str | None = None
        self._voice_state = "silent"
        self._voice_failure_kind: str | None = None
        self._failure_kind: str | None = None
        self._fatal_service_failure = False
        self._status_write_failed = False
        self._status_writer = _CoalescingStatusWriter(
            config.status_path,
            self._dependencies.status_file_writer,
        )
        self._realtime_thread: threading.Thread | None = None

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    @property
    def current_mode(self) -> SessionMode | None:
        with self._lock:
            session = self._session
            terminal = self._session_terminal
        if terminal:
            return SessionMode.STOPPED
        if session is None:
            return None
        try:
            mode = session.coordinator.current_mode
            return mode if isinstance(mode, SessionMode) else None
        except Exception:
            return None

    @property
    def last_failure_kind(self) -> str | None:
        with self._lock:
            return self._failure_kind or self._voice_failure_kind

    def wait_for_status(self, timeout: float | None = 1.0) -> bool:
        """Wait only for diagnostics in tests/tools; safety paths never call it."""

        return self._status_writer.wait_idle(timeout)

    def start_local_stop_monitor(self) -> None:
        """Start and confirm the Pi-local physical-stop input exactly once."""

        with self._lifecycle_lock:
            if self._shutdown.is_set():
                raise RuntimeError("remote pose service is stopped")
            with self._lock:
                if self._stop_monitor_started:
                    return
                self._stop_monitor_started = True
            try:
                monitor = self._dependencies.stop_monitor_factory(
                    config=GpioStopConfig(line_offset=self.config.button_gpio),
                    on_stop=self.request_local_stop,
                    on_status=self._on_stop_input_status,
                )
            except Exception:
                with self._lock:
                    self._stop_input_failure_kind = "GPIOFactoryError"
                self.request_local_stop(
                    StopInputTrigger.INPUT_UNAVAILABLE,
                    failure_kind="GPIOFactoryError",
                )
                return
            with self._lock:
                self._stop_monitor = monitor
            try:
                monitor.start()
            except Exception:
                with self._lock:
                    self._stop_input_failure_kind = "GPIOStartError"
                self.request_local_stop(
                    StopInputTrigger.INPUT_UNAVAILABLE,
                    failure_kind="GPIOStartError",
                )
                return

        # The listener is not created until the hardware boundary has either
        # become available or failed closed. This wait is bounded and does not
        # use the injected pose clock.
        if not self._stop_input_ready.wait(_GPIO_START_TIMEOUT_SECONDS):
            self.request_local_stop(
                StopInputTrigger.INPUT_UNAVAILABLE,
                failure_kind="GPIOStartupTimeout",
            )
            return
        with self._lock:
            state = self._stop_input_state
            failure_kind = self._stop_input_failure_kind
        if state is not StopInputState.AVAILABLE:
            self.request_local_stop(
                StopInputTrigger.INPUT_UNAVAILABLE,
                failure_kind=failure_kind or "GPIOInputUnavailable",
            )

    def request_local_stop(
        self,
        trigger: StopInputTrigger = StopInputTrigger.BUTTON_PRESSED,
        *,
        failure_kind: str | None = None,
    ) -> None:
        """Apply one local physical-stop edge without relying on the peer."""

        if not isinstance(trigger, StopInputTrigger):
            raise TypeError("trigger must be a StopInputTrigger")
        if failure_kind is not None and (not isinstance(failure_kind, str) or not failure_kind):
            raise TypeError("failure_kind must be a non-empty string or None")
        if self._shutdown.is_set():
            return

        # Mark the lane terminal and interrupt remote I/O before waiting for a
        # possibly busy session lifecycle. A session being constructed checks
        # the epoch before installation, so a button cannot be lost behind a
        # bounded cloud handshake.
        with self._lock:
            if self._shutdown.is_set():
                return
            if (
                trigger is StopInputTrigger.BUTTON_PRESSED
                and self._session_terminal
                and self._session is None
            ):
                return
            if trigger is StopInputTrigger.INPUT_UNAVAILABLE and self._input_terminal:
                return
            self._stop_request_epoch += 1
            if trigger is StopInputTrigger.INPUT_UNAVAILABLE:
                self._input_terminal = True
                self._stop_input_state = StopInputState.UNAVAILABLE
                selected_failure = _gpio_failure_kind(failure_kind or self._stop_input_failure_kind)
                self._service_state = "failed"
            else:
                selected_failure = "PhysicalStop"
            self._failure_kind = selected_failure
            self._resume_armed = False
            if self._voice_state == "connected":
                self._voice_state = "silent"
            session = self._session
            speaker = self._speaker
            # Permanently close the captured speaker's admission gate at the
            # terminal edge and before any lifecycle wait. An analysis already
            # inside ``_lifecycle_lock`` or a Realtime receiver returning from
            # a blocking pump may still hold the old callback, but it can no
            # longer enqueue or start another clip.
            _quiesce_speaker(speaker)
            connection = self._connection
            listener = self._listener if self._input_terminal else None
            should_request_stop = (
                session is not None
                and not self._session_terminal
                and not self._session_stop_requested
            )
            if session is not None:
                self._session_terminal = True
                if self._session_id is not None:
                    self._retired_session_ids.add(self._session_id)
                self._session = None
                self._speaker = None
            if should_request_stop:
                self._session_stop_requested = True

        if connection is not None:
            _close_quietly(connection)
        if listener is not None:
            _close_quietly(listener)
        if speaker is not None:
            try:
                speaker.preempt()
            except Exception:
                pass

        with self._lifecycle_lock:
            if should_request_stop and session is not None:
                try:
                    session.request_physical_stop()
                except Exception:
                    with self._lock:
                        self._failure_kind = "SessionStopError"
            if session is not None and not _session_ended(session):
                transition = getattr(session.coordinator, "transition_to", None)
                if callable(transition):
                    try:
                        transition(SessionMode.STOPPED)
                    except Exception:
                        pass
            if speaker is not None:
                try:
                    speaker.close()
                except Exception:
                    pass
        self._publish_status()

    def _on_stop_input_status(self, snapshot: StopInputSnapshot) -> None:
        if not isinstance(snapshot, StopInputSnapshot):
            return
        if not isinstance(snapshot.state, StopInputState):
            self.request_local_stop(
                StopInputTrigger.INPUT_UNAVAILABLE,
                failure_kind="GPIOInputUnavailable",
            )
            return
        with self._lock:
            shutting_down = self._shutdown.is_set()
            if self._input_terminal and not (
                shutting_down and snapshot.state is StopInputState.CLOSED
            ):
                return
            if shutting_down:
                self._stop_input_state = snapshot.state
                self._stop_input_failure_kind = (
                    _gpio_failure_kind(snapshot.failure_kind)
                    if snapshot.failure_kind is not None
                    else None
                )
                return
            self._stop_input_state = snapshot.state
            self._stop_input_failure_kind = (
                _gpio_failure_kind(snapshot.failure_kind)
                if snapshot.failure_kind is not None
                else None
            )
            if snapshot.state is not StopInputState.STARTING:
                self._stop_input_ready.set()
            unexpected_close = snapshot.state is StopInputState.CLOSED
        if unexpected_close:
            self.request_local_stop(
                StopInputTrigger.INPUT_UNAVAILABLE,
                failure_kind="GPIOInputClosed",
            )
        else:
            self._publish_status()

    def serve_forever(self) -> None:
        """Run the explicitly selected pose lane until stop or fatal failure."""

        self.start_local_stop_monitor()
        with self._lock:
            input_terminal = self._input_terminal
        if input_terminal:
            self.shutdown()
            raise RuntimeError("physical stop input is unavailable")
        try:
            if self.config.pose_source is PoseSourceMode.LOCAL:
                self._serve_local_pose_forever()
            else:
                self._serve_remote_pose_forever()
        finally:
            if not self._shutdown.is_set():
                self.shutdown()

    def _serve_remote_pose_forever(self) -> None:
        """Bind the configured tailnet address and serialize signed peers."""

        listener: _Listener | None = None
        try:
            listener = self._dependencies.listener_factory()
            with self._lock:
                self._listener = listener
            try:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except (AttributeError, OSError):
                pass
            listener.bind((self.config.bind_host, self.config.port))
            listener.listen(1)
            listener.settimeout(self.config.socket_poll_seconds)
            with self._lock:
                self._service_state = "listening"
                self._failure_kind = None
            self._publish_status()
            next_status_heartbeat = self._status_now() + _STATUS_HEARTBEAT_SECONDS

            while not self._shutdown.is_set():
                with self._lock:
                    if self._input_terminal:
                        break
                try:
                    connection, address = listener.accept()
                except TimeoutError:
                    heartbeat_now = self._status_now()
                    if heartbeat_now >= next_status_heartbeat:
                        # Refresh an otherwise identical listening snapshot so
                        # the separate status endpoint can distinguish a live
                        # idle service from a dead producer. This diagnostic
                        # clock never participates in Guardian freshness.
                        self._publish_status(force=True)
                        next_status_heartbeat = heartbeat_now + _STATUS_HEARTBEAT_SECONDS
                    continue
                except OSError:
                    with self._lock:
                        input_terminal = self._input_terminal
                    if self._shutdown.is_set() or input_terminal:
                        break
                    with self._lock:
                        self._failure_kind = "AcceptError"
                    self._publish_status()
                    raise RuntimeError("pose listener accept failed") from None
                peer_host = _peer_host(address)
                self.handle_connection(connection, peer_host=peer_host)
                next_status_heartbeat = self._status_now() + _STATUS_HEARTBEAT_SECONDS
        finally:
            if listener is not None:
                try:
                    listener.close()
                except Exception:
                    pass
                with self._lock:
                    if self._listener is listener:
                        self._listener = None
        with self._lock:
            input_terminal = self._input_terminal
        if input_terminal:
            raise RuntimeError("physical stop input became unavailable")

    def _serve_local_pose_forever(self) -> None:
        """Consume only the newest process-local numeric pose observation."""

        worker = _LocalPoseWorker(
            self._dependencies.local_pose_source_factory,
            clock=self._dependencies.clock,
        )
        worker.start()
        if not worker.wait_ready(_LOCAL_POSE_START_TIMEOUT_SECONDS):
            self._record_fatal_service_failure("LocalPoseStartupTimeout")
            worker.stop(_LOCAL_POSE_CLEANUP_TIMEOUT_SECONDS)
            raise RuntimeError("local pose source startup timed out")
        failure_kind = worker.failure_kind
        if failure_kind is not None:
            self._record_fatal_service_failure(failure_kind)
            worker.stop(_LOCAL_POSE_CLEANUP_TIMEOUT_SECONDS)
            raise RuntimeError("local pose source could not start")

        started_at = self._now()
        start_message = RemotePoseMessage(
            kind=RemotePoseKind.START,
            session_id=_LOCAL_SESSION_ID,
            server_nonce="0" * 64,
            service_epoch="0" * 64,
        )
        try:
            self._accept_start(start_message, receipt=started_at, peer_host="local")
        except _ConnectionRejected as exc:
            worker.stop(_LOCAL_POSE_CLEANUP_TIMEOUT_SECONDS)
            if exc.failure_kind == "PhysicalStop":
                return
            self._record_fatal_service_failure("LocalSessionStartError")
            raise RuntimeError("local Guardian session could not start") from None
        with self._lock:
            self._peer = None
            self._service_state = "local"
            # Explicit local-mode selection is the one initial in-process
            # authorization. It survives missing/non-standing startup frames,
            # is consumed by the first eligible standing attempt, and is never
            # recreated automatically after a Guardian pause.
            self._resume_armed = True
        self._publish_status()
        next_status_heartbeat = self._status_now() + _STATUS_HEARTBEAT_SECONDS

        awaiting_since = started_at
        try:
            while not self._shutdown.is_set():
                with self._lock:
                    if self._input_terminal:
                        raise RuntimeError("physical stop input became unavailable")
                    if self._session_terminal:
                        return
                sample = worker.take_latest(self.config.socket_poll_seconds)
                now = self._now()
                heartbeat_now = self._status_now()
                if heartbeat_now >= next_status_heartbeat:
                    # Local pose can produce the same sanitized snapshot for
                    # minutes while no person is visible. Refresh it on the
                    # diagnostic clock so status freshness proves the producer
                    # is alive without changing Guardian evidence timing.
                    self._publish_status(force=True)
                    next_status_heartbeat = heartbeat_now + _STATUS_HEARTBEAT_SECONDS
                if sample is not None:
                    total_age = max(0.0, now - sample.acquisition_started)
                    if total_age >= self.config.watchdog_seconds:
                        with self._lock:
                            self._last_pose_age_ms = math.ceil(total_age * 1000.0)
                        self._pause_for_pose_loss(
                            "LocalPoseStale",
                            now=now,
                            force_stale=True,
                        )
                        self._record_fatal_service_failure("LocalPoseStale")
                        raise RuntimeError("local pose evidence exceeded the age limit")
                    try:
                        self._apply_local_analysis(
                            sample.analysis,
                            receipt=sample.acquisition_started,
                        )
                    except Exception:
                        with self._lock:
                            stopped = self._session_terminal or self._input_terminal
                        if stopped:
                            return
                        self._record_fatal_service_failure("LocalSessionProcessingError")
                        raise RuntimeError("local Guardian processing failed") from None
                    self._tick_session()
                    self._watchdog(self._now())
                    awaiting_since = now
                    continue

                failure_kind = worker.failure_kind
                if failure_kind is not None or worker.done:
                    selected = failure_kind or "LocalPoseStopped"
                    self._pause_for_pose_loss(selected, now=now, force_stale=True)
                    self._record_fatal_service_failure(selected)
                    raise RuntimeError("local pose source stopped")
                self._watchdog(now)
                self._tick_session()
                if now - awaiting_since >= self.config.watchdog_seconds:
                    self._record_fatal_service_failure("LocalPoseTimeout")
                    raise RuntimeError("local pose source missed the age deadline")
        finally:
            stopped = worker.stop(_LOCAL_POSE_CLEANUP_TIMEOUT_SECONDS)
            if not stopped and not self._shutdown.is_set():
                self._record_fatal_service_failure("LocalPoseShutdownTimeout")

    def _apply_local_analysis(
        self,
        analysis: SquatAnalysis,
        *,
        receipt: float,
    ) -> None:
        if not isinstance(analysis, SquatAnalysis):
            raise TypeError("analysis must be a SquatAnalysis")
        with self._lifecycle_lock:
            if self._shutdown.is_set():
                return
            self._apply_analysis_serialized(analysis, receipt=receipt)

    def _record_fatal_service_failure(self, failure_kind: str) -> None:
        with self._lock:
            self._fatal_service_failure = True
            self._service_state = "failed"
            self._failure_kind = failure_kind
            self._resume_armed = False
        self._publish_status()

    def handle_connection(self, connection: _Connection, *, peer_host: str) -> None:
        """Handle one injected socket until EOF, stop, or protocol failure."""

        if not isinstance(peer_host, str):
            raise TypeError("peer_host must be a string")
        token = self._token
        if self.config.pose_source is not PoseSourceMode.REMOTE or token is None:
            _close_quietly(connection)
            return
        if peer_host != self.config.allowed_peer:
            with self._lock:
                self._failure_kind = "PeerRejected"
            self._publish_status()
            _close_quietly(connection)
            return

        with self._lock:
            if not self._stop_monitor_started or self._input_terminal:
                if self._failure_kind is None:
                    self._failure_kind = "PhysicalStopUnavailable"
                self._publish_status_locked()
                _close_quietly(connection)
                return
            if self._connection is not None and self._connection is not connection:
                self._failure_kind = "PeerBusy"
                self._publish_status_locked()
                _close_quietly(connection)
                return
            self._connection = connection
        try:
            connection.settimeout(self.config.socket_poll_seconds)
        except Exception:
            with self._lock:
                self._failure_kind = "SocketSetupError"
                if self._connection is connection:
                    self._connection = None
            self._publish_status()
            _close_quietly(connection)
            return

        try:
            challenge = RemotePoseChallenge(
                service_epoch=self._service_epoch,
                server_nonce=self._dependencies.server_nonce_factory(),
            )
            connection.sendall(encode_remote_pose_challenge(challenge, token))
        except Exception:
            with self._lock:
                self._failure_kind = "ChallengeSendError"
                if self._connection is connection:
                    self._connection = None
            self._publish_status()
            _close_quietly(connection)
            return

        buffer = b""
        bound = False
        stopped = False
        outstanding_request: tuple[RemotePoseRequest, float] | None = None
        failure_kind: str | None = None
        partial_line_started: float | None = None
        next_status_heartbeat = self._status_now() + _STATUS_HEARTBEAT_SECONDS

        def publish_status_heartbeat_if_due() -> None:
            nonlocal next_status_heartbeat
            heartbeat_now = self._status_now()
            if heartbeat_now < next_status_heartbeat:
                return
            # A connected stream can leave every sanitized field unchanged.
            # Refresh that snapshot independently of pose evidence so the
            # status endpoint can distinguish a live producer from a dead one.
            self._publish_status(force=True)
            next_status_heartbeat = heartbeat_now + _STATUS_HEARTBEAT_SECONDS

        try:
            connection_started = self._now()
            last_byte_received = connection_started
            while not self._shutdown.is_set() and not stopped:
                with self._lock:
                    if self._input_terminal or (bound and self._session_terminal):
                        break
                try:
                    chunk = connection.recv(4096)
                except TimeoutError:
                    now = self._now()
                    self._watchdog(now)
                    self._tick_session()
                    publish_status_heartbeat_if_due()
                    if not bound and now - connection_started >= self.config.watchdog_seconds:
                        failure_kind = "HandshakeTimeout"
                        break
                    if bound and now - last_byte_received >= self.config.authenticated_idle_seconds:
                        failure_kind = "ConnectionIdle"
                        break
                    if (
                        bound
                        and partial_line_started is not None
                        and now - partial_line_started >= self.config.watchdog_seconds
                    ):
                        failure_kind = "RemotePoseProtocolError"
                        break
                    continue
                except OSError:
                    failure_kind = "ConnectionReadError"
                    break

                if type(chunk) is not bytes:
                    failure_kind = "ConnectionReadError"
                    break
                if not chunk:
                    failure_kind = "ConnectionLost" if not buffer else "RemotePoseProtocolError"
                    break
                chunk_receipt = self._now()
                # A peer cannot keep stale movement assessable by dripping an
                # unterminated line often enough to suppress socket timeouts.
                self._watchdog(chunk_receipt)
                if not bound and chunk_receipt - connection_started >= self.config.watchdog_seconds:
                    raise _ConnectionRejected("HandshakeTimeout")
                if partial_line_started is None:
                    partial_line_started = chunk_receipt
                elif bound and chunk_receipt - partial_line_started >= self.config.watchdog_seconds:
                    raise _ConnectionRejected("RemotePoseProtocolError")
                last_byte_received = chunk_receipt
                buffer += chunk
                while True:
                    newline = buffer.find(b"\n")
                    if newline < 0:
                        if len(buffer) >= MAX_REMOTE_POSE_PAYLOAD_BYTES:
                            raise _ConnectionRejected("RemotePoseProtocolError")
                        break
                    line = buffer[: newline + 1]
                    buffer = buffer[newline + 1 :]
                    partial_line_started = chunk_receipt if buffer else None
                    if len(line) > MAX_REMOTE_POSE_PAYLOAD_BYTES:
                        raise _ConnectionRejected("RemotePoseProtocolError")
                    receipt = chunk_receipt
                    try:
                        message = decode_remote_pose_message(line, token)
                    except (RemotePoseProtocolError, TypeError, ValueError):
                        raise _ConnectionRejected("RemotePoseProtocolError") from None
                    if not bound:
                        if message.kind is not RemotePoseKind.START:
                            raise _ConnectionRejected("StartRequired")
                        if (
                            message.service_epoch != challenge.service_epoch
                            or message.server_nonce != challenge.server_nonce
                        ):
                            raise _ConnectionRejected("ChallengeMismatch")
                        self._accept_start(message, receipt=receipt, peer_host=peer_host)
                        bound = True
                        outstanding_request = self._issue_pose_request(
                            connection,
                            session_id=message.session_id,
                            challenge=challenge,
                        )
                        self._watchdog(self._now())
                        continue
                    if message.kind is RemotePoseKind.START:
                        raise _ConnectionRejected("UnexpectedStart")
                    if (
                        message.service_epoch != challenge.service_epoch
                        or message.server_nonce != challenge.server_nonce
                    ):
                        raise _ConnectionRejected("ChallengeMismatch")
                    source_age_ms = 0
                    if message.kind is RemotePoseKind.ANALYSIS:
                        expected = outstanding_request
                        outstanding_request = None
                        if expected is None:
                            raise _ConnectionRejected("PoseRequestRequired")
                        request, request_sent = expected
                        if (
                            message.request_sequence != request.request_sequence
                            or message.request_nonce != request.request_nonce
                        ):
                            raise _ConnectionRejected("PoseRequestMismatch")
                        if receipt < request_sent:
                            raise _ConnectionRejected("PoseResponseBeforeRequest")
                        request_round_trip = receipt - request_sent
                        round_trip_ms = max(
                            0,
                            math.ceil(request_round_trip * 1000.0),
                        )
                        source_age_ms = max(
                            message.evidence_age_ms or 0,
                            round_trip_ms,
                        )
                    accepted = self._accept_sequenced(message, line=line)
                    if not accepted:
                        self._watchdog(self._now())
                        publish_status_heartbeat_if_due()
                        continue
                    stopped = self._apply_message(
                        message,
                        receipt=receipt,
                        source_age_ms=source_age_ms,
                    )
                    self._tick_session()
                    if not stopped:
                        self._watchdog(self._now())
                    if not stopped and message.kind is RemotePoseKind.ANALYSIS:
                        outstanding_request = self._issue_pose_request(
                            connection,
                            session_id=message.session_id,
                            challenge=challenge,
                        )
                    publish_status_heartbeat_if_due()
                    if stopped:
                        break
        except _ConnectionRejected as exc:
            failure_kind = exc.failure_kind
        except Exception:
            failure_kind = "ServiceProcessingError"
        finally:
            _close_quietly(connection)
            with self._lock:
                if self._connection is connection:
                    self._connection = None
                self._peer = None
            with self._lock:
                input_terminal = self._input_terminal
                session_terminal = self._session_terminal
            if (
                bound
                and not stopped
                and not self._shutdown.is_set()
                and not input_terminal
                and not session_terminal
            ):
                self._pause_for_pose_loss(failure_kind or "ConnectionLost", now=self._now())
            elif (
                failure_kind is not None
                and not self._shutdown.is_set()
                and not input_terminal
                and not session_terminal
            ):
                with self._lock:
                    self._failure_kind = failure_kind
            with self._lock:
                if not self._shutdown.is_set() and not input_terminal:
                    self._service_state = "listening"
            self._publish_status()

    def shutdown(self) -> None:
        """Stop accepting input and request the current local stop boundary."""

        with self._shutdown_lock:
            if self._shutdown_complete.is_set():
                return
            self._shutdown.set()
            with self._lock:
                listener = self._listener
                connection = self._connection
                monitor = self._stop_monitor
                session = self._session
                speaker = self._speaker
                _quiesce_speaker(speaker)
                should_request_stop = (
                    session is not None
                    and not self._session_terminal
                    and not self._session_stop_requested
                )
                self._session_terminal = self._session_terminal or session is not None
                if should_request_stop:
                    self._session_stop_requested = True
                if session is not None:
                    if self._session_id is not None:
                        self._retired_session_ids.add(self._session_id)
                    self._session = None
                    self._speaker = None
                self._service_state = (
                    "failed" if self._input_terminal or self._fatal_service_failure else "stopped"
                )
                self._peer = None
                if self._voice_state == "connected":
                    self._voice_state = "silent"

            if connection is not None:
                _close_quietly(connection)
            if listener is not None:
                _close_quietly(listener)
            if speaker is not None:
                try:
                    speaker.preempt()
                except Exception:
                    pass
            if monitor is not None:
                try:
                    monitor.close(timeout_seconds=1.0)
                except Exception:
                    # Shutdown continues fail-closed even if the GPIO worker
                    # does not acknowledge its bounded cleanup window.
                    pass
            with self._lifecycle_lock:
                if should_request_stop and session is not None:
                    try:
                        session.request_physical_stop()
                    except Exception:
                        with self._lock:
                            self._failure_kind = "SessionStopError"
                    if not _session_ended(session):
                        transition = getattr(session.coordinator, "transition_to", None)
                        if callable(transition):
                            try:
                                transition(SessionMode.STOPPED)
                            except Exception:
                                pass
                if speaker is not None:
                    try:
                        speaker.close()
                    except Exception:
                        pass
            self._publish_status()
            self._status_writer.close()
            self._shutdown_complete.set()

    def close(self) -> None:
        self.shutdown()

    def __enter__(self) -> RemotePoseService:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.shutdown()

    def _accept_start(
        self,
        message: RemotePoseMessage,
        *,
        receipt: float,
        peer_host: str,
    ) -> None:
        with self._lifecycle_lock:
            if self._shutdown.is_set():
                raise _ConnectionRejected("ServiceStopped")
            self._accept_start_serialized(
                message,
                receipt=receipt,
                peer_host=peer_host,
            )

    def _accept_start_serialized(
        self,
        message: RemotePoseMessage,
        *,
        receipt: float,
        peer_host: str,
    ) -> None:
        with self._lock:
            current_id = self._session_id
            current_session = self._session
            start_stop_epoch = self._stop_request_epoch
            monitor_started = self._stop_monitor_started
            input_terminal = self._input_terminal
        if current_session is not None and current_id != message.session_id:
            try:
                inherit_pause = current_session.coordinator.current_mode is SessionMode.PAUSED
            except Exception:
                inherit_pause = True
        else:
            inherit_pause = False
        if not monitor_started or input_terminal:
            raise _ConnectionRejected("PhysicalStopUnavailable")
        if current_id == message.session_id:
            with self._lock:
                terminal = self._session_terminal
            if current_session is None or terminal or _session_ended(current_session):
                raise _ConnectionRejected("RetiredSession")
            with self._lock:
                self._peer = peer_host
                self._service_state = "connected"
                self._failure_kind = None
                self._resume_armed = False
            self._publish_status()
            return
        with self._lock:
            if message.session_id in self._retired_session_ids:
                raise _ConnectionRejected("RetiredSession")

        self._retire_current_session()
        session, speaker, voice_state, voice_failure = self._create_session()
        try:
            session.start(
                instructions=REMOTE_POSE_SESSION_INSTRUCTIONS,
                voice=self.config.realtime_voice,
            )
            if inherit_pause:
                transition = getattr(session.coordinator, "transition_to", None)
                if not callable(transition):
                    raise RuntimeError("new session cannot inherit pause")
                transition(SessionMode.PAUSED)
        except Exception:
            _quiesce_speaker(speaker)
            try:
                speaker.preempt()
            except Exception:
                pass
            try:
                session.request_physical_stop()
            except Exception:
                pass
            try:
                speaker.close()
            except Exception:
                pass
            if voice_state == "connected":
                session, speaker = self._create_silent_session()
                try:
                    session.start(
                        instructions=REMOTE_POSE_SESSION_INSTRUCTIONS,
                        voice=self.config.realtime_voice,
                    )
                    if inherit_pause:
                        transition = getattr(session.coordinator, "transition_to", None)
                        if not callable(transition):
                            raise RuntimeError("new session cannot inherit pause")
                        transition(SessionMode.PAUSED)
                except Exception:
                    _quiesce_speaker(speaker)
                    try:
                        speaker.preempt()
                    except Exception:
                        pass
                    try:
                        session.request_physical_stop()
                    except Exception:
                        pass
                    try:
                        speaker.close()
                    except Exception:
                        pass
                    raise _ConnectionRejected("SessionStartError") from None
                voice_state = "silent"
                voice_failure = "RealtimeSessionStartError"
            else:
                raise _ConnectionRejected("SessionStartError") from None

        with self._lock:
            interrupted_by_local_stop = (
                self._input_terminal or self._stop_request_epoch != start_stop_epoch
            )
            if not interrupted_by_local_stop:
                self._session = session
                self._speaker = speaker
                self._session_id = message.session_id
                self._session_terminal = False
                self._session_stop_requested = False
                self._last_sequence = 0
                self._last_message_digest = None
                self._last_analysis = None
                self._last_analysis_receipt = receipt
                self._last_pose_age_ms = None
                self._resume_armed = False
                self._loss_injected = False
                self._exercise_activated = False
                self._peer = peer_host
                self._service_state = "connected"
                self._voice_state = voice_state
                self._voice_failure_kind = voice_failure
                self._failure_kind = None
        if interrupted_by_local_stop:
            _quiesce_speaker(speaker)
            try:
                speaker.preempt()
            except Exception:
                pass
            try:
                session.request_physical_stop()
            except Exception:
                pass
            try:
                speaker.close()
            except Exception:
                pass
            raise _ConnectionRejected("PhysicalStop")
        failure_binder = getattr(speaker, "bind_failure_callback", None)
        if callable(failure_binder):
            failure_binder(lambda: self._on_speaker_failure(session))
        success_binder = getattr(speaker, "bind_playback_succeeded_callback", None)
        if callable(success_binder):
            success_binder(lambda cue_id: self._on_speaker_playback_succeeded(session, cue_id))
        if voice_state == "connected":
            self._start_realtime_receiver(session)
        self._publish_status()

    def _accept_sequenced(self, message: RemotePoseMessage, *, line: bytes) -> bool:
        assert message.sequence is not None
        digest = hashlib.sha256(line).digest()
        with self._lock:
            if message.session_id != self._session_id:
                raise _ConnectionRejected("SessionMismatch")
            if message.sequence == self._last_sequence:
                if self._last_message_digest is not None and digest == self._last_message_digest:
                    return False
                raise _ConnectionRejected("SequenceConflict")
            if message.sequence < self._last_sequence:
                raise _ConnectionRejected("SequenceReplay")
            self._last_sequence = message.sequence
            self._last_message_digest = digest
        return True

    def _issue_pose_request(
        self,
        connection: _Connection,
        *,
        session_id: str,
        challenge: RemotePoseChallenge,
    ) -> tuple[RemotePoseRequest, float]:
        with self._lock:
            self._next_request_sequence += 1
            request_sequence = self._next_request_sequence
        try:
            request = RemotePoseRequest(
                session_id=session_id,
                service_epoch=challenge.service_epoch,
                server_nonce=challenge.server_nonce,
                request_sequence=request_sequence,
                request_nonce=self._dependencies.request_nonce_factory(),
            )
            sent_at = self._now()
            token = self._token
            if token is None:
                raise RuntimeError("remote token is unavailable")
            connection.sendall(encode_remote_pose_request(request, token))
        except _ConnectionRejected:
            raise
        except Exception:
            raise _ConnectionRejected("PoseRequestSendError") from None
        return request, sent_at

    def _apply_message(
        self,
        message: RemotePoseMessage,
        *,
        receipt: float,
        source_age_ms: int = 0,
    ) -> bool:
        with self._lifecycle_lock:
            if self._shutdown.is_set():
                return True
            return self._apply_message_serialized(
                message,
                receipt=receipt,
                source_age_ms=source_age_ms,
            )

    def _apply_message_serialized(
        self,
        message: RemotePoseMessage,
        *,
        receipt: float,
        source_age_ms: int = 0,
    ) -> bool:
        with self._lock:
            session = self._session
            terminal = self._session_terminal
            input_terminal = self._input_terminal
        if session is None:
            raise _ConnectionRejected("SessionUnavailable")
        if terminal or input_terminal:
            raise _ConnectionRejected("SessionTerminated")
        if message.kind is RemotePoseKind.STOP:
            with self._lock:
                self._resume_armed = False
                self._failure_kind = None
                if self._voice_state == "connected":
                    self._voice_state = "silent"
                speaker = self._speaker
                _quiesce_speaker(speaker)
                self._session_terminal = True
                should_request_stop = not self._session_stop_requested
                self._session_stop_requested = True
                if self._session_id is not None:
                    self._retired_session_ids.add(self._session_id)
                self._session = None
                self._speaker = None
            if speaker is not None:
                try:
                    speaker.preempt()
                except Exception:
                    pass
            if should_request_stop:
                try:
                    session.request_physical_stop()
                except Exception:
                    # The one-shot stop signal is committed before cleanup inside
                    # LaptopSquatSession. Never retry or weaken that boundary.
                    pass
            if not _session_ended(session):
                with self._lock:
                    self._failure_kind = "SessionStopError"
                transition = getattr(session.coordinator, "transition_to", None)
                if callable(transition):
                    try:
                        transition(SessionMode.STOPPED)
                    except Exception:
                        pass
            if speaker is not None:
                try:
                    speaker.close()
                except Exception:
                    pass
            with self._lock:
                self._service_state = "connected"
            self._publish_status()
            return True
        if message.kind is RemotePoseKind.RESUME:
            with self._lock:
                self._resume_armed = True
                self._failure_kind = None
            self._publish_status()
            return False
        if message.kind is not RemotePoseKind.ANALYSIS or message.analysis is None:
            raise _ConnectionRejected("RemotePoseProtocolError")
        self._apply_analysis_serialized(
            message.analysis,
            receipt=receipt,
            source_age_ms=source_age_ms,
        )
        return False

    def _apply_analysis_serialized(
        self,
        analysis: SquatAnalysis,
        *,
        receipt: float,
        source_age_ms: int = 0,
    ) -> None:
        if not isinstance(analysis, SquatAnalysis):
            raise _ConnectionRejected("RemotePoseProtocolError")
        if (
            isinstance(source_age_ms, bool)
            or not isinstance(source_age_ms, int)
            or source_age_ms < 0
        ):
            raise _ConnectionRejected("RemotePoseProtocolError")
        with self._lock:
            session = self._session
            terminal = self._session_terminal
            input_terminal = self._input_terminal
            stop_input_state = self._stop_input_state
        if session is None:
            raise _ConnectionRejected("SessionUnavailable")
        if terminal or input_terminal:
            raise _ConnectionRejected("SessionTerminated")

        now = self._now()
        evidence_time = receipt - (source_age_ms / 1000.0)
        elapsed = max(0.0, now - evidence_time)
        pose_age_ms = max(0, math.ceil(elapsed * 1000.0))
        pose_fresh = elapsed < self.config.watchdog_seconds
        if not pose_fresh:
            pose_age_ms = max(session.plan.max_pose_age_ms + 1, pose_age_ms)
        input_available = stop_input_state is StopInputState.AVAILABLE
        if not input_available:
            # Numeric camera evidence cannot become permission to move while
            # the independent physical-stop input is not confirmed usable.
            pose_age_ms = max(session.plan.max_pose_age_ms + 1, pose_age_ms)
            pose_fresh = False

        if not pose_fresh:
            with self._lock:
                self._last_analysis_receipt = evidence_time
                self._last_pose_age_ms = pose_age_ms
                self._resume_armed = False
                self._failure_kind = "PoseTimeout"
            self._pause_for_pose_loss_serialized(
                "PoseTimeout",
                now=now,
                force_stale=True,
            )
            self._publish_status()
            return

        mode = session.coordinator.current_mode
        activation_eligible = analysis.assessable and analysis.phase is SquatPhase.STANDING
        with self._lock:
            self._last_analysis = analysis
            self._last_analysis_receipt = evidence_time
            self._last_pose_age_ms = pose_age_ms
            self._loss_injected = False
            resume_armed = self._resume_armed
            # A local camera commonly starts before a person is visible. Keep
            # the one startup authorization across missing/non-standing
            # observations; consume it only when activation/resume can
            # actually be attempted. Never let an arm linger while already
            # active or in another non-activation mode.
            preserve_local_arm = (
                self.config.pose_source is PoseSourceMode.LOCAL
                and not activation_eligible
                and mode in {SessionMode.IDLE, SessionMode.CHECK_IN, SessionMode.PAUSED}
            )
            if resume_armed and not preserve_local_arm:
                self._resume_armed = False
            self._failure_kind = None

        if mode in {SessionMode.IDLE, SessionMode.CHECK_IN} and resume_armed and input_available:
            activated = session.activate_exercise(
                analysis,
                pose_age_ms=pose_age_ms,
            )
            if activated:
                with self._lock:
                    if session is self._session:
                        self._exercise_activated = True
        elif mode is SessionMode.PAUSED and resume_armed and input_available:
            session.resume_after_assessable_pose(
                analysis,
                pose_age_ms=pose_age_ms,
            )
        if not _session_ended(session):
            session.process_analysis(analysis, pose_age_ms=pose_age_ms)
        if _session_ended(session):
            with self._lock:
                if session is self._session:
                    self._session_terminal = True
        self._sync_voice_failure(session)
        self._publish_status()

    def _watchdog(self, now: float) -> None:
        with self._lock:
            receipt = self._last_analysis_receipt
            session = self._session
            loss_injected = self._loss_injected
            terminal = self._session_terminal
        if receipt is None or session is None or terminal or _session_ended(session):
            return
        elapsed = max(0.0, now - receipt)
        age_ms = max(0, math.ceil(elapsed * 1000.0))
        with self._lock:
            self._last_pose_age_ms = age_ms
        if elapsed >= self.config.watchdog_seconds and not loss_injected:
            self._pause_for_pose_loss("PoseTimeout", now=now, force_stale=True)
        else:
            self._publish_status()

    def _pause_for_pose_loss(
        self,
        failure_kind: str,
        *,
        now: float,
        force_stale: bool = False,
    ) -> None:
        with self._lifecycle_lock:
            if self._shutdown.is_set():
                return
            self._pause_for_pose_loss_serialized(
                failure_kind,
                now=now,
                force_stale=force_stale,
            )

    def _pause_for_pose_loss_serialized(
        self,
        failure_kind: str,
        *,
        now: float,
        force_stale: bool = False,
    ) -> None:
        with self._lock:
            session = self._session
            if session is None or self._session_terminal or _session_ended(session):
                if self._failure_kind not in {
                    "SessionPauseError",
                    "SessionProcessingError",
                    "SessionStopError",
                }:
                    self._failure_kind = failure_kind
                return
            if self._loss_injected:
                if self._failure_kind not in {
                    "SessionPauseError",
                    "SessionProcessingError",
                    "SessionStopError",
                }:
                    self._failure_kind = failure_kind
                self._resume_armed = False
                return
            analysis = self._last_analysis
            receipt = self._last_analysis_receipt
            self._loss_injected = True
            self._resume_armed = False
            self._failure_kind = failure_kind
        if session.coordinator.current_mode is not SessionMode.ACTIVE_EXERCISE:
            return
        if analysis is None:
            timestamp_ms = 0
            rep_count = 0
        else:
            timestamp_ms = analysis.timestamp_ms
            rep_count = analysis.rep_count
        elapsed = 0.0 if receipt is None else max(0.0, now - receipt)
        pose_age_ms = max(0, math.ceil(elapsed * 1000.0))
        if force_stale:
            pose_age_ms = max(session.plan.max_pose_age_ms + 1, pose_age_ms)
        timeout_analysis = SquatAnalysis(
            timestamp_ms=timestamp_ms,
            assessable=False,
            phase=SquatPhase.UNKNOWN,
            rep_count=rep_count,
            events=(),
            issues=(SquatAssessmentIssue.CAMERA_TIMEOUT,),
            confidence=0.0,
            knee_angle_degrees=None,
            arms_in_t=None,
        )
        try:
            session.process_analysis(timeout_analysis, pose_age_ms=pose_age_ms)
        except Exception:
            with self._lock:
                self._failure_kind = "SessionProcessingError"
            self._stop_after_pause_failure(session)
        else:
            try:
                still_active = session.coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
            except Exception:
                still_active = True
            if still_active:
                with self._lock:
                    self._failure_kind = "SessionPauseError"
                self._stop_after_pause_failure(session)
        with self._lock:
            self._last_pose_age_ms = pose_age_ms
        self._publish_status()

    def _stop_after_pause_failure(self, session: _RemoteSquatSession) -> None:
        """Escalate a failed Guardian-pause application to the local stop edge."""

        with self._lock:
            if session is self._session:
                self._session_terminal = True
                if self._voice_state == "connected":
                    self._voice_state = "silent"
                should_request_stop = not self._session_stop_requested
                self._session_stop_requested = True
                speaker = self._speaker
                _quiesce_speaker(speaker)
                if self._session_id is not None:
                    self._retired_session_ids.add(self._session_id)
                self._session = None
                self._speaker = None
            else:
                should_request_stop = True
                speaker = None
        if speaker is not None:
            try:
                speaker.preempt()
            except Exception:
                pass
        if should_request_stop:
            try:
                session.request_physical_stop()
            except Exception:
                pass
        if not _session_ended(session):
            transition = getattr(session.coordinator, "transition_to", None)
            if callable(transition):
                try:
                    transition(SessionMode.STOPPED)
                except Exception:
                    pass
        if speaker is not None:
            try:
                speaker.close()
            except Exception:
                pass

    def _create_session(
        self,
    ) -> tuple[_RemoteSquatSession, _CueSpeaker, str, str | None]:
        if not self.config.audio_enabled:
            session, speaker = self._create_silent_session()
            return session, speaker, "silent", None
        api_key: str | None = None
        try:
            supplied = self._dependencies.credential_provider()
            if supplied is not None:
                if not isinstance(supplied, str):
                    raise TypeError("credential provider must return a string or None")
                api_key = supplied.strip() or None
        except Exception:
            api_key = None
            credential_failure = "CredentialProviderError"
        else:
            credential_failure = None
        if api_key is None:
            session, speaker = self._create_silent_session()
            return session, speaker, "silent", credential_failure

        speaker: _CueSpeaker | None = None
        try:
            speaker = self._dependencies.speaker_factory(
                AlsaCommandConfig(playback_device=self.config.playback_device)
            )
            session = self._dependencies.session_factory(
                api_key=api_key,
                on_cue_audio=speaker.enqueue,
                on_audio_preempt=speaker.preempt,
            )
            return session, speaker, "connected", None
        except Exception:
            failure_kind = "RealtimeConnectError"
            if speaker is not None:
                _quiesce_speaker(speaker)
                try:
                    speaker.preempt()
                except Exception:
                    pass
                try:
                    speaker.close()
                except Exception:
                    pass
            session, silent = self._create_silent_session()
            return session, silent, "silent", failure_kind
        finally:
            api_key = None

    def _create_silent_session(self) -> tuple[_RemoteSquatSession, _CueSpeaker]:
        speaker: _CueSpeaker = _SilentCueSpeaker()
        session = self._dependencies.session_factory(
            api_key=None,
            on_cue_audio=speaker.enqueue,
            on_audio_preempt=speaker.preempt,
        )
        return session, speaker

    def _retire_current_session(self) -> None:
        with self._lock:
            session = self._session
            speaker = self._speaker
            session_id = self._session_id
            terminal = self._session_terminal
            stop_requested = self._session_stop_requested
            _quiesce_speaker(speaker)
            if session_id is not None:
                self._retired_session_ids.add(session_id)
            self._session = None
            self._speaker = None
            self._session_terminal = False
            self._session_stop_requested = False
        if speaker is not None:
            try:
                speaker.preempt()
            except Exception:
                pass
        if (
            session is not None
            and not terminal
            and not stop_requested
            and not _session_ended(session)
        ):
            try:
                session.request_physical_stop()
            except Exception:
                pass
        if speaker is not None:
            try:
                speaker.close()
            except Exception:
                pass

    def _start_realtime_receiver(self, session: _RemoteSquatSession) -> None:
        pump_once = getattr(session, "pump_once", None)
        if not callable(pump_once):
            return

        def receive() -> None:
            while self._session_is_live(session):
                try:
                    result = pump_once()
                except Exception:
                    self._record_voice_failure(session, "RealtimeReceiveError")
                    return
                failure_kind = getattr(result, "failure_kind", None)
                if failure_kind is not None:
                    self._record_voice_failure(session, "RealtimeProviderError")
                    return
                if getattr(result, "end_signal", None) is not None:
                    self._publish_status()
                    return

        thread = threading.Thread(
            target=receive,
            name="recoverybox-pi-realtime-receiver",
            daemon=True,
        )
        with self._lock:
            self._realtime_thread = thread
        thread.start()

    def _on_speaker_failure(self, session: _RemoteSquatSession) -> None:
        with self._lifecycle_lock:
            self._on_speaker_failure_serialized(session)

    def _on_speaker_playback_succeeded(
        self,
        session: _RemoteSquatSession,
        cue_id: CueId,
    ) -> None:
        """Continue an authorized check-in only after physical playback ends."""

        with self._lifecycle_lock:
            with self._lock:
                connected_lane = (
                    self.config.pose_source is PoseSourceMode.LOCAL
                    and self._service_state == "local"
                ) or (self.config.pose_source is PoseSourceMode.REMOTE and self._peer is not None)
                if (
                    session is not self._session
                    or self._session_terminal
                    or _session_ended(session)
                    or not connected_lane
                ):
                    return
            notifier = getattr(session, "notify_cue_playback_succeeded", None)
            if not callable(notifier):
                self._on_speaker_failure_serialized(session)
                return
            try:
                armed = notifier(cue_id)
            except Exception:
                self._on_speaker_failure_serialized(session)
                return
            if armed:
                # The original explicit RESUME was consumed while requesting
                # the detection cue. Successful physical playback continues
                # that same authorization onto the required later pose; it is
                # not a fresh authorization and cannot survive pause/stop.
                with self._lock:
                    if session is self._session and not self._session_terminal:
                        self._resume_armed = True

    def _on_speaker_failure_serialized(self, session: _RemoteSquatSession) -> None:
        with self._lock:
            if session is not self._session or self._session_terminal or _session_ended(session):
                return
        reporter = getattr(session, "report_speaker_failure", None)
        if callable(reporter):
            try:
                reporter()
            except Exception:
                pass
        self._record_voice_failure(session, "SpeakerPlaybackError")

    def _record_voice_failure(
        self,
        session: _RemoteSquatSession,
        failure_kind: str,
    ) -> None:
        with self._lifecycle_lock:
            self._record_voice_failure_serialized(session, failure_kind)

    def _record_voice_failure_serialized(
        self,
        session: _RemoteSquatSession,
        failure_kind: str,
    ) -> None:
        with self._lock:
            if session is not self._session or self._session_terminal or _session_ended(session):
                return
            replace_unstarted = not self._exercise_activated
            self._voice_state = "failed"
            if self._voice_failure_kind is None:
                self._voice_failure_kind = failure_kind
        if replace_unstarted and self._replace_unstarted_voice_session(
            session,
            failure_kind,
        ):
            return
        self._publish_status()

    def _replace_unstarted_voice_session(
        self,
        failed_session: _RemoteSquatSession,
        failure_kind: str,
    ) -> bool:
        """Restore an IDLE local lane if cloud voice dies before activation."""

        replacement: _RemoteSquatSession | None = None
        silent_speaker: _CueSpeaker | None = None
        try:
            replacement, silent_speaker = self._create_silent_session()
            replacement.start(
                instructions=REMOTE_POSE_SESSION_INSTRUCTIONS,
                voice=self.config.realtime_voice,
            )
        except Exception:
            if silent_speaker is not None:
                _quiesce_speaker(silent_speaker)
                try:
                    silent_speaker.preempt()
                except Exception:
                    pass
            if replacement is not None:
                try:
                    replacement.request_physical_stop()
                except Exception:
                    pass
            if silent_speaker is not None:
                try:
                    silent_speaker.close()
                except Exception:
                    pass
            return False
        assert replacement is not None
        assert silent_speaker is not None
        with self._lock:
            if (
                failed_session is not self._session
                or self._session_terminal
                or self._exercise_activated
                or self._shutdown.is_set()
                or self._input_terminal
            ):
                keep_replacement = False
                old_speaker = None
            else:
                keep_replacement = True
                old_speaker = self._speaker
                _quiesce_speaker(old_speaker)
                self._session = replacement
                self._speaker = silent_speaker
                self._voice_state = "silent"
                self._voice_failure_kind = failure_kind
                # Explicit local-pose mode is itself the one startup
                # authorization. If its voice lane dies while CHECK_IN is
                # consuming that authorization, the silent IDLE replacement
                # must be allowed to consume exactly one later fresh pose.
                # Remote mode has no equivalent local authority and remains
                # disarmed until another authenticated RESUME arrives.
                self._resume_armed = self.config.pose_source is PoseSourceMode.LOCAL
        if not keep_replacement:
            _quiesce_speaker(silent_speaker)
            try:
                replacement.request_physical_stop()
            except Exception:
                pass
            try:
                silent_speaker.close()
            except Exception:
                pass
            return False
        try:
            failed_session.request_physical_stop()
        except Exception:
            pass
        if old_speaker is not None:
            try:
                old_speaker.close()
            except Exception:
                pass
        self._publish_status()
        return True

    def _sync_voice_failure(self, session: _RemoteSquatSession) -> None:
        with self._lock:
            if self._voice_state != "connected" or session is not self._session:
                return
        failure_kind = getattr(session, "realtime_failure_kind", None)
        if failure_kind is not None:
            self._record_voice_failure(session, "RealtimeProviderError")

    def _tick_session(self) -> None:
        with self._lifecycle_lock:
            self._tick_session_serialized()

    def _tick_session_serialized(self) -> None:
        with self._lock:
            session = self._session
            terminal = self._session_terminal
        if session is None or terminal or _session_ended(session):
            return
        try:
            session.tick()
        except Exception:
            self._record_voice_failure(session, "CueTickError")
        self._sync_voice_failure(session)

    def _session_is_live(self, session: _RemoteSquatSession) -> bool:
        with self._lock:
            return (
                not self._shutdown.is_set()
                and session is self._session
                and not self._session_terminal
                and not _session_ended(session)
            )

    def _now(self) -> float:
        value = self._dependencies.clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise _ConnectionRejected("ClockError")
        return float(value)

    def _status_now(self) -> float:
        value = self._dependencies.status_clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RuntimeError("status heartbeat clock is invalid")
        return float(value)

    def _publish_status(self, *, force: bool = False) -> None:
        with self._lock:
            self._publish_status_locked(force=force)

    def _publish_status_locked(self, *, force: bool = False) -> None:
        session = self._session
        try:
            if session is None:
                mode = (
                    SessionMode.STOPPED.value
                    if self._session_terminal and self._session_id is not None
                    else None
                )
            elif self._session_terminal:
                mode = SessionMode.STOPPED.value
            else:
                current_mode = session.coordinator.current_mode
                mode = current_mode.value if isinstance(current_mode, SessionMode) else None
        except Exception:
            mode = None
        rep_count = 0 if self._last_analysis is None else self._last_analysis.rep_count
        status: dict[str, object] = {
            "service": self._service_state,
            "peer": self._peer,
            "session": self._session_id,
            "mode": mode,
            "rep": rep_count,
            "age": self._last_pose_age_ms,
            "voice": self._voice_state,
            "failure": self._failure_kind or self._voice_failure_kind,
            "button": self._stop_input_state.value,
        }
        assert frozenset(status) == _STATUS_FIELDS
        self._status_writer.publish(status, force=force)


def _session_ended(session: _RemoteSquatSession) -> bool:
    try:
        return bool(session.ended) or session.coordinator.current_mode in {
            SessionMode.STOPPED,
            SessionMode.COMPLETE,
        }
    except Exception:
        return True


def _peer_host(address: object) -> str:
    if isinstance(address, tuple) and address and isinstance(address[0], str):
        return address[0]
    return ""


def _gpio_failure_kind(value: object) -> str:
    if isinstance(value, str) and value in _GPIO_FAILURE_KINDS:
        return value
    return "GPIOInputUnavailable"


def _close_quietly(value: object) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _atomic_write_status(path: Path, status: Mapping[str, object]) -> None:
    if frozenset(status) != _STATUS_FIELDS:
        raise ValueError("status contains fields outside the sanitized schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as output:
            descriptor = -1
            json.dump(
                dict(status),
                output,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def run_remote_pose_service(
    *,
    environment: Mapping[str, str] | None = None,
    output: TextIO = sys.stderr,
    dependencies: RemotePoseServiceDependencies | None = None,
) -> int:
    """Load the file-only secret and run the headless receiver until stopped."""

    env = os.environ if environment is None else environment
    selected = dependencies or RemotePoseServiceDependencies()
    if environment is not None and dependencies is None:
        selected = replace(
            selected,
            credential_provider=lambda: _credential_provider_from_environment(env),
            local_pose_source_factory=lambda: _local_pose_source_from_environment(env),
        )
    service: RemotePoseService | None = None
    previous_sigterm_handler: object | None = None
    sigterm_handler_installed = False
    try:
        config = RemotePoseServiceConfig.from_environment(env)
        token = (
            selected.token_loader(config.token_file)
            if config.pose_source is PoseSourceMode.REMOTE
            else None
        )
        service = RemotePoseService(config, token=token, dependencies=selected)
        if threading.current_thread() is threading.main_thread():
            previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

            def stop_on_sigterm(signum: int, frame: object) -> None:
                del signum, frame
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, stop_on_sigterm)
            sigterm_handler_installed = True
        service.serve_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"[remote-pose] service failed ({type(exc).__name__})", file=output, flush=True)
        return 2
    finally:
        if service is not None:
            service.shutdown()
        if sigterm_handler_installed:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)


def main(argv: Sequence[str] | None = None) -> int:
    """Environment-configured console entrypoint."""

    if argv:
        print("remote pose service accepts configuration through environment only", file=sys.stderr)
        return 2
    return run_remote_pose_service()


if __name__ == "__main__":  # pragma: no cover - module execution shim
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "DEFAULT_REMOTE_POSE_PORT",
    "DEFAULT_REMOTE_POSE_STATUS_PATH",
    "DEFAULT_REMOTE_POSE_WATCHDOG_SECONDS",
    "MAX_OPENAI_API_CREDENTIAL_BYTES",
    "OPENAI_API_CREDENTIAL_FILE_ENV",
    "PoseSourceMode",
    "RemotePoseService",
    "RemotePoseServiceConfig",
    "RemotePoseServiceConfigurationError",
    "RemotePoseServiceDependencies",
    "load_openai_api_key_credential",
    "main",
    "run_remote_pose_service",
]
