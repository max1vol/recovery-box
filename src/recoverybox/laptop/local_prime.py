"""Development-only Mac launcher for the Pi-owned remote-pose service.

This launcher exists solely for exercising the real two-process topology on a
single Mac.  It binds the production service to the Mac's own Tailscale IPv4
address, replaces only the physical GPIO input with an always-available
virtual monitor, and plays already-approved cue clips through ``afplay``.
Production configuration and its Tailscale/GPIO validation are unchanged.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TextIO

from recoverybox.core import CueId
from recoverybox.device.gpio_stop import (
    GpioStopConfig,
    StopInputSnapshot,
    StopInputState,
    StopInputTrigger,
)
from recoverybox.device.remote_pose_service import (
    DEFAULT_REMOTE_POSE_PORT,
    RemotePoseServiceDependencies,
    run_remote_pose_service,
)
from recoverybox.laptop.audio import MacOSAudioPlayer, PlaybackCancelledError
from recoverybox.realtime import ReleasedCueAudio


class LocalPrimeConfigurationError(ValueError):
    """The explicit local-development launcher configuration is incomplete."""


class _PlaybackTicket(Protocol):
    def result(self, timeout: float | None = None) -> None: ...


class _AudioPlayer(Protocol):
    def play(self, pcm: bytes) -> _PlaybackTicket: ...

    def stop(self, timeout: float | None = None) -> None: ...

    def close(self) -> None: ...


class VirtualStopMonitor:
    """Always-available stop-input shim used only by this Mac test launcher."""

    def __init__(
        self,
        *,
        config: GpioStopConfig,
        on_stop: Callable[[StopInputTrigger], None],
        on_status: Callable[[StopInputSnapshot], None],
    ) -> None:
        self.config = config
        self._on_stop = on_stop
        self._on_status = on_status
        self._snapshot = StopInputSnapshot(
            state=StopInputState.STARTING,
            stop_count=0,
            failure_kind=None,
        )
        self._closed = False

    @property
    def snapshot(self) -> StopInputSnapshot:
        return self._snapshot

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("virtual stop monitor is closed")
        self._publish(StopInputState.AVAILABLE)

    def close(self, *, timeout_seconds: float = 1.0) -> None:
        del timeout_seconds
        if self._closed:
            return
        self._closed = True
        self._publish(StopInputState.CLOSED)

    def _publish(self, state: StopInputState) -> None:
        self._snapshot = StopInputSnapshot(
            state=state,
            stop_count=self._snapshot.stop_count,
            failure_kind=None,
        )
        self._on_status(self._snapshot)


class _MacOSCueSpeaker:
    """Adapt approved complete cue clips to the existing macOS player."""

    def __init__(self, player: _AudioPlayer) -> None:
        self._player = player
        self._lock = threading.RLock()
        self._accepting = True
        self._closed = False
        self._failure_callback: Callable[[], None] | None = None
        self._playback_succeeded_callback: Callable[[CueId], None] | None = None

    def bind_playback_succeeded_callback(self, callback: Callable[[CueId], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            if self._playback_succeeded_callback is not None:
                raise RuntimeError("speaker playback callback is already bound")
            self._playback_succeeded_callback = callback

    def bind_failure_callback(self, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            if self._failure_callback is not None:
                raise RuntimeError("speaker failure callback is already bound")
            self._failure_callback = callback

    def enqueue(self, clip: ReleasedCueAudio) -> None:
        if not isinstance(clip, ReleasedCueAudio):
            raise TypeError("clip must be ReleasedCueAudio")
        with self._lock:
            if not self._accepting:
                raise RuntimeError("cue speaker is quiesced")
            ticket = self._player.play(clip.pcm16_mono_24khz)
        threading.Thread(
            target=self._watch_ticket,
            args=(ticket, clip.authorization.cue_id),
            name=f"recoverybox-local-prime-cue-{clip.ticket_id}",
            daemon=True,
        ).start()

    def quiesce(self) -> None:
        with self._lock:
            self._accepting = False

    def preempt(self) -> None:
        self._player.stop()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._accepting = False
        self._player.close()

    def _watch_ticket(self, ticket: _PlaybackTicket, cue_id: CueId) -> None:
        try:
            ticket.result()
        except PlaybackCancelledError:
            return
        except Exception:
            with self._lock:
                callback = self._failure_callback
            if callback is not None:
                callback()
            return
        if cue_id is not CueId.SQUAT_PERSON_DETECTED:
            return
        with self._lock:
            callback = self._playback_succeeded_callback
        if callback is None:
            return
        try:
            callback(cue_id)
        except Exception:
            with self._lock:
                failure_callback = self._failure_callback
            if failure_callback is not None:
                failure_callback()


@dataclass(frozen=True, slots=True)
class LocalPrimeConfig:
    """Inputs for one explicit same-Mac Prime simulation."""

    tailscale_ip: str
    token_file: Path
    env_file: Path = Path(".env")
    status_file: Path = Path("/tmp/recoverybox-local-prime-status.json")
    port: int = DEFAULT_REMOTE_POSE_PORT
    voice: str = "marin"

    def __post_init__(self) -> None:
        if not isinstance(self.tailscale_ip, str) or not self.tailscale_ip.strip():
            raise LocalPrimeConfigurationError("tailscale_ip must not be blank")
        if not isinstance(self.voice, str) or not self.voice.strip():
            raise LocalPrimeConfigurationError("voice must not be blank")
        token_file = Path(self.token_file).expanduser().resolve()
        # Keep the final path component unresolved so the credential loader can
        # reject a caller-supplied symlink.  ``absolute()`` retains the existing
        # absolute-path behavior without following the selected file.
        env_file = Path(self.env_file).expanduser().absolute()
        status_file = Path(self.status_file).expanduser().resolve()
        object.__setattr__(self, "tailscale_ip", self.tailscale_ip.strip())
        object.__setattr__(self, "voice", self.voice.strip())
        object.__setattr__(self, "token_file", token_file)
        object.__setattr__(self, "env_file", env_file)
        object.__setattr__(self, "status_file", status_file)


def load_local_openai_api_key(
    env_file: str | Path,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Load the local key, preferring the explicitly selected project ``.env``."""

    selected_environment = os.environ if environment is None else environment
    path = Path(env_file)
    from_file: str | None = None
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        lines: Sequence[str] = ()
    except OSError as exc:
        raise LocalPrimeConfigurationError("the selected .env file could not be read") from exc
    else:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise LocalPrimeConfigurationError(
                "the selected .env file must be a private regular file"
            )
        if stat.S_IMODE(metadata.st_mode) & 0o066:
            raise LocalPrimeConfigurationError(
                "the selected .env file must not be readable or writable by group or others"
            )

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise LocalPrimeConfigurationError(
                "the selected .env file could not be opened safely"
            ) from exc
        try:
            opened_metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened_metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino)
                != (opened_metadata.st_dev, opened_metadata.st_ino)
                or stat.S_IMODE(opened_metadata.st_mode) & 0o066
            ):
                raise LocalPrimeConfigurationError(
                    "the selected .env file changed before it could be read safely"
                )
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 65_536):
                chunks.append(chunk)
            after_read_metadata = os.fstat(descriptor)
        except OSError as exc:
            raise LocalPrimeConfigurationError(
                "the selected .env file could not be read safely"
            ) from exc
        finally:
            os.close(descriptor)
        if (
            (opened_metadata.st_dev, opened_metadata.st_ino)
            != (after_read_metadata.st_dev, after_read_metadata.st_ino)
            or stat.S_IMODE(after_read_metadata.st_mode) & 0o066
            or after_read_metadata.st_size != sum(map(len, chunks))
        ):
            raise LocalPrimeConfigurationError(
                "the selected .env file changed while it was being read"
            )
        try:
            lines = b"".join(chunks).decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise LocalPrimeConfigurationError("the selected .env file is not valid UTF-8") from exc
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        name, separator, raw_value = stripped.partition("=")
        if not separator or name.strip() != "OPENAI_API_KEY":
            continue
        if from_file is not None:
            raise LocalPrimeConfigurationError("OPENAI_API_KEY appears more than once in .env")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        from_file = value
    key = from_file if from_file is not None else selected_environment.get("OPENAI_API_KEY", "")
    if not isinstance(key, str) or not key or len(key) > 4_096:
        raise LocalPrimeConfigurationError("OPENAI_API_KEY is missing or invalid")
    if any(not 33 <= ord(character) <= 126 for character in key):
        raise LocalPrimeConfigurationError("OPENAI_API_KEY is missing or invalid")
    return key


def build_local_prime_dependencies(
    api_key: str,
    *,
    player_factory: Callable[[], _AudioPlayer] = MacOSAudioPlayer,
) -> RemotePoseServiceDependencies:
    """Replace only GPIO, credential, and speaker adapters for the Mac test."""

    base = RemotePoseServiceDependencies()

    def speaker_factory(config: object) -> _MacOSCueSpeaker:
        del config
        return _MacOSCueSpeaker(player_factory())

    return replace(
        base,
        speaker_factory=speaker_factory,
        credential_provider=lambda: api_key,
        stop_monitor_factory=VirtualStopMonitor,
    )


ServiceRunner = Callable[..., int]
DependenciesBuilder = Callable[[str], RemotePoseServiceDependencies]


def run_local_prime(
    config: LocalPrimeConfig,
    *,
    environment: Mapping[str, str] | None = None,
    output: TextIO = sys.stderr,
    service_runner: ServiceRunner = run_remote_pose_service,
    dependencies_builder: DependenciesBuilder = build_local_prime_dependencies,
) -> int:
    """Run the Pi-owned Guardian/Realtime/audio service on this Mac."""

    if not isinstance(config, LocalPrimeConfig):
        raise TypeError("config must be a LocalPrimeConfig")
    api_key = load_local_openai_api_key(config.env_file, environment)
    service_environment = {
        "RECOVERYBOX_POSE_SOURCE": "remote",
        "RECOVERYBOX_POSE_BIND_HOST": config.tailscale_ip,
        "RECOVERYBOX_POSE_ALLOWED_PEER": config.tailscale_ip,
        "RECOVERYBOX_POSE_TOKEN_FILE": str(config.token_file),
        "RECOVERYBOX_POSE_PORT": str(config.port),
        "RECOVERYBOX_STATUS_PATH": str(config.status_file),
        "RECOVERYBOX_REALTIME_VOICE": config.voice,
        "RECOVERYBOX_AUDIO_ENABLED": "true",
    }
    dependencies = dependencies_builder(api_key)
    api_key = ""
    print(
        f"[local-prime] listening on {config.tailscale_ip}:{config.port}; "
        f"status={config.status_file}",
        file=output,
        flush=True,
    )
    return service_runner(
        environment=service_environment,
        output=output,
        dependencies=dependencies,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recoverybox-local-prime",
        description=(
            "Development only: run the Pi-owned Guardian, Realtime, and audio service "
            "on this Mac while recoverybox-pose-client supplies camera pose results."
        ),
    )
    parser.add_argument("--tailscale-ip", required=True, help="this Mac's literal Tailscale IPv4")
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("/tmp/recoverybox-local-prime-status.json"),
    )
    parser.add_argument("--port", type=int, default=DEFAULT_REMOTE_POSE_PORT)
    parser.add_argument("--voice", default="marin")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run_local_prime(
            LocalPrimeConfig(
                tailscale_ip=args.tailscale_ip,
                token_file=args.token_file,
                env_file=args.env_file,
                status_file=args.status_file,
                port=args.port,
                voice=args.voice,
            )
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[local-prime] failed ({type(exc).__name__})", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LocalPrimeConfig",
    "LocalPrimeConfigurationError",
    "VirtualStopMonitor",
    "build_local_prime_dependencies",
    "load_local_openai_api_key",
    "main",
    "run_local_prime",
]
