"""Configurable ALSA command adapters for raw 24 kHz mono PCM audio."""

from __future__ import annotations

import math
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO

from .ports import PCM_S16LE_24K_MONO, AudioFormat


@dataclass(frozen=True, slots=True)
class AlsaCommandConfig:
    """Command names, optional ALSA devices, and process timeouts."""

    capture_executable: str = "arecord"
    playback_executable: str = "aplay"
    capture_device: str | None = None
    playback_device: str | None = None
    process_timeout_seconds: float = 2.0
    audio_format: AudioFormat = PCM_S16LE_24K_MONO

    def __post_init__(self) -> None:
        if not self.capture_executable:
            raise ValueError("capture_executable cannot be empty")
        if not self.playback_executable:
            raise ValueError("playback_executable cannot be empty")
        if (
            isinstance(self.process_timeout_seconds, bool)
            or not isinstance(self.process_timeout_seconds, (int, float))
            or not math.isfinite(self.process_timeout_seconds)
        ):
            raise ValueError("process_timeout_seconds must be a finite number")
        if self.process_timeout_seconds <= 0:
            raise ValueError("process_timeout_seconds must be positive")
        if not (
            self.audio_format.signed
            and self.audio_format.little_endian
            and self.audio_format.sample_width_bytes == 2
        ):
            raise ValueError("ALSA adapters require signed 16-bit little-endian PCM")


def _alsa_args(executable: str, device: str | None, audio_format: AudioFormat) -> list[str]:
    args = [executable, "-q"]
    if device:
        args.extend(["-D", device])
    args.extend(
        [
            "-f",
            "S16_LE",
            "-c",
            str(audio_format.channels),
            "-r",
            str(audio_format.sample_rate_hz),
            "-t",
            "raw",
        ]
    )
    return args


def _terminate_and_reap(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> Exception | None:
    """Terminate and reap a child, returning a recovered terminate error.

    If ``terminate`` itself fails, ``kill`` is attempted immediately. A failure
    before a successful wait is raised so the adapter can retain ownership and
    retry cleanup later.
    """

    terminate_error: Exception | None = None
    try:
        process.terminate()
    except Exception as exc:
        terminate_error = exc
        process.kill()

    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)
    return terminate_error


class SubprocessRecorder:
    """Capture PCM with ``arecord`` while spooling output off the pipe."""

    def __init__(self, config: AlsaCommandConfig | None = None) -> None:
        self._config = config or AlsaCommandConfig()
        self._process: subprocess.Popen[bytes] | None = None
        self._spool: BinaryIO | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("capture is already active")
        spool = tempfile.TemporaryFile(mode="w+b")
        try:
            process = subprocess.Popen(
                _alsa_args(
                    self._config.capture_executable,
                    self._config.capture_device,
                    self._config.audio_format,
                ),
                stdin=subprocess.DEVNULL,
                stdout=spool,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            spool.close()
            raise
        self._spool = spool
        self._process = process

    def stop(self) -> bytes:
        process, spool = self._active()
        return_code = process.poll()
        if return_code is not None:
            self._finalize_active(process, spool)
            raise RuntimeError(f"capture process exited before stop with {return_code}")

        try:
            terminate_error = _terminate_and_reap(
                process,
                timeout_seconds=self._config.process_timeout_seconds,
            )
        except Exception:
            # Keep both handles installed so fail-safe abort can retry.
            raise

        try:
            if terminate_error is not None:
                raise terminate_error
            spool.seek(0)
            return spool.read()
        finally:
            self._finalize_active(process, spool)

    def abort(self) -> None:
        if self._process is None:
            return
        process, spool = self._active()
        try:
            terminate_error = _terminate_and_reap(
                process,
                timeout_seconds=self._config.process_timeout_seconds,
            )
        except Exception:
            # Retain ownership after incomplete cleanup so another explicit
            # abort/recovery attempt can still reach the child.
            raise
        try:
            if terminate_error is not None:
                raise terminate_error
        finally:
            self._finalize_active(process, spool)

    def _active(self) -> tuple[subprocess.Popen[bytes], BinaryIO]:
        if self._process is None or self._spool is None:
            raise RuntimeError("capture is not active")
        return self._process, self._spool

    def _finalize_active(self, process: subprocess.Popen[bytes], spool: BinaryIO) -> None:
        if self._process is process:
            self._process = None
        if self._spool is spool:
            self._spool = None
        spool.close()


class SubprocessPlayback:
    """Stream PCM into ``aplay`` and conservatively estimate audible duration."""

    def __init__(
        self,
        config: AlsaCommandConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config or AlsaCommandConfig()
        self._clock = clock
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._stream_id: str | None = None
        self._started_at: float | None = None
        self._submitted_bytes = 0
        self._stopping = False

    def start(self, stream_id: str) -> None:
        with self._lock:
            if self._process is not None:
                raise RuntimeError("playback is already active")
            self._process = subprocess.Popen(
                _alsa_args(
                    self._config.playback_executable,
                    self._config.playback_device,
                    self._config.audio_format,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._stream_id = stream_id
            self._started_at = self._clock()
            self._submitted_bytes = 0
            self._stopping = False

    def write(self, stream_id: str, pcm: bytes) -> None:
        # stop deliberately does not take _write_lock: terminating the child
        # breaks a full pipe and preempts this call.
        with self._write_lock:
            with self._lock:
                process = self._process
                if (
                    process is None
                    or self._stopping
                    or process.stdin is None
                    or stream_id != self._stream_id
                ):
                    raise RuntimeError("playback is not active")
            if process.poll() is not None:
                raise RuntimeError(f"playback process exited with {process.returncode}")
            process.stdin.write(pcm)
            process.stdin.flush()
            with self._lock:
                if self._process is process:
                    self._submitted_bytes += len(pcm)

    def finish(self) -> None:
        with self._write_lock:
            with self._lock:
                process = self._active_locked()
                self._stopping = True
            close_error: Exception | None = None
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except Exception as exc:
                close_error = exc
            try:
                self._wait_or_kill(process, require_success=True)
            except Exception:
                # A timeout/nonzero status can be reported after a successful
                # reap. Clear only in that confirmed case; otherwise retain the
                # handle so fail-safe stop can retry.
                if process.poll() is not None:
                    with self._lock:
                        self._finalize_active_locked(process)
                raise
            with self._lock:
                self._finalize_active_locked(process)
            if close_error is not None:
                raise close_error

    def stop(self) -> int:
        with self._lock:
            if self._process is None:
                return 0
            played_ms = self._estimated_played_ms_locked()
            process = self._process
            self._stopping = True

        try:
            terminate_error = _terminate_and_reap(
                process,
                timeout_seconds=self._config.process_timeout_seconds,
            )
        except Exception:
            # An incomplete kill/wait leaves the handle installed for retry.
            raise

        close_error: Exception | None = None
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                # Expected when termination closes the child's read end.
                pass
            except Exception as exc:
                close_error = exc
        with self._lock:
            self._finalize_active_locked(process)
        if terminate_error is not None:
            raise terminate_error
        if close_error is not None:
            raise close_error
        return played_ms

    def _estimated_played_ms_locked(self) -> int:
        if self._started_at is None:
            return 0
        elapsed_ms = max(0, int((self._clock() - self._started_at) * 1000))
        queued_ms = int(1000 * self._submitted_bytes / self._config.audio_format.bytes_per_second)
        return min(elapsed_ms, queued_ms)

    def _active_locked(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError("playback is not active")
        return self._process

    def _finalize_active_locked(self, process: subprocess.Popen[bytes]) -> None:
        if self._process is not process:
            return
        self._process = None
        self._stream_id = None
        self._started_at = None
        self._submitted_bytes = 0
        self._stopping = False

    def _wait_or_kill(
        self, process: subprocess.Popen[bytes], *, require_success: bool = False
    ) -> None:
        timed_out = False
        try:
            process.wait(timeout=self._config.process_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=self._config.process_timeout_seconds)
        if timed_out and require_success:
            raise TimeoutError("playback process did not finish before timeout")
        if require_success and process.returncode != 0:
            raise RuntimeError(f"playback process exited with {process.returncode}")
