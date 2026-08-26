"""macOS playback for complete PCM clips already approved by a safety gate.

This module deliberately has no prompt, phrase, TTS, or OpenAI surface.  Its
only input is one complete raw-audio clip.  The caller remains responsible for
ensuring that model audio passed the appropriate RecoveryBox quarantine before
calling :meth:`MacOSAudioPlayer.play`.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
import wave
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Protocol, Self

from recoverybox.device import PCM_S16LE_24K_MONO, AudioFormat


class LaptopAudioError(RuntimeError):
    """Base error for laptop playback failures."""


class PlaybackCancelledError(LaptopAudioError):
    """Raised by a ticket whose clip was stopped or preempted."""


class _Process(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


ProcessFactory = Callable[[Sequence[str]], _Process]
TempFileFactory = Callable[[], BinaryIO]


@dataclass(frozen=True, slots=True)
class MacOSAudioConfig:
    """Command and bounded shutdown timing for the macOS player."""

    afplay_executable: str = "/usr/bin/afplay"
    cancellation_poll_seconds: float = 0.01
    terminate_timeout_seconds: float = 1.0
    close_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not self.afplay_executable:
            raise ValueError("afplay_executable must not be blank")
        for name in (
            "cancellation_poll_seconds",
            "terminate_timeout_seconds",
            "close_timeout_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")


class PlaybackTicket:
    """Thread-safe completion state for one submitted clip."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._cancelled = False
        self._error: BaseException | None = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for terminal state and return whether it was reached."""

        return self._done.wait(timeout)

    def result(self, timeout: float | None = None) -> None:
        """Wait for success, or raise cancellation, timeout, or playback error."""

        if not self._done.wait(timeout):
            raise TimeoutError("audio playback did not complete before timeout")
        with self._lock:
            cancelled = self._cancelled
            error = self._error
        if cancelled:
            raise PlaybackCancelledError("audio playback was cancelled")
        if error is not None:
            raise LaptopAudioError("audio playback failed") from error

    def _finish(
        self,
        *,
        cancelled: bool = False,
        error: BaseException | None = None,
    ) -> None:
        with self._lock:
            if self._done.is_set():
                return
            self._cancelled = cancelled
            self._error = error
            self._done.set()


@dataclass(slots=True)
class _Clip:
    pcm: bytes
    ticket: PlaybackTicket = field(default_factory=PlaybackTicket)
    cancel_requested: threading.Event = field(default_factory=threading.Event)

    def discard_pcm(self) -> None:
        self.pcm = b""


def _default_process_factory(command: Sequence[str]) -> _Process:
    return subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _default_temp_file_factory() -> BinaryIO:
    return tempfile.NamedTemporaryFile(mode="w+b", suffix=".wav", delete=False)


class MacOSAudioPlayer:
    """Serialize complete 24 kHz mono S16LE clips through macOS ``afplay``.

    ``play`` never interrupts earlier submissions. ``preempt`` cancels the
    active clip and every queued clip, then places its replacement at the head
    of the now-empty queue. ``stop`` cancels active and queued work and waits
    until the active child has been reaped. No audio bytes are logged or stored
    on a ticket.
    """

    def __init__(
        self,
        config: MacOSAudioConfig | None = None,
        *,
        process_factory: ProcessFactory = _default_process_factory,
        temp_file_factory: TempFileFactory = _default_temp_file_factory,
    ) -> None:
        self._config = config or MacOSAudioConfig()
        self._process_factory = process_factory
        self._temp_file_factory = temp_file_factory
        self._condition = threading.Condition(threading.RLock())
        self._queue: deque[_Clip] = deque()
        self._active: _Clip | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="recoverybox-afplay",
            daemon=True,
        )
        self._worker.start()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def active(self) -> bool:
        with self._condition:
            return self._active is not None

    @property
    def queued_clip_count(self) -> int:
        with self._condition:
            return len(self._queue)

    def play(
        self,
        pcm: bytes,
        *,
        audio_format: AudioFormat = PCM_S16LE_24K_MONO,
    ) -> PlaybackTicket:
        """Queue one complete, already-approved PCM clip for playback."""

        clip = self._new_clip(pcm, audio_format)
        with self._condition:
            self._require_open_locked()
            self._queue.append(clip)
            self._condition.notify()
        return clip.ticket

    def preempt(
        self,
        pcm: bytes,
        *,
        audio_format: AudioFormat = PCM_S16LE_24K_MONO,
    ) -> PlaybackTicket:
        """Cancel all existing work and queue this approved clip next.

        Validation happens before any current audio is interrupted, so an
        invalid replacement cannot accidentally silence a valid cue.
        """

        replacement = self._new_clip(pcm, audio_format)
        with self._condition:
            self._require_open_locked()
            self._cancel_all_locked()
            self._queue.appendleft(replacement)
            self._condition.notify_all()
        return replacement.ticket

    def stop(self, timeout: float | None = None) -> None:
        """Cancel current and queued playback, waiting for audible output to stop."""

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        with self._condition:
            active_ticket = self._active.ticket if self._active is not None else None
            self._cancel_all_locked()
            self._condition.notify_all()
        if active_ticket is not None:
            wait_timeout = self._config.close_timeout_seconds if timeout is None else timeout
            if not active_ticket.wait(wait_timeout):
                raise TimeoutError("active audio did not stop before timeout")

    def close(self) -> None:
        """Permanently stop playback and join the background worker."""

        if threading.current_thread() is self._worker:
            raise RuntimeError("the audio worker cannot close itself")
        with self._condition:
            self._closed = True
            self._cancel_all_locked()
            self._condition.notify_all()
        self._worker.join(self._config.close_timeout_seconds)
        if self._worker.is_alive():
            raise TimeoutError("audio worker did not close before timeout")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _new_clip(self, pcm: bytes, audio_format: AudioFormat) -> _Clip:
        if not isinstance(pcm, bytes):
            raise TypeError("pcm must be bytes")
        if audio_format != PCM_S16LE_24K_MONO:
            raise ValueError("laptop playback requires 24 kHz mono signed S16LE PCM")
        if not pcm:
            raise ValueError("pcm clip must contain at least one complete sample")
        if len(pcm) % PCM_S16LE_24K_MONO.sample_width_bytes:
            raise ValueError("pcm clip must contain complete 2-byte samples")
        return _Clip(pcm=pcm)

    def _require_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("audio player is closed")

    def _cancel_all_locked(self) -> None:
        if self._active is not None:
            self._active.cancel_requested.set()
        while self._queue:
            clip = self._queue.popleft()
            clip.cancel_requested.set()
            clip.discard_pcm()
            clip.ticket._finish(cancelled=True)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed and not self._queue:
                    return
                clip = self._queue.popleft()
                self._active = clip

            cancelled = False
            error: BaseException | None = None
            try:
                cancelled = self._play_clip(clip)
            except BaseException as exc:
                error = exc
            finally:
                clip.discard_pcm()
                with self._condition:
                    if self._active is clip:
                        self._active = None
                    if error is not None:
                        clip.ticket._finish(error=error)
                    else:
                        clip.ticket._finish(cancelled=cancelled)
                    self._condition.notify_all()

    def _play_clip(self, clip: _Clip) -> bool:
        if clip.cancel_requested.is_set():
            return True

        path: Path | None = None
        try:
            path = self._write_wav(clip.pcm)
            clip.discard_pcm()
            if clip.cancel_requested.is_set():
                return True
            process = self._process_factory([self._config.afplay_executable, os.fspath(path)])
            return self._wait_for_process(process, clip.cancel_requested)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def _write_wav(self, pcm: bytes) -> Path:
        temp_file = self._temp_file_factory()
        raw_name = getattr(temp_file, "name", None)
        if not isinstance(raw_name, (str, os.PathLike)):
            temp_file.close()
            raise TypeError("temporary audio file must expose a filesystem path")
        path = Path(raw_name)
        try:
            with wave.open(temp_file, "wb") as output:
                output.setnchannels(PCM_S16LE_24K_MONO.channels)
                output.setsampwidth(PCM_S16LE_24K_MONO.sample_width_bytes)
                output.setframerate(PCM_S16LE_24K_MONO.sample_rate_hz)
                output.writeframes(pcm)
            temp_file.flush()
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            temp_file.close()
        return path

    def _wait_for_process(
        self,
        process: _Process,
        cancel_requested: threading.Event,
    ) -> bool:
        while process.poll() is None:
            if cancel_requested.wait(self._config.cancellation_poll_seconds):
                self._terminate_and_reap(process)
                return True
        return_code = process.wait()
        if return_code != 0:
            raise LaptopAudioError(f"afplay exited with status {return_code}")
        return False

    def _terminate_and_reap(self, process: _Process) -> None:
        try:
            process.terminate()
        except BaseException:
            process.kill()
        try:
            process.wait(timeout=self._config.terminate_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self._config.terminate_timeout_seconds)
