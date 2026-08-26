"""Manual push-to-talk capture for the laptop runtime.

The native ``sounddevice`` dependency is deliberately imported only by the
default stream factory when :meth:`LaptopMicrophoneRecorder.start` is called.
Importing :mod:`recoverybox.laptop` and running the ordinary test suite remain
hardware-free; tests inject a fake ``RawInputStream`` factory.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from recoverybox.device import PCM_S16LE_24K_MONO, AudioFormat


class LaptopMicrophoneError(RuntimeError):
    """Raised when a capture cannot be trusted and must be discarded."""


class _RawInputStream(Protocol):
    """Small subset of ``sounddevice.RawInputStream`` used by the adapter."""

    def start(self) -> object: ...

    def stop(self) -> object: ...

    def close(self) -> object: ...


RawInputCallback = Callable[[object, int, object, object], None]
RawInputStreamFactory = Callable[..., _RawInputStream]


def _is_supported_format(audio_format: AudioFormat) -> bool:
    return audio_format == PCM_S16LE_24K_MONO


@dataclass(frozen=True, slots=True)
class LaptopMicrophoneConfig:
    """Fixed PCM format and memory bound for one manual audio turn."""

    max_capture_seconds: float = 30.0
    blocksize_frames: int = 0
    device: int | str | None = None
    audio_format: AudioFormat = PCM_S16LE_24K_MONO

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_capture_seconds, bool)
            or not isinstance(self.max_capture_seconds, (int, float))
            or not math.isfinite(self.max_capture_seconds)
            or self.max_capture_seconds <= 0
        ):
            raise ValueError("max_capture_seconds must be a positive finite number")
        if (
            isinstance(self.blocksize_frames, bool)
            or not isinstance(self.blocksize_frames, int)
            or self.blocksize_frames < 0
        ):
            raise ValueError("blocksize_frames must be a non-negative integer")
        if self.device is not None and (
            isinstance(self.device, bool) or not isinstance(self.device, (int, str))
        ):
            raise TypeError("device must be an integer, string, or None")
        if isinstance(self.device, str) and not self.device.strip():
            raise ValueError("device must not be blank")
        if not _is_supported_format(self.audio_format):
            raise ValueError("laptop capture requires 24 kHz mono signed S16LE PCM")
        if self.max_capture_bytes < self.audio_format.sample_width_bytes:
            raise ValueError("max_capture_seconds must allow at least one complete frame")

    @property
    def max_capture_bytes(self) -> int:
        """Maximum bytes retained for a turn, rounded down to a whole frame."""

        frame_bytes = self.audio_format.channels * self.audio_format.sample_width_bytes
        byte_count = int(self.max_capture_seconds * self.audio_format.bytes_per_second)
        return byte_count - (byte_count % frame_bytes)


def _default_raw_input_stream_factory(**kwargs: object) -> _RawInputStream:
    """Create the native stream, importing ``sounddevice`` only on demand."""

    try:
        import sounddevice  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise LaptopMicrophoneError(
            "sounddevice is not installed; install the RecoveryBox laptop extra"
        ) from exc
    return sounddevice.RawInputStream(**kwargs)


class LaptopMicrophoneRecorder:
    """Bounded, thread-safe 24 kHz mono S16LE push-to-talk recorder.

    Any callback status, malformed block, callback conversion failure, memory
    limit breach, or stream shutdown error invalidates the entire turn. The
    recorder never returns partial audio after such a failure. Captured bytes
    are cleared from the adapter immediately after ``stop`` or ``abort``.
    """

    def __init__(
        self,
        config: LaptopMicrophoneConfig | None = None,
        *,
        stream_factory: RawInputStreamFactory | None = None,
    ) -> None:
        self._config = config or LaptopMicrophoneConfig()
        self._stream_factory = stream_factory or _default_raw_input_stream_factory
        self._control_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stream: _RawInputStream | None = None
        self._capturing = False
        self._pcm = bytearray()
        self._capture_error: LaptopMicrophoneError | None = None

    @property
    def audio_format(self) -> AudioFormat:
        return self._config.audio_format

    @property
    def active(self) -> bool:
        with self._state_lock:
            return self._capturing

    def start(self) -> None:
        """Start one manual capture turn."""

        with self._control_lock:
            with self._state_lock:
                if self._stream is not None or self._capturing:
                    raise RuntimeError("capture is already active")
                self._clear_turn_locked()

            stream: _RawInputStream | None = None
            try:
                stream = self._stream_factory(
                    samplerate=self._config.audio_format.sample_rate_hz,
                    channels=self._config.audio_format.channels,
                    dtype="int16",
                    blocksize=self._config.blocksize_frames,
                    device=self._config.device,
                    callback=self._on_audio,
                )
                with self._state_lock:
                    self._stream = stream
                    self._capturing = True
                stream.start()
            except Exception as exc:
                with self._state_lock:
                    self._capturing = False
                    if self._stream is stream:
                        self._stream = None
                    self._clear_turn_locked()
                if stream is not None:
                    self._best_effort_shutdown(stream)
                if isinstance(exc, LaptopMicrophoneError):
                    raise
                raise LaptopMicrophoneError("could not start microphone capture") from exc

    def stop(self) -> bytes:
        """Stop capture and return trusted PCM, or discard it and fail closed."""

        with self._control_lock:
            stream = self._detach_active_stream()
            shutdown_error = self._shutdown(stream)

            with self._state_lock:
                capture_error = self._capture_error
                pcm = bytes(self._pcm) if capture_error is None and shutdown_error is None else b""
                self._clear_turn_locked()

            if capture_error is not None:
                raise capture_error
            if shutdown_error is not None:
                raise LaptopMicrophoneError("could not stop microphone capture") from shutdown_error
            return pcm

    def abort(self) -> None:
        """Stop and discard the active turn; repeated aborts are harmless."""

        with self._control_lock:
            with self._state_lock:
                stream = self._stream
                self._stream = None
                self._capturing = False
                self._clear_turn_locked()
            if stream is None:
                return
            shutdown_error = self._shutdown(stream)
            if shutdown_error is not None:
                raise LaptopMicrophoneError(
                    "could not abort microphone capture"
                ) from shutdown_error

    def _detach_active_stream(self) -> _RawInputStream:
        with self._state_lock:
            stream = self._stream
            if stream is None or not self._capturing:
                raise RuntimeError("capture is not active")
            self._stream = None
            self._capturing = False
            return stream

    def _on_audio(self, indata: object, frames: int, time_info: object, status: object) -> None:
        """Copy one callback block without allowing exceptions into PortAudio."""

        del time_info
        try:
            with self._state_lock:
                if not self._capturing or self._capture_error is not None:
                    return
                if bool(status):
                    self._invalidate_locked("microphone reported an input status error")
                    return
                if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
                    self._invalidate_locked("microphone returned an invalid frame count")
                    return
                chunk = bytes(indata)  # type: ignore[arg-type]
                expected_bytes = (
                    frames
                    * self._config.audio_format.bytes_per_second
                    // (self._config.audio_format.sample_rate_hz)
                )
                if len(chunk) != expected_bytes:
                    self._invalidate_locked("microphone returned a malformed PCM block")
                    return
                if len(self._pcm) + len(chunk) > self._config.max_capture_bytes:
                    self._invalidate_locked("microphone capture exceeded its configured limit")
                    return
                self._pcm.extend(chunk)
        except Exception:
            # Native audio callbacks must never receive a Python exception.
            # Store only a generic error, not callback data or provider status.
            with self._state_lock:
                if self._capturing:
                    self._invalidate_locked("microphone callback failed")

    def _invalidate_locked(self, message: str) -> None:
        self._pcm.clear()
        self._capture_error = LaptopMicrophoneError(message)

    def _clear_turn_locked(self) -> None:
        self._pcm.clear()
        self._capture_error = None

    @staticmethod
    def _shutdown(stream: _RawInputStream) -> Exception | None:
        first_error: Exception | None = None
        try:
            stream.stop()
        except Exception as exc:
            first_error = exc
        try:
            stream.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
        return first_error

    @classmethod
    def _best_effort_shutdown(cls, stream: _RawInputStream) -> None:
        cls._shutdown(stream)
