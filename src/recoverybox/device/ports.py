"""Ports and shared types for RecoveryBox device I/O.

The controller depends on these small interfaces instead of GPIO, ALSA, or a
particular realtime client.  This keeps the safety-critical interaction logic
deterministic and straightforward to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """Raw PCM format exchanged by the recorder, player, and realtime client."""

    sample_rate_hz: int = 24_000
    channels: int = 1
    sample_width_bytes: int = 2
    signed: bool = True
    little_endian: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.sample_width_bytes <= 0:
            raise ValueError("sample_width_bytes must be positive")

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate_hz * self.channels * self.sample_width_bytes


PCM_S16LE_24K_MONO = AudioFormat()


class DeviceState(StrEnum):
    IDLE = "idle"
    RECORDING = "recording"
    WAITING = "waiting"
    SPEAKING = "speaking"
    ENDED = "ended"
    ERROR = "error"


class LedMode(StrEnum):
    OFF = "off"
    SOLID = "solid"
    BLINK = "blink"
    FAST_BLINK = "fast_blink"


@runtime_checkable
class LedPort(Protocol):
    def set_mode(self, mode: LedMode) -> None:
        """Set the visible LED mode."""


@runtime_checkable
class RecorderPort(Protocol):
    def start(self) -> None:
        """Start capturing raw PCM."""

    def stop(self) -> bytes:
        """Stop capturing and return all raw PCM from this turn."""

    def abort(self) -> None:
        """Stop capturing and discard the current turn, if any."""


@runtime_checkable
class PlaybackPort(Protocol):
    def start(self, stream_id: str) -> None:
        """Start a new raw PCM output stream."""

    def write(self, stream_id: str, pcm: bytes) -> None:
        """Append correlated PCM; ``stop`` may safely preempt a blocked call."""

    def finish(self) -> None:
        """Let the active stream finish normally."""

    def stop(self) -> int:
        """Preempt any write and return the estimated played duration in ms."""


@runtime_checkable
class ConversationPort(Protocol):
    def send_audio_turn(self, pcm: bytes, *, audio_format: AudioFormat) -> str:
        """Commit a manual audio turn and return its local correlation ID."""

    def cancel_response(self, turn_id: str, response_id: str | None) -> None:
        """Cancel generation for a committed turn."""

    def truncate_assistant(self, item_id: str, audio_end_ms: int) -> None:
        """Trim the remote assistant item to audio that was actually played."""

    def clear_and_end(self) -> None:
        """Clear conversation state and end the current device session."""
