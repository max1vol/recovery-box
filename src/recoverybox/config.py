"""Environment-backed configuration with safety-critical invariants."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REALTIME_MODEL = "gpt-realtime-2.1"
REALTIME_WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
PCM_SAMPLE_RATE_HZ = 24_000
DEFAULT_POSE_MODEL_PATH = Path("models/mediapipe/pose_landmarker_lite-v1.task")


class ConfigurationError(ValueError):
    """Raised when runtime configuration violates a fixed contract."""


def _integer(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _boolean(environment: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environment.get(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


@dataclass(frozen=True, slots=True)
class RealtimeSettings:
    model: str = REALTIME_MODEL
    voice: str = "marin"
    sample_rate_hz: int = PCM_SAMPLE_RATE_HZ

    def validate(self) -> None:
        if self.model != REALTIME_MODEL:
            raise ConfigurationError(
                f"RECOVERYBOX_REALTIME_MODEL must be pinned to {REALTIME_MODEL}"
            )
        if self.sample_rate_hz != PCM_SAMPLE_RATE_HZ:
            raise ConfigurationError(
                "RECOVERYBOX_AUDIO_RATE_HZ must be 24000 for the Realtime PCM contract"
            )
        if not self.voice.strip():
            raise ConfigurationError("RECOVERYBOX_REALTIME_VOICE cannot be empty")

    @property
    def websocket_url(self) -> str:
        return REALTIME_WEBSOCKET_URL


@dataclass(frozen=True, slots=True)
class AudioSettings:
    capture_device: str = "default"
    playback_device: str = "default"
    sample_rate_hz: int = PCM_SAMPLE_RATE_HZ

    def validate(self) -> None:
        if self.sample_rate_hz != PCM_SAMPLE_RATE_HZ:
            raise ConfigurationError("device audio must be mono 24 kHz PCM")
        if not self.capture_device or not self.playback_device:
            raise ConfigurationError("audio device names cannot be empty")


@dataclass(frozen=True, slots=True)
class HardwareSettings:
    button_gpio: int = 23
    led_gpio: int = 25
    minimum_hold_ms: int = 180

    def validate(self) -> None:
        if self.button_gpio == self.led_gpio:
            raise ConfigurationError("button and LED GPIO pins must differ")
        if min(self.button_gpio, self.led_gpio) < 0:
            raise ConfigurationError("GPIO pins cannot be negative")
        if not 50 <= self.minimum_hold_ms <= 5_000:
            raise ConfigurationError("minimum button hold must be between 50 and 5000 ms")


@dataclass(frozen=True, slots=True)
class PrivacySettings:
    feature_store: Path = Path("data/features.jsonl")
    retain_transcripts: bool = False

    def validate(self) -> None:
        if self.retain_transcripts:
            raise ConfigurationError(
                "transcript retention is disabled for the hackathon safety profile"
            )


@dataclass(frozen=True, slots=True)
class PoseSettings:
    """Laptop camera selection; model integrity is checked at demo startup."""

    camera_index: int = 0
    model_path: Path = DEFAULT_POSE_MODEL_PATH
    preview: bool = True

    def validate(self) -> None:
        if (
            isinstance(self.camera_index, bool)
            or not isinstance(self.camera_index, int)
            or self.camera_index < 0
        ):
            raise ConfigurationError("RECOVERYBOX_CAMERA_INDEX must be a non-negative integer")
        if not str(self.model_path).strip():
            raise ConfigurationError("RECOVERYBOX_POSE_MODEL_PATH cannot be empty")
        if not isinstance(self.preview, bool):
            raise ConfigurationError("RECOVERYBOX_CAMERA_PREVIEW must be true or false")


@dataclass(frozen=True, slots=True)
class Settings:
    realtime: RealtimeSettings
    audio: AudioSettings
    hardware: HardwareSettings
    privacy: PrivacySettings
    pose: PoseSettings
    openai_api_key_present: bool

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environment is None else environment
        if "RECOVERYBOX_REALTIME_URL" in env:
            raise ConfigurationError(
                "RECOVERYBOX_REALTIME_URL is not configurable; the endpoint and model are pinned"
            )
        rate = _integer(env, "RECOVERYBOX_AUDIO_RATE_HZ", PCM_SAMPLE_RATE_HZ)
        settings = cls(
            realtime=RealtimeSettings(
                model=env.get("RECOVERYBOX_REALTIME_MODEL", REALTIME_MODEL),
                voice=env.get("RECOVERYBOX_REALTIME_VOICE", "marin"),
                sample_rate_hz=rate,
            ),
            audio=AudioSettings(
                capture_device=env.get("RECOVERYBOX_CAPTURE_DEVICE", "default"),
                playback_device=env.get("RECOVERYBOX_PLAYBACK_DEVICE", "default"),
                sample_rate_hz=rate,
            ),
            hardware=HardwareSettings(
                button_gpio=_integer(env, "RECOVERYBOX_BUTTON_GPIO", 23),
                led_gpio=_integer(env, "RECOVERYBOX_LED_GPIO", 25),
                minimum_hold_ms=_integer(env, "RECOVERYBOX_MIN_HOLD_MS", 180),
            ),
            privacy=PrivacySettings(
                feature_store=Path(env.get("RECOVERYBOX_FEATURE_STORE", "data/features.jsonl")),
                retain_transcripts=_boolean(env, "RECOVERYBOX_TRANSCRIPT_RETENTION", False),
            ),
            pose=PoseSettings(
                camera_index=_integer(env, "RECOVERYBOX_CAMERA_INDEX", 0),
                model_path=Path(
                    env.get("RECOVERYBOX_POSE_MODEL_PATH", str(DEFAULT_POSE_MODEL_PATH))
                ),
                preview=_boolean(env, "RECOVERYBOX_CAMERA_PREVIEW", True),
            ),
            openai_api_key_present=bool(env.get("OPENAI_API_KEY", "").strip()),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        self.realtime.validate()
        self.audio.validate()
        self.hardware.validate()
        self.privacy.validate()
        self.pose.validate()
