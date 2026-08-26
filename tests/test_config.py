from pathlib import Path

import pytest

from recoverybox.config import ConfigurationError, Settings


def test_safe_defaults_are_pinned() -> None:
    settings = Settings.from_environment({})

    assert settings.realtime.model == "gpt-realtime-2.1"
    assert settings.realtime.sample_rate_hz == 24_000
    assert settings.realtime.websocket_url.endswith("model=gpt-realtime-2.1")
    assert settings.privacy.feature_store == Path("data/features.jsonl")
    assert settings.privacy.retain_transcripts is False
    assert settings.pose.camera_index == 0
    assert settings.pose.model_path == Path("models/mediapipe/pose_landmarker_lite-v1.task")
    assert settings.pose.preview is True
    assert settings.openai_api_key_present is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RECOVERYBOX_REALTIME_MODEL", "another-model"),
        ("RECOVERYBOX_REALTIME_URL", "wss://example.invalid/realtime"),
        ("RECOVERYBOX_AUDIO_RATE_HZ", "16000"),
        ("RECOVERYBOX_TRANSCRIPT_RETENTION", "true"),
        ("RECOVERYBOX_CAMERA_INDEX", "-1"),
        ("RECOVERYBOX_CAMERA_PREVIEW", "sometimes"),
    ],
)
def test_safety_contract_cannot_be_weakened_by_environment(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_environment({name: value})


def test_api_key_is_reported_without_being_stored() -> None:
    settings = Settings.from_environment({"OPENAI_API_KEY": "secret-value"})

    assert settings.openai_api_key_present is True
    assert "secret-value" not in repr(settings)


def test_pose_settings_are_configurable_without_opening_camera(tmp_path: Path) -> None:
    model_path = tmp_path / "pose.task"
    settings = Settings.from_environment(
        {
            "RECOVERYBOX_CAMERA_INDEX": "2",
            "RECOVERYBOX_POSE_MODEL_PATH": str(model_path),
            "RECOVERYBOX_CAMERA_PREVIEW": "false",
        }
    )

    assert settings.pose.camera_index == 2
    assert settings.pose.model_path == model_path
    assert settings.pose.preview is False
    assert not model_path.exists()
