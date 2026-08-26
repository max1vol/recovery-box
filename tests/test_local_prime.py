from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest

from recoverybox.core import CueId, CueKind, GuardianReason
from recoverybox.device.gpio_stop import GpioStopConfig, StopInputState
from recoverybox.device.remote_pose_service import RemotePoseServiceDependencies
from recoverybox.laptop.local_prime import (
    LocalPrimeConfig,
    LocalPrimeConfigurationError,
    VirtualStopMonitor,
    _MacOSCueSpeaker,
    build_local_prime_dependencies,
    load_local_openai_api_key,
    run_local_prime,
)
from recoverybox.realtime import ReleasedCueAudio
from recoverybox.session import ApprovedCuePlaybackAuthorization


def test_virtual_stop_monitor_reports_available_then_closed() -> None:
    snapshots = []
    monitor = VirtualStopMonitor(
        config=GpioStopConfig(line_offset=23),
        on_stop=lambda trigger: pytest.fail(f"unexpected stop: {trigger}"),
        on_status=snapshots.append,
    )

    monitor.start()
    monitor.close()

    assert [snapshot.state for snapshot in snapshots] == [
        StopInputState.AVAILABLE,
        StopInputState.CLOSED,
    ]
    assert monitor.snapshot.state is StopInputState.CLOSED


def test_dotenv_key_overrides_inherited_environment_without_exposing_value(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY='new-local-key'\n", encoding="utf-8")

    assert (
        load_local_openai_api_key(
            env_file,
            {"OPENAI_API_KEY": "stale-inherited-key"},
        )
        == "new-local-key"
    )


def test_key_loader_falls_back_to_environment_when_dotenv_has_no_key(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RECOVERYBOX_REALTIME_VOICE=marin\n", encoding="utf-8")

    assert (
        load_local_openai_api_key(
            env_file,
            {"OPENAI_API_KEY": "environment-key"},
        )
        == "environment-key"
    )


def test_key_loader_rejects_missing_key_without_echoing_secret(tmp_path: Path) -> None:
    with pytest.raises(LocalPrimeConfigurationError, match="missing or invalid") as raised:
        load_local_openai_api_key(tmp_path / "missing.env", {})

    assert "OPENAI_API_KEY=" not in str(raised.value)


def test_local_prime_uses_same_tailscale_ip_and_keeps_key_out_of_service_env(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    captured: dict[str, object] = {}
    expected_dependencies = RemotePoseServiceDependencies()

    def build_dependencies(api_key: str) -> RemotePoseServiceDependencies:
        captured["api_key"] = api_key
        return expected_dependencies

    def run_service(*, environment, output, dependencies) -> int:
        captured["environment"] = environment
        captured["output"] = output
        captured["dependencies"] = dependencies
        return 17

    output = io.StringIO()
    result = run_local_prime(
        LocalPrimeConfig(
            tailscale_ip="100.70.100.93",
            token_file=tmp_path / "pose-token.hex",
            env_file=env_file,
            status_file=tmp_path / "status.json",
        ),
        environment={"OPENAI_API_KEY": "stale-key"},
        output=output,
        service_runner=run_service,
        dependencies_builder=build_dependencies,
    )

    assert result == 17
    assert captured["api_key"] == "file-key"
    service_environment = captured["environment"]
    assert isinstance(service_environment, dict)
    assert service_environment["RECOVERYBOX_POSE_BIND_HOST"] == "100.70.100.93"
    assert service_environment["RECOVERYBOX_POSE_ALLOWED_PEER"] == "100.70.100.93"
    assert service_environment["RECOVERYBOX_AUDIO_ENABLED"] == "true"
    assert "OPENAI_API_KEY" not in service_environment
    assert "file-key" not in output.getvalue()
    assert captured["dependencies"] is expected_dependencies


def test_dependency_builder_replaces_only_local_development_boundaries() -> None:
    dependencies = build_local_prime_dependencies(
        "local-key",
        player_factory=lambda: pytest.fail("player must remain lazy"),
    )

    assert dependencies.credential_provider() == "local-key"
    assert dependencies.stop_monitor_factory is VirtualStopMonitor


def test_mac_speaker_reports_person_detected_only_after_player_success() -> None:
    class ImmediateTicket:
        def result(self, timeout: float | None = None) -> None:
            del timeout

    class FakePlayer:
        def play(self, pcm: bytes) -> ImmediateTicket:
            assert pcm == b"\x01\x00"
            return ImmediateTicket()

        def stop(self, timeout: float | None = None) -> None:
            del timeout

        def close(self) -> None:
            return

    authorization = ApprovedCuePlaybackAuthorization(
        cue_id=CueId.SQUAT_PERSON_DETECTED,
        cue_kind=CueKind.STATUS,
        catalog_version="prompt-cues-v3",
        guardian_rule_version="guardian-rules-v1",
        reason_codes=(GuardianReason.LOCAL_CUE_ACCEPTED,),
    )
    clip = ReleasedCueAudio(
        ticket_id=1,
        authorization=authorization,
        response_id="response-1",
        item_id="item-1",
        content_index=0,
        pcm16_mono_24khz=b"\x01\x00",
        queued_at_seconds=0.0,
        requested_at_seconds=0.0,
        released_at_seconds=0.1,
    )
    completed = threading.Event()
    speaker = _MacOSCueSpeaker(FakePlayer())
    speaker.bind_playback_succeeded_callback(
        lambda cue_id: (
            completed.set()
            if cue_id is CueId.SQUAT_PERSON_DETECTED
            else pytest.fail("unexpected cue")
        )
    )

    speaker.enqueue(clip)

    assert completed.wait(1.0)
