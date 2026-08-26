from __future__ import annotations

import io
import json
from dataclasses import dataclass

from recoverybox import cli
from recoverybox.core import DEFAULT_CUE_CATALOG
from recoverybox.device import PCM_S16LE_24K_MONO, DeviceState, LedMode
from recoverybox.voice_cli import (
    CHECKIN_INSTRUCTIONS,
    DeveloperVoiceCheckinApplication,
    TerminalStatus,
    build_voice_checkin_application,
    run_voice_checkin,
)


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    def send_event(self, event) -> None:
        self.sent.append(dict(event))

    def receive_event(self):
        raise TimeoutError("bounded test receive")

    def close(self) -> None:
        self.closed = True


class FakeRecorder:
    def __init__(self) -> None:
        self.active = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> bytes:
        self.active = False
        return b"\x01\x00" * 12_000

    def abort(self) -> None:
        self.active = False


class FakePlayback:
    def start(self, stream_id: str) -> None:
        del stream_id

    def write(self, stream_id: str, pcm: bytes) -> None:
        del stream_id, pcm

    def finish(self) -> None:
        pass

    def stop(self) -> int:
        return 0


def test_build_composes_manual_24k_realtime_without_exposing_key() -> None:
    secret = "test-key-must-never-be-printed"
    transport = FakeTransport()
    output = io.StringIO()
    seen_audio_configs = []

    def connect(*, api_key: str, timeout_seconds: float):
        assert api_key == secret
        assert timeout_seconds == 5.0
        return transport

    def recorder_factory(config):
        seen_audio_configs.append(config)
        return FakeRecorder()

    def playback_factory(config):
        seen_audio_configs.append(config)
        return FakePlayback()

    application = build_voice_checkin_application(
        environment={
            "OPENAI_API_KEY": secret,
            "RECOVERYBOX_REALTIME_MODEL": "gpt-realtime-2.1",
            "RECOVERYBOX_AUDIO_RATE_HZ": "24000",
        },
        output=output,
        transport_factory=connect,
        recorder_factory=recorder_factory,
        playback_factory=playback_factory,
    )

    session_update = transport.sent[0]
    session = session_update["session"]
    assert session_update["type"] == "session.update"
    assert session["model"] == "gpt-realtime-2.1"
    assert session["audio"]["input"]["turn_detection"] is None
    assert session["audio"]["input"]["format"]["rate"] == 24_000
    assert session["audio"]["output"]["format"]["rate"] == 24_000
    assert "Never coach an exercise in progress" in CHECKIN_INSTRUCTIONS
    assert session["instructions"].startswith(f"{CHECKIN_INSTRUCTIONS.strip()}\n\n")
    for cue in DEFAULT_CUE_CATALOG.values():
        cue_line = f"- {cue.cue_id}: {json.dumps(cue.spoken_text, ensure_ascii=False)}"
        assert session["instructions"].count(cue_line) == 1
    assert all(config.audio_format == PCM_S16LE_24K_MONO for config in seen_audio_configs)
    assert secret not in output.getvalue()
    assert secret not in json.dumps(transport.sent)

    application.shutdown()
    assert transport.closed


@dataclass
class FakeController:
    state: DeviceState = DeviceState.IDLE
    pressed: int = 0
    released: int = 0
    ended: int = 0
    ticks: int = 0

    def on_button_pressed(self) -> None:
        self.pressed += 1
        self.state = DeviceState.RECORDING

    def on_button_released(self) -> None:
        self.released += 1
        self.state = DeviceState.WAITING

    def on_double_click(self) -> None:
        self.ended += 1
        self.state = DeviceState.ENDED

    def on_tick(self) -> None:
        self.ticks += 1


class FakeAdapter:
    def __init__(self, outcomes: list[Exception | None] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.pumps = 0

    def pump_once(self) -> None:
        self.pumps += 1
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if outcome is not None:
                raise outcome


def test_enter_toggles_press_release_and_q_shuts_down() -> None:
    controller = FakeController()
    output = io.StringIO()
    application = DeveloperVoiceCheckinApplication(
        controller=controller,  # type: ignore[arg-type]
        adapter=FakeAdapter(),  # type: ignore[arg-type]
        status=TerminalStatus(output),
    )

    assert application.handle_line("")
    assert controller.state is DeviceState.RECORDING
    assert controller.pressed == 1

    assert application.handle_line("")
    assert controller.state is DeviceState.WAITING
    assert controller.released == 1

    assert not application.handle_line("q")
    assert controller.state is DeviceState.ENDED
    assert controller.ended == 1
    assert "Recording" in output.getvalue()
    assert "Turn sent" in output.getvalue()


def test_pump_timeout_is_recoverable_but_connection_loss_is_redacted() -> None:
    controller = FakeController()
    adapter = FakeAdapter([TimeoutError("provider details"), ConnectionResetError("details")])
    application = DeveloperVoiceCheckinApplication(
        controller=controller,  # type: ignore[arg-type]
        adapter=adapter,  # type: ignore[arg-type]
        status=TerminalStatus(io.StringIO()),
    )

    assert application.pump_once()
    assert application.pump_failure_kind is None
    assert not application.pump_once()
    assert application.pump_failure_kind == "ConnectionResetError"


def test_missing_environment_key_fails_before_external_factories() -> None:
    output = io.StringIO()
    factory_called = False

    def connect(*, api_key: str, timeout_seconds: float):
        nonlocal factory_called
        del api_key, timeout_seconds
        factory_called = True
        return FakeTransport()

    assert (
        run_voice_checkin(
            environment={},
            output=output,
            transport_factory=connect,
        )
        == 2
    )
    assert not factory_called
    assert "OPENAI_API_KEY must be set" in output.getvalue()


def test_terminal_led_outputs_only_mode() -> None:
    output = io.StringIO()
    status = TerminalStatus(output)
    status.set_mode(LedMode.BLINK)
    assert output.getvalue() == "[LED] blink\n"


def test_cli_dispatches_voice_checkin(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_voice_checkin", lambda: calls.append("run") or 7)
    assert cli.main(["voice-checkin"]) == 7
    assert calls == ["run"]
