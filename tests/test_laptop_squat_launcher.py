from __future__ import annotations

import io
import threading
from collections import deque
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import pytest

from recoverybox import cli
from recoverybox.core import SessionMode
from recoverybox.exercise import (
    SquatAnalysis,
    SquatEvent,
    SquatEventType,
    SquatPhase,
)
from recoverybox.laptop.audio import PlaybackCancelledError
from recoverybox.laptop.squat_launcher import (
    LaptopRuntimePinError,
    SquatDemoCommand,
    SquatDemoConfig,
    SquatDemoDependencies,
    SquatDemoEndReason,
    _CueSpeakerBridge,
    build_squat_demo,
    run_squat_demo,
    validate_laptop_runtime_pins,
)


def analysis(
    timestamp_ms: int,
    *,
    rep_count: int = 0,
    events: tuple[SquatEvent, ...] = (),
) -> SquatAnalysis:
    return SquatAnalysis(
        timestamp_ms=timestamp_ms,
        assessable=True,
        phase=SquatPhase.STANDING,
        rep_count=rep_count,
        events=events,
        issues=(),
        confidence=0.95,
        knee_angle_degrees=170.0,
        arms_in_t=True,
    )


@dataclass(frozen=True)
class FakeSample:
    timestamp_ms: int
    pose: object | None
    quit_requested: bool = False


class FakePoseSource:
    def __init__(
        self,
        samples: list[FakeSample],
        *,
        order: list[str] | None = None,
        before_read: threading.Event | None = None,
        wait_before_read_number: int = 1,
    ) -> None:
        self.samples = deque(samples)
        self.order = order
        self.before_read = before_read
        self.wait_before_read_number = wait_before_read_number
        self.read_count = 0
        self.opened = False
        self.closed = False
        self.preview_lines: list[tuple[str, ...]] = []

    def open(self) -> FakePoseSource:
        if self.order is not None:
            self.order.append("open")
        self.opened = True
        return self

    def read(self, *, preview_lines=()):
        self.read_count += 1
        if self.before_read is not None and self.read_count == self.wait_before_read_number:
            assert self.before_read.wait(1)
        self.preview_lines.append(tuple(preview_lines))
        return self.samples.popleft()

    def close(self) -> None:
        self.closed = True


class FakeTracker:
    def __init__(self, analyses: list[SquatAnalysis]) -> None:
        self.analyses = deque(analyses)
        self.calls = 0
        self._rep_count = 0

    @property
    def rep_count(self) -> int:
        return self._rep_count

    def _next(self) -> SquatAnalysis:
        self.calls += 1
        result = self.analyses.popleft()
        self._rep_count = result.rep_count
        return result

    def update(self, frame: object) -> SquatAnalysis:
        assert frame is not None
        return self._next()

    def update_missing(self, timestamp_ms: int) -> SquatAnalysis:
        assert timestamp_ms >= 0
        return self._next()


class BlockingTransport:
    def __init__(self, incoming: list[dict] | None = None) -> None:
        self.sent: list[dict] = []
        self.incoming = deque(incoming or [])
        self.closed = False
        self.close_event = threading.Event()
        self._condition = threading.Condition()

    def send_event(self, event) -> None:
        if self.closed:
            raise RuntimeError("closed")
        self.sent.append(dict(event))

    def receive_event(self):
        with self._condition:
            while not self.incoming and not self.closed:
                self._condition.wait()
            if self.incoming:
                return self.incoming.popleft()
            raise EOFError("closed")

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self.close_event.set()
            self._condition.notify_all()


class FailingReceiveTransport(BlockingTransport):
    def __init__(self) -> None:
        super().__init__()
        self.receive_failed = threading.Event()

    def receive_event(self):
        self.receive_failed.set()
        raise OSError("provider payload must remain redacted")


class ControlFinishTransport(BlockingTransport):
    def send_event(self, event) -> None:
        super().send_event(event)
        if event.get("type") != "response.create":
            return
        with self._condition:
            self.incoming.extend(
                (
                    {
                        "type": "response.created",
                        "event_id": "control-created",
                        "response": {"id": "control-1", "status": "in_progress"},
                    },
                    {
                        "type": "response.function_call_arguments.done",
                        "event_id": "tool-finish",
                        "response_id": "control-1",
                        "item_id": "tool-item-1",
                        "name": "finish_session",
                        "call_id": "finish-call-1",
                        "arguments": "{}",
                    },
                )
            )
            self._condition.notify_all()


class FakeTicket:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.observed = threading.Event()

    def result(self, timeout: float | None = None) -> None:
        del timeout
        self.observed.set()
        if self.failure is not None:
            raise self.failure


class FakePlayer:
    def __init__(self, *, ticket_failure: Exception | None = None) -> None:
        self.played: list[bytes] = []
        self.stops = 0
        self.closed = False
        self.ticket_failure = ticket_failure
        self.last_ticket: FakeTicket | None = None

    def play(self, pcm: bytes) -> FakeTicket:
        self.played.append(pcm)
        self.last_ticket = FakeTicket(self.ticket_failure)
        return self.last_ticket

    def stop(self, timeout: float | None = None) -> None:
        del timeout
        self.stops += 1

    def close(self) -> None:
        self.closed = True


class FakeMicrophone:
    def __init__(self) -> None:
        self.active = False
        self.starts = 0
        self.stops = 0
        self.aborts = 0

    def start(self) -> None:
        self.starts += 1
        self.active = True

    def stop(self) -> bytes:
        self.stops += 1
        self.active = False
        return b"\x01\x00" * 200

    def abort(self) -> None:
        self.aborts += 1
        self.active = False


class FakeCommands:
    def __init__(self, batches: list[tuple[SquatDemoCommand, ...]] | None = None) -> None:
        self.batches = deque(batches or [])
        self.closed = False

    def poll(self) -> tuple[SquatDemoCommand, ...]:
        return self.batches.popleft() if self.batches else ()

    def close(self) -> None:
        self.closed = True


def dependencies(
    *,
    model_path: Path,
    source: FakePoseSource,
    tracker: FakeTracker,
    transport: BlockingTransport | None = None,
    player: FakePlayer | None = None,
    microphone: FakeMicrophone | None = None,
    commands: FakeCommands | None = None,
    order: list[str] | None = None,
    connect_calls: list[str] | None = None,
) -> SquatDemoDependencies:
    def validate_runtime() -> dict[str, str]:
        if order is not None:
            order.append("runtime")
        return {
            "mediapipe": "0.10.35",
            "opencv-contrib-python": "4.14.0.94",
        }

    def validate(path: str | Path) -> Path:
        assert Path(path) == model_path
        if order is not None:
            order.append("validate")
        return model_path

    def source_factory(config):
        assert config.model_asset_path == model_path.resolve()
        if order is not None:
            order.append("construct_source")
        return source

    def connect(*, api_key: str):
        assert api_key == "secret-test-key"
        if connect_calls is not None:
            connect_calls.append("connect")
        assert transport is not None
        return transport

    return SquatDemoDependencies(
        validate_runtime=validate_runtime,
        validate_model=validate,
        transport_factory=connect,
        pose_source_factory=source_factory,
        tracker_factory=lambda: tracker,
        audio_player_factory=lambda: player or FakePlayer(),
        microphone_factory=lambda: microphone or FakeMicrophone(),
        command_source_factory=lambda: commands or FakeCommands(),
    )


def test_bounded_launcher_validates_model_before_camera_and_uses_one_connection(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "pose.task"
    order: list[str] = []
    connect_calls: list[str] = []
    source = FakePoseSource(
        [FakeSample(100, object()), FakeSample(133, object())],
        order=order,
    )
    tracker = FakeTracker(
        [
            analysis(100),
            analysis(
                133,
                rep_count=1,
                events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
            ),
        ]
    )
    transport = BlockingTransport()
    player = FakePlayer()
    app = build_squat_demo(
        SquatDemoConfig(
            model_asset_path=model_path,
            preview=False,
            microphone_enabled=False,
            max_frames=2,
        ),
        environment={"OPENAI_API_KEY": "secret-test-key"},
        output=io.StringIO(),
        dependencies=dependencies(
            model_path=model_path,
            source=source,
            tracker=tracker,
            transport=transport,
            player=player,
            order=order,
            connect_calls=connect_calls,
        ),
    )

    assert order == ["runtime", "validate", "construct_source"]
    result = app.run()

    assert order == ["runtime", "validate", "construct_source", "open"]
    assert connect_calls == ["connect"]
    assert result.frames_processed == 2
    assert result.pose_frames == 2
    assert result.assessable_frames == 2
    assert result.rep_count == 1
    assert result.end_reason is SquatDemoEndReason.MAX_FRAMES
    assert result.final_mode is SessionMode.STOPPED
    assert result.voice_enabled is True
    assert result.voice_connected is True
    assert set(result.runtime_versions) == {
        "mediapipe",
        "opencv-contrib-python",
        "sounddevice",
    }
    assert connect_calls == ["connect"]
    assert [event["type"] for event in transport.sent].count("session.update") == 1
    assert [event["type"] for event in transport.sent].count("response.create") == 1
    assert transport.closed
    assert source.closed
    assert player.closed


def test_no_voice_camera_run_never_constructs_network_or_native_audio(tmp_path: Path) -> None:
    model_path = tmp_path / "pose.task"
    source = FakePoseSource([FakeSample(100, object(), quit_requested=True)])
    tracker = FakeTracker([analysis(100)])
    calls: list[str] = []
    deps = SquatDemoDependencies(
        validate_runtime=lambda: {
            "mediapipe": "0.10.35",
            "opencv-contrib-python": "4.14.0.94",
        },
        validate_model=lambda _: model_path,
        transport_factory=lambda **_: calls.append("transport"),  # type: ignore[arg-type]
        pose_source_factory=lambda _: source,
        tracker_factory=lambda: tracker,
        audio_player_factory=lambda: calls.append("speaker"),  # type: ignore[arg-type]
        microphone_factory=lambda: calls.append("microphone"),  # type: ignore[arg-type]
        command_source_factory=FakeCommands,
    )
    app = build_squat_demo(
        SquatDemoConfig(
            model_asset_path=model_path,
            preview=False,
            voice_enabled=False,
            microphone_enabled=False,
        ),
        environment={},
        output=io.StringIO(),
        dependencies=deps,
    )

    result = app.run()

    assert calls == []
    assert result.end_reason is SquatDemoEndReason.PHYSICAL_STOP
    assert result.voice_enabled is False
    assert result.voice_connected is False
    assert tracker.calls == 0
    assert source.closed


def test_no_voice_preserves_rep_events_for_guardian_without_dispatching_cues(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "pose.task"
    source = FakePoseSource([FakeSample(100, object()), FakeSample(133, object())])
    tracker = FakeTracker(
        [
            analysis(100),
            analysis(
                133,
                rep_count=1,
                events=(SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),),
            ),
        ]
    )
    output = io.StringIO()
    app = build_squat_demo(
        SquatDemoConfig(
            model_asset_path=model_path,
            preview=False,
            voice_enabled=False,
            microphone_enabled=False,
            max_frames=2,
        ),
        environment={},
        output=output,
        dependencies=dependencies(
            model_path=model_path,
            source=source,
            tracker=tracker,
        ),
    )

    result = app.run()

    assert result.rep_count == 1
    assert result.realtime_failure_kind is None
    assert app.session.cue_delivery.snapshot.requested_count == 0
    assert app.session.cue_delivery_enabled is False
    assert source.preview_lines[0][-1] == "q/Esc stop | r+Enter resume"
    assert "mode paused" not in output.getvalue()


def test_receive_failure_pauses_voice_but_camera_loop_keeps_processing(tmp_path: Path) -> None:
    model_path = tmp_path / "pose.task"
    transport = FailingReceiveTransport()
    source = FakePoseSource(
        [FakeSample(100, object()), FakeSample(133, object())],
        before_read=transport.receive_failed,
    )
    tracker = FakeTracker([analysis(100), analysis(133)])
    output = io.StringIO()
    app = build_squat_demo(
        SquatDemoConfig(
            model_asset_path=model_path,
            preview=False,
            microphone_enabled=False,
            max_frames=2,
        ),
        environment={"OPENAI_API_KEY": "secret-test-key"},
        output=output,
        dependencies=dependencies(
            model_path=model_path,
            source=source,
            tracker=tracker,
            transport=transport,
        ),
    )

    result = app.run()

    assert result.frames_processed == 2
    assert tracker.calls == 2
    assert result.realtime_failure_kind == "OSError"
    assert result.end_reason is SquatDemoEndReason.MAX_FRAMES
    assert "provider payload" not in output.getvalue()
    assert "local tracking continues" in output.getvalue()


def test_validated_finish_tool_ends_same_persistent_session(tmp_path: Path) -> None:
    model_path = tmp_path / "pose.task"
    transport = ControlFinishTransport()
    microphone = FakeMicrophone()
    commands = FakeCommands(
        [
            (SquatDemoCommand.TOGGLE_MICROPHONE,),
            (SquatDemoCommand.TOGGLE_MICROPHONE,),
        ]
    )
    source = FakePoseSource(
        [FakeSample(100, object()), FakeSample(133, object(), quit_requested=True)],
        before_read=transport.close_event,
        wait_before_read_number=2,
    )
    tracker = FakeTracker([analysis(100)])
    app = build_squat_demo(
        SquatDemoConfig(
            model_asset_path=model_path,
            preview=False,
        ),
        environment={"OPENAI_API_KEY": "secret-test-key"},
        output=io.StringIO(),
        dependencies=dependencies(
            model_path=model_path,
            source=source,
            tracker=tracker,
            transport=transport,
            microphone=microphone,
            commands=commands,
        ),
    )

    result = app.run()

    assert result.end_reason is SquatDemoEndReason.VALIDATED_TOOL_CALL
    assert app.session.end_controller.end_signal is not None
    assert app.session.end_controller.end_signal.tool_call_id == "finish-call-1"
    assert microphone.starts == 1
    assert microphone.stops == 1
    assert tracker.calls == 1
    assert transport.closed


def test_terminal_push_to_talk_is_no_audio_control_turn_then_q_stops(tmp_path: Path) -> None:
    model_path = tmp_path / "pose.task"
    transport = BlockingTransport()
    microphone = FakeMicrophone()
    commands = FakeCommands(
        [
            (SquatDemoCommand.TOGGLE_MICROPHONE,),
            (SquatDemoCommand.TOGGLE_MICROPHONE,),
            (SquatDemoCommand.STOP,),
        ]
    )
    source = FakePoseSource([FakeSample(100, object()), FakeSample(133, object())])
    tracker = FakeTracker([analysis(100), analysis(133)])
    app = build_squat_demo(
        SquatDemoConfig(model_asset_path=model_path, preview=False),
        environment={"OPENAI_API_KEY": "secret-test-key"},
        output=io.StringIO(),
        dependencies=dependencies(
            model_path=model_path,
            source=source,
            tracker=tracker,
            transport=transport,
            microphone=microphone,
            commands=commands,
        ),
    )

    result = app.run()

    event_types = [event["type"] for event in transport.sent]
    assert microphone.starts == 1
    assert microphone.stops == 1
    assert "input_audio_buffer.append" in event_types
    assert "input_audio_buffer.commit" in event_types
    assert event_types.count("response.create") == 1
    assert result.end_reason is SquatDemoEndReason.PHYSICAL_STOP


def test_async_speaker_ticket_failure_is_reported_to_session() -> None:
    player = FakePlayer(ticket_failure=RuntimeError("native details"))
    bridge = _CueSpeakerBridge(player)
    failure_kinds: list[str] = []
    bridge.bind_failure_callback(failure_kinds.append)

    class FakeSession:
        ended = False

        def __init__(self) -> None:
            self.reported = threading.Event()

        def report_speaker_failure(self) -> None:
            self.reported.set()

    session = FakeSession()
    bridge.bind(session)  # type: ignore[arg-type]
    bridge.enqueue(type("Clip", (), {"ticket_id": 1, "pcm16_mono_24khz": b"\x00\x00"})())  # type: ignore[arg-type]

    assert session.reported.wait(1)
    assert player.played == [b"\x00\x00"]
    assert failure_kinds == ["SpeakerPlaybackError"]


def test_intentional_playback_cancellation_is_not_reported_as_speaker_failure() -> None:
    player = FakePlayer(ticket_failure=PlaybackCancelledError("expected preemption"))
    bridge = _CueSpeakerBridge(player)
    failure_kinds: list[str] = []
    bridge.bind_failure_callback(failure_kinds.append)

    class FakeSession:
        ended = False

        def __init__(self) -> None:
            self.reported = threading.Event()

        def report_speaker_failure(self) -> None:
            self.reported.set()

    session = FakeSession()
    bridge.bind(session)  # type: ignore[arg-type]
    bridge.enqueue(type("Clip", (), {"ticket_id": 2, "pcm16_mono_24khz": b"\x00\x00"})())  # type: ignore[arg-type]

    assert player.last_ticket is not None
    assert player.last_ticket.observed.wait(1)
    assert not session.reported.is_set()
    assert failure_kinds == []


def test_stop_outranks_microphone_toggle_in_same_terminal_batch(tmp_path: Path) -> None:
    model_path = tmp_path / "pose.task"
    transport = BlockingTransport()
    microphone = FakeMicrophone()
    microphone.active = True
    commands = FakeCommands([(SquatDemoCommand.TOGGLE_MICROPHONE, SquatDemoCommand.STOP)])
    source = FakePoseSource([FakeSample(100, object())])
    tracker = FakeTracker([analysis(100)])
    app = build_squat_demo(
        SquatDemoConfig(model_asset_path=model_path, preview=False),
        environment={"OPENAI_API_KEY": "secret-test-key"},
        output=io.StringIO(),
        dependencies=dependencies(
            model_path=model_path,
            source=source,
            tracker=tracker,
            transport=transport,
            microphone=microphone,
            commands=commands,
        ),
    )

    result = app.run()

    assert result.end_reason is SquatDemoEndReason.PHYSICAL_STOP
    assert microphone.stops == 0
    assert microphone.aborts == 1
    assert tracker.calls == 0
    assert [event["type"] for event in transport.sent] == ["session.update"]


def test_runtime_pin_preflight_is_metadata_only_and_rejects_second_cv2_wheel(
    monkeypatch,
) -> None:
    queried: list[str] = []
    installed = {
        "mediapipe": "0.10.35",
        "opencv-contrib-python": "4.14.0.94",
    }

    def versions(distribution: str) -> str:
        queried.append(distribution)
        if distribution not in installed:
            raise metadata.PackageNotFoundError(distribution)
        return installed[distribution]

    def native_import_forbidden(*args, **kwargs):
        raise AssertionError(f"native import attempted: {args} {kwargs}")

    monkeypatch.setattr("importlib.import_module", native_import_forbidden)
    assert validate_laptop_runtime_pins(version_provider=versions) == installed
    assert queried == [
        "mediapipe",
        "opencv-contrib-python",
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    ]

    installed["opencv-python"] = "4.14.0.94"
    with pytest.raises(LaptopRuntimePinError, match="second distribution"):
        validate_laptop_runtime_pins(version_provider=versions)

    installed.pop("opencv-python")
    installed["mediapipe"] = "1.0.1"
    with pytest.raises(LaptopRuntimePinError, match="exact pins"):
        validate_laptop_runtime_pins(version_provider=versions)


def test_runtime_pin_failure_happens_before_model_or_camera_construction(tmp_path: Path) -> None:
    calls: list[str] = []

    def fail_runtime() -> dict[str, str]:
        calls.append("runtime")
        raise LaptopRuntimePinError("wrong version")

    with pytest.raises(LaptopRuntimePinError):
        build_squat_demo(
            SquatDemoConfig(
                model_asset_path=tmp_path / "pose.task",
                voice_enabled=False,
                microphone_enabled=False,
            ),
            environment={},
            dependencies=SquatDemoDependencies(
                validate_runtime=fail_runtime,
                validate_model=lambda _: calls.append("model"),  # type: ignore[arg-type]
                pose_source_factory=lambda _: calls.append("camera"),  # type: ignore[arg-type]
            ),
        )

    assert calls == ["runtime"]


def test_cli_forwards_named_squat_demo_flags_without_opening_hardware(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "run_squat_demo", lambda **kwargs: calls.append(kwargs) or 7)

    result = cli.main(
        [
            "squat-demo",
            "--camera-index",
            "2",
            "--model-path",
            "models/test.task",
            "--voice",
            "marin",
            "--no-preview",
            "--no-voice",
            "--no-mic",
            "--max-frames",
            "12",
        ]
    )

    assert result == 7
    assert calls == [
        {
            "camera_index": 2,
            "model_path": "models/test.task",
            "voice": "marin",
            "no_preview": True,
            "no_voice": True,
            "no_mic": True,
            "max_frames": 12,
        }
    ]


def test_run_squat_demo_redacts_model_validation_details(tmp_path: Path) -> None:
    model_path = tmp_path / "pose.task"
    output = io.StringIO()

    def fail(_: str | Path) -> Path:
        raise RuntimeError("secret provider detail")

    result = run_squat_demo(
        environment={
            "RECOVERYBOX_POSE_MODEL_PATH": str(model_path),
            "RECOVERYBOX_CAMERA_PREVIEW": "false",
        },
        output=output,
        no_voice=True,
        no_mic=True,
        max_frames=1,
        dependencies=SquatDemoDependencies(
            validate_runtime=lambda: {
                "mediapipe": "0.10.35",
                "opencv-contrib-python": "4.14.0.94",
            },
            validate_model=fail,
        ),
    )

    assert result == 2
    assert "secret provider detail" not in output.getvalue()
    assert "RuntimeError" in output.getvalue()


@pytest.mark.parametrize("target_reps", [1, 9, 11])
def test_launcher_rejects_unreviewed_rep_targets(tmp_path: Path, target_reps: int) -> None:
    with pytest.raises(ValueError, match="exactly 10 reps"):
        SquatDemoConfig(model_asset_path=tmp_path / "pose.task", target_reps=target_reps)
