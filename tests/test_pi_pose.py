from __future__ import annotations

import ctypes
import io
import json
import struct
from pathlib import Path

import pytest

from recoverybox.device.pi_pose import (
    MOVENET_INPUT_BYTES,
    MOVENET_KEYPOINT_COUNT,
    MOVENET_OUTPUT_VALUES,
    FfmpegRawCamera,
    MoveNetKeypoint,
    MoveNetPoint,
    PiPoseConfig,
    PiPoseConfigurationError,
    PiPoseObservation,
    PiPoseRuntimeError,
    PiPoseSource,
    TFLiteMoveNet,
    main,
    movenet_to_mediapipe_frame,
    run_movenet_self_check,
    run_pi_pose_check,
    validate_movenet_model,
)
from recoverybox.exercise import MediaPipePoseLandmark, SquatAssessmentIssue


def _standing_points(*, score: float = 0.99) -> tuple[MoveNetPoint, ...]:
    coordinates = [(0.15, 0.50)] * MOVENET_KEYPOINT_COUNT
    placements = {
        MoveNetKeypoint.LEFT_SHOULDER: (0.30, 0.35),
        MoveNetKeypoint.RIGHT_SHOULDER: (0.30, 0.65),
        MoveNetKeypoint.LEFT_ELBOW: (0.30, 0.20),
        MoveNetKeypoint.RIGHT_ELBOW: (0.30, 0.80),
        MoveNetKeypoint.LEFT_WRIST: (0.30, 0.05),
        MoveNetKeypoint.RIGHT_WRIST: (0.30, 0.95),
        MoveNetKeypoint.LEFT_HIP: (0.55, 0.43),
        MoveNetKeypoint.RIGHT_HIP: (0.55, 0.57),
        MoveNetKeypoint.LEFT_KNEE: (0.75, 0.43),
        MoveNetKeypoint.RIGHT_KNEE: (0.75, 0.57),
        MoveNetKeypoint.LEFT_ANKLE: (0.95, 0.43),
        MoveNetKeypoint.RIGHT_ANKLE: (0.95, 0.57),
    }
    for keypoint, position in placements.items():
        coordinates[keypoint] = position
    return tuple(MoveNetPoint(y=y, x=x, score=score) for y, x in coordinates)


def test_config_loads_closed_environment() -> None:
    config = PiPoseConfig.from_environment(
        {
            "RECOVERYBOX_CAMERA_DEVICE": "/dev/video4",
            "RECOVERYBOX_MOVENET_MODEL_PATH": "/srv/model.tflite",
            "RECOVERYBOX_TFLITE_LIBRARY_PATH": "/srv/libtensorflow-lite.so",
            "RECOVERYBOX_CAMERA_WIDTH": "800",
            "RECOVERYBOX_CAMERA_HEIGHT": "600",
            "RECOVERYBOX_CAMERA_FPS": "4",
            "RECOVERYBOX_POSE_FRAME_TIMEOUT_SECONDS": "0.25",
            "RECOVERYBOX_TFLITE_THREADS": "3",
            "RECOVERYBOX_FFMPEG_BINARY": "/opt/bin/ffmpeg",
        }
    )

    assert config.camera_device == Path("/dev/video4")
    assert config.model_path == Path("/srv/model.tflite")
    assert config.tflite_library_path == Path("/srv/libtensorflow-lite.so")
    assert (config.capture_width, config.capture_height) == (800, 600)
    assert config.frames_per_second == 4
    assert config.frame_timeout_seconds == 0.25
    assert config.inference_threads == 3
    assert config.ffmpeg_binary == "/opt/bin/ffmpeg"


@pytest.mark.parametrize("timeout", [0.0, -0.1, 0.5001, float("inf")])
def test_config_rejects_timeout_outside_guardian_bound(timeout: float) -> None:
    with pytest.raises(PiPoseConfigurationError, match="frame_timeout_seconds"):
        PiPoseConfig(frame_timeout_seconds=timeout)


def test_model_validator_rejects_symlink_before_hashing(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.write_bytes(b"not the model")
    link = tmp_path / "model-link"
    link.symlink_to(target)

    with pytest.raises(PiPoseConfigurationError, match="non-symlink"):
        validate_movenet_model(link)


def test_model_validator_rejects_wrong_size_without_exposing_digest(tmp_path: Path) -> None:
    path = tmp_path / "model.tflite"
    path.write_bytes(b"wrong")

    with pytest.raises(PiPoseConfigurationError, match="size") as error:
        validate_movenet_model(path)

    assert "wrong" not in str(error.value)


def test_movenet_mapping_populates_only_supported_landmarks() -> None:
    points = _standing_points(score=0.8)

    frame = movenet_to_mediapipe_frame(points, timestamp_ms=123)

    left_knee = frame.landmark(MediaPipePoseLandmark.LEFT_KNEE)
    assert (left_knee.x, left_knee.y) == (0.43, 0.75)
    assert (left_knee.visibility, left_knee.presence) == (0.8, 0.8)
    unsupported = frame.landmark(MediaPipePoseLandmark.LEFT_FOOT_INDEX)
    assert (unsupported.visibility, unsupported.presence) == (0.0, 0.0)
    assert len(frame.landmarks) == 33


def test_movenet_mapping_rejects_wrong_count() -> None:
    with pytest.raises(PiPoseRuntimeError, match="17"):
        movenet_to_mediapipe_frame(_standing_points()[:-1], timestamp_ms=1)


def test_movenet_point_clamps_confidence_but_rejects_nonfinite() -> None:
    assert MoveNetPoint(y=0.2, x=0.3, score=1.2).score == 1.0
    assert MoveNetPoint(y=0.2, x=0.3, score=-0.2).score == 0.0
    with pytest.raises(ValueError, match="finite"):
        MoveNetPoint(y=float("nan"), x=0.3, score=0.5)


class _FakeNativeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *arguments):
        return self.callback(*arguments)


class _FakeTFLiteLibrary:
    INPUT = 101
    OUTPUT = 102

    def __init__(self, values: tuple[float, ...]) -> None:
        self.deleted: list[tuple[str, int]] = []
        self.copied_input_size: int | None = None
        self.TfLiteModelCreateFromFile = _FakeNativeFunction(lambda path: 11)
        self.TfLiteModelDelete = _FakeNativeFunction(
            lambda pointer: self.deleted.append(("model", pointer))
        )
        self.TfLiteInterpreterOptionsCreate = _FakeNativeFunction(lambda: 12)
        self.TfLiteInterpreterOptionsDelete = _FakeNativeFunction(
            lambda pointer: self.deleted.append(("options", pointer))
        )
        self.TfLiteInterpreterOptionsSetNumThreads = _FakeNativeFunction(
            lambda options, threads: None
        )
        self.TfLiteInterpreterCreate = _FakeNativeFunction(lambda model, options: 13)
        self.TfLiteInterpreterDelete = _FakeNativeFunction(
            lambda pointer: self.deleted.append(("interpreter", pointer))
        )
        self.TfLiteInterpreterAllocateTensors = _FakeNativeFunction(lambda interpreter: 0)
        self.TfLiteInterpreterGetInputTensorCount = _FakeNativeFunction(lambda interpreter: 1)
        self.TfLiteInterpreterGetOutputTensorCount = _FakeNativeFunction(lambda interpreter: 1)
        self.TfLiteInterpreterGetInputTensor = _FakeNativeFunction(
            lambda interpreter, index: self.INPUT
        )
        self.TfLiteInterpreterGetOutputTensor = _FakeNativeFunction(
            lambda interpreter, index: self.OUTPUT
        )
        self.TfLiteTensorType = _FakeNativeFunction(lambda tensor: 3 if tensor == self.INPUT else 1)
        self.TfLiteTensorNumDims = _FakeNativeFunction(lambda tensor: 4)
        input_shape = (1, 192, 192, 3)
        output_shape = (1, 1, 17, 3)
        self.TfLiteTensorDim = _FakeNativeFunction(
            lambda tensor, index: (input_shape if tensor == self.INPUT else output_shape)[index]
        )
        self.TfLiteTensorByteSize = _FakeNativeFunction(
            lambda tensor: (
                MOVENET_INPUT_BYTES if tensor == self.INPUT else MOVENET_OUTPUT_VALUES * 4
            )
        )

        def copy_from(tensor, pointer, size):
            self.copied_input_size = size
            return 0

        packed = struct.pack(f"={MOVENET_OUTPUT_VALUES}f", *values)

        def copy_to(tensor, pointer, size):
            ctypes.memmove(pointer, packed, size)
            return 0

        self.TfLiteTensorCopyFromBuffer = _FakeNativeFunction(copy_from)
        self.TfLiteTensorCopyToBuffer = _FakeNativeFunction(copy_to)
        self.TfLiteInterpreterInvoke = _FakeNativeFunction(lambda interpreter: 0)


def test_tflite_c_api_infers_and_releases_all_handles(tmp_path: Path) -> None:
    values = tuple(
        component for point in _standing_points() for component in (point.y, point.x, point.score)
    )
    library = _FakeTFLiteLibrary(values)
    model = tmp_path / "model.tflite"
    model.write_bytes(b"injected")
    estimator = TFLiteMoveNet(
        model_path=model,
        library_path="/runtime/libtensorflow-lite.so",
        library_loader=lambda path: library,
        model_validator=lambda path: Path(path),
    )

    points = estimator.infer(bytes(MOVENET_INPUT_BYTES))
    estimator.close()

    assert len(points) == 17
    assert points[MoveNetKeypoint.LEFT_HIP].x == pytest.approx(0.43)
    assert library.copied_input_size == MOVENET_INPUT_BYTES
    assert library.deleted == [
        ("interpreter", 13),
        ("options", 12),
        ("model", 11),
    ]


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.BytesIO()
        self.terminated = False
        self.killed = False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0 if self.returncode is None else self.returncode


def test_ffmpeg_camera_uses_pipe_only_and_disables_audio() -> None:
    process = _FakeProcess()
    invocation: dict[str, object] = {}

    def factory(command, **options):
        invocation["command"] = command
        invocation["options"] = options
        return process

    camera = FfmpegRawCamera(
        PiPoseConfig(),
        process_factory=factory,
        frame_reader=lambda stream, size, timeout: bytes(size),
    )
    camera.open()
    frame = camera.read_frame()
    camera.close()

    command = invocation["command"]
    assert isinstance(command, list)
    assert "-an" in command
    assert command[-1] == "pipe:1"
    assert not any(value.endswith((".jpg", ".jpeg", ".png", ".raw")) for value in command)
    assert invocation["options"]["shell"] is False  # type: ignore[index]
    assert frame == bytes(MOVENET_INPUT_BYTES)
    assert process.terminated


class _FakeCamera:
    def __init__(self, frames: list[bytes | None]) -> None:
        self.frames = frames
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read_frame(self) -> bytes | None:
        return self.frames.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeEstimator:
    def __init__(self, points: tuple[MoveNetPoint, ...]) -> None:
        self.points = points
        self.inputs: list[bytes] = []
        self.closed = False

    def infer(self, rgb_frame: bytes) -> tuple[MoveNetPoint, ...]:
        self.inputs.append(rgb_frame)
        return self.points

    def close(self) -> None:
        self.closed = True


def test_pose_source_returns_only_sanitized_analysis_and_drops_frame_reference() -> None:
    raw = b"private-camera-frame" + bytes(MOVENET_INPUT_BYTES - 20)
    camera = _FakeCamera([raw])
    estimator = _FakeEstimator(_standing_points())
    ticks = iter((1.0, 1.01, 1.02))
    source = PiPoseSource(
        camera=camera,
        estimator=estimator,
        clock=lambda: next(ticks),
    )

    source.open()
    observation = source.read()
    source.close()

    assert observation.analysis.assessable
    assert observation.frame_received
    assert not hasattr(observation, "frame")
    assert raw not in repr(observation).encode()
    assert estimator.inputs == [raw]
    assert camera.closed and estimator.closed


def test_pose_source_timeout_is_immediately_nonassessable() -> None:
    camera = _FakeCamera([None])
    estimator = _FakeEstimator(_standing_points())
    source = PiPoseSource(camera=camera, estimator=estimator, clock=lambda: 2.0)

    source.open()
    observation = source.read()
    source.close()

    assert not observation.analysis.assessable
    assert observation.analysis.issues == (SquatAssessmentIssue.CAMERA_TIMEOUT,)
    assert observation.inference_ms is None
    assert not observation.frame_received
    assert estimator.inputs == []


class _FakeCheckSource:
    def __init__(self, observations: list[PiPoseObservation]) -> None:
        self.observations = observations
        self.closed = False

    def open(self) -> None:
        return

    def read(self) -> PiPoseObservation:
        return self.observations.pop(0)

    def close(self) -> None:
        self.closed = True


def test_bounded_check_reports_numeric_only_status() -> None:
    camera = _FakeCamera([bytes(MOVENET_INPUT_BYTES)])
    estimator = _FakeEstimator(_standing_points())
    source = PiPoseSource(camera=camera, estimator=estimator, clock=lambda: 3.0)
    source.open()
    observation = source.read()
    source.close()
    fake = _FakeCheckSource([observation])

    result = run_pi_pose_check(
        PiPoseConfig(),
        max_frames=1,
        source_factory=lambda config: fake,  # type: ignore[arg-type]
    )

    assert result == {
        "service": "recoverybox-pi-pose-check/v1",
        "frames": 1,
        "assessable": 1,
        "timeouts": 0,
        "inference_ms_max": 0.0,
        "raw_frames_persisted": 0,
        "audio": "disabled",
    }
    assert fake.closed
    assert "private" not in json.dumps(result)


def test_model_self_check_uses_only_synthetic_bytes() -> None:
    estimator = _FakeEstimator(_standing_points())
    ticks = iter((1.0, 1.01, 2.0, 2.02))

    result = run_movenet_self_check(
        PiPoseConfig(),
        iterations=2,
        estimator_factory=lambda **options: estimator,  # type: ignore[arg-type]
        clock=lambda: next(ticks),
    )

    assert result["first_inference_ms"] == pytest.approx(10.0)
    assert result["warm_inference_ms_max"] == pytest.approx(20.0)
    assert result["camera_used"] is False
    assert result["raw_frames_persisted"] == 0
    assert estimator.inputs == [bytes(MOVENET_INPUT_BYTES)] * 2
    assert estimator.closed


def test_main_scrubs_runtime_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "recoverybox.device.pi_pose.run_pi_pose_check",
        lambda config, max_frames: (_ for _ in ()).throw(PiPoseRuntimeError("secret path")),
    )

    assert main(["--max-frames", "1"]) == 1
    status = json.loads(capsys.readouterr().out)
    assert status["failure"] == "PiPoseRuntimeError"
    assert "secret path" not in json.dumps(status)
    assert status["audio"] == "disabled"
