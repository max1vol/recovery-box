from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recoverybox.exercise import (
    MediaPipePoseFrame,
    NormalizedLandmark,
    SquatAssessmentIssue,
    SquatTracker,
)
from recoverybox.vision import (
    MEDIAPIPE_POSE_LANDMARK_COUNT,
    MediaPipeResultError,
    VisionDependencyError,
    WebcamPoseConfig,
    WebcamPoseSource,
    WebcamReadError,
    WebcamUnavailableError,
    pose_frame_from_mediapipe_result,
    webcam_output_field_names,
)
from recoverybox.vision import webcam as webcam_module


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


def fake_landmarks(count: int = MEDIAPIPE_POSE_LANDMARK_COUNT) -> list[FakeLandmark]:
    return [
        FakeLandmark(
            x=(index - 1) / 100.0,
            y=(index + 1) / 100.0,
            z=-index / 200.0,
            visibility=0.9,
            presence=0.8,
        )
        for index in range(count)
    ]


def fake_result(*, pose_count: int = 1, landmark_count: int = 33) -> SimpleNamespace:
    return SimpleNamespace(
        pose_landmarks=[fake_landmarks(landmark_count) for _ in range(pose_count)]
    )


def adapt(result: SimpleNamespace, *, timestamp_ms: int = 10) -> MediaPipePoseFrame | None:
    return pose_frame_from_mediapipe_result(
        result,
        timestamp_ms=timestamp_ms,
        image_width=640,
        image_height=480,
    )


def test_adapter_preserves_all_33_landmarks_in_mediapipe_index_order() -> None:
    frame = adapt(fake_result(), timestamp_ms=1_234)

    assert isinstance(frame, MediaPipePoseFrame)
    assert frame.timestamp_ms == 1_234
    assert frame.image_width == 640
    assert frame.image_height == 480
    assert type(frame.landmarks) is tuple
    assert len(frame.landmarks) == 33
    assert all(isinstance(landmark, NormalizedLandmark) for landmark in frame.landmarks)
    assert frame.landmarks[0] == NormalizedLandmark(
        x=-0.01,
        y=0.01,
        z=0.0,
        visibility=0.9,
        presence=0.8,
    )
    assert frame.landmarks[32].x == pytest.approx(0.31)


def test_adapter_returns_none_when_no_pose_is_detected() -> None:
    assert adapt(fake_result(pose_count=0)) is None


@pytest.mark.parametrize(
    ("image_width", "image_height", "message"),
    [
        (0, 480, "image_width must be a positive integer"),
        (640, -1, "image_height must be a positive integer"),
        (True, 480, "image_width must be a positive integer"),
    ],
)
def test_adapter_rejects_invalid_image_dimensions(
    image_width: object,
    image_height: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pose_frame_from_mediapipe_result(
            fake_result(),
            timestamp_ms=10,
            image_width=image_width,  # type: ignore[arg-type]
            image_height=image_height,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("landmark_count", [0, 32, 34])
def test_adapter_rejects_any_noncanonical_landmark_width(landmark_count: int) -> None:
    with pytest.raises(MediaPipeResultError, match="expected 33 pose landmarks"):
        adapt(fake_result(landmark_count=landmark_count))


def test_adapter_rejects_ambiguous_multiple_people() -> None:
    with pytest.raises(MediaPipeResultError, match="at most one pose"):
        adapt(fake_result(pose_count=2))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("x", float("nan"), "x must be finite"),
        ("y", True, "y must be a real number"),
        ("visibility", 1.01, "visibility must be between 0 and 1"),
        ("presence", -0.01, "presence must be between 0 and 1"),
    ],
)
def test_adapter_rejects_invalid_landmark_numbers(
    field_name: str,
    value: object,
    message: str,
) -> None:
    result = fake_result()
    setattr(result.pose_landmarks[0][7], field_name, value)

    with pytest.raises(MediaPipeResultError, match=message):
        adapt(result)


def test_adapter_does_not_clamp_extrapolated_normalized_coordinates() -> None:
    result = fake_result()
    result.pose_landmarks[0][0].x = -0.2
    result.pose_landmarks[0][0].y = 1.1

    frame = adapt(result)

    assert frame is not None
    assert (frame.landmarks[0].x, frame.landmarks[0].y) == (-0.2, 1.1)


def test_source_construction_and_missing_model_do_not_import_native_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr(
        webcam_module.importlib,
        "import_module",
        lambda name: imported.append(name),
    )
    source = WebcamPoseSource(
        WebcamPoseConfig(model_asset_path=tmp_path / "not-present.task", preview=False)
    )

    assert imported == []
    with pytest.raises(FileNotFoundError, match=r"provide a local \.task model"):
        source.open()
    assert imported == []


def test_open_reports_missing_optional_native_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "pose_landmarker.task"
    model_path.write_bytes(b"test placeholder")

    def missing_import(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(webcam_module.importlib, "import_module", missing_import)

    with pytest.raises(VisionDependencyError, match="opencv-contrib-python"):
        WebcamPoseSource(WebcamPoseConfig(model_path, preview=False)).open()


def test_webcam_source_returns_only_numeric_pose_data_and_uses_strict_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "pose_landmarker.task"
    model_path.write_bytes(b"test placeholder")
    runtime = FakeVisionRuntime(results=[fake_result(), fake_result()])
    monkeypatch.setattr(webcam_module, "_load_runtime_modules", runtime.modules)
    times = iter((1_000_000_000, 1_000_000_000))

    source = WebcamPoseSource(
        WebcamPoseConfig(model_path, preview=False, capture_width=640, capture_height=480),
        _clock_ns=lambda: next(times),
    )
    with source:
        first = source.read()
        second = source.read()

    assert first.timestamp_ms == 1_000
    assert first.pose is not None and len(first.pose.landmarks) == 33
    assert (first.pose.image_width, first.pose.image_height) == (640, 480)
    assert second.timestamp_ms == 1_001
    assert second.pose is not None and second.pose.timestamp_ms == 1_001
    assert webcam_output_field_names() == {"timestamp_ms", "pose", "quit_requested"}
    assert not hasattr(first, "frame")
    assert not hasattr(first, "image")
    assert not hasattr(source, "last_frame")
    assert runtime.detector.timestamps == [1_000, 1_001]
    assert runtime.capture.settings == [(3, 640), (4, 480)]
    assert runtime.options.num_poses == 1
    assert runtime.options.output_segmentation_masks is False
    assert runtime.options.base_options.model_asset_path == str(model_path)
    assert runtime.detector.closed
    assert runtime.capture.released


def test_no_pose_sample_drives_explicit_fail_closed_tracker_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "pose_landmarker.task"
    model_path.write_bytes(b"test placeholder")
    runtime = FakeVisionRuntime(results=[fake_result(pose_count=0)])
    monkeypatch.setattr(webcam_module, "_load_runtime_modules", runtime.modules)

    with WebcamPoseSource(
        WebcamPoseConfig(model_path, preview=False),
        _clock_ns=lambda: 1_500_000_000,
    ) as source:
        sample = source.read()

    assert sample.pose is None
    analysis = SquatTracker().update_missing(sample.timestamp_ms)
    assert not analysis.assessable
    assert analysis.issues == (SquatAssessmentIssue.NO_POSE,)


def test_preview_stays_internal_and_q_requests_clean_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "pose_landmarker.task"
    model_path.write_bytes(b"test placeholder")
    runtime = FakeVisionRuntime(results=[fake_result()], key_code=ord("q"))
    monkeypatch.setattr(webcam_module, "_load_runtime_modules", runtime.modules)

    with WebcamPoseSource(
        WebcamPoseConfig(model_path, preview=True, mirror_preview=True),
        _clock_ns=lambda: 2_000_000_000,
        _performance_clock_ns=iter((100_000_000, 105_000_000, 106_000_000, 126_000_000)).__next__,
    ) as source:
        sample = source.read(preview_lines=("Squats: 1", "Stand tall"))
        assert sample.quit_requested
        assert source.quit_requested
        with pytest.raises(RuntimeError, match="requested shutdown"):
            source.read()

    assert runtime.cv2.imshow_calls == 1
    assert runtime.cv2.flip_calls == 1
    assert runtime.cv2.line_calls > 0
    assert runtime.cv2.circle_calls > 0
    assert runtime.cv2.text_lines == [
        "Squats: 1",
        "Stand tall",
        "Frames: 0.0 FPS | capture 5.0 ms",
        "Pose model: 50.0 FPS | inference 20.0 ms",
    ]
    assert runtime.cv2.operations.index("flip") < runtime.cv2.operations.index("text")
    assert runtime.cv2.destroyed_windows == ["RecoveryBox squat tracker"]


def test_preview_performance_metrics_are_finite_deterministic_and_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "pose_landmarker.task"
    model_path.write_bytes(b"test placeholder")
    runtime = FakeVisionRuntime(results=[fake_result(), fake_result(), fake_result()])
    monkeypatch.setattr(webcam_module, "_load_runtime_modules", runtime.modules)
    timestamps = iter((1_000_000_000, 1_001_000_000, 1_002_000_000))
    performance_times = iter(
        (
            # First frame: a zero-duration model sample remains finite.
            100_000_000,
            105_000_000,
            106_000_000,
            106_000_000,
            # Second frame: 25 FPS between completed frames, 50 FPS model.
            120_000_000,
            125_000_000,
            126_000_000,
            146_000_000,
            # Reopened source: frame-rate history must start cleanly again.
            200_000_000,
            205_000_000,
            206_000_000,
            226_000_000,
        )
    )
    source = WebcamPoseSource(
        WebcamPoseConfig(model_path, preview=True),
        _clock_ns=timestamps.__next__,
        _performance_clock_ns=performance_times.__next__,
    )

    source.open()
    first = source.read()
    second = source.read()
    source.close()
    source.open()
    reopened = source.read()
    source.close()

    assert runtime.cv2.text_lines == [
        "Frames: 0.0 FPS | capture 5.0 ms",
        "Pose model: 0.0 FPS | inference 0.0 ms",
        "Frames: 25.0 FPS | capture 5.0 ms",
        "Pose model: 50.0 FPS | inference 20.0 ms",
        "Frames: 0.0 FPS | capture 5.0 ms",
        "Pose model: 50.0 FPS | inference 20.0 ms",
    ]
    assert all(
        "nan" not in line.lower() and "inf fps" not in line.lower() and "inf ms" not in line.lower()
        for line in runtime.cv2.text_lines
    )
    assert webcam_output_field_names() == {"timestamp_ms", "pose", "quit_requested"}
    for sample in (first, second, reopened):
        assert set(sample.__dataclass_fields__) == {
            "timestamp_ms",
            "pose",
            "quit_requested",
        }
        assert not hasattr(sample, "frame")
        assert not hasattr(sample, "image")
    assert not hasattr(source, "last_frame")


def test_camera_open_and_read_failures_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "pose_landmarker.task"
    model_path.write_bytes(b"test placeholder")
    unavailable = FakeVisionRuntime(results=[], camera_open=False)
    monkeypatch.setattr(webcam_module, "_load_runtime_modules", unavailable.modules)
    with pytest.raises(WebcamUnavailableError, match="camera index 0"):
        WebcamPoseSource(WebcamPoseConfig(model_path, preview=False)).open()
    assert unavailable.capture.released

    unreadable = FakeVisionRuntime(results=[], frames=[])
    monkeypatch.setattr(webcam_module, "_load_runtime_modules", unreadable.modules)
    with WebcamPoseSource(WebcamPoseConfig(model_path, preview=False)) as source:
        with pytest.raises(WebcamReadError, match="did not return a frame"):
            source.read()


class FakeFrame:
    shape = (480, 640, 3)

    def copy(self) -> FakeFrame:
        return FakeFrame()


class FakeCapture:
    def __init__(self, *, frames: list[FakeFrame], camera_open: bool) -> None:
        self.frames = frames
        self.camera_open = camera_open
        self.released = False
        self.settings: list[tuple[int, int]] = []

    def isOpened(self) -> bool:
        return self.camera_open

    def read(self) -> tuple[bool, FakeFrame | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def set(self, prop: int, value: int) -> None:
        self.settings.append((prop, value))

    def release(self) -> None:
        self.released = True


class FakeDetector:
    def __init__(self, results: list[SimpleNamespace]) -> None:
        self.results = results
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, image: object, timestamp_ms: int) -> SimpleNamespace:
        self.timestamps.append(timestamp_ms)
        return self.results.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    COLOR_BGR2RGB = 5
    LINE_AA = 16
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self, capture: FakeCapture, *, key_code: int) -> None:
        self.capture = capture
        self.key_code = key_code
        self.imshow_calls = 0
        self.flip_calls = 0
        self.line_calls = 0
        self.circle_calls = 0
        self.text_lines: list[str] = []
        self.destroyed_windows: list[str] = []
        self.operations: list[str] = []

    def VideoCapture(self, camera_index: int) -> FakeCapture:
        return self.capture

    @staticmethod
    def cvtColor(frame: FakeFrame, conversion: int) -> FakeFrame:
        return frame

    def line(self, *args: object) -> None:
        self.line_calls += 1
        self.operations.append("line")

    def circle(self, *args: object) -> None:
        self.circle_calls += 1
        self.operations.append("circle")

    def putText(
        self,
        frame: FakeFrame,
        line: str,
        *args: object,
    ) -> None:
        self.text_lines.append(line)
        self.operations.append("text")

    def flip(self, frame: FakeFrame, direction: int) -> FakeFrame:
        self.flip_calls += 1
        self.operations.append("flip")
        return frame

    def imshow(self, window_name: str, frame: FakeFrame) -> None:
        self.imshow_calls += 1

    def waitKey(self, delay_ms: int) -> int:
        return self.key_code

    def destroyWindow(self, window_name: str) -> None:
        self.destroyed_windows.append(window_name)


class OptionBag:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)


class FakeVisionRuntime:
    def __init__(
        self,
        *,
        results: list[SimpleNamespace],
        frames: list[FakeFrame] | None = None,
        camera_open: bool = True,
        key_code: int = -1,
    ) -> None:
        default_frames = [FakeFrame() for _ in results]
        self.capture = FakeCapture(
            frames=default_frames if frames is None else frames,
            camera_open=camera_open,
        )
        self.detector = FakeDetector(results)
        self.cv2 = FakeCv2(self.capture, key_code=key_code)
        self.options: OptionBag | None = None

    def modules(self) -> tuple[FakeCv2, SimpleNamespace]:
        owner = self

        class PoseLandmarkerOptions(OptionBag):
            def __init__(self, **values: object) -> None:
                super().__init__(**values)
                owner.options = self

        class PoseLandmarker:
            @staticmethod
            def create_from_options(options: OptionBag) -> FakeDetector:
                return owner.detector

        media_pipe = SimpleNamespace(
            tasks=SimpleNamespace(
                BaseOptions=OptionBag,
                vision=SimpleNamespace(
                    PoseLandmarkerOptions=PoseLandmarkerOptions,
                    PoseLandmarker=PoseLandmarker,
                    RunningMode=SimpleNamespace(VIDEO="video"),
                ),
            ),
            Image=OptionBag,
            ImageFormat=SimpleNamespace(SRGB="srgb"),
        )
        return self.cv2, media_pipe
