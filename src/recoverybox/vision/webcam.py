"""Laptop webcam capture with a raw-frame-tight MediaPipe boundary.

``WebcamPoseSource`` is the only object in the application that sees OpenCV
frames.  Its public output contains timestamps and numeric pose landmarks
only.  Camera frames are consumed, optionally previewed, and discarded during
each call to :meth:`WebcamPoseSource.read`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
from time import monotonic_ns
from typing import Any

from recoverybox.exercise import MediaPipePoseFrame

from .mediapipe_adapter import pose_frame_from_mediapipe_result

# Canonical MediaPipe pose topology, expressed only as numeric landmark indices.
_POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


class VisionDependencyError(RuntimeError):
    """Raised when optional laptop-vision packages are unavailable."""


class WebcamUnavailableError(RuntimeError):
    """Raised when the configured webcam cannot be opened."""


class WebcamReadError(RuntimeError):
    """Raised when an opened webcam stops returning frames."""


@dataclass(frozen=True, slots=True)
class WebcamPoseConfig:
    """Local-only configuration for the MediaPipe laptop prototype."""

    model_asset_path: Path
    camera_index: int = 0
    preview: bool = True
    mirror_preview: bool = True
    preview_window_name: str = "RecoveryBox squat tracker"
    quit_key: str = "q"
    capture_width: int | None = None
    capture_height: int | None = None
    minimum_pose_detection_confidence: float = 0.5
    minimum_pose_presence_confidence: float = 0.5
    minimum_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        path = Path(self.model_asset_path).expanduser().resolve()
        if isinstance(self.camera_index, bool) or not isinstance(self.camera_index, int):
            raise TypeError("camera_index must be an integer")
        if self.camera_index < 0:
            raise ValueError("camera_index must be non-negative")
        if type(self.preview) is not bool or type(self.mirror_preview) is not bool:
            raise TypeError("preview options must be booleans")
        if not self.preview_window_name.strip():
            raise ValueError("preview_window_name must not be blank")
        if len(self.quit_key) != 1 or ord(self.quit_key) > 255:
            raise ValueError("quit_key must be one single-byte character")
        _validate_optional_dimension(self.capture_width, "capture_width")
        _validate_optional_dimension(self.capture_height, "capture_height")
        for name in (
            "minimum_pose_detection_confidence",
            "minimum_pose_presence_confidence",
            "minimum_tracking_confidence",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            converted = float(value)
            if not isfinite(converted) or not 0.0 <= converted <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
            object.__setattr__(self, name, converted)
        object.__setattr__(self, "model_asset_path", path)


@dataclass(frozen=True, slots=True)
class WebcamPoseSample:
    """Raw-media-free result of processing one captured webcam frame.

    ``pose is None`` is explicit missing evidence, not a frame to skip.  The
    exercise loop must pass ``timestamp_ms`` to ``SquatTracker.update_missing``
    so disappearance immediately invalidates any partial repetition.
    """

    timestamp_ms: int
    pose: MediaPipePoseFrame | None
    quit_requested: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_ms, bool)
            or not isinstance(self.timestamp_ms, int)
            or self.timestamp_ms < 0
        ):
            raise ValueError("timestamp_ms must be a non-negative integer")
        if self.pose is not None and self.pose.timestamp_ms != self.timestamp_ms:
            raise ValueError("pose timestamp must match its webcam sample")
        if type(self.quit_requested) is not bool:
            raise TypeError("quit_requested must be a boolean")


def webcam_output_field_names() -> frozenset[str]:
    """Expose the auditable public output boundary without constructing a camera."""

    return frozenset(field.name for field in fields(WebcamPoseSample))


class WebcamPoseSource:
    """Own one OpenCV camera and emit only numeric MediaPipe pose samples.

    Imports and native resources are acquired only by :meth:`open`, never at
    import time.  The caller must provide an already-downloaded MediaPipe
    ``.task`` model; this class performs no network access.
    """

    def __init__(
        self,
        config: WebcamPoseConfig,
        *,
        _clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if not isinstance(config, WebcamPoseConfig):
            raise TypeError("config must be a WebcamPoseConfig")
        self.config = config
        self._clock_ns = _clock_ns
        self._cv2: Any | None = None
        self._mediapipe: Any | None = None
        self._capture: Any | None = None
        self._landmarker: Any | None = None
        self._last_timestamp_ms: int | None = None
        self._quit_requested = False
        self._preview_visible = False

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._landmarker is not None

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def open(self) -> WebcamPoseSource:
        if self.is_open:
            raise RuntimeError("webcam pose source is already open")
        if not self.config.model_asset_path.is_file():
            raise FileNotFoundError(
                "MediaPipe pose model not found at "
                f"{self.config.model_asset_path}; provide a local .task model"
            )

        cv2, mediapipe = _load_runtime_modules()
        capture = cv2.VideoCapture(self.config.camera_index)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise WebcamUnavailableError(
                f"could not open webcam at camera index {self.config.camera_index}"
            )

        try:
            if self.config.capture_width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture_width)
            if self.config.capture_height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture_height)

            options = mediapipe.tasks.vision.PoseLandmarkerOptions(
                base_options=mediapipe.tasks.BaseOptions(
                    model_asset_path=str(self.config.model_asset_path)
                ),
                running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=(self.config.minimum_pose_detection_confidence),
                min_pose_presence_confidence=(self.config.minimum_pose_presence_confidence),
                min_tracking_confidence=self.config.minimum_tracking_confidence,
                output_segmentation_masks=False,
            )
            landmarker = mediapipe.tasks.vision.PoseLandmarker.create_from_options(options)
        except BaseException:
            capture.release()
            raise

        self._cv2 = cv2
        self._mediapipe = mediapipe
        self._capture = capture
        self._landmarker = landmarker
        self._last_timestamp_ms = None
        self._quit_requested = False
        self._preview_visible = False
        return self

    def read(self, *, preview_lines: Sequence[str] = ()) -> WebcamPoseSample:
        """Capture, infer, optionally preview, and discard one raw frame."""

        if not self.is_open:
            raise RuntimeError("webcam pose source is not open")
        if self._quit_requested:
            raise RuntimeError("preview has already requested shutdown")
        if isinstance(preview_lines, (bytes, bytearray, memoryview, str)):
            raise TypeError("preview_lines must be a sequence of complete lines")
        lines = tuple(str(line) for line in preview_lines)

        cv2 = self._cv2
        mediapipe = self._mediapipe
        capture = self._capture
        landmarker = self._landmarker
        assert cv2 is not None and mediapipe is not None
        assert capture is not None and landmarker is not None

        success, camera_frame = capture.read()
        if not success or camera_frame is None:
            raise WebcamReadError("webcam did not return a frame")
        try:
            image_height, image_width = camera_frame.shape[:2]
        except (AttributeError, TypeError, ValueError) as exc:
            raise WebcamReadError("webcam returned a frame without valid dimensions") from exc
        if (
            isinstance(image_width, bool)
            or not isinstance(image_width, int)
            or image_width <= 0
            or isinstance(image_height, bool)
            or not isinstance(image_height, int)
            or image_height <= 0
        ):
            raise WebcamReadError("webcam returned a frame without valid dimensions")

        timestamp_ms = self._next_timestamp_ms()
        rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        media_image = mediapipe.Image(
            image_format=mediapipe.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = landmarker.detect_for_video(media_image, timestamp_ms)
        pose = pose_frame_from_mediapipe_result(
            result,
            timestamp_ms=timestamp_ms,
            image_width=image_width,
            image_height=image_height,
        )

        quit_requested = False
        if self.config.preview:
            quit_requested = self._preview(camera_frame, pose, lines)
            self._quit_requested = quit_requested

        # camera_frame, rgb_frame, and media_image remain local variables and
        # are not stored on the source or included in the returned sample.
        return WebcamPoseSample(
            timestamp_ms=timestamp_ms,
            pose=pose,
            quit_requested=quit_requested,
        )

    def __iter__(self) -> Iterator[WebcamPoseSample]:
        while not self._quit_requested:
            sample = self.read()
            yield sample
            if sample.quit_requested:
                break

    def close(self) -> None:
        """Release the detector, camera, and preview window idempotently."""

        landmarker, self._landmarker = self._landmarker, None
        capture, self._capture = self._capture, None
        cv2, self._cv2 = self._cv2, None
        self._mediapipe = None

        first_error: BaseException | None = None
        if landmarker is not None:
            try:
                landmarker.close()
            except BaseException as exc:  # pragma: no cover - native cleanup failure
                first_error = exc
        if capture is not None:
            try:
                capture.release()
            except BaseException as exc:  # pragma: no cover - native cleanup failure
                first_error = first_error or exc
        if cv2 is not None and self._preview_visible:
            try:
                cv2.destroyWindow(self.config.preview_window_name)
            except BaseException as exc:  # pragma: no cover - backend-specific cleanup
                first_error = first_error or exc

        self._preview_visible = False
        if first_error is not None:
            raise first_error

    def __enter__(self) -> WebcamPoseSource:
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _next_timestamp_ms(self) -> int:
        raw_ns = self._clock_ns()
        if isinstance(raw_ns, bool) or not isinstance(raw_ns, int) or raw_ns < 0:
            raise RuntimeError("monotonic clock returned an invalid value")
        candidate = raw_ns // 1_000_000
        if self._last_timestamp_ms is not None and candidate <= self._last_timestamp_ms:
            candidate = self._last_timestamp_ms + 1
        self._last_timestamp_ms = candidate
        return candidate

    def _preview(
        self,
        camera_frame: Any,
        pose: MediaPipePoseFrame | None,
        lines: tuple[str, ...],
    ) -> bool:
        cv2 = self._cv2
        assert cv2 is not None
        preview = camera_frame.copy()

        if pose is not None:
            height, width = preview.shape[:2]
            points: list[tuple[int, int] | None] = []
            for landmark in pose.landmarks:
                x = round(landmark.x * width)
                y = round(landmark.y * height)
                point = (x, y) if 0 <= x < width and 0 <= y < height else None
                points.append(point)

            for start_index, end_index in _POSE_CONNECTIONS:
                start = points[start_index]
                end = points[end_index]
                if start is not None and end is not None:
                    cv2.line(preview, start, end, (40, 220, 120), 2, cv2.LINE_AA)
            for point, landmark in zip(points, pose.landmarks, strict=True):
                if point is not None and landmark.visibility >= 0.5:
                    cv2.circle(preview, point, 3, (255, 190, 40), -1, cv2.LINE_AA)

        # Mirror the camera and skeleton together, then draw status text so the
        # user-facing labels remain readable rather than being mirrored.
        if self.config.mirror_preview:
            preview = cv2.flip(preview, 1)

        for row, line in enumerate(lines):
            cv2.putText(
                preview,
                line,
                (16, 32 + row * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.config.preview_window_name, preview)
        self._preview_visible = True
        key_code = cv2.waitKey(1) & 0xFF
        return key_code in (ord(self.config.quit_key), 27)


def _validate_optional_dimension(value: int | None, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer when provided")


def _load_runtime_modules() -> tuple[Any, Any]:
    """Load native optional dependencies only when a camera is opened."""

    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise VisionDependencyError(
            "webcam support requires the pinned optional package 'opencv-contrib-python==4.14.0.94'"
        ) from exc
    try:
        mediapipe = importlib.import_module("mediapipe")
    except ImportError as exc:
        raise VisionDependencyError(
            "pose inference requires the optional package 'mediapipe'"
        ) from exc
    return cv2, mediapipe
