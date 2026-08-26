"""Silent request-gated laptop pose client for a Pi-owned Guardian.

The laptop opens one outbound TCP connection to a literal Tailscale IPv4
address.  A process-local camera worker continuously replaces one raw-frame
slot so camera startup and backend buffering are outside the request path.
Inference and preview still occur only after an authenticated request and only
on a frame whose completion timestamp is strictly newer than that request.
The response contains only a numeric ``SquatAnalysis``; raw images and
landmarks remain process-local.

Native MediaPipe, OpenCV, and camera resources are acquired only by ``run``.
Importing this module and its ordinary tests remains hardware-free.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Condition, Event, Thread, current_thread, main_thread
from time import monotonic, monotonic_ns
from typing import Any, Protocol

from recoverybox.exercise import MediaPipePoseFrame, SquatAnalysis, SquatTracker
from recoverybox.remote_pose import (
    MAX_REMOTE_POSE_EVIDENCE_AGE_MS,
    RemotePosePublisher,
    RemotePoseRequest,
    load_remote_pose_token,
)
from recoverybox.vision import WebcamPoseConfig, WebcamPoseSample
from recoverybox.vision.mediapipe_adapter import pose_frame_from_mediapipe_result

from .doctor import MEDIAPIPE_EXPECTED_VERSION, OPENCV_EXPECTED_VERSION
from .pose_model import DEFAULT_POSE_MODEL_PATH, validate_pose_model
from .squat_launcher import validate_laptop_runtime_pins

DEFAULT_REQUEST_WAIT_SECONDS = 0.1
CAMERA_OPEN_TIMEOUT_SECONDS = 5.0
CAMERA_FRAME_TIMEOUT_SECONDS = 0.3
CAMERA_CLOSE_TIMEOUT_SECONDS = 1.0

# Canonical MediaPipe pose topology.  It is kept local to the capture process:
# only the derived ``SquatAnalysis`` is handed to the network publisher.
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


class _PoseSource(Protocol):
    def open(self) -> object: ...

    def read(self, *, preview_timing: PoseRequestTiming | None = None) -> WebcamPoseSample: ...

    def close(self) -> None: ...


class _Tracker(Protocol):
    @property
    def rep_count(self) -> int: ...

    def update(self, frame: object) -> SquatAnalysis: ...

    def update_missing(self, timestamp_ms: int) -> SquatAnalysis: ...


class _Publisher(Protocol):
    @property
    def connected(self) -> bool: ...

    @property
    def failure_kind(self) -> str | None: ...

    def start(self) -> None: ...

    def wait_for_request(
        self, timeout_seconds: float | None = None
    ) -> RemotePoseRequest | None: ...

    def submit(
        self,
        analysis: SquatAnalysis,
        *,
        request: RemotePoseRequest,
        evidence_age_ms: int,
    ) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PoseClientConfig:
    """Validated settings for one silent laptop client process."""

    peer: str
    token_file: Path
    model_asset_path: Path = DEFAULT_POSE_MODEL_PATH
    camera_index: int = 0
    fixture_paths: tuple[Path, ...] = ()
    preview: bool = True
    authorize: bool = False
    max_requests: int | None = None
    request_wait_seconds: float = DEFAULT_REQUEST_WAIT_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.peer, str) or not self.peer.strip():
            raise ValueError("peer must be a non-blank literal Tailscale IPv4:port")
        if isinstance(self.camera_index, bool) or not isinstance(self.camera_index, int):
            raise TypeError("camera_index must be an integer")
        if self.camera_index < 0:
            raise ValueError("camera_index must be non-negative")
        if type(self.preview) is not bool:
            raise TypeError("preview must be a boolean")
        if type(self.authorize) is not bool:
            raise TypeError("authorize must be a boolean")
        if self.max_requests is not None and (
            isinstance(self.max_requests, bool)
            or not isinstance(self.max_requests, int)
            or self.max_requests <= 0
        ):
            raise ValueError("max_requests must be a positive integer")
        if (
            isinstance(self.request_wait_seconds, bool)
            or not isinstance(self.request_wait_seconds, (int, float))
            or not isfinite(float(self.request_wait_seconds))
            or not 0.0 < float(self.request_wait_seconds) <= 0.5
        ):
            raise ValueError("request_wait_seconds must be between 0 and 0.5 seconds")
        token_file = Path(self.token_file).expanduser()
        model_path = Path(self.model_asset_path).expanduser()
        fixtures = tuple(Path(path).expanduser() for path in self.fixture_paths)
        object.__setattr__(self, "peer", self.peer.strip())
        object.__setattr__(self, "token_file", token_file)
        object.__setattr__(self, "model_asset_path", model_path)
        object.__setattr__(self, "fixture_paths", fixtures)
        object.__setattr__(self, "request_wait_seconds", float(self.request_wait_seconds))


@dataclass(frozen=True, slots=True)
class PoseClientResult:
    """Content-free result suitable for a terminal or deployment health check."""

    requests_processed: int
    assessable_responses: int
    rep_count: int
    source: str
    authorized: bool
    connected: bool
    failure_kind: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "requests_processed": self.requests_processed,
            "assessable_responses": self.assessable_responses,
            "rep_count": self.rep_count,
            "source": self.source,
            "authorized": self.authorized,
            "connected": self.connected,
            "failure_kind": self.failure_kind,
        }


@dataclass(frozen=True, slots=True)
class PoseRequestTiming:
    """Content-free timing context passed into a request-gated capture."""

    request_received_ns: int
    request_interval_ns: int | None = None

    def __post_init__(self) -> None:
        _validate_non_negative_ns(self.request_received_ns, "request_received_ns")
        if self.request_interval_ns is not None:
            _validate_positive_ns(self.request_interval_ns, "request_interval_ns")


@dataclass(frozen=True, slots=True)
class PosePreviewMetrics:
    """Local-only performance values rendered into the debug preview."""

    frame_fps: float | None
    capture_latency_ms: float
    model_fps: float | None
    model_latency_ms: float
    request_fps: float | None
    e2e_latency_ms: float

    def __post_init__(self) -> None:
        for name in ("frame_fps", "model_fps", "request_fps"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        for name in ("capture_latency_ms", "model_latency_ms", "e2e_latency_ms"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")


def pose_preview_lines(
    metrics: PosePreviewMetrics,
    analysis: SquatAnalysis | None = None,
) -> tuple[str, ...]:
    """Format deterministic, content-free debug overlay lines."""

    if not isinstance(metrics, PosePreviewMetrics):
        raise TypeError("metrics must be PosePreviewMetrics")
    if analysis is not None and not isinstance(analysis, SquatAnalysis):
        raise TypeError("analysis must be SquatAnalysis or None")
    frame_fps = "--" if metrics.frame_fps is None else f"{metrics.frame_fps:.1f}"
    model_fps = "--" if metrics.model_fps is None else f"{metrics.model_fps:.1f}"
    request_fps = "--" if metrics.request_fps is None else f"{metrics.request_fps:.1f}"
    status_lines = (
        ()
        if analysis is None
        else (f"Squats: {analysis.rep_count} | Phase: {analysis.phase.value}",)
    )
    return (
        *status_lines,
        f"Frames: {frame_fps} FPS | capture {metrics.capture_latency_ms:.1f} ms",
        f"Pose model: {model_fps} FPS | inference {metrics.model_latency_ms:.1f} ms",
        f"Requests: {request_fps} FPS | E2E ready {metrics.e2e_latency_ms:.1f} ms",
        "q/Esc stop",
    )


@dataclass(frozen=True, slots=True)
class FixturePoseConfig:
    """Explicit process-local image sequence for integration verification."""

    model_asset_path: Path
    image_paths: tuple[Path, ...]
    preview: bool = True
    preview_window_name: str = "RecoveryBox pose client"

    def __post_init__(self) -> None:
        model = Path(self.model_asset_path).expanduser().resolve()
        images = tuple(Path(path).expanduser().resolve() for path in self.image_paths)
        if not images:
            raise ValueError("fixture mode requires at least one explicit image path")
        if type(self.preview) is not bool:
            raise TypeError("preview must be a boolean")
        if not isinstance(self.preview_window_name, str) or not self.preview_window_name.strip():
            raise ValueError("preview_window_name must not be blank")
        object.__setattr__(self, "model_asset_path", model)
        object.__setattr__(self, "image_paths", images)
        object.__setattr__(self, "preview_window_name", self.preview_window_name.strip())


def _validate_non_negative_ns(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_positive_ns(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _read_clock_ns(clock_ns: Callable[[], int]) -> int:
    value = clock_ns()
    _validate_non_negative_ns(value, "monotonic clock result")
    return value


def _elapsed_ns(start_ns: int, end_ns: int) -> int:
    if end_ns < start_ns:
        raise RuntimeError("monotonic clock moved backwards")
    return end_ns - start_ns


def _fps_from_interval_ns(interval_ns: int | None) -> float | None:
    if interval_ns is None or interval_ns <= 0:
        return None
    return 1_000_000_000.0 / interval_ns


def _preview_metrics(
    *,
    capture_started_ns: int,
    capture_completed_ns: int,
    inference_started_ns: int,
    inference_completed_ns: int,
    last_frame_completed_ns: int | None,
    request_timing: PoseRequestTiming | None,
) -> PosePreviewMetrics:
    capture_ns = _elapsed_ns(capture_started_ns, capture_completed_ns)
    model_ns = _elapsed_ns(inference_started_ns, inference_completed_ns)
    request_received_ns = (
        capture_started_ns if request_timing is None else request_timing.request_received_ns
    )
    e2e_ns = _elapsed_ns(request_received_ns, inference_completed_ns)
    frame_interval_ns = (
        None
        if last_frame_completed_ns is None
        else _elapsed_ns(last_frame_completed_ns, capture_completed_ns)
    )
    request_interval_ns = None if request_timing is None else request_timing.request_interval_ns
    return PosePreviewMetrics(
        frame_fps=_fps_from_interval_ns(frame_interval_ns),
        capture_latency_ms=capture_ns / 1_000_000.0,
        model_fps=_fps_from_interval_ns(model_ns),
        model_latency_ms=model_ns / 1_000_000.0,
        request_fps=_fps_from_interval_ns(request_interval_ns),
        e2e_latency_ms=e2e_ns / 1_000_000.0,
    )


def _show_pose_preview(
    cv2: Any,
    camera_frame: Any,
    pose: MediaPipePoseFrame | None,
    metrics: PosePreviewMetrics,
    *,
    analysis: SquatAnalysis | None = None,
    mirror: bool,
    window_name: str,
) -> bool:
    """Render and discard one frame entirely inside the capture boundary."""

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

    if mirror:
        preview = cv2.flip(preview, 1)
    for row, line in enumerate(pose_preview_lines(metrics, analysis)):
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
    cv2.imshow(window_name, preview)
    key_code = cv2.waitKey(1) & 0xFF
    return key_code in (ord("q"), 27)


def _require_main_thread_for_preview(preview: bool) -> None:
    """Keep macOS OpenCV window operations on the process main thread."""

    if preview and current_thread() is not main_thread():
        raise RuntimeError("pose preview must run on the process main thread")


class _CapturedCameraFrame:
    """One process-local raw frame with content-free capture timing."""

    __slots__ = ("completed_ns", "frame", "generation", "interval_ns")

    def __init__(
        self,
        frame: Any,
        *,
        completed_ns: int,
        generation: int,
        interval_ns: int | None,
    ) -> None:
        self.frame = frame
        self.completed_ns = completed_ns
        self.generation = generation
        self.interval_ns = interval_ns


class _LatestCameraFrameWorker:
    """Own a camera on one daemon thread and retain at most its latest frame.

    No raw frame crosses this private boundary except when :meth:`take_after`
    transfers the sole slot to the process-local inference caller.  Native
    camera open, read, and release all execute on the worker thread.  Public
    methods use wall-clock deadlines so an injected or stalled capture clock
    cannot make lifecycle operations wait indefinitely.
    """

    def __init__(
        self,
        cv2: Any,
        config: WebcamPoseConfig,
        *,
        clock_ns: Callable[[], int],
        open_timeout_seconds: float,
        frame_timeout_seconds: float,
        close_timeout_seconds: float,
    ) -> None:
        self._cv2 = cv2
        self._config = config
        self._clock_ns = clock_ns
        self._open_timeout_seconds = _validate_timeout_seconds(
            open_timeout_seconds, "open_timeout_seconds"
        )
        self._frame_timeout_seconds = _validate_timeout_seconds(
            frame_timeout_seconds, "frame_timeout_seconds"
        )
        self._close_timeout_seconds = _validate_timeout_seconds(
            close_timeout_seconds, "close_timeout_seconds"
        )
        self._condition = Condition()
        self._stop = Event()
        self._thread: Thread | None = None
        self._opened = False
        self._failure: str | None = None
        self._release_failure: str | None = None
        self._read_generation = 0
        self._request_barrier_generation = 0
        self._slot: _CapturedCameraFrame | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("camera capture worker has already started")
            thread = Thread(
                target=self._run,
                name="recoverybox-camera-capture",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            deadline = monotonic() + self._open_timeout_seconds
            while not (self._opened and self._slot is not None):
                if self._failure is not None:
                    raise RuntimeError(self._failure)
                if not thread.is_alive():
                    raise RuntimeError("camera capture worker stopped during startup")
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    self._stop.set()
                    self._slot = None
                    self._condition.notify_all()
                    raise TimeoutError("webcam did not open and produce a frame in time")
                self._condition.wait(remaining)

    def take_after(self, request_received_ns: int) -> _CapturedCameraFrame:
        """Consume a post-request frame from a post-claim camera read.

        The generation barrier closes a scheduling race in which native
        ``capture.read`` returns before the request, but the worker does not
        record its completion timestamp until after the request.  A frame must
        both complete after the request timestamp and come from a read begun
        after this method claims the barrier under the worker condition.
        """

        _validate_non_negative_ns(request_received_ns, "request_received_ns")
        deadline = monotonic() + self._frame_timeout_seconds
        with self._condition:
            barrier_generation = self._read_generation
            self._request_barrier_generation = barrier_generation
            self._condition.notify_all()
            while True:
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("webcam did not return a post-request frame in time")
                slot = self._slot
                if slot is not None:
                    if (
                        slot.generation > barrier_generation
                        and slot.completed_ns > request_received_ns
                    ):
                        self._slot = None
                        return slot
                    # Do not retain or infer on a pre-request frame.  There is
                    # never a queue: the worker will replace this empty slot.
                    self._slot = None
                if self._failure is not None:
                    raise RuntimeError(self._failure)
                thread = self._thread
                if thread is None or not thread.is_alive():
                    raise RuntimeError("camera capture worker is not running")
                self._condition.wait(remaining)

    def close(self) -> None:
        with self._condition:
            thread = self._thread
            self._stop.set()
            self._slot = None
            self._condition.notify_all()
        if thread is None:
            if self._release_failure is not None:
                raise RuntimeError(self._release_failure)
            return
        thread.join(self._close_timeout_seconds)
        with self._condition:
            self._slot = None
            if thread.is_alive():
                raise TimeoutError("camera capture worker did not stop in time")
            self._thread = None
            if self._release_failure is not None:
                raise RuntimeError(self._release_failure)

    def _run(self) -> None:
        capture: Any | None = None
        last_completed_ns: int | None = None
        try:
            capture = self._cv2.VideoCapture(self._config.camera_index)
            if capture is None or not capture.isOpened():
                self._fail("configured webcam could not be opened")
                return
            self._configure_capture(capture)
            with self._condition:
                self._opened = True
                self._condition.notify_all()

            while not self._stop.is_set():
                with self._condition:
                    if self._stop.is_set():
                        break
                    self._read_generation += 1
                    generation = self._read_generation
                    self._condition.notify_all()
                success, frame = capture.read()
                if self._stop.is_set():
                    frame = None
                    break
                if not success or frame is None:
                    self._fail("webcam capture failed before a fresh frame was available")
                    return
                completed_ns = _read_clock_ns(self._clock_ns)
                if last_completed_ns is not None and completed_ns < last_completed_ns:
                    frame = None
                    self._fail("camera capture monotonic clock moved backwards")
                    return
                interval_ns = (
                    None
                    if last_completed_ns is None or completed_ns == last_completed_ns
                    else completed_ns - last_completed_ns
                )
                last_completed_ns = completed_ns
                captured = _CapturedCameraFrame(
                    frame,
                    completed_ns=completed_ns,
                    generation=generation,
                    interval_ns=interval_ns,
                )
                frame = None
                with self._condition:
                    if self._stop.is_set():
                        break
                    # Replacing this reference discards the previous frame;
                    # there is deliberately no list, deque, callback, or log.
                    self._slot = captured
                    self._condition.notify_all()
        except BaseException:
            self._fail("camera capture worker failed")
        finally:
            if capture is not None:
                try:
                    capture.release()
                except BaseException:
                    with self._condition:
                        self._release_failure = "camera capture release failed"
                        self._condition.notify_all()
            with self._condition:
                self._condition.notify_all()

    def _configure_capture(self, capture: Any) -> None:
        buffer_property = getattr(self._cv2, "CAP_PROP_BUFFERSIZE", None)
        if buffer_property is not None:
            capture.set(buffer_property, 1)
        if self._config.capture_width is not None:
            capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, self._config.capture_width)
        if self._config.capture_height is not None:
            capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self._config.capture_height)

    def _fail(self, message: str) -> None:
        with self._condition:
            if not self._stop.is_set() and self._failure is None:
                self._failure = message
            self._slot = None
            self._condition.notify_all()


def _validate_timeout_seconds(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field_name} must be a finite positive number")
    return float(value)


class RequestFreshWebcamPoseSource:
    """Infer only on a one-slot camera frame completed after the Pi request.

    The worker warms one persistent ``VideoCapture`` before the network client
    connects.  It continuously replaces a single process-local slot, avoiding
    both camera-open latency and backend queues.  :meth:`read` discards any
    slot that completed at or before the authenticated request and waits a
    bounded time for a strictly newer frame.  MediaPipe inference and preview
    stay on the caller thread; the worker alone owns camera open/read/release.
    """

    def __init__(
        self,
        config: WebcamPoseConfig,
        *,
        runtime_loader: Callable[[], tuple[Any, Any]] | None = None,
        clock_ns: Callable[[], int] = monotonic_ns,
        open_timeout_seconds: float = CAMERA_OPEN_TIMEOUT_SECONDS,
        frame_timeout_seconds: float = CAMERA_FRAME_TIMEOUT_SECONDS,
        close_timeout_seconds: float = CAMERA_CLOSE_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(config, WebcamPoseConfig):
            raise TypeError("config must be a WebcamPoseConfig")
        self.config = config
        self._runtime_loader = runtime_loader or _load_fixture_runtime
        self._clock_ns = clock_ns
        self._open_timeout_seconds = _validate_timeout_seconds(
            open_timeout_seconds, "open_timeout_seconds"
        )
        self._frame_timeout_seconds = _validate_timeout_seconds(
            frame_timeout_seconds, "frame_timeout_seconds"
        )
        self._close_timeout_seconds = _validate_timeout_seconds(
            close_timeout_seconds, "close_timeout_seconds"
        )
        self._cv2: Any | None = None
        self._mediapipe: Any | None = None
        self._landmarker: Any | None = None
        self._capture_worker: _LatestCameraFrameWorker | None = None
        self._last_timestamp_ms: int | None = None
        self._preview_visible = False

    def open(self) -> RequestFreshWebcamPoseSource:
        if self._landmarker is not None or self._capture_worker is not None:
            raise RuntimeError("request-fresh webcam pose source is already open")
        if not self.config.model_asset_path.is_file():
            raise FileNotFoundError("pose model is not installed")
        cv2, mediapipe = self._runtime_loader()
        options = mediapipe.tasks.vision.PoseLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(
                model_asset_path=str(self.config.model_asset_path)
            ),
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.minimum_pose_detection_confidence,
            min_pose_presence_confidence=self.config.minimum_pose_presence_confidence,
            min_tracking_confidence=self.config.minimum_tracking_confidence,
            output_segmentation_masks=False,
        )
        landmarker = mediapipe.tasks.vision.PoseLandmarker.create_from_options(options)
        worker = _LatestCameraFrameWorker(
            cv2,
            self.config,
            clock_ns=self._clock_ns,
            open_timeout_seconds=self._open_timeout_seconds,
            frame_timeout_seconds=self._frame_timeout_seconds,
            close_timeout_seconds=self._close_timeout_seconds,
        )
        try:
            worker.start()
        except BaseException:
            try:
                worker.close()
            except BaseException:
                pass
            landmarker.close()
            raise
        self._cv2 = cv2
        self._mediapipe = mediapipe
        self._landmarker = landmarker
        self._capture_worker = worker
        self._last_timestamp_ms = None
        self._preview_visible = False
        return self

    def read(self, *, preview_timing: PoseRequestTiming | None = None) -> WebcamPoseSample:
        sample, _ = self._read(preview_timing=preview_timing, tracker=None)
        return sample

    def read_analyzed(
        self,
        tracker: _Tracker,
        *,
        preview_timing: PoseRequestTiming | None = None,
    ) -> tuple[WebcamPoseSample, SquatAnalysis]:
        """Track and preview one frame before its process-local image is discarded."""

        sample, analysis = self._read(preview_timing=preview_timing, tracker=tracker)
        assert analysis is not None
        return sample, analysis

    def _read(
        self,
        *,
        preview_timing: PoseRequestTiming | None,
        tracker: _Tracker | None,
    ) -> tuple[WebcamPoseSample, SquatAnalysis | None]:
        if preview_timing is not None and not isinstance(preview_timing, PoseRequestTiming):
            raise TypeError("preview_timing must be PoseRequestTiming")
        _require_main_thread_for_preview(self.config.preview)
        cv2 = self._cv2
        mediapipe = self._mediapipe
        landmarker = self._landmarker
        worker = self._capture_worker
        if cv2 is None or mediapipe is None or landmarker is None or worker is None:
            raise RuntimeError("request-fresh webcam pose source is not open")
        capture_started_ns = (
            _read_clock_ns(self._clock_ns)
            if preview_timing is None
            else preview_timing.request_received_ns
        )
        captured = worker.take_after(capture_started_ns)
        camera_frame = captured.frame
        capture_completed_ns = captured.completed_ns
        timestamp_ms = self._next_timestamp_ms(capture_completed_ns)
        try:
            image_height, image_width = camera_frame.shape[:2]
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("webcam returned invalid frame dimensions") from exc
        rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        media_image = mediapipe.Image(
            image_format=mediapipe.ImageFormat.SRGB,
            data=rgb_frame,
        )
        inference_started_ns = _read_clock_ns(self._clock_ns)
        result = landmarker.detect_for_video(media_image, timestamp_ms)
        inference_completed_ns = _read_clock_ns(self._clock_ns)
        pose = pose_frame_from_mediapipe_result(
            result,
            timestamp_ms=timestamp_ms,
            image_width=image_width,
            image_height=image_height,
        )
        metrics = _preview_metrics(
            capture_started_ns=capture_started_ns,
            capture_completed_ns=capture_completed_ns,
            inference_started_ns=inference_started_ns,
            inference_completed_ns=inference_completed_ns,
            last_frame_completed_ns=(
                None
                if captured.interval_ns is None
                else capture_completed_ns - captured.interval_ns
            ),
            request_timing=preview_timing,
        )
        analysis = (
            None
            if tracker is None
            else (
                tracker.update(pose) if pose is not None else tracker.update_missing(timestamp_ms)
            )
        )
        quit_requested = False
        if self.config.preview:
            quit_requested = _show_pose_preview(
                cv2,
                camera_frame,
                pose,
                metrics,
                analysis=analysis,
                mirror=self.config.mirror_preview,
                window_name=self.config.preview_window_name,
            )
            self._preview_visible = True
        return (
            WebcamPoseSample(
                timestamp_ms=timestamp_ms,
                pose=pose,
                quit_requested=quit_requested,
            ),
            analysis,
        )

    def close(self) -> None:
        worker = self._capture_worker
        landmarker, self._landmarker = self._landmarker, None
        cv2, self._cv2 = self._cv2, None
        self._mediapipe = None
        close_error: BaseException | None = None
        if worker is not None:
            try:
                worker.close()
            except BaseException as exc:
                close_error = exc
            else:
                self._capture_worker = None
        try:
            if landmarker is not None:
                landmarker.close()
        finally:
            if cv2 is not None and self._preview_visible:
                cv2.destroyWindow(self.config.preview_window_name)
            self._preview_visible = False
        if close_error is not None:
            raise close_error

    def _next_timestamp_ms(self, raw_ns: int) -> int:
        timestamp_ms = raw_ns // 1_000_000
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms


class MediaPipeFixturePoseSource:
    """Cycle local fixture images through the real MediaPipe pose landmarker.

    No camera is constructed.  Each image is loaded only after the caller has
    received a verified Pi request, processed into numeric landmarks, and then
    discarded.  Tests replace the runtime loader; ordinary pytest never imports
    the native modules.
    """

    def __init__(
        self,
        config: FixturePoseConfig,
        *,
        runtime_loader: Callable[[], tuple[Any, Any]] | None = None,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if not isinstance(config, FixturePoseConfig):
            raise TypeError("config must be a FixturePoseConfig")
        self.config = config
        self._runtime_loader = runtime_loader or _load_fixture_runtime
        self._clock_ns = clock_ns
        self._cv2: Any | None = None
        self._mediapipe: Any | None = None
        self._landmarker: Any | None = None
        self._next_image = 0
        self._last_timestamp_ms: int | None = None
        self._last_frame_completed_ns: int | None = None
        self._preview_visible = False

    def open(self) -> MediaPipeFixturePoseSource:
        if self._landmarker is not None:
            raise RuntimeError("fixture pose source is already open")
        if not self.config.model_asset_path.is_file():
            raise FileNotFoundError("pose model is not installed")
        if not all(path.is_file() for path in self.config.image_paths):
            raise FileNotFoundError("one or more fixture image paths do not exist")
        cv2, mediapipe = self._runtime_loader()
        options = mediapipe.tasks.vision.PoseLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(
                model_asset_path=str(self.config.model_asset_path)
            ),
            running_mode=mediapipe.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        landmarker = mediapipe.tasks.vision.PoseLandmarker.create_from_options(options)
        self._cv2 = cv2
        self._mediapipe = mediapipe
        self._landmarker = landmarker
        self._next_image = 0
        self._last_timestamp_ms = None
        self._last_frame_completed_ns = None
        self._preview_visible = False
        return self

    def read(self, *, preview_timing: PoseRequestTiming | None = None) -> WebcamPoseSample:
        sample, _ = self._read(preview_timing=preview_timing, tracker=None)
        return sample

    def read_analyzed(
        self,
        tracker: _Tracker,
        *,
        preview_timing: PoseRequestTiming | None = None,
    ) -> tuple[WebcamPoseSample, SquatAnalysis]:
        """Track and preview one fixture before its process-local image is discarded."""

        sample, analysis = self._read(preview_timing=preview_timing, tracker=tracker)
        assert analysis is not None
        return sample, analysis

    def _read(
        self,
        *,
        preview_timing: PoseRequestTiming | None,
        tracker: _Tracker | None,
    ) -> tuple[WebcamPoseSample, SquatAnalysis | None]:
        if preview_timing is not None and not isinstance(preview_timing, PoseRequestTiming):
            raise TypeError("preview_timing must be PoseRequestTiming")
        _require_main_thread_for_preview(self.config.preview)
        cv2 = self._cv2
        mediapipe = self._mediapipe
        landmarker = self._landmarker
        if cv2 is None or mediapipe is None or landmarker is None:
            raise RuntimeError("fixture pose source is not open")
        path = self.config.image_paths[self._next_image]
        self._next_image = (self._next_image + 1) % len(self.config.image_paths)
        capture_started_ns = _read_clock_ns(self._clock_ns)
        timestamp_ms = self._next_timestamp_ms(capture_started_ns)
        camera_frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if camera_frame is None:
            raise RuntimeError("fixture image could not be decoded")
        capture_completed_ns = _read_clock_ns(self._clock_ns)
        try:
            image_height, image_width = camera_frame.shape[:2]
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("fixture image has invalid dimensions") from exc
        rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        media_image = mediapipe.Image(
            image_format=mediapipe.ImageFormat.SRGB,
            data=rgb_frame,
        )
        inference_started_ns = _read_clock_ns(self._clock_ns)
        result = landmarker.detect(media_image)
        inference_completed_ns = _read_clock_ns(self._clock_ns)
        pose = pose_frame_from_mediapipe_result(
            result,
            timestamp_ms=timestamp_ms,
            image_width=image_width,
            image_height=image_height,
        )
        metrics = _preview_metrics(
            capture_started_ns=capture_started_ns,
            capture_completed_ns=capture_completed_ns,
            inference_started_ns=inference_started_ns,
            inference_completed_ns=inference_completed_ns,
            last_frame_completed_ns=self._last_frame_completed_ns,
            request_timing=preview_timing,
        )
        self._last_frame_completed_ns = capture_completed_ns
        analysis = (
            None
            if tracker is None
            else (
                tracker.update(pose) if pose is not None else tracker.update_missing(timestamp_ms)
            )
        )
        quit_requested = False
        if self.config.preview:
            quit_requested = _show_pose_preview(
                cv2,
                camera_frame,
                pose,
                metrics,
                analysis=analysis,
                mirror=False,
                window_name=self.config.preview_window_name,
            )
            self._preview_visible = True
        return (
            WebcamPoseSample(
                timestamp_ms=timestamp_ms,
                pose=pose,
                quit_requested=quit_requested,
            ),
            analysis,
        )

    def close(self) -> None:
        landmarker, self._landmarker = self._landmarker, None
        cv2, self._cv2 = self._cv2, None
        self._mediapipe = None
        if landmarker is not None:
            landmarker.close()
        if cv2 is not None and self._preview_visible:
            cv2.destroyWindow(self.config.preview_window_name)
        self._preview_visible = False

    def _next_timestamp_ms(self, raw_ns: int) -> int:
        timestamp_ms = raw_ns // 1_000_000
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms


PublisherFactory = Callable[..., _Publisher]
SourceFactory = Callable[[PoseClientConfig, Path], _PoseSource]


@dataclass(frozen=True, slots=True)
class PoseClientDependencies:
    """Replaceable native and network boundaries for hardware-free tests."""

    validate_runtime: Callable[[], Mapping[str, str]] = validate_laptop_runtime_pins
    validate_model: Callable[[str | Path], Path] = validate_pose_model
    token_loader: Callable[[str | Path], bytes] = load_remote_pose_token
    publisher_factory: PublisherFactory = RemotePosePublisher
    source_factory: SourceFactory | None = None
    tracker_factory: Callable[[], _Tracker] = SquatTracker
    clock_ns: Callable[[], int] = monotonic_ns


def _load_fixture_runtime() -> tuple[Any, Any]:
    cv2 = importlib.import_module("cv2")
    mediapipe = importlib.import_module("mediapipe")
    return cv2, mediapipe


def _default_source_factory(config: PoseClientConfig, model_path: Path) -> _PoseSource:
    if config.fixture_paths:
        return MediaPipeFixturePoseSource(
            FixturePoseConfig(
                model_asset_path=model_path,
                image_paths=config.fixture_paths,
                preview=config.preview,
            )
        )
    return RequestFreshWebcamPoseSource(
        WebcamPoseConfig(
            model_asset_path=model_path,
            camera_index=config.camera_index,
            preview=config.preview,
        )
    )


def _capture_to_submit_age_ms(clock_ns: Callable[[], int], timestamp_ms: int) -> int:
    try:
        now_ns = clock_ns()
    except Exception:
        return MAX_REMOTE_POSE_EVIDENCE_AGE_MS
    if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
        return MAX_REMOTE_POSE_EVIDENCE_AGE_MS
    now_ceiling_ms = (now_ns + 999_999) // 1_000_000
    if now_ceiling_ms < timestamp_ms:
        return MAX_REMOTE_POSE_EVIDENCE_AGE_MS
    return min(now_ceiling_ms - timestamp_ms, MAX_REMOTE_POSE_EVIDENCE_AGE_MS)


def run_pose_client(
    config: PoseClientConfig,
    *,
    dependencies: PoseClientDependencies | None = None,
) -> PoseClientResult:
    """Run until stopped, epoch changes, or the optional request bound is met."""

    if not isinstance(config, PoseClientConfig):
        raise TypeError("config must be a PoseClientConfig")
    selected = dependencies or PoseClientDependencies()
    selected.validate_runtime()
    model_path = selected.validate_model(config.model_asset_path)
    token = selected.token_loader(config.token_file)
    publisher = selected.publisher_factory(
        config.peer,
        token,
        authorize_initial_epoch=config.authorize,
    )
    token = b""
    source_factory = selected.source_factory or _default_source_factory
    source = source_factory(config, model_path)
    tracker = selected.tracker_factory()
    analyzed_reader = getattr(source, "read_analyzed", None)
    processed = 0
    assessable = 0
    connected = False
    opened = False
    last_request_received_ns: int | None = None
    try:
        source.open()
        opened = True
        publisher.start()
        while config.max_requests is None or processed < config.max_requests:
            connected = connected or publisher.connected
            if publisher.failure_kind == "ServiceEpochChanged":
                break
            request = publisher.wait_for_request(config.request_wait_seconds)
            if request is None:
                continue
            request_received_ns = _read_clock_ns(selected.clock_ns)
            if (
                last_request_received_ns is not None
                and request_received_ns < last_request_received_ns
            ):
                raise RuntimeError("monotonic clock moved backwards")
            request_interval_ns = (
                None
                if last_request_received_ns is None
                or request_received_ns == last_request_received_ns
                else request_received_ns - last_request_received_ns
            )
            last_request_received_ns = request_received_ns
            # This is the inference boundary: the camera worker has at most one
            # process-local frame, and read accepts only a frame completed after
            # this authenticated request was claimed exactly once.
            preview_timing = PoseRequestTiming(
                request_received_ns=request_received_ns,
                request_interval_ns=request_interval_ns,
            )
            if callable(analyzed_reader):
                # The repository-owned sources update the deterministic tracker
                # before discarding the raw frame, so the local overlay describes
                # the exact frame beneath its bones. Third-party/test sources keep
                # the original numeric-only read contract and are tracked here.
                sample, analysis = analyzed_reader(
                    tracker,
                    preview_timing=preview_timing,
                )
            else:
                sample = source.read(preview_timing=preview_timing)
                analysis = (
                    tracker.update(sample.pose)
                    if sample.pose is not None
                    else tracker.update_missing(sample.timestamp_ms)
                )
            publisher.submit(
                analysis,
                request=request,
                evidence_age_ms=_capture_to_submit_age_ms(
                    selected.clock_ns,
                    sample.timestamp_ms,
                ),
            )
            processed += 1
            if analysis.assessable:
                assessable += 1
            if sample.quit_requested:
                break
    finally:
        try:
            publisher.close()
        finally:
            if opened:
                source.close()
    return PoseClientResult(
        requests_processed=processed,
        assessable_responses=assessable,
        rep_count=tracker.rep_count,
        source="fixtures" if config.fixture_paths else "camera",
        authorized=config.authorize,
        connected=connected,
        failure_kind=publisher.failure_kind,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recoverybox-pose-client",
        description=(
            "Silent request-gated Mac pose client. It opens no microphone or speaker. "
            "Use --authorize only for an intentional fresh Pi service-epoch session."
        ),
    )
    parser.add_argument("--peer", required=True, help="literal Tailscale IPv4:port")
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_POSE_MODEL_PATH)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="disable the local pose-bones and performance preview window",
    )
    parser.add_argument(
        "--fixture",
        dest="fixtures",
        action="append",
        type=Path,
        default=[],
        help="explicit local image; repeat to define the request-by-request cycle",
    )
    parser.add_argument(
        "--authorize",
        action="store_true",
        help="send one explicit signed RESUME for this newly observed Pi service epoch",
    )
    parser.add_argument("--max-requests", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_pose_client(
            PoseClientConfig(
                peer=args.peer,
                token_file=args.token_file,
                model_asset_path=args.model,
                camera_index=args.camera_index,
                fixture_paths=tuple(args.fixtures),
                preview=args.preview,
                authorize=args.authorize,
                max_requests=args.max_requests,
            )
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "failure_kind": type(exc).__name__,
                    "voice": False,
                    "microphone": False,
                    "speaker": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "ok": result.failure_kind is None,
                **result.as_dict(),
                "runtime": {
                    "mediapipe": MEDIAPIPE_EXPECTED_VERSION,
                    "opencv": OPENCV_EXPECTED_VERSION,
                },
                "voice": False,
                "microphone": False,
                "speaker": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.failure_kind is None else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FixturePoseConfig",
    "MediaPipeFixturePoseSource",
    "PoseClientConfig",
    "PoseClientDependencies",
    "PoseClientResult",
    "PosePreviewMetrics",
    "PoseRequestTiming",
    "RequestFreshWebcamPoseSource",
    "main",
    "pose_preview_lines",
    "run_pose_client",
]
