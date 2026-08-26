from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread, current_thread, main_thread
from time import monotonic, monotonic_ns
from types import SimpleNamespace

import pytest

from recoverybox.exercise import (
    MediaPipePoseFrame,
    NormalizedLandmark,
    SquatAnalysis,
    SquatPhase,
)
from recoverybox.laptop.pose_client import (
    FixturePoseConfig,
    MediaPipeFixturePoseSource,
    PoseClientConfig,
    PoseClientDependencies,
    PosePreviewMetrics,
    PoseRequestTiming,
    RequestFreshWebcamPoseSource,
    _parser,
    _show_pose_preview,
    pose_preview_lines,
    run_pose_client,
)
from recoverybox.remote_pose import RemotePoseRequest
from recoverybox.vision import WebcamPoseConfig

SERVICE_EPOCH = "a" * 64
SERVER_NONCE = "b" * 64


def analysis(
    timestamp_ms: int,
    *,
    phase: SquatPhase = SquatPhase.STANDING,
    rep_count: int = 0,
) -> SquatAnalysis:
    return SquatAnalysis(
        timestamp_ms=timestamp_ms,
        assessable=True,
        phase=phase,
        rep_count=rep_count,
        events=(),
        issues=(),
        confidence=0.9,
        knee_angle_degrees=170.0,
        arms_in_t=True,
    )


@dataclass(frozen=True)
class FakeSample:
    timestamp_ms: int
    pose: object | None
    quit_requested: bool = False


class PreviewFrame:
    shape = (480, 640, 3)

    def copy(self) -> PreviewFrame:
        return self


class PreviewCapture:
    def __init__(self, frame: PreviewFrame) -> None:
        self.frame = frame
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, property_id, value) -> None:
        del property_id, value

    def read(self):
        return True, self.frame

    def release(self) -> None:
        self.released = True


class PreviewCv2:
    IMREAD_COLOR = 1
    COLOR_BGR2RGB = 2
    CAP_PROP_BUFFERSIZE = 3
    LINE_AA = 4
    FONT_HERSHEY_SIMPLEX = 5

    def __init__(self, *, key_code: int) -> None:
        self.key_code = key_code
        self.frame = PreviewFrame()
        self.capture = PreviewCapture(self.frame)
        self.line_calls = 0
        self.circle_calls = 0
        self.flip_calls = 0
        self.text_lines: list[str] = []
        self.imshow_calls = 0
        self.imshow_thread_ids: list[int | None] = []
        self.wait_key_calls = 0
        self.wait_key_thread_ids: list[int | None] = []
        self.destroyed_windows: list[str] = []

    def VideoCapture(self, camera_index: int) -> PreviewCapture:
        del camera_index
        return self.capture

    def imread(self, path: str, mode: int) -> PreviewFrame:
        del path, mode
        return self.frame

    def cvtColor(self, frame: PreviewFrame, conversion: int) -> PreviewFrame:
        del conversion
        return frame

    def line(self, *args: object) -> None:
        del args
        self.line_calls += 1

    def circle(self, *args: object) -> None:
        del args
        self.circle_calls += 1

    def flip(self, frame: PreviewFrame, direction: int) -> PreviewFrame:
        del direction
        self.flip_calls += 1
        return frame

    def putText(self, frame: PreviewFrame, line: str, *args: object) -> None:
        del frame, args
        self.text_lines.append(line)

    def imshow(self, window_name: str, frame: PreviewFrame) -> None:
        del window_name, frame
        self.imshow_calls += 1
        self.imshow_thread_ids.append(current_thread().ident)

    def waitKey(self, delay_ms: int) -> int:
        del delay_ms
        self.wait_key_calls += 1
        self.wait_key_thread_ids.append(current_thread().ident)
        return self.key_code

    def destroyWindow(self, window_name: str) -> None:
        self.destroyed_windows.append(window_name)


def raw_pose_landmarks() -> list[SimpleNamespace]:
    return [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9, presence=0.9) for _ in range(33)]


def numeric_pose(timestamp_ms: int) -> MediaPipePoseFrame:
    landmark = NormalizedLandmark(
        x=0.5,
        y=0.5,
        z=0.0,
        visibility=0.9,
        presence=0.9,
    )
    return MediaPipePoseFrame(
        timestamp_ms=timestamp_ms,
        image_width=640,
        image_height=480,
        landmarks=(landmark,) * 33,
    )


def fake_mediapipe(landmarker: object) -> SimpleNamespace:
    return SimpleNamespace(
        Image=lambda **kwargs: kwargs,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: kwargs,
            vision=SimpleNamespace(
                RunningMode=SimpleNamespace(IMAGE="image", VIDEO="video"),
                PoseLandmarkerOptions=lambda **kwargs: kwargs,
                PoseLandmarker=SimpleNamespace(create_from_options=lambda options: landmarker),
            ),
        ),
    )


class FakeSource:
    def __init__(self, samples: list[FakeSample], order: list[str]) -> None:
        self.samples = deque(samples)
        self.order = order
        self.closed = False
        self.last_timestamp_ms = 0
        self.preview_timings: list[PoseRequestTiming | None] = []

    def open(self) -> FakeSource:
        self.order.append("source_open")
        return self

    def read(self, *, preview_timing=None):
        self.order.append("source_read")
        self.preview_timings.append(preview_timing)
        sample = self.samples.popleft()
        self.last_timestamp_ms = sample.timestamp_ms
        return sample

    def close(self) -> None:
        self.order.append("source_close")
        self.closed = True


class FakeAnalyzedSource(FakeSource):
    def __init__(self, samples: list[FakeSample], order: list[str]) -> None:
        super().__init__(samples, order)
        self.rendered_analyses: list[SquatAnalysis] = []

    def read(self, *, preview_timing=None):
        del preview_timing
        raise AssertionError("run_pose_client should use the analyzed preview path")

    def read_analyzed(self, tracker, *, preview_timing=None):
        self.order.append("source_read")
        self.preview_timings.append(preview_timing)
        sample = self.samples.popleft()
        self.last_timestamp_ms = sample.timestamp_ms
        current = (
            tracker.update(sample.pose)
            if sample.pose is not None
            else tracker.update_missing(sample.timestamp_ms)
        )
        self.rendered_analyses.append(current)
        return sample, current


class FakeTracker:
    def __init__(self, analyses: list[SquatAnalysis], order: list[str]) -> None:
        self.analyses = deque(analyses)
        self.order = order
        self._rep_count = 0

    @property
    def rep_count(self) -> int:
        return self._rep_count

    def update(self, frame: object) -> SquatAnalysis:
        assert frame is not None
        self.order.append("track")
        return self.analyses.popleft()

    def update_missing(self, timestamp_ms: int) -> SquatAnalysis:
        self.order.append("track_missing")
        return self.analyses.popleft()


class FakePublisher:
    def __init__(self, requests: list[RemotePoseRequest], order: list[str]) -> None:
        self.requests = deque(requests)
        self.order = order
        self.connected = False
        self.failure_kind: str | None = None
        self.submissions: list[tuple[SquatAnalysis, RemotePoseRequest, int]] = []
        self.closed = False

    def start(self) -> None:
        self.order.append("publisher_start")
        self.connected = True

    def wait_for_request(self, timeout_seconds=None) -> RemotePoseRequest | None:
        del timeout_seconds
        self.order.append("wait_for_request")
        return self.requests.popleft() if self.requests else None

    def submit(
        self,
        submitted: SquatAnalysis,
        *,
        request: RemotePoseRequest,
        evidence_age_ms: int,
    ) -> None:
        self.order.append("submit")
        self.submissions.append((submitted, request, evidence_age_ms))

    def close(self) -> None:
        self.order.append("publisher_close")
        self.closed = True
        self.connected = False


def pose_request(sequence: int) -> RemotePoseRequest:
    return RemotePoseRequest(
        session_id="0" * 32,
        service_epoch=SERVICE_EPOCH,
        server_nonce=SERVER_NONCE,
        request_sequence=sequence,
        request_nonce=f"{sequence:064x}",
    )


def test_pose_client_acquires_each_frame_only_after_authenticated_request(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    model = tmp_path / "pose.task"
    token_file = tmp_path / "pose.token"
    source = FakeAnalyzedSource(
        [FakeSample(100, object()), FakeSample(133, object())],
        order,
    )
    tracker = FakeTracker([analysis(100), analysis(133)], order)
    publisher = FakePublisher([pose_request(1), pose_request(2)], order)
    factory_calls: list[tuple[str, bytes, bool]] = []

    def publisher_factory(peer: str, token: bytes, *, authorize_initial_epoch: bool):
        factory_calls.append((peer, token, authorize_initial_epoch))
        return publisher

    result = run_pose_client(
        PoseClientConfig(
            peer="100.106.237.106:45873",
            token_file=token_file,
            model_asset_path=model,
            authorize=True,
            max_requests=2,
        ),
        dependencies=PoseClientDependencies(
            validate_runtime=lambda: {
                "mediapipe": "0.10.35",
                "opencv-contrib-python": "4.14.0.94",
            },
            validate_model=lambda path: Path(path),
            token_loader=lambda path: b"t" * 32,
            publisher_factory=publisher_factory,
            source_factory=lambda config, path: source,
            tracker_factory=lambda: tracker,
            clock_ns=lambda: source.last_timestamp_ms * 1_000_000,
        ),
    )

    assert factory_calls == [("100.106.237.106:45873", b"t" * 32, True)]
    assert order == [
        "source_open",
        "publisher_start",
        "wait_for_request",
        "source_read",
        "track",
        "submit",
        "wait_for_request",
        "source_read",
        "track",
        "submit",
        "publisher_close",
        "source_close",
    ]
    assert [item[1].request_sequence for item in publisher.submissions] == [1, 2]
    assert [item[2] for item in publisher.submissions] == [0, 0]
    assert source.preview_timings == [
        PoseRequestTiming(request_received_ns=0),
        PoseRequestTiming(
            request_received_ns=100_000_000,
            request_interval_ns=100_000_000,
        ),
    ]
    assert source.rendered_analyses == [analysis(100), analysis(133)]
    assert result.requests_processed == 2
    assert result.assessable_responses == 2
    assert result.source == "camera"
    assert result.authorized is True
    assert result.connected is True
    assert publisher.closed


def test_preview_quit_submits_current_numeric_sample_then_exits_cleanly(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    source = FakeSource([FakeSample(100, object(), quit_requested=True)], order)
    tracker = FakeTracker([analysis(100)], order)
    publisher = FakePublisher([pose_request(1), pose_request(2)], order)

    result = run_pose_client(
        PoseClientConfig(
            peer="100.106.237.106:45873",
            token_file=tmp_path / "pose.token",
            model_asset_path=tmp_path / "pose.task",
            max_requests=2,
        ),
        dependencies=PoseClientDependencies(
            validate_runtime=lambda: {},
            validate_model=lambda path: Path(path),
            token_loader=lambda path: b"t" * 32,
            publisher_factory=lambda peer, token, **kwargs: publisher,
            source_factory=lambda config, path: source,
            tracker_factory=lambda: tracker,
            clock_ns=lambda: source.last_timestamp_ms * 1_000_000,
        ),
    )

    assert len(publisher.submissions) == 1
    assert result.requests_processed == 1
    assert order == [
        "source_open",
        "publisher_start",
        "wait_for_request",
        "source_read",
        "track",
        "submit",
        "publisher_close",
        "source_close",
    ]


def test_pose_client_does_not_open_native_modules_at_import_or_config_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"native import: {name}")),
    )
    config = PoseClientConfig(
        peer="100.106.237.106:45873",
        token_file=tmp_path / "pose.token",
        fixture_paths=(tmp_path / "standing.jpg",),
    )
    assert config.fixture_paths == (tmp_path / "standing.jpg",)
    assert config.preview is True
    assert config.authorize is False


def test_cli_preview_is_on_by_default_and_has_explicit_headless_switch() -> None:
    parser = _parser()
    default_args = parser.parse_args(
        ["--peer", "100.106.237.106:45873", "--token-file", "pose.token"]
    )
    headless_args = parser.parse_args(
        [
            "--peer",
            "100.106.237.106:45873",
            "--token-file",
            "pose.token",
            "--no-preview",
        ]
    )

    assert default_args.preview is True
    assert headless_args.preview is False


def test_pose_preview_draws_bones_fps_text_and_accepts_q_or_escape() -> None:
    metrics = PosePreviewMetrics(
        frame_fps=20.0,
        capture_latency_ms=5.0,
        model_fps=50.0,
        model_latency_ms=20.0,
        request_fps=10.0,
        e2e_latency_ms=36.0,
    )
    expected_lines = (
        "Squats: 2 | Phase: down",
        "Frames: 20.0 FPS | capture 5.0 ms",
        "Pose model: 50.0 FPS | inference 20.0 ms",
        "Requests: 10.0 FPS | E2E ready 36.0 ms",
        "q/Esc stop",
    )

    current_analysis = analysis(100, phase=SquatPhase.DOWN, rep_count=2)
    assert pose_preview_lines(metrics, current_analysis) == expected_lines
    for key_code in (ord("q"), 27):
        cv2 = PreviewCv2(key_code=key_code)

        quit_requested = _show_pose_preview(
            cv2,
            cv2.frame,
            numeric_pose(100),
            metrics,
            analysis=current_analysis,
            mirror=True,
            window_name="RecoveryBox pose client",
        )

        assert quit_requested is True
        assert cv2.line_calls > 0
        assert cv2.circle_calls == 33
        assert cv2.flip_calls == 1
        assert cv2.text_lines == list(expected_lines)
        assert cv2.imshow_calls == 1
        assert cv2.wait_key_calls == 1


def test_fixture_source_cycles_explicit_images_without_constructing_a_camera(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pose.task"
    standing = tmp_path / "standing.jpg"
    down = tmp_path / "down.jpg"
    model.write_bytes(b"model")
    standing.write_bytes(b"standing")
    down.write_bytes(b"down")
    loaded: list[str] = []

    class FakeFrame:
        shape = (480, 640, 3)

    class FakeLandmarker:
        def __init__(self) -> None:
            self.closed = False

        def detect(self, image):
            assert image is not None
            return SimpleNamespace(pose_landmarks=[])

        def close(self) -> None:
            self.closed = True

    landmarker = FakeLandmarker()
    cv2 = SimpleNamespace(
        IMREAD_COLOR=1,
        COLOR_BGR2RGB=2,
        imread=lambda path, mode: loaded.append(path) or FakeFrame(),
        cvtColor=lambda frame, conversion: frame,
    )
    pose_landmarker = SimpleNamespace(create_from_options=lambda options: landmarker)
    mediapipe = SimpleNamespace(
        Image=lambda **kwargs: kwargs,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: kwargs,
            vision=SimpleNamespace(
                RunningMode=SimpleNamespace(IMAGE="image"),
                PoseLandmarkerOptions=lambda **kwargs: kwargs,
                PoseLandmarker=pose_landmarker,
            ),
        ),
    )
    ticks = iter(
        (
            100_000_000,
            101_000_000,
            102_000_000,
            103_000_000,
            133_000_000,
            134_000_000,
            135_000_000,
            136_000_000,
            166_000_000,
            167_000_000,
            168_000_000,
            169_000_000,
        )
    )
    source = MediaPipeFixturePoseSource(
        FixturePoseConfig(model, (standing, down), preview=False),
        runtime_loader=lambda: (cv2, mediapipe),
        clock_ns=lambda: next(ticks),
    )

    source.open()
    samples = [source.read(), source.read(), source.read()]
    source.close()

    assert loaded == [str(standing.resolve()), str(down.resolve()), str(standing.resolve())]
    assert [sample.timestamp_ms for sample in samples] == [100, 133, 166]
    assert all(sample.pose is None for sample in samples)
    assert landmarker.closed


def test_fixture_preview_stays_local_and_reports_current_performance(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pose.task"
    fixture = tmp_path / "standing.jpg"
    model.write_bytes(b"model")
    fixture.write_bytes(b"fixture")
    cv2 = PreviewCv2(key_code=ord("q"))

    class Landmarker:
        def detect(self, image):
            assert image is not None
            return SimpleNamespace(pose_landmarks=[raw_pose_landmarks()])

        def close(self) -> None:
            pass

    ticks = iter((1_000_000_000, 1_005_000_000, 1_006_000_000, 1_026_000_000))
    source = MediaPipeFixturePoseSource(
        FixturePoseConfig(model, (fixture,)),
        runtime_loader=lambda: (cv2, fake_mediapipe(Landmarker())),
        clock_ns=lambda: next(ticks),
    )
    tracker = FakeTracker(
        [analysis(1000, phase=SquatPhase.DOWN, rep_count=2)],
        [],
    )

    source.open()
    sample, current_analysis = source.read_analyzed(
        tracker,
        preview_timing=PoseRequestTiming(
            request_received_ns=990_000_000,
            request_interval_ns=100_000_000,
        ),
    )
    source.close()

    assert sample.timestamp_ms == 1000
    assert sample.pose is not None
    assert sample.quit_requested is True
    assert current_analysis.phase is SquatPhase.DOWN
    assert current_analysis.rep_count == 2
    assert set(sample.__dataclass_fields__) == {
        "timestamp_ms",
        "pose",
        "quit_requested",
    }
    assert cv2.text_lines == [
        "Squats: 2 | Phase: down",
        "Frames: -- FPS | capture 5.0 ms",
        "Pose model: 50.0 FPS | inference 20.0 ms",
        "Requests: 10.0 FPS | E2E ready 36.0 ms",
        "q/Esc stop",
    ]
    assert cv2.flip_calls == 0
    assert cv2.destroyed_windows == ["RecoveryBox pose client"]


class ScriptedCapture:
    """Hardware-free capture whose blocking reads model a native backend."""

    def __init__(
        self,
        frames: Queue[object],
        *,
        read_timeout_seconds: float = 0.05,
    ) -> None:
        self.frames = frames
        self.read_timeout_seconds = read_timeout_seconds
        self.thread_events: list[tuple[str, str]] = []
        self.read_failed = Event()
        self.released = Event()

    def isOpened(self) -> bool:
        self.thread_events.append(("is_opened", current_thread().name))
        return True

    def set(self, property_id, value) -> None:
        del property_id, value
        self.thread_events.append(("set", current_thread().name))

    def read(self):
        self.thread_events.append(("read", current_thread().name))
        try:
            frame = self.frames.get(timeout=self.read_timeout_seconds)
        except Empty:
            self.read_failed.set()
            return False, None
        return True, frame

    def release(self) -> None:
        self.thread_events.append(("release", current_thread().name))
        self.released.set()


def feed_after_request_generation_barrier(
    source: RequestFreshWebcamPoseSource,
    frames: Queue[object],
    *delivered_frames: object,
    delay_seconds: float = 0.0,
) -> Thread:
    """Feed a blocked fake capture only after ``read`` claims its barrier."""

    worker = source._capture_worker
    assert worker is not None
    previous_barrier = worker._request_barrier_generation

    def feed() -> None:
        with worker._condition:
            reached = worker._condition.wait_for(
                lambda: worker._request_barrier_generation > previous_barrier,
                timeout=0.2,
            )
        assert reached
        if delay_seconds:
            Event().wait(delay_seconds)
        for frame in delivered_frames:
            frames.put(frame)

    thread = Thread(target=feed, name="test-frame-feeder")
    thread.start()
    return thread


def test_request_fresh_webcam_uses_one_camera_and_only_infers_post_request_frames(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")
    frames: Queue[object] = Queue()
    stale = PreviewFrame()
    in_flight_one = PreviewFrame()
    post_request_one = PreviewFrame()
    in_flight_two = PreviewFrame()
    post_request_two = PreviewFrame()
    frames.put(stale)
    capture = ScriptedCapture(frames)
    camera_opens: list[tuple[int, str]] = []
    inferred_frames: list[object] = []

    class Landmarker:
        def detect_for_video(self, image, timestamp_ms):
            assert timestamp_ms >= 0
            inferred_frames.append(image["data"])
            return SimpleNamespace(pose_landmarks=[])

        def close(self) -> None:
            pass

    def video_capture(index: int) -> ScriptedCapture:
        camera_opens.append((index, current_thread().name))
        return capture

    cv2 = SimpleNamespace(
        VideoCapture=video_capture,
        CAP_PROP_BUFFERSIZE=1,
        COLOR_BGR2RGB=2,
        cvtColor=lambda frame, conversion: frame,
    )
    source = RequestFreshWebcamPoseSource(
        WebcamPoseConfig(model_asset_path=model, preview=False, camera_index=3),
        runtime_loader=lambda: (cv2, fake_mediapipe(Landmarker())),
        clock_ns=monotonic_ns,
    )

    source.open()  # Waits for ``stale`` to warm the persistent camera.
    first_request_ns = monotonic_ns()
    first_feeder = feed_after_request_generation_barrier(
        source,
        frames,
        in_flight_one,
        post_request_one,
    )
    first = source.read(preview_timing=PoseRequestTiming(request_received_ns=first_request_ns))
    first_feeder.join(0.2)
    second_request_ns = monotonic_ns()
    second_feeder = feed_after_request_generation_barrier(
        source,
        frames,
        in_flight_two,
        post_request_two,
    )
    second = source.read(
        preview_timing=PoseRequestTiming(
            request_received_ns=second_request_ns,
            request_interval_ns=second_request_ns - first_request_ns,
        )
    )
    second_feeder.join(0.2)
    source.close()

    assert inferred_frames == [post_request_one, post_request_two]
    assert stale not in inferred_frames
    assert in_flight_one not in inferred_frames
    assert in_flight_two not in inferred_frames
    assert camera_opens == [(3, "recoverybox-camera-capture")]
    assert capture.released.is_set()
    assert {thread for _, thread in capture.thread_events} == {"recoverybox-camera-capture"}
    assert first.timestamp_ms <= second.timestamp_ms
    assert first.pose is None and second.pose is None
    assert set(first.__dataclass_fields__) == {
        "timestamp_ms",
        "pose",
        "quit_requested",
    }
    assert not hasattr(first, "frame")


def test_request_fresh_webcam_preview_mirrors_pose_and_uses_camera_frame_fps(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")
    frames: Queue[object] = Queue()
    frames.put(PreviewFrame())
    capture = ScriptedCapture(frames)
    cv2 = PreviewCv2(key_code=27)
    cv2.VideoCapture = lambda camera_index: capture
    detection_thread_ids: list[int | None] = []

    class Landmarker:
        def detect_for_video(self, image, timestamp_ms):
            assert image is not None
            assert timestamp_ms >= 0
            detection_thread_ids.append(current_thread().ident)
            return SimpleNamespace(pose_landmarks=[raw_pose_landmarks()])

        def close(self) -> None:
            pass

    source = RequestFreshWebcamPoseSource(
        WebcamPoseConfig(model_asset_path=model, preview=True, camera_index=2),
        runtime_loader=lambda: (cv2, fake_mediapipe(Landmarker())),
        clock_ns=monotonic_ns,
    )

    source.open()
    request_ns = monotonic_ns()
    feeder = feed_after_request_generation_barrier(
        source,
        frames,
        PreviewFrame(),
        PreviewFrame(),
    )
    sample = source.read(
        preview_timing=PoseRequestTiming(
            request_received_ns=request_ns,
            request_interval_ns=100_000_000,
        )
    )
    feeder.join(0.2)
    source.close()

    assert sample.quit_requested is True
    assert sample.pose is not None
    assert capture.released.is_set()
    assert cv2.line_calls > 0
    assert cv2.circle_calls == 33
    assert cv2.flip_calls == 1
    assert cv2.text_lines[0].startswith("Frames: ")
    assert "-- FPS" not in cv2.text_lines[0]
    assert cv2.text_lines[-1] == "q/Esc stop"
    assert cv2.destroyed_windows == ["RecoveryBox squat tracker"]
    assert current_thread() is main_thread()
    assert detection_thread_ids == [current_thread().ident]
    assert cv2.imshow_thread_ids == [current_thread().ident]
    assert cv2.wait_key_thread_ids == [current_thread().ident]


def test_request_fresh_webcam_propagates_capture_failure_and_cleans_up(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")
    frames: Queue[object] = Queue()
    frames.put(PreviewFrame())
    capture = ScriptedCapture(frames, read_timeout_seconds=0.02)
    cv2 = SimpleNamespace(
        VideoCapture=lambda index: capture,
        CAP_PROP_BUFFERSIZE=1,
        COLOR_BGR2RGB=2,
        cvtColor=lambda frame, conversion: frame,
    )

    class Landmarker:
        def detect_for_video(self, image, timestamp_ms):
            raise AssertionError("failed capture must not reach inference")

        def close(self) -> None:
            pass

    source = RequestFreshWebcamPoseSource(
        WebcamPoseConfig(model_asset_path=model, preview=False),
        runtime_loader=lambda: (cv2, fake_mediapipe(Landmarker())),
        frame_timeout_seconds=0.1,
    )

    source.open()
    assert capture.read_failed.wait(0.2)
    with pytest.raises(RuntimeError, match="capture failed"):
        source.read(preview_timing=PoseRequestTiming(request_received_ns=monotonic_ns()))
    source.close()

    assert capture.released.is_set()


def test_request_fresh_webcam_rejects_preview_outside_main_thread(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")
    frames: Queue[object] = Queue()
    frames.put(PreviewFrame())
    capture = ScriptedCapture(frames)
    cv2 = PreviewCv2(key_code=27)
    cv2.VideoCapture = lambda camera_index: capture

    class Landmarker:
        def close(self) -> None:
            pass

    source = RequestFreshWebcamPoseSource(
        WebcamPoseConfig(model_asset_path=model, preview=True),
        runtime_loader=lambda: (cv2, fake_mediapipe(Landmarker())),
    )
    source.open()
    failures: list[BaseException] = []

    def read_off_main_thread() -> None:
        try:
            source.read()
        except BaseException as exc:
            failures.append(exc)

    reader = Thread(target=read_off_main_thread, name="not-main")
    reader.start()
    reader.join(0.2)
    source.close()

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "process main thread" in str(failures[0])
    assert cv2.imshow_calls == 0
    assert cv2.wait_key_calls == 0


def test_request_fresh_webcam_propagates_release_failure(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")
    frames: Queue[object] = Queue()
    frames.put(PreviewFrame())

    class ReleaseFailCapture(ScriptedCapture):
        def release(self) -> None:
            super().release()
            raise RuntimeError("native release failed")

    capture = ReleaseFailCapture(frames, read_timeout_seconds=0.02)
    cv2 = SimpleNamespace(VideoCapture=lambda index: capture, COLOR_BGR2RGB=2)

    class Landmarker:
        def close(self) -> None:
            pass

    source = RequestFreshWebcamPoseSource(
        WebcamPoseConfig(model_asset_path=model, preview=False),
        runtime_loader=lambda: (cv2, fake_mediapipe(Landmarker())),
    )
    source.open()

    with pytest.raises(RuntimeError, match="capture release failed"):
        source.close()

    assert capture.released.is_set()


def test_request_fresh_webcam_open_has_a_hard_deadline(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")
    unblock = Event()

    class BlockingCapture:
        def isOpened(self) -> bool:
            return True

        def set(self, property_id, value) -> None:
            del property_id, value

        def read(self):
            unblock.wait(0.5)
            return False, None

        def release(self) -> None:
            pass

    landmarker_closed = Event()

    class Landmarker:
        def close(self) -> None:
            landmarker_closed.set()

    capture = BlockingCapture()
    cv2 = SimpleNamespace(VideoCapture=lambda index: capture, COLOR_BGR2RGB=2)
    source = RequestFreshWebcamPoseSource(
        WebcamPoseConfig(model_asset_path=model, preview=False),
        runtime_loader=lambda: (cv2, fake_mediapipe(Landmarker())),
        open_timeout_seconds=0.02,
        close_timeout_seconds=0.02,
    )

    started = monotonic()
    with pytest.raises(TimeoutError, match="open and produce"):
        source.open()
    elapsed = monotonic() - started
    unblock.set()

    assert elapsed < 0.2
    assert landmarker_closed.is_set()


def test_request_fresh_webcam_rejects_a_slot_arriving_after_read_deadline(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")
    frames: Queue[object] = Queue()
    frames.put(PreviewFrame())
    capture = ScriptedCapture(frames, read_timeout_seconds=0.05)
    cv2 = SimpleNamespace(
        VideoCapture=lambda index: capture,
        COLOR_BGR2RGB=2,
        cvtColor=lambda frame, conversion: frame,
    )

    class Landmarker:
        def detect_for_video(self, image, timestamp_ms):
            raise AssertionError("a late frame must not reach inference")

        def close(self) -> None:
            pass

    source = RequestFreshWebcamPoseSource(
        WebcamPoseConfig(model_asset_path=model, preview=False),
        runtime_loader=lambda: (cv2, fake_mediapipe(Landmarker())),
        frame_timeout_seconds=0.01,
        close_timeout_seconds=0.1,
    )
    source.open()
    feeder = feed_after_request_generation_barrier(
        source,
        frames,
        PreviewFrame(),
        PreviewFrame(),
        delay_seconds=0.03,
    )

    started = monotonic()
    with pytest.raises(TimeoutError, match="post-request frame"):
        source.read(preview_timing=PoseRequestTiming(request_received_ns=monotonic_ns()))
    elapsed = monotonic() - started
    feeder.join(0.2)
    source.close()

    assert elapsed < 0.03


def test_request_fresh_webcam_read_and_close_have_hard_deadlines(
    tmp_path: Path,
) -> None:
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")
    block_after_warmup = Event()
    released = Event()

    class BlockingAfterWarmupCapture:
        def __init__(self) -> None:
            self.reads = 0

        def isOpened(self) -> bool:
            return True

        def set(self, property_id, value) -> None:
            del property_id, value

        def read(self):
            self.reads += 1
            if self.reads == 1:
                return True, PreviewFrame()
            block_after_warmup.wait(0.5)
            return False, None

        def release(self) -> None:
            released.set()

    capture = BlockingAfterWarmupCapture()
    cv2 = SimpleNamespace(
        VideoCapture=lambda index: capture,
        COLOR_BGR2RGB=2,
        cvtColor=lambda frame, conversion: frame,
    )

    class Landmarker:
        def close(self) -> None:
            pass

    source = RequestFreshWebcamPoseSource(
        WebcamPoseConfig(model_asset_path=model, preview=False),
        runtime_loader=lambda: (cv2, fake_mediapipe(Landmarker())),
        frame_timeout_seconds=0.02,
        close_timeout_seconds=0.02,
    )

    source.open()
    started = monotonic()
    with pytest.raises(TimeoutError, match="post-request frame"):
        source.read(preview_timing=PoseRequestTiming(request_received_ns=monotonic_ns()))
    assert monotonic() - started < 0.2

    started = monotonic()
    with pytest.raises(TimeoutError, match="did not stop"):
        source.close()
    assert monotonic() - started < 0.2

    block_after_warmup.set()
    assert released.wait(0.2)
    source.close()
