from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import recoverybox.device.pi_pose_v4l2 as pose_module
import recoverybox.device.remote_pose_service as service_module
from recoverybox.device.pi_pose import (
    MOVENET_KEYPOINT_COUNT,
    MoveNetKeypoint,
    MoveNetPoint,
    PiPoseConfigurationError,
    PiPoseRuntimeError,
)
from recoverybox.device.pi_pose_ncnn import PoseInferenceResult
from recoverybox.device.pi_pose_v4l2 import (
    LibyuvYuy2ToBgra,
    V4L2Dependencies,
    V4L2MmapCamera,
    V4L2NcnnPoseConfig,
    V4L2NcnnPoseObservation,
    V4L2NcnnPoseSource,
    run_v4l2_ncnn_pose_check,
)
from recoverybox.exercise import (
    SquatAnalysis,
    SquatAssessmentIssue,
    SquatPhase,
    SquatTracker,
    SquatTrackerConfig,
)


def _analysis(
    timestamp_ms: int,
    *,
    assessable: bool = False,
    rep_count: int = 0,
) -> SquatAnalysis:
    return SquatAnalysis(
        timestamp_ms=timestamp_ms,
        assessable=assessable,
        phase=SquatPhase.STANDING if assessable else SquatPhase.UNKNOWN,
        rep_count=rep_count,
        events=(),
        issues=() if assessable else (SquatAssessmentIssue.CAMERA_TIMEOUT,),
        confidence=0.9 if assessable else 0.0,
        knee_angle_degrees=175.0 if assessable else None,
        arms_in_t=True if assessable else None,
    )


def _observation(
    timestamp_ms: int,
    *,
    assessable: bool = False,
    rep_count: int = 0,
) -> V4L2NcnnPoseObservation:
    return V4L2NcnnPoseObservation(
        analysis=_analysis(timestamp_ms, assessable=assessable, rep_count=rep_count),
        frame_received=True,
        inference_ms=20.0,
        evidence_age_ms=30.0,
        person_score=0.9 if assessable else None,
        timed_out=not assessable,
    )


def _points(*, down: bool = False) -> tuple[MoveNetPoint, ...]:
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
        MoveNetKeypoint.LEFT_ANKLE: (0.75 if down else 0.95, 0.62 if down else 0.43),
        MoveNetKeypoint.RIGHT_ANKLE: (0.75 if down else 0.95, 0.38 if down else 0.57),
    }
    for keypoint, coordinate in placements.items():
        coordinates[keypoint] = coordinate
    return tuple(MoveNetPoint(y=y, x=x, score=0.99) for y, x in coordinates)


def test_environment_defaults_are_deployed_opt_paths() -> None:
    config = V4L2NcnnPoseConfig.from_environment({})

    assert config.camera_device == Path("/dev/video0")
    assert config.libyuv_library_path == Path("/usr/lib/arm-linux-gnueabihf/libyuv.so.0")
    assert config.ncnn.runtime_path == Path("/opt/recoverybox/runtime/ncnn")
    assert config.ncnn.rtmpose_param_path == Path("/opt/recoverybox/models/ncnn/rtmpose-t.param")
    assert config.ncnn.nanodet_bin_path == Path("/opt/recoverybox/models/ncnn/nanodet-m-int8.bin")
    assert config.frames_per_second == 10
    assert config.buffer_count == 8


@pytest.mark.parametrize("timeout", [0, -0.1, 0.5001, float("inf")])
def test_all_runtime_read_timeouts_are_capped_at_half_second(timeout: float) -> None:
    with pytest.raises(PiPoseConfigurationError, match=r"at most 0\.5"):
        V4L2NcnnPoseConfig(worker_timeout_seconds=timeout)


def test_yuyv_capture_requires_even_width_and_bounded_buffer_count() -> None:
    with pytest.raises(PiPoseConfigurationError, match="even"):
        V4L2NcnnPoseConfig(width=321)
    with pytest.raises(PiPoseConfigurationError, match="between 2 and 8"):
        V4L2NcnnPoseConfig(buffer_count=9)


class _FakeMapping(bytearray):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeV4L2:
    def __init__(
        self,
        *,
        readable: bool = True,
        kernel_timestamp: float = 9.0,
        readiness: list[bool] | None = None,
        dequeue_frames: list[tuple[int, float]] | None = None,
        clocks: list[float] | None = None,
        returned_buffer_count: int | None = None,
        negotiated_fps: int | None = None,
        monotonic_timestamps: bool = True,
    ) -> None:
        self.readable = readable
        self.kernel_timestamp = kernel_timestamp
        self.readiness = readiness
        self.dequeue_frames = dequeue_frames
        self.clocks = clocks
        self.returned_buffer_count = returned_buffer_count
        self.negotiated_fps = negotiated_fps
        self.monotonic_timestamps = monotonic_timestamps
        self.calls: list[int] = []
        self.queued: list[int] = []
        self.closed_fds: list[int] = []
        self.mappings: list[_FakeMapping] = []
        self.frame_size = 320 * 240 * 2

    def open(self, path: str, flags: int) -> int:
        assert path == "/dev/video0"
        assert flags & pose_module.os.O_NONBLOCK
        return 17

    def close(self, fd: int) -> None:
        self.closed_fds.append(fd)

    @staticmethod
    def fstat(fd: int):
        assert fd == 17
        return SimpleNamespace(st_mode=stat.S_IFCHR | 0o600)

    def mmap(self, fd: int, length: int, flags: int, protection: int, offset: int):
        del flags, protection
        assert fd == 17
        assert length == self.frame_size
        assert offset % self.frame_size == 0
        mapped = _FakeMapping(length)
        self.mappings.append(mapped)
        return mapped

    def wait(self, fd: int, timeout: float) -> bool:
        assert fd == 17
        assert 0 <= timeout <= 0.5
        if self.readiness is not None:
            return self.readiness.pop(0)
        if timeout == 0:
            return False
        return self.readable

    def clock(self) -> float:
        if self.clocks is None:
            return 10.0
        return self.clocks.pop(0)

    def ioctl(self, fd: int, request: int, value) -> None:
        assert fd == 17
        self.calls.append(request)
        if request == pose_module.VIDIOC_QUERYCAP:
            value.capabilities = pose_module.V4L2_CAP_VIDEO_CAPTURE | pose_module.V4L2_CAP_STREAMING
        elif request == pose_module.VIDIOC_S_FMT:
            value.fmt.pix.bytesperline = 320 * 2
            value.fmt.pix.sizeimage = self.frame_size
        elif request == pose_module.VIDIOC_S_PARM:
            assert value.parm.capture.timeperframe.denominator == 10
            if self.negotiated_fps is not None:
                value.parm.capture.timeperframe.numerator = 1
                value.parm.capture.timeperframe.denominator = self.negotiated_fps
        elif request == pose_module.VIDIOC_REQBUFS:
            if self.returned_buffer_count is not None:
                value.count = self.returned_buffer_count
        elif request == pose_module.VIDIOC_QUERYBUF:
            value.length = self.frame_size
            value.m.offset = value.index * self.frame_size
        elif request == pose_module.VIDIOC_QBUF:
            self.queued.append(value.index)
        elif request == pose_module.VIDIOC_DQBUF:
            if self.dequeue_frames is None:
                index, timestamp = 0, self.kernel_timestamp
            else:
                index, timestamp = self.dequeue_frames.pop(0)
            value.index = index
            value.bytesused = self.frame_size
            value.flags = (
                pose_module.V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC if self.monotonic_timestamps else 0
            )
            value.timestamp.tv_sec = int(timestamp)
            value.timestamp.tv_usec = int((timestamp % 1) * 1_000_000)

    def dependencies(self) -> V4L2Dependencies:
        return V4L2Dependencies(
            open_device=self.open,
            close_device=self.close,
            fstat=self.fstat,  # type: ignore[arg-type]
            ioctl=self.ioctl,
            mmap_buffer=self.mmap,
            wait_readable=self.wait,
            clock=self.clock,
        )


def test_v4l2_streams_yuyv_with_mmap_and_requeues_after_lease() -> None:
    backend = _FakeV4L2(kernel_timestamp=9.25)
    config = V4L2NcnnPoseConfig(width=320, height=240)
    camera = V4L2MmapCamera(config, dependencies=backend.dependencies())

    camera.open()
    lease = camera.dequeue()

    assert lease is not None
    assert lease.captured_monotonic == pytest.approx(9.25)
    assert camera.bytes_per_line == 640
    assert backend.queued == list(range(8))
    lease.release()
    assert backend.queued == [*range(8), 0]
    camera.close()
    assert backend.closed_fds == [17]
    assert all(mapped.closed for mapped in backend.mappings)
    assert pose_module.VIDIOC_STREAMON in backend.calls
    assert pose_module.VIDIOC_STREAMOFF in backend.calls


def test_v4l2_rejects_fewer_buffers_or_faster_cadence_than_configured() -> None:
    fewer = _FakeV4L2(returned_buffer_count=4)
    camera = V4L2MmapCamera(
        V4L2NcnnPoseConfig(width=320, height=240),
        dependencies=fewer.dependencies(),
    )
    with pytest.raises(PiPoseRuntimeError, match="unsafe mmap buffer count"):
        camera.open()

    faster = _FakeV4L2(negotiated_fps=30)
    camera = V4L2MmapCamera(
        V4L2NcnnPoseConfig(width=320, height=240),
        dependencies=faster.dependencies(),
    )
    with pytest.raises(PiPoseRuntimeError, match="cadence exceeds"):
        camera.open()


def test_capture_timestamp_never_claims_newer_than_pre_wait_clock() -> None:
    backend = _FakeV4L2(kernel_timestamp=11.0)
    camera = V4L2MmapCamera(
        V4L2NcnnPoseConfig(width=320, height=240),
        dependencies=backend.dependencies(),
    )
    camera.open()

    lease = camera.dequeue()

    assert lease is not None
    assert lease.captured_monotonic == 10.0
    lease.release()
    camera.close()


def test_dequeue_drops_ready_backlog_and_returns_only_newest_buffer() -> None:
    backend = _FakeV4L2(
        readiness=[True, True, False],
        dequeue_frames=[(0, 9.6), (1, 9.7)],
        clocks=[10.0, 10.01],
    )
    camera = V4L2MmapCamera(
        V4L2NcnnPoseConfig(width=320, height=240),
        dependencies=backend.dependencies(),
    )
    camera.open()

    lease = camera.dequeue()

    assert lease is not None
    assert lease.index == 1
    assert lease.captured_monotonic == pytest.approx(9.7)
    # The older ready buffer was returned before the newest lease escaped.
    assert backend.queued == [*range(8), 0]
    lease.release()
    assert backend.queued == [*range(8), 0, 1]
    camera.close()


def test_drained_buffers_without_kernel_monotonic_flag_remain_conservatively_stale() -> None:
    backend = _FakeV4L2(
        readiness=[True, True, False],
        dequeue_frames=[(0, 9.6), (1, 9.7)],
        clocks=[10.0, 10.01],
        monotonic_timestamps=False,
    )
    camera = V4L2MmapCamera(
        V4L2NcnnPoseConfig(width=320, height=240),
        dependencies=backend.dependencies(),
    )
    camera.open()

    lease = camera.dequeue()

    assert lease is not None
    assert lease.index == 1
    assert lease.captured_monotonic == 0.0
    lease.release()
    camera.close()


def test_select_timeout_is_nonblocking_missing_frame_without_dequeue() -> None:
    backend = _FakeV4L2(readable=False)
    camera = V4L2MmapCamera(
        V4L2NcnnPoseConfig(width=320, height=240, capture_timeout_seconds=0.2),
        dependencies=backend.dependencies(),
    )
    camera.open()

    assert camera.dequeue() is None
    assert pose_module.VIDIOC_DQBUF not in backend.calls
    camera.close()


class _FakeNativeFunction:
    def __init__(self) -> None:
        self.argtypes = None
        self.restype = None
        self.calls: list[tuple[int, int, int, int]] = []

    def __call__(self, source, source_stride, output, output_stride, width, height) -> int:
        del source, output
        self.calls.append((source_stride, output_stride, width, height))
        return 0


class _FakeLibyuv:
    def __init__(self) -> None:
        self.YUY2ToARGB = _FakeNativeFunction()


def test_libyuv_conversion_is_same_process_and_exact_bgra_size() -> None:
    library = _FakeLibyuv()
    converter = LibyuvYuy2ToBgra(
        Path("/usr/lib/arm-linux-gnueabihf/libyuv.so.0"),
        library_loader=lambda path: library,
    )
    yuyv = _FakeMapping(4 * 2 * 2)

    bgra = converter.convert(yuyv, source_stride=8, width=4, height=2)

    assert type(bgra) is bytes
    assert len(bgra) == 4 * 2 * 4
    assert library.YUY2ToARGB.calls == [(8, 16, 4, 2)]


class _FakeLease:
    def __init__(self, marker: bytes, timestamp: float) -> None:
        self.mapped = _FakeMapping(320 * 240 * 2)
        self.mapped[: len(marker)] = marker
        self.captured_monotonic = timestamp
        self.released = False

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.released = True


class _FakeCamera:
    bytes_per_line = 640

    def __init__(self, leases: list[_FakeLease | None]) -> None:
        self.leases = leases
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def dequeue(self):
        return self.leases.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeConverter:
    def __init__(self, marker: bytes) -> None:
        self.marker = marker

    def convert(self, source, *, source_stride: int, width: int, height: int) -> bytes:
        assert bytes(source[: len(self.marker)]) == self.marker
        assert (source_stride, width, height) == (640, 320, 240)
        return self.marker + bytes(width * height * 4 - len(self.marker))


class _FakeEstimator:
    def __init__(self, points: list[tuple[MoveNetPoint, ...] | None]) -> None:
        self.points = points
        self.inputs: list[bytes] = []
        self.closed = False
        self.expected_released_lease: _FakeLease | None = None

    def infer_bgra(self, frame: bytes, *, width: int, height: int, captured_monotonic: float):
        assert (width, height) == (320, 240)
        if self.expected_released_lease is not None:
            assert self.expected_released_lease.released
        self.inputs.append(frame)
        points = self.points.pop(0)
        return PoseInferenceResult(
            points=points,
            person_score=0.9 if points is not None else None,
            detector_ms=5.0,
            pose_ms=5.0 if points is not None else None,
            total_ms=10.0,
            evidence_age_ms=20.0,
            timed_out=False,
        )

    def close(self) -> None:
        self.closed = True


def test_child_side_source_composes_ncnn_and_tracker_without_frame_output() -> None:
    marker = b"PRIVATE_YUYV_MARKER_3719"
    leases = [_FakeLease(marker, timestamp) for timestamp in (1.0, 1.2, 1.4)]
    camera = _FakeCamera(list(leases))
    estimator = _FakeEstimator([_points(), _points(down=True), _points()])
    tracker = SquatTracker(SquatTrackerConfig(phase_confirmation_frames=1))
    source = pose_module._InProcessV4L2NcnnPoseSource(
        V4L2NcnnPoseConfig(width=320, height=240),
        camera=camera,  # type: ignore[arg-type]
        converter=_FakeConverter(marker),
        estimator=estimator,
        tracker=tracker,
        clock=lambda: 1.0,
    )

    source.open()
    observations = []
    for lease in leases:
        estimator.expected_released_lease = lease
        observations.append(source.read())
    source.close()

    assert observations[-1].analysis.rep_count == 1
    assert all(observation.analysis.assessable for observation in observations)
    assert all(not hasattr(observation, "frame") for observation in observations)
    assert marker not in repr(observations).encode()
    assert all(lease.released for lease in leases)
    assert camera.closed and estimator.closed


class _FakeWorker:
    def __init__(self, observations: list[V4L2NcnnPoseObservation | None]) -> None:
        self.observations = observations
        self.start_timeout: float | None = None
        self.read_timeouts: list[float] = []
        self.closed = False
        self.failure_kind: str | None = None

    def start(self, timeout_seconds: float) -> None:
        self.start_timeout = timeout_seconds

    def read(self, timeout_seconds: float):
        self.read_timeouts.append(timeout_seconds)
        return self.observations.pop(0)

    def close(self) -> None:
        self.closed = True


def test_public_source_enforces_bounded_worker_timeout_and_fails_closed() -> None:
    worker = _FakeWorker([None])
    source = V4L2NcnnPoseSource(
        V4L2NcnnPoseConfig(worker_timeout_seconds=0.25),
        worker=worker,
        clock=lambda: 2.0,
    )
    source.open()

    observation = source.read()

    assert not observation.analysis.assessable
    assert observation.analysis.issues == (SquatAssessmentIssue.CAMERA_TIMEOUT,)
    assert worker.read_timeouts == [0.25]
    with pytest.raises(PiPoseRuntimeError, match="unavailable"):
        source.read()
    source.close()
    assert worker.closed


def test_public_source_rejects_stale_child_analysis_and_preserves_rep_count() -> None:
    worker = _FakeWorker([_observation(1_000, assessable=True, rep_count=2)])
    ticks = iter((1.6, 1.6))
    source = V4L2NcnnPoseSource(
        V4L2NcnnPoseConfig(),
        worker=worker,
        clock=lambda: next(ticks),
    )
    source.open()

    observation = source.read()

    assert not observation.analysis.assessable
    assert observation.analysis.rep_count == 0
    assert observation.analysis.issues == (SquatAssessmentIssue.CAMERA_TIMEOUT,)
    source.close()


class _FakeChildConnection:
    def __init__(self, commands: list[str]) -> None:
        self.commands = commands
        self.sent: list[object] = []
        self.closed = False

    def send(self, value: object) -> None:
        self.sent.append(value)

    def recv(self) -> object:
        return self.commands.pop(0)

    def poll(self, timeout: float = 0.0) -> bool:
        del timeout
        return bool(self.commands)

    def close(self) -> None:
        self.closed = True


class _FakeChildSource:
    def __init__(self, observation: V4L2NcnnPoseObservation) -> None:
        self.observation = observation
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read(self) -> V4L2NcnnPoseObservation:
        return self.observation

    def close(self) -> None:
        self.closed = True


def test_child_ipc_transmits_only_numeric_observation(monkeypatch) -> None:
    child_source = _FakeChildSource(_observation(1_000, assessable=True))
    monkeypatch.setattr(
        pose_module._InProcessV4L2NcnnPoseSource,
        "from_config",
        lambda config: child_source,
    )
    connection = _FakeChildConnection(["read", "close"])

    pose_module._run_pose_child(connection, V4L2NcnnPoseConfig())

    assert connection.sent == [
        ("ready", None),
        ("observation", child_source.observation),
    ]
    assert all(not isinstance(payload, bytes) for message in connection.sent for payload in message)
    assert child_source.opened and child_source.closed and connection.closed


class _FakeParentConnection:
    def __init__(self) -> None:
        self.poll_results = [True, False]
        self.poll_timeouts: list[float] = []
        self.sent: list[object] = []
        self.closed = False

    def send(self, value: object) -> None:
        self.sent.append(value)

    @staticmethod
    def recv() -> object:
        return ("ready", None)

    def poll(self, timeout: float = 0.0) -> bool:
        self.poll_timeouts.append(timeout)
        return self.poll_results.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    exitcode = None

    def __init__(self) -> None:
        self.alive = False
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_timeouts: list[float | None] = []

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False


class _FakeProcessContext:
    def __init__(self) -> None:
        self.parent = _FakeParentConnection()
        self.child = _FakeChildConnection([])
        self.process = _FakeProcess()
        self.process_options: dict[str, object] | None = None

    def Pipe(self, duplex: bool = True):
        assert duplex
        return self.parent, self.child

    def Process(self, **kwargs: object):
        self.process_options = kwargs
        return self.process


def test_pose_child_timeout_terminates_without_extending_read_deadline() -> None:
    context = _FakeProcessContext()
    worker = pose_module._MultiprocessingPoseWorker(
        V4L2NcnnPoseConfig(),
        context=context,  # type: ignore[arg-type]
    )
    worker.start(2.0)

    assert worker.read(0.5) is None

    assert worker.failure_kind == "LocalPoseWorkerTimeout"
    assert context.parent.poll_timeouts == [2.0, 0.5]
    assert context.process.terminate_calls == 1
    assert context.process.join_timeouts == [0]
    assert context.process_options is not None
    assert context.process_options["target"] is pose_module._run_pose_child
    assert context.process_options["daemon"] is True
    worker.close()
    assert context.parent.closed


def test_remote_service_local_mode_selects_bounded_v4l2_ncnn_factory(monkeypatch) -> None:
    sentinel = object()
    environment = {"RECOVERYBOX_CAMERA_DEVICE": "/dev/video5"}
    monkeypatch.setattr(
        pose_module.V4L2NcnnPoseSource,
        "from_environment",
        lambda selected: sentinel,
    )

    source = service_module._local_pose_source_from_environment(environment)

    assert source is sentinel


class _FakeCheckSource:
    def __init__(self, observations: list[V4L2NcnnPoseObservation]) -> None:
        self.observations = observations
        self.closed = False

    def open(self) -> None:
        return

    def read(self) -> V4L2NcnnPoseObservation:
        return self.observations.pop(0)

    def close(self) -> None:
        self.closed = True


def test_bounded_check_status_is_numeric_and_contains_no_frame() -> None:
    source = _FakeCheckSource([_observation(1_000, assessable=True)])

    report = run_v4l2_ncnn_pose_check(
        V4L2NcnnPoseConfig(),
        max_frames=1,
        source_factory=lambda config: source,  # type: ignore[arg-type]
    )

    assert report["frames"] == 1
    assert report["frames_received"] == 1
    assert report["fresh_frames"] == 1
    assert report["assessable"] == 1
    assert report["inference_ms_max"] == 20.0
    assert report["raw_frames_persisted"] == 0
    assert report["audio"] == "disabled"
    assert "frame" not in json.dumps(report).lower().replace("frames", "")
    assert source.closed


def test_check_main_scrubs_runtime_details(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        pose_module,
        "run_v4l2_ncnn_pose_check",
        lambda config, max_frames: (_ for _ in ()).throw(
            PiPoseRuntimeError("PRIVATE_FRAME_OR_PATH")
        ),
    )

    assert pose_module.main(["--max-frames", "1"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "audio": "disabled",
        "failure": "PiPoseRuntimeError",
        "raw_frames_persisted": 0,
        "service": "recoverybox-pi-v4l2-ncnn-check/v1",
    }
    assert "PRIVATE" not in json.dumps(report)


def test_check_main_fails_when_no_camera_frame_reaches_ncnn(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        pose_module,
        "run_v4l2_ncnn_pose_check",
        lambda config, max_frames: {
            "service": "recoverybox-pi-v4l2-ncnn-check/v1",
            "frames": 3,
            "frames_received": 0,
            "fresh_frames": 0,
            "raw_frames_persisted": 0,
            "audio": "disabled",
        },
    )

    assert pose_module.main(["--max-frames", "3"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["failure"] == "FreshPoseEvidenceUnavailable"


def test_check_main_fails_when_received_frames_are_all_stale(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        pose_module,
        "run_v4l2_ncnn_pose_check",
        lambda config, max_frames: {
            "service": "recoverybox-pi-v4l2-ncnn-check/v1",
            "frames": 3,
            "frames_received": 3,
            "fresh_frames": 0,
            "timeouts": 3,
            "raw_frames_persisted": 0,
            "audio": "disabled",
        },
    )

    assert pose_module.main(["--max-frames", "3"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["failure"] == "FreshPoseEvidenceUnavailable"
