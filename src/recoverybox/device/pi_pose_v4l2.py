"""Single-worker V4L2, libyuv, and NCNN pose source for Raspberry Pi 3.

The camera is opened directly and streamed with V4L2 ``mmap`` buffers.  A
dequeued YUYV buffer is converted to a private BGRA copy by libyuv, immediately
requeued, and then consumed by NCNN in the same child worker.  There is no
separate camera-only process or raw-frame transport, and no raw frame is
returned by the public source, written to disk, logged, or placed in status.

Only :class:`V4L2NcnnPoseObservation`, whose public payload is numeric, crosses
the capture/inference boundary.  Hardware and C-library operations are
injectable so the ordinary test suite never opens a camera or loads NCNN.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import json
import math
import mmap
import multiprocessing
import os
import select
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol

from recoverybox.device.pi_pose import (
    PiPoseConfigurationError,
    PiPoseRuntimeError,
    movenet_to_mediapipe_frame,
)
from recoverybox.device.pi_pose_ncnn import (
    NcnnPersonPoseEstimator,
    NcnnPoseConfig,
    PoseInferenceResult,
)
from recoverybox.exercise import (
    SquatAnalysis,
    SquatAssessmentIssue,
    SquatPhase,
    SquatTracker,
)

DEFAULT_CAMERA_DEVICE: Final = Path("/dev/video0")
DEFAULT_LIBYUV_LIBRARY_PATH: Final = Path("/usr/lib/arm-linux-gnueabihf/libyuv.so.0")
DEFAULT_NCNN_RUNTIME_PATH: Final = Path("/opt/recoverybox/runtime/ncnn")
DEFAULT_NCNN_MODEL_DIRECTORY: Final = Path("/opt/recoverybox/models/ncnn")

V4L2_BUF_TYPE_VIDEO_CAPTURE: Final = 1
V4L2_MEMORY_MMAP: Final = 1
V4L2_FIELD_NONE: Final = 1
V4L2_CAP_VIDEO_CAPTURE: Final = 0x0000_0001
V4L2_CAP_STREAMING: Final = 0x0400_0000
V4L2_CAP_DEVICE_CAPS: Final = 0x8000_0000
V4L2_BUF_FLAG_ERROR: Final = 0x0000_0040
V4L2_BUF_FLAG_TIMESTAMP_MASK: Final = 0x0000_E000
V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC: Final = 0x0000_2000

_IOC_WRITE: Final = 1
_IOC_READ: Final = 2
_IOC_NRSHIFT: Final = 0
_IOC_TYPESHIFT: Final = 8
_IOC_SIZESHIFT: Final = 16
_IOC_DIRSHIFT: Final = 30
_MAX_CAPTURE_TIMEOUT_SECONDS: Final = 0.5
_DEFAULT_WORKER_START_TIMEOUT_SECONDS: Final = 25.0
_MAX_WORKER_START_TIMEOUT_SECONDS: Final = 29.0
_MIN_MMAP_BUFFERS: Final = 2
_MAX_MMAP_BUFFERS: Final = 8
_NCNN_QUEUE_COVERAGE_SECONDS: Final = 0.4


def _fourcc(a: str, b: str, c: str, d: str) -> int:
    return ord(a) | ord(b) << 8 | ord(c) << 16 | ord(d) << 24


V4L2_PIX_FMT_YUYV: Final = _fourcc("Y", "U", "Y", "V")


class _V4L2Capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_ubyte * 16),
        ("card", ctypes.c_ubyte * 32),
        ("bus_info", ctypes.c_ubyte * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class _V4L2PixFormat(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class _V4L2FormatUnion(ctypes.Union):
    _fields_ = [("pix", _V4L2PixFormat), ("raw_data", ctypes.c_ubyte * 200)]


class _V4L2Format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", _V4L2FormatUnion)]


class _V4L2RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("flags", ctypes.c_ubyte),
        ("reserved", ctypes.c_ubyte * 3),
    ]


class _Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _V4L2Timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_ubyte),
        ("seconds", ctypes.c_ubyte),
        ("minutes", ctypes.c_ubyte),
        ("hours", ctypes.c_ubyte),
        ("userbits", ctypes.c_ubyte * 4),
    ]


class _V4L2BufferMemory(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("planes", ctypes.c_void_p),
        ("fd", ctypes.c_int32),
    ]


class _V4L2Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", _Timeval),
        ("timecode", _V4L2Timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", _V4L2BufferMemory),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request_fd", ctypes.c_int32),
    ]


class _V4L2Fract(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


class _V4L2CaptureParm(ctypes.Structure):
    _fields_ = [
        ("capability", ctypes.c_uint32),
        ("capturemode", ctypes.c_uint32),
        ("timeperframe", _V4L2Fract),
        ("extendedmode", ctypes.c_uint32),
        ("readbuffers", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class _V4L2StreamParmUnion(ctypes.Union):
    _fields_ = [("capture", _V4L2CaptureParm), ("raw_data", ctypes.c_ubyte * 200)]


class _V4L2StreamParm(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("parm", _V4L2StreamParmUnion)]


def _ioc(direction: int, kind: str, number: int, size: int) -> int:
    return (
        direction << _IOC_DIRSHIFT
        | ord(kind) << _IOC_TYPESHIFT
        | number << _IOC_NRSHIFT
        | size << _IOC_SIZESHIFT
    )


def _ior(kind: str, number: int, structure: type[ctypes.Structure]) -> int:
    return _ioc(_IOC_READ, kind, number, ctypes.sizeof(structure))


def _iow(kind: str, number: int, structure: type[ctypes._SimpleCData]) -> int:
    return _ioc(_IOC_WRITE, kind, number, ctypes.sizeof(structure))


def _iowr(kind: str, number: int, structure: type[ctypes.Structure]) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, kind, number, ctypes.sizeof(structure))


VIDIOC_QUERYCAP: Final = _ior("V", 0, _V4L2Capability)
VIDIOC_S_FMT: Final = _iowr("V", 5, _V4L2Format)
VIDIOC_REQBUFS: Final = _iowr("V", 8, _V4L2RequestBuffers)
VIDIOC_QUERYBUF: Final = _iowr("V", 9, _V4L2Buffer)
VIDIOC_QBUF: Final = _iowr("V", 15, _V4L2Buffer)
VIDIOC_DQBUF: Final = _iowr("V", 17, _V4L2Buffer)
VIDIOC_STREAMON: Final = _iow("V", 18, ctypes.c_int)
VIDIOC_STREAMOFF: Final = _iow("V", 19, ctypes.c_int)
VIDIOC_S_PARM: Final = _iowr("V", 22, _V4L2StreamParm)


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PiPoseConfigurationError(f"{field_name} must be a positive integer")
    return value


def _finite_timeout(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PiPoseConfigurationError(f"{field_name} must be a number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0 or timeout > _MAX_CAPTURE_TIMEOUT_SECONDS:
        raise PiPoseConfigurationError(f"{field_name} must be positive and at most 0.5")
    return timeout


def _absolute_path(value: object, *, field_name: str) -> Path:
    try:
        path = Path(value).expanduser()  # type: ignore[arg-type]
    except TypeError as exc:
        raise PiPoseConfigurationError(f"{field_name} must be a path") from exc
    if not path.is_absolute():
        raise PiPoseConfigurationError(f"{field_name} must be absolute")
    return path


def _env_text(environment: Mapping[str, str], name: str, default: Path) -> str:
    raw = environment.get(name, str(default))
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
        raise PiPoseConfigurationError(f"{name} must be configured without whitespace")
    if any(ord(character) < 32 for character in raw):
        raise PiPoseConfigurationError(f"{name} contains invalid characters")
    return raw


def _env_integer(environment: Mapping[str, str], name: str, default: int) -> int:
    try:
        value = int(environment.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise PiPoseConfigurationError(f"{name} must be an integer") from exc
    return _positive_integer(value, field_name=name)


def _env_float(environment: Mapping[str, str], name: str, default: float) -> float:
    try:
        value = float(environment.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise PiPoseConfigurationError(f"{name} must be a number") from exc
    return _finite_timeout(value, field_name=name)


def _environment_start_timeout(environment: Mapping[str, str]) -> float:
    name = "RECOVERYBOX_LOCAL_POSE_START_TIMEOUT_SECONDS"
    try:
        value = float(environment.get(name, str(_DEFAULT_WORKER_START_TIMEOUT_SECONDS)))
    except (TypeError, ValueError) as exc:
        raise PiPoseConfigurationError(f"{name} must be a number") from exc
    if not math.isfinite(value) or not 0 < value <= _MAX_WORKER_START_TIMEOUT_SECONDS:
        raise PiPoseConfigurationError(f"{name} must be positive and at most 29")
    return value


def _default_ncnn_config() -> NcnnPoseConfig:
    return NcnnPoseConfig(
        runtime_path=DEFAULT_NCNN_RUNTIME_PATH,
        rtmpose_param_path=DEFAULT_NCNN_MODEL_DIRECTORY / "rtmpose-t.param",
        rtmpose_bin_path=DEFAULT_NCNN_MODEL_DIRECTORY / "rtmpose-t.bin",
        nanodet_param_path=DEFAULT_NCNN_MODEL_DIRECTORY / "nanodet-m-int8.param",
        nanodet_bin_path=DEFAULT_NCNN_MODEL_DIRECTORY / "nanodet-m-int8.bin",
    )


@dataclass(frozen=True, slots=True)
class V4L2NcnnPoseConfig:
    """Closed capture/conversion/inference configuration for the Pi service."""

    camera_device: Path = DEFAULT_CAMERA_DEVICE
    libyuv_library_path: Path = DEFAULT_LIBYUV_LIBRARY_PATH
    width: int = 640
    height: int = 480
    frames_per_second: int = 10
    capture_timeout_seconds: float = 0.1
    worker_timeout_seconds: float = 0.5
    worker_start_timeout_seconds: float = _DEFAULT_WORKER_START_TIMEOUT_SECONDS
    buffer_count: int = 8
    ncnn: NcnnPoseConfig = field(default_factory=lambda: _default_ncnn_config())

    def __post_init__(self) -> None:
        for field_name in ("camera_device", "libyuv_library_path"):
            object.__setattr__(
                self,
                field_name,
                _absolute_path(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("width", "height", "frames_per_second", "buffer_count"):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name=field_name),
            )
        if self.width % 2:
            raise PiPoseConfigurationError("width must be even for YUYV capture")
        if not _MIN_MMAP_BUFFERS <= self.buffer_count <= _MAX_MMAP_BUFFERS:
            raise PiPoseConfigurationError("buffer_count must be between 2 and 8")
        minimum_buffers = math.ceil(_NCNN_QUEUE_COVERAGE_SECONDS * self.frames_per_second) + 1
        if self.buffer_count < minimum_buffers:
            raise PiPoseConfigurationError(
                "buffer_count is too small for the configured FPS and NCNN budget"
            )
        object.__setattr__(
            self,
            "capture_timeout_seconds",
            _finite_timeout(
                self.capture_timeout_seconds,
                field_name="capture_timeout_seconds",
            ),
        )
        object.__setattr__(
            self,
            "worker_timeout_seconds",
            _finite_timeout(self.worker_timeout_seconds, field_name="worker_timeout_seconds"),
        )
        if (
            isinstance(self.worker_start_timeout_seconds, bool)
            or not isinstance(self.worker_start_timeout_seconds, (int, float))
            or not math.isfinite(self.worker_start_timeout_seconds)
            or not 0 < self.worker_start_timeout_seconds <= _MAX_WORKER_START_TIMEOUT_SECONDS
        ):
            raise PiPoseConfigurationError(
                "worker_start_timeout_seconds must be positive and at most 29"
            )
        object.__setattr__(
            self,
            "worker_start_timeout_seconds",
            float(self.worker_start_timeout_seconds),
        )
        if not isinstance(self.ncnn, NcnnPoseConfig):
            raise TypeError("ncnn must be an NcnnPoseConfig")

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> V4L2NcnnPoseConfig:
        env = os.environ if environment is None else environment
        ncnn_environment = dict(env)
        ncnn_environment.setdefault("RECOVERYBOX_NCNN_RUNTIME_PATH", str(DEFAULT_NCNN_RUNTIME_PATH))
        ncnn_environment.setdefault(
            "RECOVERYBOX_RTMPOSE_PARAM_PATH",
            str(DEFAULT_NCNN_MODEL_DIRECTORY / "rtmpose-t.param"),
        )
        ncnn_environment.setdefault(
            "RECOVERYBOX_RTMPOSE_BIN_PATH",
            str(DEFAULT_NCNN_MODEL_DIRECTORY / "rtmpose-t.bin"),
        )
        ncnn_environment.setdefault(
            "RECOVERYBOX_NANODET_PARAM_PATH",
            str(DEFAULT_NCNN_MODEL_DIRECTORY / "nanodet-m-int8.param"),
        )
        ncnn_environment.setdefault(
            "RECOVERYBOX_NANODET_BIN_PATH",
            str(DEFAULT_NCNN_MODEL_DIRECTORY / "nanodet-m-int8.bin"),
        )
        return cls(
            camera_device=Path(_env_text(env, "RECOVERYBOX_CAMERA_DEVICE", DEFAULT_CAMERA_DEVICE)),
            libyuv_library_path=Path(
                _env_text(env, "RECOVERYBOX_LIBYUV_LIBRARY_PATH", DEFAULT_LIBYUV_LIBRARY_PATH)
            ),
            width=_env_integer(env, "RECOVERYBOX_CAMERA_WIDTH", 640),
            height=_env_integer(env, "RECOVERYBOX_CAMERA_HEIGHT", 480),
            frames_per_second=_env_integer(env, "RECOVERYBOX_CAMERA_FPS", 10),
            capture_timeout_seconds=_env_float(
                env,
                "RECOVERYBOX_POSE_FRAME_TIMEOUT_SECONDS",
                0.1,
            ),
            worker_timeout_seconds=_env_float(
                env,
                "RECOVERYBOX_LOCAL_POSE_WORKER_TIMEOUT_SECONDS",
                0.5,
            ),
            worker_start_timeout_seconds=_environment_start_timeout(env),
            buffer_count=_env_integer(env, "RECOVERYBOX_V4L2_BUFFER_COUNT", 8),
            ncnn=NcnnPoseConfig.from_environment(ncnn_environment),
        )


Ioctl = Callable[[int, int, ctypes.Structure | ctypes._SimpleCData], None]
OpenDevice = Callable[[str, int], int]
CloseDevice = Callable[[int], None]
Fstat = Callable[[int], os.stat_result]
MmapFactory = Callable[[int, int, int, int, int], mmap.mmap]
ReadableWaiter = Callable[[int, float], bool]
LibraryLoader = Callable[[str], object]


def _ioctl(fd: int, request: int, value: ctypes.Structure | ctypes._SimpleCData) -> None:
    fcntl.ioctl(fd, request, value, True)


def _mmap_buffer(fd: int, length: int, flags: int, protection: int, offset: int) -> mmap.mmap:
    return mmap.mmap(fd, length, flags=flags, prot=protection, offset=offset)


def _wait_readable(fd: int, timeout_seconds: float) -> bool:
    readable, _, _ = select.select([fd], [], [], timeout_seconds)
    return bool(readable)


@dataclass(frozen=True, slots=True)
class V4L2Dependencies:
    """Injectable kernel boundaries; defaults stay in the current process."""

    open_device: OpenDevice = os.open
    close_device: CloseDevice = os.close
    fstat: Fstat = os.fstat
    ioctl: Ioctl = _ioctl
    mmap_buffer: MmapFactory = _mmap_buffer
    wait_readable: ReadableWaiter = _wait_readable
    clock: Callable[[], float] = time.monotonic


class _MappedBuffer(Protocol):
    def __len__(self) -> int: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _FrameLease:
    """Private, short-lived view of one dequeued kernel buffer."""

    owner: V4L2MmapCamera
    index: int
    mapped: _MappedBuffer
    bytes_used: int
    captured_monotonic: float
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.owner._requeue(self.index)

    def __enter__(self) -> _FrameLease:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class V4L2MmapCamera:
    """Direct nonblocking YUYV streaming capture with mmap buffer ownership."""

    def __init__(
        self,
        config: V4L2NcnnPoseConfig,
        *,
        dependencies: V4L2Dependencies | None = None,
    ) -> None:
        if not isinstance(config, V4L2NcnnPoseConfig):
            raise TypeError("config must be a V4L2NcnnPoseConfig")
        selected_dependencies = dependencies if dependencies is not None else V4L2Dependencies()
        if not isinstance(selected_dependencies, V4L2Dependencies):
            raise TypeError("dependencies must be V4L2Dependencies")
        self._config = config
        self._dependencies = selected_dependencies
        self._fd: int | None = None
        self._buffers: list[_MappedBuffer] = []
        self._streaming = False
        self._bytes_per_line = 0

    @property
    def bytes_per_line(self) -> int:
        if not self._streaming:
            raise PiPoseRuntimeError("V4L2 camera is not streaming")
        return self._bytes_per_line

    def _call_ioctl(
        self,
        request: int,
        value: ctypes.Structure | ctypes._SimpleCData,
        *,
        failure: str,
    ) -> None:
        if self._fd is None:
            raise PiPoseRuntimeError("V4L2 camera is not open")
        try:
            self._dependencies.ioctl(self._fd, request, value)
        except OSError as exc:
            raise PiPoseRuntimeError(failure) from exc

    @staticmethod
    def _buffer(index: int) -> _V4L2Buffer:
        value = _V4L2Buffer()
        value.index = index
        value.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        value.memory = V4L2_MEMORY_MMAP
        return value

    def open(self) -> None:
        if self._fd is not None:
            raise RuntimeError("V4L2 camera is already open")
        flags = (
            os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fd = self._dependencies.open_device(str(self._config.camera_device), flags)
        except OSError as exc:
            raise PiPoseRuntimeError("V4L2 camera could not be opened") from exc
        self._fd = fd
        try:
            metadata = self._dependencies.fstat(fd)
            if not stat.S_ISCHR(metadata.st_mode):
                raise PiPoseRuntimeError("V4L2 camera must be a character device")
            capability = _V4L2Capability()
            self._call_ioctl(VIDIOC_QUERYCAP, capability, failure="V4L2 capability query failed")
            caps = (
                capability.device_caps
                if capability.capabilities & V4L2_CAP_DEVICE_CAPS
                else capability.capabilities
            )
            required = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING
            if caps & required != required:
                raise PiPoseRuntimeError("V4L2 device lacks capture or streaming support")

            pixel_format = _V4L2Format()
            pixel_format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            pixel_format.fmt.pix.width = self._config.width
            pixel_format.fmt.pix.height = self._config.height
            pixel_format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV
            pixel_format.fmt.pix.field = V4L2_FIELD_NONE
            self._call_ioctl(VIDIOC_S_FMT, pixel_format, failure="V4L2 YUYV format setup failed")
            negotiated = pixel_format.fmt.pix
            if (
                negotiated.width != self._config.width
                or negotiated.height != self._config.height
                or negotiated.pixelformat != V4L2_PIX_FMT_YUYV
            ):
                raise PiPoseRuntimeError("V4L2 device did not accept the exact YUYV format")
            minimum_stride = self._config.width * 2
            self._bytes_per_line = max(minimum_stride, int(negotiated.bytesperline))
            minimum_image_size = self._bytes_per_line * self._config.height
            if int(negotiated.sizeimage) < minimum_image_size:
                raise PiPoseRuntimeError("V4L2 negotiated image size is truncated")

            parameters = _V4L2StreamParm()
            parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            parameters.parm.capture.timeperframe.numerator = 1
            parameters.parm.capture.timeperframe.denominator = self._config.frames_per_second
            self._call_ioctl(VIDIOC_S_PARM, parameters, failure="V4L2 frame-rate setup failed")
            numerator = int(parameters.parm.capture.timeperframe.numerator)
            denominator = int(parameters.parm.capture.timeperframe.denominator)
            if numerator < 1 or denominator < 1:
                raise PiPoseRuntimeError("V4L2 returned an invalid capture cadence")
            negotiated_fps = denominator / numerator
            if (
                not math.isfinite(negotiated_fps)
                or negotiated_fps > self._config.frames_per_second * 1.01
            ):
                raise PiPoseRuntimeError("V4L2 capture cadence exceeds the configured bound")

            requested = _V4L2RequestBuffers()
            requested.count = self._config.buffer_count
            requested.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            requested.memory = V4L2_MEMORY_MMAP
            self._call_ioctl(VIDIOC_REQBUFS, requested, failure="V4L2 mmap request failed")
            if requested.count < self._config.buffer_count or requested.count > _MAX_MMAP_BUFFERS:
                raise PiPoseRuntimeError("V4L2 returned an unsafe mmap buffer count")
            for index in range(requested.count):
                query = self._buffer(index)
                self._call_ioctl(VIDIOC_QUERYBUF, query, failure="V4L2 buffer query failed")
                if query.length < minimum_image_size:
                    raise PiPoseRuntimeError("V4L2 mmap buffer is too short")
                try:
                    mapped = self._dependencies.mmap_buffer(
                        fd,
                        query.length,
                        mmap.MAP_SHARED,
                        mmap.PROT_READ | mmap.PROT_WRITE,
                        query.m.offset,
                    )
                except (OSError, ValueError) as exc:
                    raise PiPoseRuntimeError("V4L2 buffer mapping failed") from exc
                self._buffers.append(mapped)
                self._call_ioctl(VIDIOC_QBUF, query, failure="V4L2 initial queue failed")

            stream_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
            self._call_ioctl(VIDIOC_STREAMON, stream_type, failure="V4L2 stream start failed")
            self._streaming = True
        except BaseException:
            self.close()
            raise

    def dequeue(self) -> _FrameLease | None:
        fd = self._fd
        if fd is None or not self._streaming:
            raise PiPoseRuntimeError("V4L2 camera is not streaming")
        acquisition_started = self._checked_clock()
        deadline = acquisition_started + self._config.capture_timeout_seconds
        # Only a kernel-declared monotonic timestamp is usable as fresh pose
        # evidence.  Zero is the conservative lower bound for an older driver;
        # at ordinary uptime NCNN will reject it as stale rather than letting an
        # optimistic dequeue-time estimate advance exercise state.
        fallback_timestamp = 0.0
        candidate = self._wait_and_dequeue(
            self._config.capture_timeout_seconds,
            fallback_timestamp=fallback_timestamp,
            timestamp_upper_bound=acquisition_started,
        )
        if candidate is None:
            return None

        # NCNN takes longer than one camera interval on Pi 3.  Drain every
        # buffer that is already ready without blocking, returning only the
        # newest timestamped lease and immediately requeueing all older ones.
        # The estimator's independent 500 ms age gate remains authoritative if
        # even that newest frame is too old.
        for _ in range(max(0, len(self._buffers) - 1)):
            if not self._wait_for_readable(0.0):
                break
            newer = self._dequeue_one(
                fallback_timestamp=fallback_timestamp,
                timestamp_upper_bound=acquisition_started,
            )
            if newer is None:
                break
            candidate.release()
            candidate = newer
        finished = self._checked_clock()
        if finished < acquisition_started:
            candidate.release()
            raise PiPoseRuntimeError("V4L2 monotonic clock moved backwards")
        if finished > deadline:
            candidate.release()
            return None
        return candidate

    def _checked_clock(self) -> float:
        value = self._dependencies.clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise PiPoseRuntimeError("V4L2 monotonic clock is invalid")
        return float(value)

    def _wait_for_readable(self, timeout_seconds: float) -> bool:
        fd = self._fd
        if fd is None:
            raise PiPoseRuntimeError("V4L2 camera is not open")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 <= timeout_seconds <= _MAX_CAPTURE_TIMEOUT_SECONDS
        ):
            raise PiPoseRuntimeError("V4L2 readiness timeout is invalid")
        try:
            return self._dependencies.wait_readable(fd, float(timeout_seconds))
        except (OSError, ValueError) as exc:
            raise PiPoseRuntimeError("V4L2 readiness wait failed") from exc

    def _wait_and_dequeue(
        self,
        timeout_seconds: float,
        *,
        fallback_timestamp: float,
        timestamp_upper_bound: float,
    ) -> _FrameLease | None:
        if not self._wait_for_readable(timeout_seconds):
            return None
        return self._dequeue_one(
            fallback_timestamp=fallback_timestamp,
            timestamp_upper_bound=timestamp_upper_bound,
        )

    def _dequeue_one(
        self,
        *,
        fallback_timestamp: float,
        timestamp_upper_bound: float,
    ) -> _FrameLease | None:
        fd = self._fd
        if fd is None:
            raise PiPoseRuntimeError("V4L2 camera is not open")
        buffer = self._buffer(0)
        try:
            self._dependencies.ioctl(fd, VIDIOC_DQBUF, buffer)
        except OSError as exc:
            if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                return None
            raise PiPoseRuntimeError("V4L2 dequeue failed") from exc
        if buffer.index >= len(self._buffers):
            raise PiPoseRuntimeError("V4L2 dequeued an invalid buffer index")
        mapped = self._buffers[buffer.index]
        required = self._bytes_per_line * self._config.height
        if buffer.bytesused < required or buffer.bytesused > len(mapped):
            self._requeue(buffer.index)
            raise PiPoseRuntimeError("V4L2 dequeued a truncated frame")
        if buffer.flags & V4L2_BUF_FLAG_ERROR:
            self._requeue(buffer.index)
            return None
        captured = fallback_timestamp
        if buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC:
            kernel_timestamp = (
                float(buffer.timestamp.tv_sec) + float(buffer.timestamp.tv_usec) / 1e6
            )
            if math.isfinite(kernel_timestamp) and kernel_timestamp >= 0:
                # Never make evidence newer than the instant immediately before
                # the bounded wait.  That is conservative even when a driver
                # stamps a buffer during the wait or has a slightly skewed clock.
                captured = min(timestamp_upper_bound, kernel_timestamp)
        return _FrameLease(
            owner=self,
            index=buffer.index,
            mapped=mapped,
            bytes_used=buffer.bytesused,
            captured_monotonic=captured,
        )

    def _requeue(self, index: int) -> None:
        if self._fd is None or not self._streaming:
            return
        self._call_ioctl(
            VIDIOC_QBUF,
            self._buffer(index),
            failure="V4L2 buffer requeue failed",
        )

    def close(self) -> None:
        fd = self._fd
        if fd is None:
            return
        if self._streaming:
            try:
                stream_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
                self._dependencies.ioctl(fd, VIDIOC_STREAMOFF, stream_type)
            except OSError:
                pass
        self._streaming = False
        for mapped in self._buffers:
            try:
                mapped.close()
            except (BufferError, OSError):
                pass
        self._buffers.clear()
        self._fd = None
        try:
            self._dependencies.close_device(fd)
        except OSError:
            pass


class LibyuvYuy2ToBgra:
    """Same-process libyuv YUY2-to-BGRA conversion."""

    def __init__(
        self,
        library_path: Path,
        *,
        library_loader: LibraryLoader = ctypes.CDLL,
    ) -> None:
        checked = _absolute_path(library_path, field_name="library_path")
        try:
            library = library_loader(str(checked))
            convert = library.YUY2ToARGB  # type: ignore[attr-defined]
            convert.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            convert.restype = ctypes.c_int
        except (AttributeError, OSError, TypeError) as exc:
            raise PiPoseConfigurationError("libyuv YUY2 conversion is unavailable") from exc
        self._convert = convert

    def convert(
        self,
        source: _MappedBuffer,
        *,
        source_stride: int,
        width: int,
        height: int,
    ) -> bytes:
        frame_width = _positive_integer(width, field_name="width")
        frame_height = _positive_integer(height, field_name="height")
        stride = _positive_integer(source_stride, field_name="source_stride")
        if frame_width % 2 or stride < frame_width * 2 or len(source) < stride * frame_height:
            raise PiPoseRuntimeError("YUYV frame layout is invalid")
        try:
            source_array = (ctypes.c_uint8 * len(source)).from_buffer(source)  # type: ignore[arg-type]
        except (TypeError, BufferError, ValueError) as exc:
            raise PiPoseRuntimeError("YUYV mmap buffer is inaccessible") from exc
        output = bytearray(frame_width * frame_height * 4)
        output_array = (ctypes.c_uint8 * len(output)).from_buffer(output)
        result = self._convert(
            source_array,
            stride,
            output_array,
            frame_width * 4,
            frame_width,
            frame_height,
        )
        del source_array, output_array
        if result != 0:
            raise PiPoseRuntimeError("libyuv YUY2 conversion failed")
        # libyuv's ARGB byte layout on the little-endian Pi is BGRA, matching
        # the explicit NCNN PIXEL_BGRA input contract.
        return bytes(output)


class _Camera(Protocol):
    @property
    def bytes_per_line(self) -> int: ...

    def open(self) -> None: ...

    def dequeue(self) -> _FrameLease | None: ...

    def close(self) -> None: ...


class _Converter(Protocol):
    def convert(
        self,
        source: _MappedBuffer,
        *,
        source_stride: int,
        width: int,
        height: int,
    ) -> bytes: ...


class _Estimator(Protocol):
    def infer_bgra(
        self,
        bgra_frame: bytes,
        *,
        width: int,
        height: int,
        captured_monotonic: float,
    ) -> PoseInferenceResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class V4L2NcnnPoseObservation:
    """Numeric-only output of one local capture/inference/tracking cycle."""

    analysis: SquatAnalysis
    frame_received: bool
    detector_ms: float | None
    pose_ms: float | None
    inference_ms: float | None
    evidence_age_ms: float | None
    person_score: float | None
    timed_out: bool
    capture_missed: bool
    worker_timed_out: bool
    parent_stale: bool

    def __post_init__(self) -> None:
        if not isinstance(self.analysis, SquatAnalysis):
            raise TypeError("analysis must be a SquatAnalysis")
        for field_name in (
            "frame_received",
            "timed_out",
            "capture_missed",
            "worker_timed_out",
            "parent_stale",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError("observation flags must be booleans")
        for field_name in (
            "detector_ms",
            "pose_ms",
            "inference_ms",
            "evidence_age_ms",
            "person_score",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            converted = float(value)
            if not math.isfinite(converted) or converted < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            if field_name == "person_score" and converted > 1:
                raise ValueError("person_score must not exceed one")
            object.__setattr__(self, field_name, converted)
        if self.capture_missed and self.frame_received:
            raise ValueError("a missed capture may not claim a received frame")
        if self.worker_timed_out and self.frame_received:
            raise ValueError("a worker timeout may not claim a received frame")
        failure_flagged = self.capture_missed or self.worker_timed_out or self.parent_stale
        if failure_flagged and not self.timed_out:
            raise ValueError("failed or stale observations must be timed out")
        if self.parent_stale and self.person_score is not None:
            raise ValueError("stale observations may not expose a person score")
        if self.frame_received:
            if (
                self.detector_ms is None
                or self.inference_ms is None
                or self.evidence_age_ms is None
            ):
                raise ValueError("a received frame requires complete detector and age timing")
            if (
                self.detector_ms > self.inference_ms
                or (self.pose_ms is not None and self.pose_ms > self.inference_ms)
                or self.inference_ms > self.evidence_age_ms
            ):
                raise ValueError("observation timing order is invalid")
        elif any(
            value is not None for value in (self.detector_ms, self.pose_ms, self.inference_ms)
        ) or (self.evidence_age_ms is not None and not self.parent_stale):
            raise ValueError("an unreceived frame may not expose inference timing")


class _InProcessV4L2NcnnPoseSource:
    """Child-only composition of V4L2, libyuv, NCNN, and squat tracking."""

    def __init__(
        self,
        config: V4L2NcnnPoseConfig,
        *,
        camera: _Camera,
        converter: _Converter,
        estimator: _Estimator,
        tracker: SquatTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, V4L2NcnnPoseConfig):
            raise TypeError("config must be a V4L2NcnnPoseConfig")
        self._config = config
        self._camera = camera
        self._converter = converter
        self._estimator = estimator
        self._tracker = tracker if tracker is not None else SquatTracker()
        self._clock = clock
        self._opened = False

    @classmethod
    def from_config(cls, config: V4L2NcnnPoseConfig) -> _InProcessV4L2NcnnPoseSource:
        if not isinstance(config, V4L2NcnnPoseConfig):
            raise TypeError("config must be a V4L2NcnnPoseConfig")
        return cls(
            config,
            camera=V4L2MmapCamera(config),
            converter=LibyuvYuy2ToBgra(config.libyuv_library_path),
            estimator=NcnnPersonPoseEstimator(config.ncnn),
        )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> _InProcessV4L2NcnnPoseSource:
        return cls.from_config(V4L2NcnnPoseConfig.from_environment(environment))

    @staticmethod
    def _timestamp_ms(value: float) -> int:
        if not math.isfinite(value) or value < 0:
            raise PiPoseRuntimeError("capture timestamp is invalid")
        return int(value * 1000)

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("pose source is already open")
        try:
            self._camera.open()
        except BaseException:
            self._estimator.close()
            raise
        self._opened = True

    def read(self) -> V4L2NcnnPoseObservation:
        if not self._opened:
            raise RuntimeError("pose source is not open")
        acquisition_started = self._clock()
        if (
            isinstance(acquisition_started, bool)
            or not isinstance(acquisition_started, (int, float))
            or not math.isfinite(acquisition_started)
            or acquisition_started < 0
        ):
            raise PiPoseRuntimeError("pose source clock is invalid")
        lease = self._camera.dequeue()
        if lease is None:
            timestamp_ms = self._timestamp_ms(float(acquisition_started))
            return V4L2NcnnPoseObservation(
                analysis=self._tracker.update_missing(
                    timestamp_ms,
                    issue=SquatAssessmentIssue.CAMERA_TIMEOUT,
                ),
                frame_received=False,
                detector_ms=None,
                pose_ms=None,
                inference_ms=None,
                evidence_age_ms=None,
                person_score=None,
                timed_out=True,
                capture_missed=True,
                worker_timed_out=False,
                parent_stale=False,
            )
        with lease:
            bgra = self._converter.convert(
                lease.mapped,
                source_stride=self._camera.bytes_per_line,
                width=self._config.width,
                height=self._config.height,
            )
        # Conversion made an immutable BGRA copy.  Requeue the mmap lease before
        # the slow NCNN call so all kernel buffers can capture newer frames
        # while inference runs; raw bytes still remain in this child process.
        result = self._estimator.infer_bgra(
            bgra,
            width=self._config.width,
            height=self._config.height,
            captured_monotonic=lease.captured_monotonic,
        )
        timestamp_ms = self._timestamp_ms(lease.captured_monotonic)
        if result.points is None:
            analysis = self._tracker.update_missing(
                timestamp_ms,
                issue=(
                    SquatAssessmentIssue.CAMERA_TIMEOUT
                    if result.timed_out
                    else SquatAssessmentIssue.NO_POSE
                ),
            )
        else:
            pose_frame = movenet_to_mediapipe_frame(result.points, timestamp_ms=timestamp_ms)
            pose_frame = replace(
                pose_frame,
                image_width=self._config.width,
                image_height=self._config.height,
            )
            analysis = self._tracker.update(pose_frame)
        return V4L2NcnnPoseObservation(
            analysis=analysis,
            frame_received=True,
            detector_ms=result.detector_ms,
            pose_ms=result.pose_ms,
            inference_ms=result.total_ms,
            evidence_age_ms=result.evidence_age_ms,
            person_score=result.person_score,
            timed_out=result.timed_out,
            capture_missed=False,
            worker_timed_out=False,
            parent_stale=False,
        )

    def close(self) -> None:
        self._opened = False
        try:
            self._camera.close()
        finally:
            self._estimator.close()

    def __enter__(self) -> _InProcessV4L2NcnnPoseSource:
        self.open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()


class _PipeConnection(Protocol):
    def send(self, value: object) -> None: ...

    def recv(self) -> object: ...

    def poll(self, timeout: float = 0.0) -> bool: ...

    def close(self) -> None: ...


class _ChildProcess(Protocol):
    exitcode: int | None

    def start(self) -> None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class _ProcessContext(Protocol):
    def Pipe(self, duplex: bool = True) -> tuple[_PipeConnection, _PipeConnection]: ...

    def Process(self, **kwargs: object) -> _ChildProcess: ...


def _run_pose_child(connection: _PipeConnection, config: V4L2NcnnPoseConfig) -> None:
    """Own all raw camera bytes and inference state in one killable process."""

    source: _InProcessV4L2NcnnPoseSource | None = None
    try:
        try:
            source = _InProcessV4L2NcnnPoseSource.from_config(config)
            source.open()
        except BaseException as exc:
            try:
                connection.send(("failure", type(exc).__name__))
            except (BrokenPipeError, EOFError, OSError):
                pass
            return
        connection.send(("ready", None))
        while True:
            try:
                command = connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                return
            if command == "close":
                return
            if command != "read":
                connection.send(("failure", "LocalPoseCommandError"))
                return
            try:
                observation = source.read()
            except BaseException as exc:
                try:
                    connection.send(("failure", type(exc).__name__))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                return
            # The IPC contract admits only the immutable numeric observation;
            # mmap buffers and converted bytes remain unreachable in this child.
            connection.send(("observation", observation))
    finally:
        if source is not None:
            try:
                source.close()
            except BaseException:
                pass
        connection.close()


class _NumericPoseWorker(Protocol):
    @property
    def failure_kind(self) -> str | None: ...

    def start(self, timeout_seconds: float) -> None: ...

    def read(self, timeout_seconds: float) -> V4L2NcnnPoseObservation | None: ...

    def close(self) -> None: ...


class _MultiprocessingPoseWorker:
    """Bounded parent endpoint for the child that owns capture and NCNN."""

    def __init__(
        self,
        config: V4L2NcnnPoseConfig,
        *,
        context: _ProcessContext | None = None,
    ) -> None:
        self._config = config
        self._context = (
            context if context is not None else multiprocessing.get_context("spawn")  # type: ignore[assignment]
        )
        self._connection: _PipeConnection | None = None
        self._process: _ChildProcess | None = None
        self._failure_kind: str | None = None

    @property
    def failure_kind(self) -> str | None:
        return self._failure_kind

    def _terminate_and_reap(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(0.5)
        if process.is_alive():
            process.kill()
            process.join(0.5)
        else:
            process.join(0)

    def _fail(self, kind: str, *, reap: bool = True) -> None:
        self._failure_kind = kind
        if reap:
            self._terminate_and_reap()
            return
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(0)

    @staticmethod
    def _decode_message(value: object) -> tuple[str, object]:
        if type(value) is not tuple or len(value) != 2 or not isinstance(value[0], str):
            raise PiPoseRuntimeError("local pose child returned an invalid message")
        return value[0], value[1]

    def start(self, timeout_seconds: float) -> None:
        if self._process is not None:
            raise RuntimeError("local pose child is already started")
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_run_pose_child,
            args=(child, self._config),
            name="recoverybox-v4l2-ncnn",
            daemon=True,
        )
        self._connection = parent
        self._process = process
        try:
            process.start()
        except BaseException as exc:
            parent.close()
            child.close()
            self._failure_kind = "LocalPoseWorkerStartError"
            raise PiPoseRuntimeError("local pose child could not start") from exc
        child.close()
        try:
            ready = parent.poll(timeout_seconds)
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._fail("LocalPoseWorkerStartError")
            raise PiPoseRuntimeError("local pose child failed during startup") from exc
        if not ready:
            self._fail("LocalPoseWorkerStartTimeout")
            raise PiPoseRuntimeError("local pose child did not become ready")
        try:
            kind, payload = self._decode_message(parent.recv())
        except (BrokenPipeError, EOFError, OSError, PiPoseRuntimeError) as exc:
            self._fail("LocalPoseWorkerStartError")
            raise PiPoseRuntimeError("local pose child failed during startup") from exc
        if kind != "ready" or payload is not None:
            failure = payload if kind == "failure" and isinstance(payload, str) else None
            self._fail(failure or "LocalPoseWorkerStartError")
            raise PiPoseRuntimeError("local pose child failed during startup")

    def read(self, timeout_seconds: float) -> V4L2NcnnPoseObservation | None:
        connection = self._connection
        process = self._process
        if connection is None or process is None or self._failure_kind is not None:
            return None
        if not process.is_alive():
            self._fail("LocalPoseWorkerExited", reap=False)
            return None
        try:
            connection.send("read")
            if not connection.poll(timeout_seconds):
                # Do not spend more of the Guardian deadline waiting for reap;
                # terminate now, return a numeric missing observation, and let
                # close() perform the bounded join/kill cleanup.
                self._fail("LocalPoseWorkerTimeout", reap=False)
                return None
            kind, payload = self._decode_message(connection.recv())
        except (BrokenPipeError, EOFError, OSError, PiPoseRuntimeError):
            self._fail("LocalPoseWorkerProtocolError", reap=False)
            return None
        if kind == "failure":
            self._fail(
                payload if isinstance(payload, str) else "LocalPoseWorkerReadError",
                reap=False,
            )
            return None
        if kind != "observation" or not isinstance(payload, V4L2NcnnPoseObservation):
            self._fail("LocalPoseWorkerContractError", reap=False)
            return None
        return payload

    def close(self) -> None:
        connection = self._connection
        process = self._process
        if connection is not None and process is not None and process.is_alive():
            try:
                connection.send("close")
                process.join(0.25)
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._terminate_and_reap()
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        self._connection = None


class V4L2NcnnPoseSource:
    """Bounded numeric proxy to a child owning V4L2 capture and NCNN.

    NCNN's Python extraction can retain the GIL indefinitely.  Keeping both raw
    capture and inference in one dedicated process preserves the raw-frame
    boundary while allowing this parent to enforce a hard 500 ms read deadline.
    """

    def __init__(
        self,
        config: V4L2NcnnPoseConfig,
        *,
        worker: _NumericPoseWorker,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, V4L2NcnnPoseConfig):
            raise TypeError("config must be a V4L2NcnnPoseConfig")
        self._config = config
        self._worker = worker
        self._clock = clock
        self._opened = False
        self._failed = False
        self._last_analysis: SquatAnalysis | None = None
        self._last_timestamp_ms: int | None = None
        self._last_clock: float | None = None

    @classmethod
    def from_config(cls, config: V4L2NcnnPoseConfig) -> V4L2NcnnPoseSource:
        if not isinstance(config, V4L2NcnnPoseConfig):
            raise TypeError("config must be a V4L2NcnnPoseConfig")
        return cls(config, worker=_MultiprocessingPoseWorker(config))

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> V4L2NcnnPoseSource:
        return cls.from_config(V4L2NcnnPoseConfig.from_environment(environment))

    def _now_ms(self) -> int:
        now = self._clock()
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
            or now < 0
        ):
            raise PiPoseRuntimeError("local pose proxy clock is invalid")
        checked = float(now)
        if self._last_clock is not None and checked < self._last_clock:
            raise PiPoseRuntimeError("local pose proxy clock moved backwards")
        self._last_clock = checked
        timestamp = int(checked * 1000)
        if self._last_timestamp_ms is not None:
            timestamp = max(timestamp, self._last_timestamp_ms + 1)
        return timestamp

    def _missing_analysis(self, timestamp_ms: int) -> SquatAnalysis:
        analysis = SquatAnalysis(
            timestamp_ms=timestamp_ms,
            assessable=False,
            phase=SquatPhase.UNKNOWN,
            rep_count=(self._last_analysis.rep_count if self._last_analysis is not None else 0),
            events=(),
            issues=(SquatAssessmentIssue.CAMERA_TIMEOUT,),
            confidence=0.0,
            knee_angle_degrees=None,
            arms_in_t=None,
        )
        self._last_analysis = analysis
        self._last_timestamp_ms = timestamp_ms
        return analysis

    def _missing(self, *, worker_timed_out: bool) -> V4L2NcnnPoseObservation:
        timestamp_ms = self._now_ms()
        return V4L2NcnnPoseObservation(
            analysis=self._missing_analysis(timestamp_ms),
            frame_received=False,
            detector_ms=None,
            pose_ms=None,
            inference_ms=None,
            evidence_age_ms=None,
            person_score=None,
            timed_out=True,
            capture_missed=False,
            worker_timed_out=worker_timed_out,
            parent_stale=False,
        )

    def _reject_stale(
        self,
        observation: V4L2NcnnPoseObservation,
        *,
        now_ms: int,
    ) -> V4L2NcnnPoseObservation:
        child_timestamp_ms = observation.analysis.timestamp_ms
        evidence_age_ms = observation.evidence_age_ms
        if child_timestamp_ms <= now_ms:
            parent_observed_age_ms = float(now_ms - child_timestamp_ms)
            evidence_age_ms = max(
                parent_observed_age_ms,
                evidence_age_ms if evidence_age_ms is not None else 0.0,
            )
        return V4L2NcnnPoseObservation(
            analysis=self._missing_analysis(now_ms),
            frame_received=observation.frame_received,
            detector_ms=observation.detector_ms,
            pose_ms=observation.pose_ms,
            inference_ms=observation.inference_ms,
            evidence_age_ms=evidence_age_ms,
            person_score=None,
            timed_out=True,
            capture_missed=observation.capture_missed,
            worker_timed_out=observation.worker_timed_out,
            parent_stale=True,
        )

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("pose source is already open")
        self._worker.start(self._config.worker_start_timeout_seconds)
        self._opened = True

    def read(self) -> V4L2NcnnPoseObservation:
        if not self._opened:
            raise RuntimeError("pose source is not open")
        if self._failed:
            raise PiPoseRuntimeError("local pose child is unavailable")
        observation = self._worker.read(self._config.worker_timeout_seconds)
        if observation is None:
            self._failed = True
            return self._missing(
                worker_timed_out=self._worker.failure_kind == "LocalPoseWorkerTimeout"
            )
        now_ms = self._now_ms()
        timestamp_ms = observation.analysis.timestamp_ms
        stale = (
            (self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms)
            or timestamp_ms > now_ms
            or now_ms - timestamp_ms >= 500
        )
        if stale:
            return self._reject_stale(observation, now_ms=now_ms)
        self._last_analysis = observation.analysis
        self._last_timestamp_ms = timestamp_ms
        return observation

    def close(self) -> None:
        self._opened = False
        self._worker.close()

    def __enter__(self) -> V4L2NcnnPoseSource:
        self.open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()


SourceFactory = Callable[[V4L2NcnnPoseConfig], V4L2NcnnPoseSource]


def run_v4l2_ncnn_pose_check(
    config: V4L2NcnnPoseConfig,
    *,
    max_frames: int,
    source_factory: SourceFactory = V4L2NcnnPoseSource.from_config,
) -> dict[str, object]:
    """Run a bounded, silent check and emit numeric-only acceptance status."""

    frame_limit = _positive_integer(max_frames, field_name="max_frames")
    source = source_factory(config)
    frames = 0
    received = 0
    fresh_frames = 0
    assessable = 0
    timeouts = 0
    capture_misses = 0
    worker_timeouts = 0
    parent_stale_count = 0
    detector_ms: list[float] = []
    pose_ms: list[float] = []
    inference_ms: list[float] = []
    evidence_age_ms: list[float] = []
    try:
        source.open()
        while frames < frame_limit:
            observation = source.read()
            frames += 1
            received += int(observation.frame_received)
            fresh_frames += int(observation.frame_received and not observation.timed_out)
            assessable += int(observation.analysis.assessable)
            timeouts += int(observation.timed_out)
            capture_misses += int(observation.capture_missed)
            worker_timeouts += int(observation.worker_timed_out)
            parent_stale_count += int(observation.parent_stale)
            if observation.detector_ms is not None:
                detector_ms.append(observation.detector_ms)
            if observation.pose_ms is not None:
                pose_ms.append(observation.pose_ms)
            if observation.inference_ms is not None:
                inference_ms.append(observation.inference_ms)
            if observation.evidence_age_ms is not None:
                evidence_age_ms.append(observation.evidence_age_ms)
    finally:
        source.close()
    return {
        "service": "recoverybox-pi-v4l2-ncnn-check/v2",
        "capture": "v4l2-mmap-yuyv",
        "conversion": "libyuv-yuy2-to-bgra",
        "estimator": "ncnn-nanodet-rtmpose",
        "frames": frames,
        "frames_received": received,
        "fresh_frames": fresh_frames,
        "assessable": assessable,
        "timeouts": timeouts,
        "capture_misses": capture_misses,
        "worker_timeouts": worker_timeouts,
        "parent_stale_count": parent_stale_count,
        "detector_ms_max": round(max(detector_ms), 3) if detector_ms else None,
        "pose_ms_max": round(max(pose_ms), 3) if pose_ms else None,
        "inference_ms_max": round(max(inference_ms), 3) if inference_ms else None,
        "evidence_age_ms_max": (round(max(evidence_age_ms), 3) if evidence_age_ms else None),
        "raw_frames_persisted": 0,
        "audio": "disabled",
    }


def _acceptance_duration(value: object, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value < 500
    ):
        raise ValueError("pose-check duration is invalid")
    return float(value)


def _successful_pose_check_report(report: Mapping[str, object], *, expected_frames: int) -> bool:
    expected = {
        "service": "recoverybox-pi-v4l2-ncnn-check/v2",
        "capture": "v4l2-mmap-yuyv",
        "conversion": "libyuv-yuy2-to-bgra",
        "estimator": "ncnn-nanodet-rtmpose",
        "frames": expected_frames,
        "frames_received": expected_frames,
        "fresh_frames": expected_frames,
        "timeouts": 0,
        "capture_misses": 0,
        "worker_timeouts": 0,
        "parent_stale_count": 0,
        "raw_frames_persisted": 0,
        "audio": "disabled",
    }
    if any(report.get(key) != value for key, value in expected.items()):
        return False
    if frozenset(report) != {
        *expected,
        "assessable",
        "detector_ms_max",
        "pose_ms_max",
        "inference_ms_max",
        "evidence_age_ms_max",
    }:
        return False
    assessable = report.get("assessable")
    if (
        isinstance(assessable, bool)
        or not isinstance(assessable, int)
        or not 0 <= assessable <= expected_frames
    ):
        return False
    try:
        detector_ms = _acceptance_duration(report.get("detector_ms_max"))
        pose_ms = _acceptance_duration(report.get("pose_ms_max"), nullable=True)
        inference_ms = _acceptance_duration(report.get("inference_ms_max"))
        evidence_age_ms = _acceptance_duration(report.get("evidence_age_ms_max"))
    except ValueError:
        return False
    assert detector_ms is not None
    assert inference_ms is not None
    assert evidence_age_ms is not None
    if assessable > 0 and pose_ms is None:
        return False
    return detector_ms <= inference_ms <= evidence_age_ms and (
        pose_ms is None or pose_ms <= inference_ms
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded silent Pi V4L2/NCNN pose check")
    parser.add_argument("--max-frames", type=int, default=3)
    arguments = parser.parse_args(argv)
    try:
        report = run_v4l2_ncnn_pose_check(
            V4L2NcnnPoseConfig.from_environment(),
            max_frames=arguments.max_frames,
        )
    except Exception as exc:
        report = {
            "service": "recoverybox-pi-v4l2-ncnn-check/v2",
            "failure": type(exc).__name__,
            "raw_frames_persisted": 0,
            "audio": "disabled",
        }
        print(json.dumps(report, sort_keys=True))
        return 1
    if not _successful_pose_check_report(report, expected_frames=arguments.max_frames):
        report = {**report, "failure": "FreshPoseEvidenceUnavailable"}
        print(json.dumps(report, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CAMERA_DEVICE",
    "DEFAULT_LIBYUV_LIBRARY_PATH",
    "DEFAULT_NCNN_MODEL_DIRECTORY",
    "DEFAULT_NCNN_RUNTIME_PATH",
    "LibyuvYuy2ToBgra",
    "V4L2Dependencies",
    "V4L2MmapCamera",
    "V4L2NcnnPoseConfig",
    "V4L2NcnnPoseObservation",
    "V4L2NcnnPoseSource",
    "main",
    "run_v4l2_ncnn_pose_check",
]
