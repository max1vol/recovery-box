"""Process-local Pi camera and MoveNet pose estimation.

The camera frame exists only as an in-memory RGB byte string between an
``ffmpeg`` pipe and TensorFlow Lite.  This module never writes, serializes, or
returns a frame.  Its public output is the existing numeric ``SquatAnalysis``
type consumed by the deterministic Guardian.

The implementation deliberately uses TensorFlow Lite's stable C API through
``ctypes``.  Raspberry Pi OS supplies that shared library on armv7, avoiding a
large Python TensorFlow/OpenCV installation on a Pi 3.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import select
import struct
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Protocol

from recoverybox.exercise import (
    MediaPipePoseFrame,
    MediaPipePoseLandmark,
    NormalizedLandmark,
    SquatAnalysis,
    SquatAssessmentIssue,
    SquatTracker,
)

MOVENET_MODEL_URL = (
    "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/"
    "tflite/int8/4?lite-format=tflite"
)
MOVENET_MODEL_SHA256 = "cd7cc22fa946e5d146a7b98d496853e1923e22828d3972d579973f27f91bb105"
MOVENET_MODEL_SIZE_BYTES = 2_894_840
MOVENET_INPUT_SIZE = 192
MOVENET_INPUT_BYTES = MOVENET_INPUT_SIZE * MOVENET_INPUT_SIZE * 3
MOVENET_KEYPOINT_COUNT = 17
MOVENET_OUTPUT_VALUES = MOVENET_KEYPOINT_COUNT * 3
DEFAULT_TFLITE_LIBRARY_PATH = Path("/usr/lib/arm-linux-gnueabihf/libtensorflow-lite.so.2.20.0")
DEFAULT_CAMERA_DEVICE = Path("/dev/video0")
DEFAULT_MOVENET_MODEL_PATH = Path(
    "/home/pi/recoverybox/models/movenet-singlepose-lightning-int8-v4.tflite"
)

_TFLITE_OK = 0
_TFLITE_FLOAT32 = 1
_TFLITE_UINT8 = 3
_EXPECTED_INPUT_DIMS = (1, MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE, 3)
_EXPECTED_OUTPUT_DIMS = (1, 1, MOVENET_KEYPOINT_COUNT, 3)


class PiPoseConfigurationError(ValueError):
    """The local camera/inference runtime is configured unsafely."""


class PiPoseRuntimeError(RuntimeError):
    """Camera or inference evidence is unavailable."""


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PiPoseConfigurationError(f"{field_name} must be a positive integer")
    return value


def _environment_text(
    environment: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    raw = environment.get(name, default)
    if not isinstance(raw, str) or not raw.strip():
        raise PiPoseConfigurationError(f"{name} must not be blank")
    value = raw.strip()
    if any(ord(character) < 32 for character in value):
        raise PiPoseConfigurationError(f"{name} contains invalid characters")
    return value


def _environment_integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw = environment.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise PiPoseConfigurationError(f"{name} must be an integer") from exc
    return _positive_integer(value, field_name=name)


def _environment_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw = environment.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise PiPoseConfigurationError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise PiPoseConfigurationError(f"{name} must be a positive finite number")
    return value


@dataclass(frozen=True, slots=True)
class PiPoseConfig:
    """Closed configuration for one Pi-local pose source."""

    camera_device: Path = DEFAULT_CAMERA_DEVICE
    model_path: Path = DEFAULT_MOVENET_MODEL_PATH
    tflite_library_path: Path = DEFAULT_TFLITE_LIBRARY_PATH
    capture_width: int = 640
    capture_height: int = 480
    frames_per_second: int = 5
    frame_timeout_seconds: float = 0.4
    inference_threads: int = 2
    ffmpeg_binary: str = "/usr/bin/ffmpeg"

    def __post_init__(self) -> None:
        for field_name in ("camera_device", "model_path", "tflite_library_path"):
            try:
                path = Path(getattr(self, field_name)).expanduser()
            except TypeError as exc:
                raise PiPoseConfigurationError(f"{field_name} must be a path") from exc
            if not path.is_absolute():
                raise PiPoseConfigurationError(f"{field_name} must be absolute")
            object.__setattr__(self, field_name, path)
        for field_name in (
            "capture_width",
            "capture_height",
            "frames_per_second",
            "inference_threads",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name=field_name),
            )
        try:
            timeout = _finite_number(
                self.frame_timeout_seconds,
                field_name="frame_timeout_seconds",
            )
        except (TypeError, ValueError) as exc:
            raise PiPoseConfigurationError(
                "frame_timeout_seconds must be a positive finite number"
            ) from exc
        if timeout <= 0 or timeout > 0.5:
            raise PiPoseConfigurationError("frame_timeout_seconds must be positive and at most 0.5")
        object.__setattr__(self, "frame_timeout_seconds", timeout)
        if not isinstance(self.ffmpeg_binary, str) or not self.ffmpeg_binary.strip():
            raise PiPoseConfigurationError("ffmpeg_binary must not be blank")
        if any(ord(character) < 32 for character in self.ffmpeg_binary):
            raise PiPoseConfigurationError("ffmpeg_binary contains invalid characters")
        if not Path(self.ffmpeg_binary).is_absolute():
            raise PiPoseConfigurationError("ffmpeg_binary must be absolute")

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> PiPoseConfig:
        env = os.environ if environment is None else environment
        return cls(
            camera_device=Path(
                _environment_text(
                    env,
                    "RECOVERYBOX_CAMERA_DEVICE",
                    str(DEFAULT_CAMERA_DEVICE),
                )
            ),
            model_path=Path(
                _environment_text(
                    env,
                    "RECOVERYBOX_MOVENET_MODEL_PATH",
                    str(DEFAULT_MOVENET_MODEL_PATH),
                )
            ),
            tflite_library_path=Path(
                _environment_text(
                    env,
                    "RECOVERYBOX_TFLITE_LIBRARY_PATH",
                    str(DEFAULT_TFLITE_LIBRARY_PATH),
                )
            ),
            capture_width=_environment_integer(
                env,
                "RECOVERYBOX_CAMERA_WIDTH",
                640,
            ),
            capture_height=_environment_integer(
                env,
                "RECOVERYBOX_CAMERA_HEIGHT",
                480,
            ),
            frames_per_second=_environment_integer(
                env,
                "RECOVERYBOX_CAMERA_FPS",
                5,
            ),
            frame_timeout_seconds=_environment_float(
                env,
                "RECOVERYBOX_POSE_FRAME_TIMEOUT_SECONDS",
                0.4,
            ),
            inference_threads=_environment_integer(
                env,
                "RECOVERYBOX_TFLITE_THREADS",
                2,
            ),
            ffmpeg_binary=_environment_text(
                env,
                "RECOVERYBOX_FFMPEG_BINARY",
                "/usr/bin/ffmpeg",
            ),
        )


def validate_movenet_model(path: str | Path) -> Path:
    """Validate the pinned public model without following a symlink."""

    model_path = Path(path).expanduser()
    try:
        metadata = model_path.lstat()
    except OSError as exc:
        raise PiPoseConfigurationError("MoveNet model is unavailable") from exc
    if model_path.is_symlink() or not model_path.is_file():
        raise PiPoseConfigurationError("MoveNet model must be a regular non-symlink file")
    if metadata.st_size != MOVENET_MODEL_SIZE_BYTES:
        raise PiPoseConfigurationError("MoveNet model size does not match the pinned asset")
    digest = hashlib.sha256()
    try:
        with model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PiPoseConfigurationError("MoveNet model cannot be read") from exc
    if digest.hexdigest() != MOVENET_MODEL_SHA256:
        raise PiPoseConfigurationError("MoveNet model digest does not match the pinned asset")
    return model_path


class MoveNetKeypoint(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


@dataclass(frozen=True, slots=True)
class MoveNetPoint:
    """One MoveNet ``[y, x, score]`` prediction in normalized coordinates."""

    y: float
    x: float
    score: float

    def __post_init__(self) -> None:
        y = _finite_number(self.y, field_name="y")
        x = _finite_number(self.x, field_name="x")
        score = _finite_number(self.score, field_name="score")
        if not -1.0 <= x <= 2.0 or not -1.0 <= y <= 2.0:
            raise PiPoseRuntimeError("MoveNet returned an implausible coordinate")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "score", max(0.0, min(1.0, score)))


_MOVENET_TO_MEDIAPIPE = {
    MoveNetKeypoint.NOSE: MediaPipePoseLandmark.NOSE,
    MoveNetKeypoint.LEFT_EYE: MediaPipePoseLandmark.LEFT_EYE,
    MoveNetKeypoint.RIGHT_EYE: MediaPipePoseLandmark.RIGHT_EYE,
    MoveNetKeypoint.LEFT_EAR: MediaPipePoseLandmark.LEFT_EAR,
    MoveNetKeypoint.RIGHT_EAR: MediaPipePoseLandmark.RIGHT_EAR,
    MoveNetKeypoint.LEFT_SHOULDER: MediaPipePoseLandmark.LEFT_SHOULDER,
    MoveNetKeypoint.RIGHT_SHOULDER: MediaPipePoseLandmark.RIGHT_SHOULDER,
    MoveNetKeypoint.LEFT_ELBOW: MediaPipePoseLandmark.LEFT_ELBOW,
    MoveNetKeypoint.RIGHT_ELBOW: MediaPipePoseLandmark.RIGHT_ELBOW,
    MoveNetKeypoint.LEFT_WRIST: MediaPipePoseLandmark.LEFT_WRIST,
    MoveNetKeypoint.RIGHT_WRIST: MediaPipePoseLandmark.RIGHT_WRIST,
    MoveNetKeypoint.LEFT_HIP: MediaPipePoseLandmark.LEFT_HIP,
    MoveNetKeypoint.RIGHT_HIP: MediaPipePoseLandmark.RIGHT_HIP,
    MoveNetKeypoint.LEFT_KNEE: MediaPipePoseLandmark.LEFT_KNEE,
    MoveNetKeypoint.RIGHT_KNEE: MediaPipePoseLandmark.RIGHT_KNEE,
    MoveNetKeypoint.LEFT_ANKLE: MediaPipePoseLandmark.LEFT_ANKLE,
    MoveNetKeypoint.RIGHT_ANKLE: MediaPipePoseLandmark.RIGHT_ANKLE,
}


def movenet_to_mediapipe_frame(
    points: Sequence[MoveNetPoint],
    *,
    timestamp_ms: int,
) -> MediaPipePoseFrame:
    """Map MoveNet's 17 numeric points to the local typed 33-point schema."""

    if len(points) != MOVENET_KEYPOINT_COUNT:
        raise PiPoseRuntimeError("MoveNet must return exactly 17 keypoints")
    if not all(isinstance(point, MoveNetPoint) for point in points):
        raise TypeError("points must contain only MoveNetPoint values")
    unavailable = NormalizedLandmark(
        x=0.0,
        y=0.0,
        z=0.0,
        visibility=0.0,
        presence=0.0,
    )
    landmarks = [unavailable] * 33
    for source, target in _MOVENET_TO_MEDIAPIPE.items():
        point = points[source]
        landmarks[target] = NormalizedLandmark(
            x=point.x,
            y=point.y,
            z=0.0,
            visibility=point.score,
            presence=point.score,
        )
    return MediaPipePoseFrame(
        timestamp_ms=timestamp_ms,
        image_width=MOVENET_INPUT_SIZE,
        image_height=MOVENET_INPUT_SIZE,
        landmarks=tuple(landmarks),
    )


class _TFLiteLibrary(Protocol):
    """Dynamic C library marker used only for injected tests."""


LibraryLoader = Callable[[str], _TFLiteLibrary]


def _bind_c_function(
    library: object,
    name: str,
    argument_types: Sequence[object],
    result_type: object,
) -> object:
    try:
        function = getattr(library, name)
        function.argtypes = list(argument_types)
        function.restype = result_type
    except (AttributeError, TypeError) as exc:
        raise PiPoseConfigurationError(f"TensorFlow Lite C API is missing {name}") from exc
    return function


class TFLiteMoveNet:
    """Long-lived MoveNet Lightning interpreter over TensorFlow Lite's C API."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        library_path: str | Path = DEFAULT_TFLITE_LIBRARY_PATH,
        threads: int = 2,
        library_loader: LibraryLoader = ctypes.CDLL,
        model_validator: Callable[[str | Path], Path] = validate_movenet_model,
    ) -> None:
        thread_count = _positive_integer(threads, field_name="threads")
        checked_model = model_validator(model_path)
        native_path = Path(library_path).expanduser()
        if not native_path.is_absolute():
            raise PiPoseConfigurationError("TensorFlow Lite library path must be absolute")
        try:
            library = library_loader(str(native_path))
        except OSError as exc:
            raise PiPoseConfigurationError("TensorFlow Lite C library is unavailable") from exc
        self._library = library
        self._model: int | None = None
        self._options: int | None = None
        self._interpreter: int | None = None
        self._input_tensor: int | None = None
        self._output_tensor: int | None = None
        self._bind_api()
        try:
            self._model = self._model_create(os.fsencode(checked_model))
            if not self._model:
                raise PiPoseRuntimeError("TensorFlow Lite rejected the MoveNet model")
            self._options = self._options_create()
            if not self._options:
                raise PiPoseRuntimeError("TensorFlow Lite could not create options")
            self._options_set_threads(self._options, thread_count)
            self._interpreter = self._interpreter_create(self._model, self._options)
            if not self._interpreter:
                raise PiPoseRuntimeError("TensorFlow Lite could not create an interpreter")
            if self._allocate(self._interpreter) != _TFLITE_OK:
                raise PiPoseRuntimeError("TensorFlow Lite could not allocate tensors")
            self._validate_tensor_contract()
        except BaseException:
            self.close()
            raise

    def _bind_api(self) -> None:
        pointer = ctypes.c_void_p
        self._model_create = _bind_c_function(
            self._library,
            "TfLiteModelCreateFromFile",
            [ctypes.c_char_p],
            pointer,
        )
        self._model_delete = _bind_c_function(
            self._library,
            "TfLiteModelDelete",
            [pointer],
            None,
        )
        self._options_create = _bind_c_function(
            self._library,
            "TfLiteInterpreterOptionsCreate",
            [],
            pointer,
        )
        self._options_delete = _bind_c_function(
            self._library,
            "TfLiteInterpreterOptionsDelete",
            [pointer],
            None,
        )
        self._options_set_threads = _bind_c_function(
            self._library,
            "TfLiteInterpreterOptionsSetNumThreads",
            [pointer, ctypes.c_int],
            None,
        )
        self._interpreter_create = _bind_c_function(
            self._library,
            "TfLiteInterpreterCreate",
            [pointer, pointer],
            pointer,
        )
        self._interpreter_delete = _bind_c_function(
            self._library,
            "TfLiteInterpreterDelete",
            [pointer],
            None,
        )
        self._allocate = _bind_c_function(
            self._library,
            "TfLiteInterpreterAllocateTensors",
            [pointer],
            ctypes.c_int,
        )
        self._input_count = _bind_c_function(
            self._library,
            "TfLiteInterpreterGetInputTensorCount",
            [pointer],
            ctypes.c_int,
        )
        self._output_count = _bind_c_function(
            self._library,
            "TfLiteInterpreterGetOutputTensorCount",
            [pointer],
            ctypes.c_int,
        )
        self._input = _bind_c_function(
            self._library,
            "TfLiteInterpreterGetInputTensor",
            [pointer, ctypes.c_int],
            pointer,
        )
        self._output = _bind_c_function(
            self._library,
            "TfLiteInterpreterGetOutputTensor",
            [pointer, ctypes.c_int],
            pointer,
        )
        self._tensor_type = _bind_c_function(
            self._library,
            "TfLiteTensorType",
            [pointer],
            ctypes.c_int,
        )
        self._tensor_dims = _bind_c_function(
            self._library,
            "TfLiteTensorNumDims",
            [pointer],
            ctypes.c_int,
        )
        self._tensor_dim = _bind_c_function(
            self._library,
            "TfLiteTensorDim",
            [pointer, ctypes.c_int],
            ctypes.c_int,
        )
        self._tensor_size = _bind_c_function(
            self._library,
            "TfLiteTensorByteSize",
            [pointer],
            ctypes.c_size_t,
        )
        self._copy_from = _bind_c_function(
            self._library,
            "TfLiteTensorCopyFromBuffer",
            [pointer, pointer, ctypes.c_size_t],
            ctypes.c_int,
        )
        self._copy_to = _bind_c_function(
            self._library,
            "TfLiteTensorCopyToBuffer",
            [pointer, pointer, ctypes.c_size_t],
            ctypes.c_int,
        )
        self._invoke = _bind_c_function(
            self._library,
            "TfLiteInterpreterInvoke",
            [pointer],
            ctypes.c_int,
        )

    def _tensor_shape(self, tensor: int) -> tuple[int, ...]:
        count = self._tensor_dims(tensor)
        if count < 0 or count > 8:
            raise PiPoseRuntimeError("TensorFlow Lite returned an invalid tensor rank")
        return tuple(self._tensor_dim(tensor, index) for index in range(count))

    def _validate_tensor_contract(self) -> None:
        assert self._interpreter is not None
        if self._input_count(self._interpreter) != 1 or self._output_count(self._interpreter) != 1:
            raise PiPoseRuntimeError("MoveNet must expose one input and one output")
        self._input_tensor = self._input(self._interpreter, 0)
        self._output_tensor = self._output(self._interpreter, 0)
        if not self._input_tensor or not self._output_tensor:
            raise PiPoseRuntimeError("MoveNet tensor handles are unavailable")
        if self._tensor_type(self._input_tensor) != _TFLITE_UINT8:
            raise PiPoseRuntimeError("MoveNet input must be uint8")
        if self._tensor_shape(self._input_tensor) != _EXPECTED_INPUT_DIMS:
            raise PiPoseRuntimeError("MoveNet input shape is not 1x192x192x3")
        if self._tensor_size(self._input_tensor) != MOVENET_INPUT_BYTES:
            raise PiPoseRuntimeError("MoveNet input byte size is invalid")
        if self._tensor_type(self._output_tensor) != _TFLITE_FLOAT32:
            raise PiPoseRuntimeError("MoveNet output must be float32")
        if self._tensor_shape(self._output_tensor) != _EXPECTED_OUTPUT_DIMS:
            raise PiPoseRuntimeError("MoveNet output shape is not 1x1x17x3")
        if self._tensor_size(self._output_tensor) != MOVENET_OUTPUT_VALUES * 4:
            raise PiPoseRuntimeError("MoveNet output byte size is invalid")

    def infer(self, rgb_frame: bytes) -> tuple[MoveNetPoint, ...]:
        if type(rgb_frame) is not bytes:
            raise TypeError("rgb_frame must be immutable bytes")
        if len(rgb_frame) != MOVENET_INPUT_BYTES:
            raise PiPoseRuntimeError("RGB frame has the wrong byte size")
        if self._interpreter is None or self._input_tensor is None or self._output_tensor is None:
            raise PiPoseRuntimeError("MoveNet interpreter is closed")
        input_buffer = (ctypes.c_ubyte * MOVENET_INPUT_BYTES).from_buffer_copy(rgb_frame)
        if (
            self._copy_from(
                self._input_tensor,
                ctypes.cast(input_buffer, ctypes.c_void_p),
                MOVENET_INPUT_BYTES,
            )
            != _TFLITE_OK
        ):
            raise PiPoseRuntimeError("TensorFlow Lite rejected the RGB frame")
        if self._invoke(self._interpreter) != _TFLITE_OK:
            raise PiPoseRuntimeError("MoveNet inference failed")
        output_buffer = (ctypes.c_float * MOVENET_OUTPUT_VALUES)()
        if (
            self._copy_to(
                self._output_tensor,
                ctypes.cast(output_buffer, ctypes.c_void_p),
                ctypes.sizeof(output_buffer),
            )
            != _TFLITE_OK
        ):
            raise PiPoseRuntimeError("TensorFlow Lite output copy failed")
        values = struct.unpack(
            f"={MOVENET_OUTPUT_VALUES}f",
            bytes(output_buffer),
        )
        return tuple(
            MoveNetPoint(
                y=values[index * 3],
                x=values[index * 3 + 1],
                score=values[index * 3 + 2],
            )
            for index in range(MOVENET_KEYPOINT_COUNT)
        )

    def close(self) -> None:
        if self._interpreter is not None:
            self._interpreter_delete(self._interpreter)
            self._interpreter = None
        if self._options is not None:
            self._options_delete(self._options)
            self._options = None
        if self._model is not None:
            self._model_delete(self._model)
            self._model = None
        self._input_tensor = None
        self._output_tensor = None

    def __enter__(self) -> TFLiteMoveNet:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()


class _Process(Protocol):
    stdout: BinaryIO | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


ProcessFactory = Callable[..., _Process]
ReadableWaiter = Callable[[int, float], bool]


def _wait_readable(file_descriptor: int, timeout_seconds: float) -> bool:
    readable, _, _ = select.select([file_descriptor], [], [], timeout_seconds)
    return bool(readable)


def _read_exact_frame(
    stream: BinaryIO,
    byte_count: int,
    timeout_seconds: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    readable_waiter: ReadableWaiter = _wait_readable,
) -> bytes | None:
    deadline = clock() + timeout_seconds
    chunks: list[bytes] = []
    received = 0
    file_descriptor = stream.fileno()
    while received < byte_count:
        remaining = deadline - clock()
        if remaining <= 0 or not readable_waiter(file_descriptor, remaining):
            if received:
                raise PiPoseRuntimeError("camera frame timed out after a partial read")
            return None
        chunk = os.read(file_descriptor, byte_count - received)
        if not chunk:
            raise PiPoseRuntimeError("camera stream ended")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class FfmpegRawCamera:
    """Read fixed-size padded RGB frames without creating an image file."""

    def __init__(
        self,
        config: PiPoseConfig,
        *,
        process_factory: ProcessFactory = subprocess.Popen,
        frame_reader: Callable[[BinaryIO, int, float], bytes | None] = _read_exact_frame,
    ) -> None:
        if not isinstance(config, PiPoseConfig):
            raise TypeError("config must be a PiPoseConfig")
        self._config = config
        self._process_factory = process_factory
        self._frame_reader = frame_reader
        self._process: _Process | None = None

    @property
    def command(self) -> tuple[str, ...]:
        config = self._config
        video_filter = (
            f"scale={MOVENET_INPUT_SIZE}:{MOVENET_INPUT_SIZE}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={MOVENET_INPUT_SIZE}:{MOVENET_INPUT_SIZE}:"
            "(ow-iw)/2:(oh-ih)/2:color=black,format=rgb24"
        )
        return (
            config.ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-video_size",
            f"{config.capture_width}x{config.capture_height}",
            "-framerate",
            str(config.frames_per_second),
            "-i",
            str(config.camera_device),
            "-an",
            "-vf",
            video_filter,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        )

    def open(self) -> None:
        if self._process is not None:
            raise RuntimeError("camera is already open")
        try:
            process = self._process_factory(
                list(self.command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                shell=False,
            )
        except OSError as exc:
            raise PiPoseRuntimeError("ffmpeg camera process could not start") from exc
        if process.stdout is None:
            try:
                process.terminate()
            finally:
                raise PiPoseRuntimeError("ffmpeg camera pipe is unavailable")
        self._process = process

    def read_frame(self) -> bytes | None:
        process = self._process
        if process is None or process.stdout is None:
            raise PiPoseRuntimeError("camera is not open")
        frame = self._frame_reader(
            process.stdout,
            MOVENET_INPUT_BYTES,
            self._config.frame_timeout_seconds,
        )
        if frame is None and process.poll() is not None:
            raise PiPoseRuntimeError("ffmpeg camera process exited")
        if frame is not None and len(frame) != MOVENET_INPUT_BYTES:
            raise PiPoseRuntimeError("ffmpeg returned a truncated RGB frame")
        return frame

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)


class _Camera(Protocol):
    def open(self) -> None: ...

    def read_frame(self) -> bytes | None: ...

    def close(self) -> None: ...


class _Estimator(Protocol):
    def infer(self, rgb_frame: bytes) -> tuple[MoveNetPoint, ...]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PiPoseObservation:
    """Sanitized numeric result from one process-local camera observation."""

    analysis: SquatAnalysis
    inference_ms: float | None
    frame_received: bool

    def __post_init__(self) -> None:
        if not isinstance(self.analysis, SquatAnalysis):
            raise TypeError("analysis must be a SquatAnalysis")
        if self.inference_ms is not None:
            inference_ms = _finite_number(self.inference_ms, field_name="inference_ms")
            if inference_ms < 0:
                raise ValueError("inference_ms must be non-negative")
            object.__setattr__(self, "inference_ms", inference_ms)
        if not isinstance(self.frame_received, bool):
            raise TypeError("frame_received must be a boolean")


class PiPoseSource:
    """Compose camera, model, and deterministic tracker without frame escape."""

    def __init__(
        self,
        *,
        camera: _Camera,
        estimator: _Estimator,
        tracker: SquatTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._camera = camera
        self._estimator = estimator
        self._tracker = tracker if tracker is not None else SquatTracker()
        self._clock = clock
        self._opened = False
        self._last_timestamp_ms: int | None = None

    @classmethod
    def from_config(cls, config: PiPoseConfig) -> PiPoseSource:
        if not isinstance(config, PiPoseConfig):
            raise TypeError("config must be a PiPoseConfig")
        return cls(
            camera=FfmpegRawCamera(config),
            estimator=TFLiteMoveNet(
                model_path=config.model_path,
                library_path=config.tflite_library_path,
                threads=config.inference_threads,
            ),
        )

    def _timestamp_ms(self) -> int:
        timestamp = max(0, int(self._clock() * 1000))
        if self._last_timestamp_ms is not None:
            timestamp = max(timestamp, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp
        return timestamp

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("pose source is already open")
        try:
            self._camera.open()
        except BaseException:
            self._estimator.close()
            raise
        self._opened = True

    def read(self) -> PiPoseObservation:
        if not self._opened:
            raise RuntimeError("pose source is not open")
        frame = self._camera.read_frame()
        timestamp_ms = self._timestamp_ms()
        if frame is None:
            return PiPoseObservation(
                analysis=self._tracker.update_missing(
                    timestamp_ms,
                    issue=SquatAssessmentIssue.CAMERA_TIMEOUT,
                ),
                inference_ms=None,
                frame_received=False,
            )
        started = self._clock()
        points = self._estimator.infer(frame)
        inference_ms = max(0.0, (self._clock() - started) * 1000.0)
        pose_frame = movenet_to_mediapipe_frame(points, timestamp_ms=timestamp_ms)
        return PiPoseObservation(
            analysis=self._tracker.update(pose_frame),
            inference_ms=inference_ms,
            frame_received=True,
        )

    def close(self) -> None:
        self._opened = False
        try:
            self._camera.close()
        finally:
            self._estimator.close()

    def __enter__(self) -> PiPoseSource:
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


def run_pi_pose_check(
    config: PiPoseConfig,
    *,
    max_frames: int,
    source_factory: Callable[[PiPoseConfig], PiPoseSource] = PiPoseSource.from_config,
) -> dict[str, object]:
    """Run a bounded, silent hardware check and return numeric-only status."""

    frame_limit = _positive_integer(max_frames, field_name="max_frames")
    frames = 0
    assessable = 0
    timeouts = 0
    inference_values: list[float] = []
    source = source_factory(config)
    try:
        source.open()
        while frames < frame_limit:
            observation = source.read()
            frames += 1
            assessable += int(observation.analysis.assessable)
            timeouts += int(not observation.frame_received)
            if observation.inference_ms is not None:
                inference_values.append(observation.inference_ms)
    finally:
        source.close()
    return {
        "service": "recoverybox-pi-pose-check/v1",
        "frames": frames,
        "assessable": assessable,
        "timeouts": timeouts,
        "inference_ms_max": (round(max(inference_values), 3) if inference_values else None),
        "raw_frames_persisted": 0,
        "audio": "disabled",
    }


def run_movenet_self_check(
    config: PiPoseConfig,
    *,
    iterations: int,
    estimator_factory: Callable[..., TFLiteMoveNet] = TFLiteMoveNet,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Benchmark the pinned model with a synthetic frame and no camera data."""

    iteration_count = _positive_integer(iterations, field_name="iterations")
    estimator = estimator_factory(
        model_path=config.model_path,
        library_path=config.tflite_library_path,
        threads=config.inference_threads,
    )
    timings: list[float] = []
    try:
        synthetic_frame = bytes(MOVENET_INPUT_BYTES)
        for _ in range(iteration_count):
            started = clock()
            points = estimator.infer(synthetic_frame)
            elapsed = max(0.0, (clock() - started) * 1000.0)
            if len(points) != MOVENET_KEYPOINT_COUNT:
                raise PiPoseRuntimeError("MoveNet self-check returned the wrong point count")
            timings.append(elapsed)
    finally:
        estimator.close()
    return {
        "service": "recoverybox-movenet-self-check/v1",
        "iterations": len(timings),
        "first_inference_ms": round(timings[0], 3),
        "warm_inference_ms_max": (round(max(timings[1:]), 3) if len(timings) > 1 else None),
        "raw_frames_persisted": 0,
        "camera_used": False,
        "audio": "disabled",
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded, silent Pi-local RecoveryBox pose check",
    )
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--model-only-inferences", type=int)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _argument_parser().parse_args(arguments)
    try:
        config = PiPoseConfig.from_environment()
        if parsed.model_only_inferences is None:
            status = run_pi_pose_check(
                config,
                max_frames=parsed.max_frames,
            )
        else:
            status = run_movenet_self_check(
                config,
                iterations=parsed.model_only_inferences,
            )
    except (PiPoseConfigurationError, PiPoseRuntimeError, OSError) as exc:
        status = {
            "service": "recoverybox-pi-pose-check/v1",
            "ok": False,
            "failure": type(exc).__name__,
            "raw_frames_persisted": 0,
            "audio": "disabled",
        }
        print(json.dumps(status, sort_keys=True, separators=(",", ":")))
        return 1
    if parsed.model_only_inferences is None:
        status["ok"] = status["frames"] == parsed.max_frames and status["timeouts"] == 0
    else:
        status["ok"] = status["iterations"] == parsed.model_only_inferences
    print(json.dumps(status, sort_keys=True, separators=(",", ":")))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
