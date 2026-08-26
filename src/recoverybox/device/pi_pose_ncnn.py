"""Pinned NCNN person detection and pose inference for Raspberry Pi 3.

This module is deliberately camera-agnostic.  A same-process capture owner may
pass one BGRA frame into :class:`NcnnPersonPoseEstimator`; the estimator returns
only immutable numeric COCO keypoints and never stores, writes, logs, or
serializes the frame.  It is not safe to connect this estimator to a camera
subprocess: raw frames must remain inside the process that owns capture.

RTMPose is a top-down model, so it is never run on an implicit whole-frame
"person".  A pinned NanoDet COCO person detector must first produce exactly one
unambiguous, fully padded in-frame person box.  Missing, multiple, clipped, or
low-confidence detections return ``points=None`` and therefore cannot advance
the deterministic exercise tracker.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from recoverybox.device.pi_pose import (
    MoveNetPoint,
    PiPoseConfigurationError,
    PiPoseRuntimeError,
)

NCNN_RUNTIME_VERSION: Final = "1.0.20260526"
NCNN_ARMV7_WHEEL_FILENAME: Final = "ncnn-1.0.20260526-cp313-cp313-manylinux_2_31_armv7l.whl"
NCNN_ARMV7_WHEEL_URL: Final = (
    "https://files.pythonhosted.org/packages/2f/2f/"
    "1a8f4c5d83213ac459865fbea4af7051ab63c5934fd2813cccfaf9bf6409/" + NCNN_ARMV7_WHEEL_FILENAME
)
NCNN_ARMV7_WHEEL_SIZE_BYTES: Final = 2_976_956
NCNN_ARMV7_WHEEL_SHA256: Final = "28c1e2a574b8f9bcbcc8c95de94d7814fb25b1a106f12dae3b7a8a4344d2db4b"
NCNN_INIT_FILENAME: Final = "ncnn/__init__.py"
NCNN_INIT_SIZE_BYTES: Final = 118
NCNN_INIT_SHA256: Final = "ef90c76a49b37e74b0cd89f1da9502e764ee6b24a8da44860f4af894dc5838fe"
NCNN_NATIVE_FILENAME: Final = "ncnn/ncnn.cpython-313-arm-linux-gnueabihf.so"
NCNN_NATIVE_SIZE_BYTES: Final = 6_611_269
NCNN_NATIVE_SHA256: Final = "b64fdd46904e1a3379fd71f4179b2a06c3109eb4a6588414a5e0fdc22c7811c9"
NCNN_LIBGOMP_FILENAME: Final = "ncnn.libs/libgomp-39027b09.so.1.0.0"
NCNN_LIBGOMP_SIZE_BYTES: Final = 184_421
NCNN_LIBGOMP_SHA256: Final = "d94a8c0b47d2371b67b9417ff21cff03204f162f6db9c8000d78b11ce389caf9"

RTMPOSE_ARCHIVE_URL: Final = (
    "https://mmdeploy-oss.openmmlab.com/model/mmpose/rtmpose-t-ncnn-155ab7.zip"
)
RTMPOSE_ARCHIVE_SIZE_BYTES: Final = 24_967_723
RTMPOSE_ARCHIVE_SHA256: Final = "1c0481b760419a2140bf814d396335711ae1ffba91566395d1a529b4a177cd5d"
RTMPOSE_PARAM_SIZE_BYTES: Final = 16_845
RTMPOSE_PARAM_SHA256: Final = "0348745dcfded9842b546a4e904c410e6c9640a09b1713b1dba055c25b646595"
RTMPOSE_BIN_SIZE_BYTES: Final = 13_332_744
RTMPOSE_BIN_SHA256: Final = "6fd3738741cfe14c82e40762434cdbe29c484b8a5a0b2bdefbc83d8bacb94c7c"

NANODET_ARCHIVE_URL: Final = (
    "https://github.com/RangiLyu/nanodet/releases/download/v0.4.0/ncnn-nanodet-m-int8.zip"
)
NANODET_ARCHIVE_SIZE_BYTES: Final = 910_094
NANODET_ARCHIVE_SHA256: Final = "2ac8f6ea9b5bb1cd52f809229c4492d7402b419daa11b068a13478b0fd33bc32"
NANODET_PARAM_SIZE_BYTES: Final = 17_127
NANODET_PARAM_SHA256: Final = "ecce99dba4f9bd9298eb753905917a1eedf84e94755a3915868b155755097f04"
NANODET_BIN_SIZE_BYTES: Final = 1_004_492
NANODET_BIN_SHA256: Final = "719679139b0762d01663508a0893b026fdec09955f881d789b3b6bbc1ca900e1"

DEFAULT_NCNN_RUNTIME_PATH: Final = Path("/opt/recoverybox/runtime/ncnn")
DEFAULT_NCNN_MODEL_DIRECTORY: Final = Path("/opt/recoverybox/models/ncnn")

RTMPOSE_INPUT_WIDTH: Final = 192
RTMPOSE_INPUT_HEIGHT: Final = 256
RTMPOSE_KEYPOINT_COUNT: Final = 17
RTMPOSE_SIMCC_X_WIDTH: Final = 384
RTMPOSE_SIMCC_Y_WIDTH: Final = 512
RTMPOSE_SIMCC_SPLIT_RATIO: Final = 2.0
NANODET_TARGET_SIZE: Final = 320
NANODET_REG_MAX: Final = 7
NANODET_CLASS_COUNT: Final = 80
NANODET_PERSON_CLASS: Final = 0
NANODET_STRIDES: Final = (8, 16, 32)
NANODET_SCORE_NAMES: Final = (
    "cls_pred_stride_8",
    "cls_pred_stride_16",
    "cls_pred_stride_32",
)
NANODET_BOX_NAMES: Final = (
    "dis_pred_stride_8",
    "dis_pred_stride_16",
    "dis_pred_stride_32",
)

_RTMPOSE_MEAN: Final = (123.675, 116.28, 103.53)
_RTMPOSE_NORM: Final = (
    1.0 / 58.395,
    1.0 / 57.12,
    1.0 / 57.375,
)
_NANODET_MEAN: Final = (103.53, 116.28, 123.675)
_NANODET_NORM: Final = (0.017429, 0.017507, 0.017125)
_MAX_HARD_RESULT_AGE_SECONDS: Final = 0.5


def _finite(value: object, *, field_name: str) -> float:
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


def _monotonic_elapsed_ms(later: object, earlier: float, *, field_name: str) -> float:
    checked_later = _finite(later, field_name=field_name)
    if checked_later < earlier:
        raise PiPoseRuntimeError("monotonic inference clock moved backwards")
    return (checked_later - earlier) * 1000.0


def _absolute_path(value: object, *, field_name: str) -> Path:
    try:
        path = Path(value).expanduser()  # type: ignore[arg-type]
    except TypeError as exc:
        raise PiPoseConfigurationError(f"{field_name} must be a path") from exc
    if not path.is_absolute():
        raise PiPoseConfigurationError(f"{field_name} must be absolute")
    return path


@dataclass(frozen=True, slots=True)
class NcnnPoseConfig:
    """Closed configuration for the optional Pi-local NCNN estimator."""

    runtime_path: Path = DEFAULT_NCNN_RUNTIME_PATH
    rtmpose_param_path: Path = DEFAULT_NCNN_MODEL_DIRECTORY / "rtmpose-t.param"
    rtmpose_bin_path: Path = DEFAULT_NCNN_MODEL_DIRECTORY / "rtmpose-t.bin"
    nanodet_param_path: Path = DEFAULT_NCNN_MODEL_DIRECTORY / "nanodet-m-int8.param"
    nanodet_bin_path: Path = DEFAULT_NCNN_MODEL_DIRECTORY / "nanodet-m-int8.bin"
    inference_threads: int = 2
    person_score_threshold: float = 0.55
    nms_iou_threshold: float = 0.3
    minimum_person_area_fraction: float = 0.05
    maximum_result_age_seconds: float = _MAX_HARD_RESULT_AGE_SECONDS

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_path",
            "rtmpose_param_path",
            "rtmpose_bin_path",
            "nanodet_param_path",
            "nanodet_bin_path",
        ):
            object.__setattr__(
                self,
                field_name,
                _absolute_path(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "inference_threads",
            _positive_integer(self.inference_threads, field_name="inference_threads"),
        )
        for field_name in (
            "person_score_threshold",
            "nms_iou_threshold",
            "minimum_person_area_fraction",
            "maximum_result_age_seconds",
        ):
            value = _finite(getattr(self, field_name), field_name=field_name)
            if not 0.0 < value <= 1.0:
                raise PiPoseConfigurationError(f"{field_name} must be in (0, 1]")
            object.__setattr__(self, field_name, value)
        if self.maximum_result_age_seconds > _MAX_HARD_RESULT_AGE_SECONDS:
            raise PiPoseConfigurationError(
                "maximum_result_age_seconds may not exceed the Guardian 0.5s limit"
            )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> NcnnPoseConfig:
        env = os.environ if environment is None else environment
        try:
            return cls(
                runtime_path=Path(
                    env.get("RECOVERYBOX_NCNN_RUNTIME_PATH", str(DEFAULT_NCNN_RUNTIME_PATH))
                ),
                rtmpose_param_path=Path(
                    env.get(
                        "RECOVERYBOX_RTMPOSE_PARAM_PATH",
                        str(DEFAULT_NCNN_MODEL_DIRECTORY / "rtmpose-t.param"),
                    )
                ),
                rtmpose_bin_path=Path(
                    env.get(
                        "RECOVERYBOX_RTMPOSE_BIN_PATH",
                        str(DEFAULT_NCNN_MODEL_DIRECTORY / "rtmpose-t.bin"),
                    )
                ),
                nanodet_param_path=Path(
                    env.get(
                        "RECOVERYBOX_NANODET_PARAM_PATH",
                        str(DEFAULT_NCNN_MODEL_DIRECTORY / "nanodet-m-int8.param"),
                    )
                ),
                nanodet_bin_path=Path(
                    env.get(
                        "RECOVERYBOX_NANODET_BIN_PATH",
                        str(DEFAULT_NCNN_MODEL_DIRECTORY / "nanodet-m-int8.bin"),
                    )
                ),
                inference_threads=int(env.get("RECOVERYBOX_NCNN_THREADS", "2")),
                person_score_threshold=float(env.get("RECOVERYBOX_PERSON_SCORE_THRESHOLD", "0.55")),
                nms_iou_threshold=float(env.get("RECOVERYBOX_PERSON_NMS_IOU", "0.3")),
                minimum_person_area_fraction=float(
                    env.get("RECOVERYBOX_MIN_PERSON_AREA_FRACTION", "0.05")
                ),
                maximum_result_age_seconds=float(
                    env.get("RECOVERYBOX_LOCAL_POSE_MAX_AGE_SECONDS", "0.5")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise PiPoseConfigurationError("NCNN pose environment is invalid") from exc


def validate_pinned_file(
    path: str | Path,
    *,
    expected_size: int,
    expected_sha256: str,
    asset_name: str,
) -> Path:
    """Validate one model/runtime asset without following a symlink."""

    checked = Path(path).expanduser()
    try:
        metadata = checked.lstat()
    except OSError as exc:
        raise PiPoseConfigurationError(f"{asset_name} is unavailable") from exc
    if checked.is_symlink() or not checked.is_file():
        raise PiPoseConfigurationError(f"{asset_name} must be a regular non-symlink file")
    if metadata.st_size != expected_size:
        raise PiPoseConfigurationError(f"{asset_name} size does not match the pin")
    digest = hashlib.sha256()
    try:
        with checked.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PiPoseConfigurationError(f"{asset_name} cannot be read") from exc
    if digest.hexdigest() != expected_sha256:
        raise PiPoseConfigurationError(f"{asset_name} digest does not match the pin")
    return checked


class _NcnnMat(Protocol):
    w: int
    h: int
    c: int

    def __getitem__(self, index: int) -> float: ...

    def substract_mean_normalize(
        self,
        mean: Sequence[float],
        norm: Sequence[float],
    ) -> None: ...


class _NcnnModule(Protocol):
    __version__: str
    __file__: str


NcnnLoader = Callable[[Path], _NcnnModule]


def _import_pinned_ncnn(runtime_path: Path) -> _NcnnModule:
    try:
        resolved_runtime = runtime_path.resolve(strict=True)
    except OSError as exc:
        raise PiPoseConfigurationError("ncnn runtime directory is unavailable") from exc
    for relative_name, expected_size, expected_digest, asset_name in (
        (
            NCNN_INIT_FILENAME,
            NCNN_INIT_SIZE_BYTES,
            NCNN_INIT_SHA256,
            "ncnn package initializer",
        ),
        (
            NCNN_NATIVE_FILENAME,
            NCNN_NATIVE_SIZE_BYTES,
            NCNN_NATIVE_SHA256,
            "ncnn native extension",
        ),
        (
            NCNN_LIBGOMP_FILENAME,
            NCNN_LIBGOMP_SIZE_BYTES,
            NCNN_LIBGOMP_SHA256,
            "ncnn OpenMP runtime",
        ),
    ):
        validate_pinned_file(
            resolved_runtime / relative_name,
            expected_size=expected_size,
            expected_sha256=expected_digest,
            asset_name=asset_name,
        )
    previous = sys.modules.get("ncnn")
    if previous is not None:
        module = previous
    else:
        sys.path.insert(0, str(resolved_runtime))
        try:
            try:
                module = importlib.import_module("ncnn")
            except (ImportError, OSError) as exc:
                raise PiPoseConfigurationError("ncnn runtime could not be imported") from exc
        finally:
            try:
                sys.path.remove(str(resolved_runtime))
            except ValueError:
                pass
    if getattr(module, "__version__", None) != NCNN_RUNTIME_VERSION:
        raise PiPoseConfigurationError("ncnn runtime version does not match the pin")
    try:
        module_file = Path(getattr(module, "__file__", "")).resolve(strict=True)
    except OSError as exc:
        raise PiPoseConfigurationError("ncnn package path is unavailable") from exc
    if module_file != resolved_runtime / NCNN_INIT_FILENAME:
        raise PiPoseConfigurationError("ncnn runtime was not loaded from the pinned path")
    native_module = sys.modules.get("ncnn.ncnn")
    try:
        native_file = Path(getattr(native_module, "__file__", "")).resolve(strict=True)
    except OSError as exc:
        raise PiPoseConfigurationError("ncnn native extension is unavailable") from exc
    if native_file != resolved_runtime / NCNN_NATIVE_FILENAME:
        raise PiPoseConfigurationError("ncnn native extension path does not match the pin")
    return module  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PersonBox:
    """One detector box in original-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    def __post_init__(self) -> None:
        for field_name in ("x1", "y1", "x2", "y2", "score"):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), field_name=field_name),
            )
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise PiPoseRuntimeError("person detector returned an invalid box")
        if not 0.0 <= self.score <= 1.0:
            raise PiPoseRuntimeError("person detector returned an invalid score")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class PoseInferenceResult:
    """Sanitized output; no raw or model tensor is retained."""

    points: tuple[MoveNetPoint, ...] | None
    person_score: float | None
    detector_ms: float
    pose_ms: float | None
    total_ms: float
    evidence_age_ms: float
    timed_out: bool

    def __post_init__(self) -> None:
        if self.points is not None and len(self.points) != RTMPOSE_KEYPOINT_COUNT:
            raise ValueError("points must contain exactly 17 values")
        if self.person_score is not None:
            score = _finite(self.person_score, field_name="person_score")
            if not 0.0 <= score <= 1.0:
                raise ValueError("person_score must be in [0, 1]")
        for field_name in ("detector_ms", "total_ms", "evidence_age_ms"):
            value = _finite(getattr(self, field_name), field_name=field_name)
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if self.pose_ms is not None:
            value = _finite(self.pose_ms, field_name="pose_ms")
            if value < 0:
                raise ValueError("pose_ms must not be negative")
        if not isinstance(self.timed_out, bool):
            raise TypeError("timed_out must be a boolean")
        if self.timed_out and self.points is not None:
            raise ValueError("a timed-out inference may not expose pose points")


def _mat_values(mat: _NcnnMat, *, width: int, height: int, name: str) -> list[float]:
    if (mat.w, mat.h, mat.c) != (width, height, 1):
        raise PiPoseRuntimeError(f"{name} tensor shape is invalid")
    values: list[float] = []
    for index in range(width * height):
        value = _finite(mat[index], field_name=name)
        values.append(value)
    return values


def _validate_mat_shape(
    mat: _NcnnMat,
    *,
    width: int,
    height: int,
    name: str,
) -> None:
    if (mat.w, mat.h, mat.c) != (width, height, 1):
        raise PiPoseRuntimeError(f"{name} tensor shape is invalid")


def _softmax_expectation(logits: Sequence[float]) -> float:
    if len(logits) != NANODET_REG_MAX + 1:
        raise PiPoseRuntimeError("NanoDet distance distribution is invalid")
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    denominator = sum(weights)
    if not math.isfinite(denominator) or denominator <= 0:
        raise PiPoseRuntimeError("NanoDet distance distribution is invalid")
    return sum(index * weight for index, weight in enumerate(weights)) / denominator


def _iou(left: PersonBox, right: PersonBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = left.area + right.area - intersection
    return 0.0 if union <= 0 else intersection / union


def _nms(boxes: Sequence[PersonBox], *, threshold: float) -> tuple[PersonBox, ...]:
    remaining = sorted(boxes, key=lambda box: box.score, reverse=True)
    selected: list[PersonBox] = []
    while remaining:
        current = remaining.pop(0)
        selected.append(current)
        remaining = [box for box in remaining if _iou(current, box) <= threshold]
    return tuple(selected)


def decode_nanodet_person_boxes(
    score_tensors: Sequence[_NcnnMat],
    box_tensors: Sequence[_NcnnMat],
    *,
    input_width: int,
    input_height: int,
    resized_scale: float,
    pad_left: int,
    pad_top: int,
    frame_width: int,
    frame_height: int,
    score_threshold: float,
    nms_iou_threshold: float,
) -> tuple[PersonBox, ...]:
    """Decode only COCO class zero (person), then apply person-only NMS."""

    if len(score_tensors) != len(NANODET_STRIDES) or len(box_tensors) != len(NANODET_STRIDES):
        raise PiPoseRuntimeError("NanoDet returned the wrong tensor count")
    if resized_scale <= 0 or not math.isfinite(resized_scale):
        raise PiPoseRuntimeError("NanoDet resize scale is invalid")
    candidates: list[PersonBox] = []
    for stride, scores, boxes in zip(
        NANODET_STRIDES,
        score_tensors,
        box_tensors,
        strict=True,
    ):
        feature_width = input_width // stride
        feature_height = input_height // stride
        row_count = feature_width * feature_height
        _validate_mat_shape(
            scores,
            width=NANODET_CLASS_COUNT,
            height=row_count,
            name=f"NanoDet score/{stride}",
        )
        _validate_mat_shape(
            boxes,
            width=4 * (NANODET_REG_MAX + 1),
            height=row_count,
            name=f"NanoDet box/{stride}",
        )
        for row in range(row_count):
            score = _finite(
                scores[row * NANODET_CLASS_COUNT + NANODET_PERSON_CLASS],
                field_name=f"NanoDet score/{stride}",
            )
            if score < score_threshold or score > 1.0:
                continue
            center_x = (row % feature_width + 0.5) * stride
            center_y = (row // feature_width + 0.5) * stride
            base = row * 4 * (NANODET_REG_MAX + 1)
            distances = []
            for side in range(4):
                side_base = base + side * (NANODET_REG_MAX + 1)
                logits = [
                    _finite(
                        boxes[side_base + index],
                        field_name=f"NanoDet box/{stride}",
                    )
                    for index in range(NANODET_REG_MAX + 1)
                ]
                distances.append(_softmax_expectation(logits) * stride)
            x1 = (center_x - distances[0] - pad_left) / resized_scale
            y1 = (center_y - distances[1] - pad_top) / resized_scale
            x2 = (center_x + distances[2] - pad_left) / resized_scale
            y2 = (center_y + distances[3] - pad_top) / resized_scale
            x1 = max(0.0, min(float(frame_width), x1))
            y1 = max(0.0, min(float(frame_height), y1))
            x2 = max(0.0, min(float(frame_width), x2))
            y2 = max(0.0, min(float(frame_height), y2))
            if x2 > x1 and y2 > y1:
                candidates.append(PersonBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score))
    return _nms(candidates, threshold=nms_iou_threshold)


@dataclass(frozen=True, slots=True)
class PoseCrop:
    """Integer, fully in-frame top-down crop and its inverse mapping."""

    x: int
    y: int
    width: int
    height: int
    frame_width: int
    frame_height: int


def person_box_to_pose_crop(
    box: PersonBox,
    *,
    frame_width: int,
    frame_height: int,
    padding: float = 1.25,
) -> PoseCrop | None:
    """Apply the canonical RTMPose padding/aspect rule, failing closed at edges."""

    if frame_width < 1 or frame_height < 1:
        raise ValueError("frame dimensions must be positive")
    center_x = (box.x1 + box.x2) / 2.0
    center_y = (box.y1 + box.y2) / 2.0
    width = (box.x2 - box.x1) * padding
    height = (box.y2 - box.y1) * padding
    target_aspect = RTMPOSE_INPUT_WIDTH / RTMPOSE_INPUT_HEIGHT
    if width > height * target_aspect:
        height = width / target_aspect
    else:
        width = height * target_aspect
    x1 = math.floor(center_x - width / 2.0)
    y1 = math.floor(center_y - height / 2.0)
    x2 = math.ceil(center_x + width / 2.0)
    y2 = math.ceil(center_y + height / 2.0)
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return PoseCrop(
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def decode_rtmpose_simcc(
    simcc_x: _NcnnMat,
    simcc_y: _NcnnMat,
    *,
    crop: PoseCrop,
    person_score: float,
) -> tuple[MoveNetPoint, ...]:
    """Decode the exact OpenMMLab SimCC contract and invert the crop."""

    x_values = _mat_values(
        simcc_x,
        width=RTMPOSE_SIMCC_X_WIDTH,
        height=RTMPOSE_KEYPOINT_COUNT,
        name="RTMPose simcc_x",
    )
    y_values = _mat_values(
        simcc_y,
        width=RTMPOSE_SIMCC_Y_WIDTH,
        height=RTMPOSE_KEYPOINT_COUNT,
        name="RTMPose simcc_y",
    )
    checked_person_score = _finite(person_score, field_name="person_score")
    if not 0.0 <= checked_person_score <= 1.0:
        raise PiPoseRuntimeError("person score is invalid")
    points: list[MoveNetPoint] = []
    for keypoint in range(RTMPOSE_KEYPOINT_COUNT):
        x_row = x_values[keypoint * RTMPOSE_SIMCC_X_WIDTH : (keypoint + 1) * RTMPOSE_SIMCC_X_WIDTH]
        y_row = y_values[keypoint * RTMPOSE_SIMCC_Y_WIDTH : (keypoint + 1) * RTMPOSE_SIMCC_Y_WIDTH]
        x_index = max(range(len(x_row)), key=x_row.__getitem__)
        y_index = max(range(len(y_row)), key=y_row.__getitem__)
        raw_score = min(x_row[x_index], y_row[y_index])
        score = max(0.0, min(1.0, raw_score, checked_person_score))
        model_x = x_index / RTMPOSE_SIMCC_SPLIT_RATIO
        model_y = y_index / RTMPOSE_SIMCC_SPLIT_RATIO
        original_x = crop.x + model_x / RTMPOSE_INPUT_WIDTH * crop.width
        original_y = crop.y + model_y / RTMPOSE_INPUT_HEIGHT * crop.height
        points.append(
            MoveNetPoint(
                y=original_y / crop.frame_height,
                x=original_x / crop.frame_width,
                score=score,
            )
        )
    return tuple(points)


class NcnnPersonPoseEstimator:
    """Long-lived detector+pose networks over the exact pinned NCNN runtime."""

    def __init__(
        self,
        config: NcnnPoseConfig,
        *,
        ncnn_loader: NcnnLoader = _import_pinned_ncnn,
        model_validator: Callable[..., Path] = validate_pinned_file,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, NcnnPoseConfig):
            raise TypeError("config must be an NcnnPoseConfig")
        self._config = config
        self._clock = clock
        assets = (
            (
                config.rtmpose_param_path,
                RTMPOSE_PARAM_SIZE_BYTES,
                RTMPOSE_PARAM_SHA256,
                "RTMPose param",
            ),
            (
                config.rtmpose_bin_path,
                RTMPOSE_BIN_SIZE_BYTES,
                RTMPOSE_BIN_SHA256,
                "RTMPose bin",
            ),
            (
                config.nanodet_param_path,
                NANODET_PARAM_SIZE_BYTES,
                NANODET_PARAM_SHA256,
                "NanoDet param",
            ),
            (
                config.nanodet_bin_path,
                NANODET_BIN_SIZE_BYTES,
                NANODET_BIN_SHA256,
                "NanoDet bin",
            ),
        )
        checked = [
            model_validator(
                path,
                expected_size=size,
                expected_sha256=digest,
                asset_name=name,
            )
            for path, size, digest, name in assets
        ]
        self._ncnn = ncnn_loader(config.runtime_path)
        self._detector = self._create_net(checked[2], checked[3])
        self._pose = self._create_net(checked[0], checked[1])
        self._closed = False

    def _create_net(self, param_path: Path, bin_path: Path) -> object:
        net = self._ncnn.Net()  # type: ignore[attr-defined]
        net.opt.num_threads = self._config.inference_threads
        net.opt.use_vulkan_compute = False
        for name in (
            "use_fp16_packed",
            "use_fp16_storage",
            "use_fp16_arithmetic",
        ):
            if hasattr(net.opt, name):
                setattr(net.opt, name, False)
        if net.load_param(str(param_path)) != 0 or net.load_model(str(bin_path)) != 0:
            raise PiPoseRuntimeError("ncnn rejected a pinned model")
        return net

    def _detector_input(self, bgra_frame: bytes, width: int, height: int):
        if width >= height:
            scale = NANODET_TARGET_SIZE / width
        else:
            scale = NANODET_TARGET_SIZE / height
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        mat = self._ncnn.Mat.from_pixels_resize(  # type: ignore[attr-defined]
            bgra_frame,
            self._ncnn.Mat.PixelType.PIXEL_BGRA2BGR,  # type: ignore[attr-defined]
            width,
            height,
            resized_width,
            resized_height,
        )
        width_padding = (resized_width + 31) // 32 * 32 - resized_width
        height_padding = (resized_height + 31) // 32 * 32 - resized_height
        left = width_padding // 2
        top = height_padding // 2
        padded = self._ncnn.copy_make_border(  # type: ignore[attr-defined]
            mat,
            top,
            height_padding - top,
            left,
            width_padding - left,
            self._ncnn.BorderType.BORDER_CONSTANT,  # type: ignore[attr-defined]
            0,
        )
        padded.substract_mean_normalize(_NANODET_MEAN, _NANODET_NORM)
        return padded, scale, left, top

    def _detect(self, bgra_frame: bytes, width: int, height: int) -> tuple[PersonBox, ...]:
        detector_input, scale, pad_left, pad_top = self._detector_input(bgra_frame, width, height)
        extractor = self._detector.create_extractor()
        if extractor.input("input.1", detector_input) != 0:
            raise PiPoseRuntimeError("NanoDet rejected the frame")
        score_tensors = []
        box_tensors = []
        for name in NANODET_SCORE_NAMES:
            status, tensor = extractor.extract(name)
            if status != 0:
                raise PiPoseRuntimeError("NanoDet score extraction failed")
            score_tensors.append(tensor)
        for name in NANODET_BOX_NAMES:
            status, tensor = extractor.extract(name)
            if status != 0:
                raise PiPoseRuntimeError("NanoDet box extraction failed")
            box_tensors.append(tensor)
        return decode_nanodet_person_boxes(
            score_tensors,
            box_tensors,
            input_width=detector_input.w,
            input_height=detector_input.h,
            resized_scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            frame_width=width,
            frame_height=height,
            score_threshold=self._config.person_score_threshold,
            nms_iou_threshold=self._config.nms_iou_threshold,
        )

    def _infer_pose(
        self,
        bgra_frame: bytes,
        width: int,
        height: int,
        box: PersonBox,
    ) -> tuple[MoveNetPoint, ...] | None:
        crop = person_box_to_pose_crop(
            box,
            frame_width=width,
            frame_height=height,
        )
        if crop is None:
            return None
        pose_input = self._ncnn.Mat.from_pixels_roi_resize(  # type: ignore[attr-defined]
            bgra_frame,
            self._ncnn.Mat.PixelType.PIXEL_BGRA2RGB,  # type: ignore[attr-defined]
            width,
            height,
            crop.x,
            crop.y,
            crop.width,
            crop.height,
            RTMPOSE_INPUT_WIDTH,
            RTMPOSE_INPUT_HEIGHT,
        )
        pose_input.substract_mean_normalize(_RTMPOSE_MEAN, _RTMPOSE_NORM)
        extractor = self._pose.create_extractor()
        if extractor.input("input", pose_input) != 0:
            raise PiPoseRuntimeError("RTMPose rejected the person crop")
        x_status, simcc_x = extractor.extract("simcc_x")
        y_status, simcc_y = extractor.extract("simcc_y")
        if x_status != 0 or y_status != 0:
            raise PiPoseRuntimeError("RTMPose output extraction failed")
        return decode_rtmpose_simcc(
            simcc_x,
            simcc_y,
            crop=crop,
            person_score=box.score,
        )

    def infer_bgra(
        self,
        bgra_frame: bytes,
        *,
        width: int,
        height: int,
        captured_monotonic: float,
    ) -> PoseInferenceResult:
        """Infer one frame, accepting points only while capture evidence is fresh."""

        if self._closed:
            raise PiPoseRuntimeError("ncnn estimator is closed")
        if type(bgra_frame) is not bytes:
            raise TypeError("bgra_frame must be immutable bytes")
        frame_width = _positive_integer(width, field_name="width")
        frame_height = _positive_integer(height, field_name="height")
        if len(bgra_frame) != frame_width * frame_height * 4:
            raise PiPoseRuntimeError("BGRA frame has the wrong byte size")
        started = _finite(self._clock(), field_name="inference_started")
        if started < 0:
            raise PiPoseRuntimeError("inference clock is invalid")
        captured = _finite(captured_monotonic, field_name="captured_monotonic")
        if captured < 0 or captured > started:
            raise PiPoseRuntimeError("capture timestamp is invalid")
        initial_age = started - captured
        if initial_age >= self._config.maximum_result_age_seconds:
            return PoseInferenceResult(
                points=None,
                person_score=None,
                detector_ms=0.0,
                pose_ms=None,
                total_ms=0.0,
                evidence_age_ms=initial_age * 1000.0,
                timed_out=True,
            )
        detector_started = started
        boxes = self._detect(bgra_frame, frame_width, frame_height)
        detector_finished = _finite(
            self._clock(),
            field_name="detector_finished",
        )
        detector_ms = _monotonic_elapsed_ms(
            detector_finished,
            detector_started,
            field_name="detector_finished",
        )
        detector_evidence_age_ms = _monotonic_elapsed_ms(
            detector_finished,
            captured,
            field_name="detector_finished",
        )
        if detector_evidence_age_ms >= self._config.maximum_result_age_seconds * 1000.0:
            return PoseInferenceResult(
                points=None,
                person_score=None,
                detector_ms=detector_ms,
                pose_ms=None,
                total_ms=detector_ms,
                evidence_age_ms=detector_evidence_age_ms,
                timed_out=True,
            )
        frame_area = frame_width * frame_height
        if (
            len(boxes) != 1
            or boxes[0].area / frame_area < self._config.minimum_person_area_fraction
        ):
            finished = _finite(self._clock(), field_name="inference_finished")
            _monotonic_elapsed_ms(
                finished,
                detector_finished,
                field_name="inference_finished",
            )
            total_ms = _monotonic_elapsed_ms(
                finished,
                started,
                field_name="inference_finished",
            )
            evidence_age_ms = _monotonic_elapsed_ms(
                finished,
                captured,
                field_name="inference_finished",
            )
            timed_out = evidence_age_ms >= self._config.maximum_result_age_seconds * 1000.0
            return PoseInferenceResult(
                points=None,
                person_score=None,
                detector_ms=detector_ms,
                pose_ms=None,
                total_ms=total_ms,
                evidence_age_ms=evidence_age_ms,
                timed_out=timed_out,
            )
        person = boxes[0]
        pose_started = _finite(self._clock(), field_name="pose_started")
        _monotonic_elapsed_ms(
            pose_started,
            detector_finished,
            field_name="pose_started",
        )
        points = self._infer_pose(
            bgra_frame,
            frame_width,
            frame_height,
            person,
        )
        finished = _finite(self._clock(), field_name="inference_finished")
        pose_ms = _monotonic_elapsed_ms(
            finished,
            pose_started,
            field_name="inference_finished",
        )
        total_ms = _monotonic_elapsed_ms(
            finished,
            started,
            field_name="inference_finished",
        )
        evidence_age_ms = _monotonic_elapsed_ms(
            finished,
            captured,
            field_name="inference_finished",
        )
        timed_out = evidence_age_ms >= self._config.maximum_result_age_seconds * 1000.0
        return PoseInferenceResult(
            points=None if timed_out else points,
            person_score=(person.score if points is not None and not timed_out else None),
            detector_ms=detector_ms,
            pose_ms=pose_ms,
            total_ms=total_ms,
            evidence_age_ms=evidence_age_ms,
            timed_out=timed_out,
        )

    def close(self) -> None:
        self._closed = True
        for net_name in ("_detector", "_pose"):
            net = getattr(self, net_name, None)
            if net is not None and hasattr(net, "clear"):
                net.clear()
            setattr(self, net_name, None)

    def __enter__(self) -> NcnnPersonPoseEstimator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "DEFAULT_NCNN_MODEL_DIRECTORY",
    "DEFAULT_NCNN_RUNTIME_PATH",
    "NANODET_ARCHIVE_SHA256",
    "NANODET_ARCHIVE_SIZE_BYTES",
    "NANODET_ARCHIVE_URL",
    "NANODET_BIN_SHA256",
    "NANODET_BIN_SIZE_BYTES",
    "NANODET_PARAM_SHA256",
    "NANODET_PARAM_SIZE_BYTES",
    "NCNN_ARMV7_WHEEL_FILENAME",
    "NCNN_ARMV7_WHEEL_SHA256",
    "NCNN_ARMV7_WHEEL_SIZE_BYTES",
    "NCNN_ARMV7_WHEEL_URL",
    "NCNN_INIT_FILENAME",
    "NCNN_INIT_SHA256",
    "NCNN_INIT_SIZE_BYTES",
    "NCNN_LIBGOMP_FILENAME",
    "NCNN_LIBGOMP_SHA256",
    "NCNN_LIBGOMP_SIZE_BYTES",
    "NCNN_NATIVE_FILENAME",
    "NCNN_NATIVE_SHA256",
    "NCNN_NATIVE_SIZE_BYTES",
    "NCNN_RUNTIME_VERSION",
    "RTMPOSE_ARCHIVE_SHA256",
    "RTMPOSE_ARCHIVE_SIZE_BYTES",
    "RTMPOSE_ARCHIVE_URL",
    "RTMPOSE_BIN_SHA256",
    "RTMPOSE_BIN_SIZE_BYTES",
    "RTMPOSE_PARAM_SHA256",
    "RTMPOSE_PARAM_SIZE_BYTES",
    "NcnnPersonPoseEstimator",
    "NcnnPoseConfig",
    "PersonBox",
    "PoseCrop",
    "PoseInferenceResult",
    "decode_nanodet_person_boxes",
    "decode_rtmpose_simcc",
    "person_box_to_pose_crop",
    "validate_pinned_file",
]
