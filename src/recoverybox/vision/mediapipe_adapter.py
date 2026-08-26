"""Pure conversion from MediaPipe Pose Landmarker results to exercise input.

This module intentionally does not import :mod:`mediapipe`.  The small
protocols below describe the result shape that Google MediaPipe exposes while
keeping landmark conversion testable without the optional camera runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import Protocol

from recoverybox.exercise import (
    MEDIAPIPE_POSE_LANDMARK_COUNT,
    MediaPipePoseFrame,
    NormalizedLandmark,
)


class MediaPipeLandmarkLike(Protocol):
    """Structural subset of a MediaPipe normalized landmark."""

    x: float
    y: float
    z: float
    visibility: float
    presence: float


class MediaPipePoseResultLike(Protocol):
    """Structural subset of ``PoseLandmarkerResult`` used by the adapter."""

    pose_landmarks: Sequence[Sequence[MediaPipeLandmarkLike]]


class MediaPipeResultError(ValueError):
    """Raised when MediaPipe returns an unexpected or unsafe result shape."""


def pose_frame_from_mediapipe_result(
    result: MediaPipePoseResultLike,
    *,
    timestamp_ms: int,
    image_width: int,
    image_height: int,
) -> MediaPipePoseFrame | None:
    """Convert one Pose Landmarker result without retaining MediaPipe objects.

    ``None`` means that MediaPipe found no person in this frame.  The webcam
    runtime is configured for one pose, so receiving multiple poses is treated
    as ambiguous rather than silently selecting a person.  Image dimensions
    travel with a detected pose so exercise geometry can undo normalized-image
    aspect-ratio distortion.
    """

    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
        raise ValueError("timestamp_ms must be a non-negative integer")
    _require_positive_image_dimension(image_width, "image_width")
    _require_positive_image_dimension(image_height, "image_height")

    try:
        poses = result.pose_landmarks
    except AttributeError as exc:
        raise MediaPipeResultError("result is missing pose_landmarks") from exc

    if isinstance(poses, (bytes, bytearray, memoryview, str)) or not isinstance(poses, Sequence):
        raise MediaPipeResultError("pose_landmarks must be a sequence of poses")
    if not poses:
        return None
    if len(poses) != 1:
        raise MediaPipeResultError("expected at most one pose from the webcam")

    source_landmarks = poses[0]
    if isinstance(source_landmarks, (bytes, bytearray, memoryview, str)) or not isinstance(
        source_landmarks, Sequence
    ):
        raise MediaPipeResultError("the detected pose must be a landmark sequence")
    if len(source_landmarks) != MEDIAPIPE_POSE_LANDMARK_COUNT:
        raise MediaPipeResultError(
            f"expected {MEDIAPIPE_POSE_LANDMARK_COUNT} pose landmarks, "
            f"received {len(source_landmarks)}"
        )

    converted: list[NormalizedLandmark] = []
    for index, landmark in enumerate(source_landmarks):
        try:
            converted.append(
                NormalizedLandmark(
                    x=_finite_landmark_value(landmark, "x"),
                    y=_finite_landmark_value(landmark, "y"),
                    z=_finite_landmark_value(landmark, "z"),
                    visibility=_unit_landmark_value(landmark, "visibility"),
                    presence=_unit_landmark_value(landmark, "presence"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise MediaPipeResultError(f"invalid pose landmark at index {index}: {exc}") from exc

    return MediaPipePoseFrame(
        timestamp_ms=timestamp_ms,
        image_width=image_width,
        image_height=image_height,
        landmarks=tuple(converted),
    )


def _finite_landmark_value(landmark: object, field_name: str) -> float:
    try:
        value = getattr(landmark, field_name)
    except AttributeError as exc:
        raise MediaPipeResultError(f"landmark is missing {field_name}") from exc
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _unit_landmark_value(landmark: object, field_name: str) -> float:
    value = _finite_landmark_value(landmark, field_name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return value


def _require_positive_image_dimension(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
