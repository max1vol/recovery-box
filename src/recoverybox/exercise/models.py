"""Typed, process-local pose inputs and semantic exercise events.

The landmark frame mirrors MediaPipe Pose's 33-landmark ordering, but has no
dependency on MediaPipe.  It is intended to live inside the camera process:
downstream components should receive only ``SquatAnalysis`` and its derived
numeric values, never the landmark tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from math import isfinite
from numbers import Real

MEDIAPIPE_POSE_LANDMARK_COUNT = 33


class MediaPipePoseLandmark(IntEnum):
    """Stable indices from the MediaPipe Pose 33-landmark schema."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def _finite_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _unit_interval(value: object, *, field_name: str) -> float:
    converted = _finite_real(value, field_name=field_name)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return converted


@dataclass(frozen=True, slots=True)
class NormalizedLandmark:
    """One numeric landmark produced by a MediaPipe-compatible pose model.

    MediaPipe can extrapolate ``x`` and ``y`` slightly outside the image, so
    coordinates must be finite but are deliberately not clamped to ``[0, 1]``.
    Visibility and presence remain closed confidence values.
    """

    x: float
    y: float
    z: float
    visibility: float
    presence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_real(self.x, field_name="x"))
        object.__setattr__(self, "y", _finite_real(self.y, field_name="y"))
        object.__setattr__(self, "z", _finite_real(self.z, field_name="z"))
        object.__setattr__(
            self,
            "visibility",
            _unit_interval(self.visibility, field_name="visibility"),
        )
        object.__setattr__(
            self,
            "presence",
            _unit_interval(self.presence, field_name="presence"),
        )


@dataclass(frozen=True, slots=True)
class MediaPipePoseFrame:
    """Exactly one person's 33 numeric landmarks at a monotonic timestamp."""

    timestamp_ms: int
    image_width: int
    image_height: int
    landmarks: tuple[NormalizedLandmark, ...]

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ms, bool) or not isinstance(self.timestamp_ms, int):
            raise TypeError("timestamp_ms must be an integer")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        for field_name in ("image_width", "image_height"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if type(self.landmarks) is not tuple:
            raise TypeError("landmarks must be an immutable tuple")
        if len(self.landmarks) != MEDIAPIPE_POSE_LANDMARK_COUNT:
            raise ValueError(
                f"landmarks must contain exactly {MEDIAPIPE_POSE_LANDMARK_COUNT} values"
            )
        if not all(isinstance(landmark, NormalizedLandmark) for landmark in self.landmarks):
            raise TypeError("landmarks must contain only NormalizedLandmark values")

    def landmark(self, name: MediaPipePoseLandmark) -> NormalizedLandmark:
        """Return one landmark by its typed MediaPipe schema index."""

        if not isinstance(name, MediaPipePoseLandmark):
            raise TypeError("name must be a MediaPipePoseLandmark")
        return self.landmarks[name]


class SquatPhase(StrEnum):
    """Last movement phase confirmed by the deterministic tracker."""

    UNKNOWN = "unknown"
    STANDING = "standing"
    DOWN = "down"


class SquatEventType(StrEnum):
    """Closed semantic events that an outer Guardian may map to cue IDs."""

    REP_COMPLETED = "rep_completed"
    ARMS_NOT_IN_T = "arms_not_in_t"


class SquatAssessmentIssue(StrEnum):
    """Closed reasons a frame was not safe to use for progression."""

    NON_MONOTONIC_TIMESTAMP = "non_monotonic_timestamp"
    STALE_FRAME_GAP = "stale_frame_gap"
    NO_POSE = "no_pose"
    CAMERA_TIMEOUT = "camera_timeout"
    LOW_VISIBILITY = "low_visibility"
    LOW_PRESENCE = "low_presence"
    OUT_OF_FRAME_LANDMARK = "out_of_frame_landmark"
    INVALID_LEG_GEOMETRY = "invalid_leg_geometry"
    BILATERAL_KNEE_DISAGREEMENT = "bilateral_knee_disagreement"
    INVALID_ARM_GEOMETRY = "invalid_arm_geometry"


@dataclass(frozen=True, slots=True)
class SquatEvent:
    """One fixed semantic event; it contains no phrase or playback command."""

    event_type: SquatEventType
    rep_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, SquatEventType):
            raise TypeError("event_type must be a SquatEventType")
        if self.event_type is SquatEventType.REP_COMPLETED:
            if (
                isinstance(self.rep_count, bool)
                or not isinstance(self.rep_count, int)
                or self.rep_count < 1
            ):
                raise ValueError("rep_completed events require a positive rep_count")
        elif self.rep_count is not None:
            raise ValueError("only rep_completed events may carry rep_count")


@dataclass(frozen=True, slots=True)
class SquatAnalysis:
    """Derived result safe to pass out of the camera/exercise process."""

    timestamp_ms: int
    assessable: bool
    phase: SquatPhase
    rep_count: int
    events: tuple[SquatEvent, ...]
    issues: tuple[SquatAssessmentIssue, ...]
    confidence: float
    knee_angle_degrees: float | None
    arms_in_t: bool | None

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ms, bool) or not isinstance(self.timestamp_ms, int):
            raise TypeError("timestamp_ms must be an integer")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if not isinstance(self.assessable, bool):
            raise TypeError("assessable must be a boolean")
        if not isinstance(self.phase, SquatPhase):
            raise TypeError("phase must be a SquatPhase")
        if isinstance(self.rep_count, bool) or not isinstance(self.rep_count, int):
            raise TypeError("rep_count must be an integer")
        if self.rep_count < 0:
            raise ValueError("rep_count must be non-negative")
        if type(self.events) is not tuple or not all(
            isinstance(event, SquatEvent) for event in self.events
        ):
            raise TypeError("events must be a tuple of SquatEvent values")
        if type(self.issues) is not tuple or not all(
            isinstance(issue, SquatAssessmentIssue) for issue in self.issues
        ):
            raise TypeError("issues must be a tuple of SquatAssessmentIssue values")
        if self.assessable == bool(self.issues):
            raise ValueError("assessable results have no issues; withheld results require issues")
        confidence = _unit_interval(self.confidence, field_name="confidence")
        object.__setattr__(self, "confidence", confidence)
        if self.knee_angle_degrees is not None:
            knee_angle = _finite_real(
                self.knee_angle_degrees,
                field_name="knee_angle_degrees",
            )
            if not 0.0 <= knee_angle <= 180.0:
                raise ValueError("knee_angle_degrees must be between 0 and 180")
            object.__setattr__(self, "knee_angle_degrees", knee_angle)
        if self.arms_in_t is not None and not isinstance(self.arms_in_t, bool):
            raise TypeError("arms_in_t must be a boolean when provided")
        if not self.assessable and self.events:
            raise ValueError("withheld results may not emit exercise events")
