"""Numeric-only pose messages exchanged across the camera worker boundary.

The camera process is expected to discard images after producing a
``PoseViewSummary``.  Keeping the boundary type deliberately small makes it
impossible for downstream code to accidentally retain a frame.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from math import isfinite
from numbers import Real
from typing import overload


class CameraView(IntEnum):
    """Stable numeric labels for the two configured camera viewpoints."""

    PRIMARY = 0
    SECONDARY = 1


class PoseAngleName(StrEnum):
    """Approved semantic positions in the knee-extension angle vector."""

    PRESCRIBED_KNEE_FLEXION = "prescribed_knee_flexion"
    PRESCRIBED_HIP_FLEXION = "prescribed_hip_flexion"
    TRUNK_FLEXION = "trunk_flexion"


class PoseAngleSchema(StrEnum):
    """Closed, versioned schemas allowed across the camera boundary."""

    SEATED_KNEE_EXTENSION_V1 = "recoverybox.seated-knee-extension-angles/v1"

    @property
    def ordered_names(self) -> tuple[PoseAngleName, ...]:
        """Return the only valid semantic order for this schema."""

        if self is PoseAngleSchema.SEATED_KNEE_EXTENSION_V1:
            return SEATED_KNEE_EXTENSION_ANGLE_NAMES
        raise AssertionError(f"unhandled pose angle schema: {self!r}")


SEATED_KNEE_EXTENSION_ANGLE_NAMES = (
    PoseAngleName.PRESCRIBED_KNEE_FLEXION,
    PoseAngleName.PRESCRIBED_HIP_FLEXION,
    PoseAngleName.TRUNK_FLEXION,
)
POSE_ANGLE_SCHEMA_ID = PoseAngleSchema.SEATED_KNEE_EXTENSION_V1.value


@dataclass(frozen=True, slots=True)
class PoseAngles(Sequence[float]):
    """Typed angle vector whose schema fixes width and semantic order.

    A dedicated type is required at every pose boundary.  The nested numeric
    tuple is deliberately strict as well: callers must materialize exactly one
    immutable vector rather than supplying bytes, text, or a lazy iterable.
    """

    schema: PoseAngleSchema
    values_degrees: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema, PoseAngleSchema):
            raise TypeError("schema must be an approved PoseAngleSchema")
        if type(self.values_degrees) is not tuple:
            raise TypeError("values_degrees must be an immutable tuple")
        if len(self.values_degrees) != len(self.schema.ordered_names):
            raise ValueError(
                f"{self.schema.value} requires exactly "
                f"{len(self.schema.ordered_names)} ordered angles"
            )

        normalized: list[float] = []
        for value in self.values_degrees:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("angle values must be real numbers, never text or booleans")
            converted = float(value)
            if not isfinite(converted) or not 0.0 <= converted <= 180.0:
                raise ValueError("angle values must be finite and between 0 and 180 degrees")
            normalized.append(converted)
        object.__setattr__(self, "values_degrees", tuple(normalized))

    @classmethod
    def seated_knee_extension(cls, values_degrees: tuple[float, ...]) -> PoseAngles:
        """Build the approved v1 knee-extension vector."""

        return cls(
            schema=PoseAngleSchema.SEATED_KNEE_EXTENSION_V1,
            values_degrees=values_degrees,
        )

    @property
    def schema_id(self) -> str:
        return self.schema.value

    @property
    def ordered_names(self) -> tuple[PoseAngleName, ...]:
        return self.schema.ordered_names

    def named_values(self) -> tuple[tuple[PoseAngleName, float], ...]:
        """Expose the unambiguous name-to-value order for diagnostics."""

        return tuple(zip(self.ordered_names, self.values_degrees, strict=True))

    def __len__(self) -> int:
        return len(self.values_degrees)

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[float, ...]: ...

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        return self.values_degrees[index]

    def __iter__(self) -> Iterator[float]:
        return iter(self.values_degrees)


class FusionIssue(StrEnum):
    """Reasons why two views cannot safely produce a movement estimate."""

    SAME_VIEW = "same_view"
    MISSING_PRIMARY = "missing_primary"
    MISSING_SECONDARY = "missing_secondary"
    STALE_PRIMARY = "stale_primary"
    STALE_SECONDARY = "stale_secondary"
    FUTURE_TIMESTAMP = "future_timestamp"
    EXCESSIVE_TIMESTAMP_SKEW = "excessive_timestamp_skew"
    LOW_PRIMARY_CONFIDENCE = "low_primary_confidence"
    LOW_SECONDARY_CONFIDENCE = "low_secondary_confidence"
    ANGLE_SCHEMA_MISMATCH = "angle_schema_mismatch"
    CAMERA_DISAGREEMENT = "camera_disagreement"


def _require_non_negative_integer(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class PoseViewSummary:
    """Derived numeric pose measurements for a single camera observation.

    ``angles_degrees`` is a typed vector with a closed, versioned semantic
    order.  It contains derived joint angles, never landmarks or pixels.  The
    monotonic timestamp is meaningful only within the device process and is
    removed before any federated feature record is created.
    """

    view: CameraView
    monotonic_timestamp_ms: int
    angles_degrees: PoseAngles
    confidence: float

    def __post_init__(self) -> None:
        if isinstance(self.view, bool):
            raise ValueError("view must identify one of the configured cameras")
        try:
            view = CameraView(self.view)
        except ValueError as exc:
            raise ValueError("view must identify one of the configured cameras") from exc
        _require_non_negative_integer(
            self.monotonic_timestamp_ms,
            field_name="monotonic_timestamp_ms",
        )
        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")

        object.__setattr__(self, "view", view)
        if not isinstance(self.angles_degrees, PoseAngles):
            raise TypeError("angles_degrees must be a validated PoseAngles value")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class FusedPoseSummary:
    """Conservative consensus produced only from an assessable camera pair."""

    monotonic_timestamp_ms: int
    angles_degrees: PoseAngles
    confidence: float
    camera_disagreement_degrees: float
    timestamp_skew_ms: int

    def __post_init__(self) -> None:
        _require_non_negative_integer(
            self.monotonic_timestamp_ms,
            field_name="monotonic_timestamp_ms",
        )
        _require_non_negative_integer(self.timestamp_skew_ms, field_name="timestamp_skew_ms")
        if not isinstance(self.angles_degrees, PoseAngles):
            raise TypeError("angles_degrees must be a validated PoseAngles value")
        confidence = float(self.confidence)
        disagreement = float(self.camera_disagreement_degrees)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        if not isfinite(disagreement) or disagreement < 0.0:
            raise ValueError("camera_disagreement_degrees must be finite and non-negative")

        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "camera_disagreement_degrees", disagreement)


@dataclass(frozen=True, slots=True)
class FusionResult:
    """A fused summary or a closed set of reasons it was withheld."""

    fused: FusedPoseSummary | None
    issues: tuple[FusionIssue, ...] = ()

    def __post_init__(self) -> None:
        normalized_issues = tuple(FusionIssue(issue) for issue in self.issues)
        if (self.fused is None) == (not normalized_issues):
            raise ValueError("result must contain either a fused summary or one or more issues")
        object.__setattr__(self, "issues", normalized_issues)

    @property
    def assessable(self) -> bool:
        return self.fused is not None
