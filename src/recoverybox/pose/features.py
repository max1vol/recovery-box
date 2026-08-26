"""Privacy-minimized pose features suitable for local federated training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .models import FusedPoseSummary, PoseAngles

POSE_FEATURE_SCHEMA_VERSION = "recoverybox.pose.v1"


@dataclass(frozen=True, slots=True)
class SanitizedPoseFeature:
    """One timestamp-free, identifier-free row of derived movement features."""

    angles_degrees: PoseAngles
    angular_velocity_degrees_per_second: tuple[float, ...]
    confidence: float
    camera_disagreement_degrees: float

    def __post_init__(self) -> None:
        if not isinstance(self.angles_degrees, PoseAngles):
            raise TypeError("angles_degrees must be a validated PoseAngles value")
        if type(self.angular_velocity_degrees_per_second) is not tuple:
            raise TypeError("angular velocities must be an immutable tuple")
        if len(self.angles_degrees) != len(self.angular_velocity_degrees_per_second):
            raise ValueError("angles and velocities must use the same schema width")
        velocities: list[float] = []
        for value in self.angular_velocity_degrees_per_second:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("angular velocities must contain only real numbers")
            converted = float(value)
            if not isfinite(converted):
                raise ValueError("angular velocities must contain only finite values")
            velocities.append(converted)
        confidence = float(self.confidence)
        disagreement = float(self.camera_disagreement_degrees)
        scalars = (*self.angles_degrees, *velocities, confidence, disagreement)
        if not all(isfinite(value) for value in scalars):
            raise ValueError("features must contain only finite numeric values")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= disagreement <= 180.0:
            raise ValueError("camera disagreement must be between 0 and 180 degrees")
        object.__setattr__(self, "angular_velocity_degrees_per_second", tuple(velocities))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "camera_disagreement_degrees",
            disagreement,
        )

    def numeric_values(self) -> tuple[float, ...]:
        """Return a stable numeric order for model input."""

        return (
            *self.angles_degrees,
            *self.angular_velocity_degrees_per_second,
            self.confidence,
            self.camera_disagreement_degrees,
        )


@dataclass(frozen=True, slots=True)
class SanitizedPoseFeatureWindow:
    """Immutable local-data snapshot; contains no absolute time or identity."""

    schema_version: str
    rows: tuple[SanitizedPoseFeature, ...]

    def __post_init__(self) -> None:
        if self.schema_version != POSE_FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported pose feature schema: {self.schema_version!r}")
        normalized_rows = tuple(self.rows)
        if normalized_rows:
            expected_schema = normalized_rows[0].angles_degrees.schema
            if any(row.angles_degrees.schema != expected_schema for row in normalized_rows):
                raise ValueError("all feature rows must use the same angle schema")
        object.__setattr__(self, "rows", normalized_rows)

    def numeric_rows(self) -> tuple[tuple[float, ...], ...]:
        """Return model-ready rows without adding metadata or identifiers."""

        return tuple(row.numeric_values() for row in self.rows)


class PoseFeatureWindowBuilder:
    """Build a bounded feature window from consecutive fused summaries.

    Device-local timestamps are used only to calculate velocity.  They are not
    copied into ``SanitizedPoseFeature`` or its immutable snapshot.
    """

    def __init__(self, *, maximum_rows: int = 64) -> None:
        if isinstance(maximum_rows, bool) or maximum_rows < 1:
            raise ValueError("maximum_rows must be a positive integer")
        self._rows: deque[SanitizedPoseFeature] = deque(maxlen=maximum_rows)
        self._previous: FusedPoseSummary | None = None

    def add(self, summary: FusedPoseSummary) -> SanitizedPoseFeature:
        previous = self._previous
        if previous is None:
            velocity = tuple(0.0 for _ in summary.angles_degrees)
        else:
            if previous.angles_degrees.schema != summary.angles_degrees.schema:
                raise ValueError("angle schema changed within a feature window")
            elapsed_ms = summary.monotonic_timestamp_ms - previous.monotonic_timestamp_ms
            if elapsed_ms <= 0:
                raise ValueError("fused pose timestamps must increase monotonically")
            elapsed_seconds = elapsed_ms / 1000.0
            velocity = tuple(
                (current - prior) / elapsed_seconds
                for current, prior in zip(
                    summary.angles_degrees,
                    previous.angles_degrees,
                    strict=True,
                )
            )

        row = SanitizedPoseFeature(
            angles_degrees=summary.angles_degrees,
            angular_velocity_degrees_per_second=velocity,
            confidence=summary.confidence,
            camera_disagreement_degrees=summary.camera_disagreement_degrees,
        )
        self._rows.append(row)
        self._previous = summary
        return row

    def snapshot(self) -> SanitizedPoseFeatureWindow:
        return SanitizedPoseFeatureWindow(
            schema_version=POSE_FEATURE_SCHEMA_VERSION,
            rows=tuple(self._rows),
        )

    def clear(self) -> None:
        self._rows.clear()
        self._previous = None
