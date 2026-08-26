"""Explicit transition from a live pose window to a post-session Flower record."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import SanitizedDataError
from .schema import (
    EXERCISE_ID,
    FEATURE_SCHEMA_VERSION,
    LABEL_DEFINITION_VERSION,
    MODEL_SCHEMA_SIGNATURE,
    validate_and_normalize_record,
)

if TYPE_CHECKING:
    from recoverybox.pose import SanitizedPoseFeatureWindow


@dataclass(frozen=True, slots=True)
class PostSessionRepSummary:
    """Sanitized rep-level values produced by deterministic local session logic."""

    range_progress: float
    rep_duration_s: float
    stability_score: float
    symmetry_score: float
    label: int
    anonymous_sample_id: str | None = None

    def __post_init__(self) -> None:
        values = (
            self.range_progress,
            self.rep_duration_s,
            self.stability_score,
            self.symmetry_score,
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise SanitizedDataError("post-session summary values must be numeric")
        if not all(isfinite(float(value)) for value in values):
            raise SanitizedDataError("post-session summary values must be finite")
        if (
            isinstance(self.label, bool)
            or not isinstance(self.label, int)
            or self.label not in (0, 1)
        ):
            raise SanitizedDataError("post-session label must be integer 0 or 1")
        if self.anonymous_sample_id is not None and (
            not isinstance(self.anonymous_sample_id, str)
            or not self.anonymous_sample_id
            or len(self.anonymous_sample_id) > 128
        ):
            raise SanitizedDataError(
                "anonymous_sample_id must be a non-empty string up to 128 chars"
            )


def pose_window_to_post_session_record(
    window: SanitizedPoseFeatureWindow,
    summary: PostSessionRepSummary,
    *,
    joint_index: int,
) -> dict[str, Any]:
    """Create one closed-schema record from an immutable numeric pose window.

    The adapter consumes only derived numeric rows. It stores the selected joint's
    mean angle and velocity, mean confidence, and maximum camera disagreement.
    Monotonic timestamps, camera identifiers, session identifiers, and media never
    enter the returned object. ``joint_index`` is the exercise configuration's
    approved angle position.
    """

    # Imported here so model/data unit tests remain independent of the live pose stack.
    from recoverybox.pose import (
        POSE_FEATURE_SCHEMA_VERSION,
        SanitizedPoseFeatureWindow,
    )

    if not isinstance(window, SanitizedPoseFeatureWindow):
        raise SanitizedDataError("window must be a SanitizedPoseFeatureWindow")
    if window.schema_version != POSE_FEATURE_SCHEMA_VERSION:
        raise SanitizedDataError(
            f"pose schema {window.schema_version!r} is incompatible; expected "
            f"{POSE_FEATURE_SCHEMA_VERSION!r}"
        )
    if isinstance(joint_index, bool) or not isinstance(joint_index, int) or joint_index < 0:
        raise SanitizedDataError("joint_index must be a non-negative integer")
    if not window.rows:
        raise SanitizedDataError("pose window must contain at least one row")
    if joint_index >= len(window.rows[0].angles_degrees):
        raise SanitizedDataError("joint_index is outside the pose angle schema")

    angles = np.asarray([row.angles_degrees[joint_index] for row in window.rows], dtype=np.float64)
    angular_velocities = np.asarray(
        [row.angular_velocity_degrees_per_second[joint_index] for row in window.rows],
        dtype=np.float64,
    )
    confidences = np.asarray([row.confidence for row in window.rows], dtype=np.float64)
    disagreements = np.asarray(
        [row.camera_disagreement_degrees for row in window.rows], dtype=np.float64
    )

    record: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "exercise_id": EXERCISE_ID,
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "model_schema_signature": MODEL_SCHEMA_SIGNATURE,
        "features": {
            "joint_angle_deg": float(np.mean(angles)),
            "angular_velocity_deg_s": float(np.mean(angular_velocities)),
            "pose_confidence": float(np.mean(confidences)),
            "camera_disagreement_deg": float(np.max(disagreements)),
            "range_progress": float(summary.range_progress),
            "rep_duration_s": float(summary.rep_duration_s),
            "stability_score": float(summary.stability_score),
            "symmetry_score": float(summary.symmetry_score),
        },
        "label": summary.label,
    }
    if summary.anonymous_sample_id is not None:
        record["sample_id"] = summary.anonymous_sample_id

    # Apply the same closed-schema validation that the on-disk reader uses before
    # allowing a record to cross the post-session boundary.
    validate_and_normalize_record(record)
    return record
