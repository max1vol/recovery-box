"""Local pose summaries, conservative fusion, and sanitized features."""

from .features import (
    POSE_FEATURE_SCHEMA_VERSION,
    PoseFeatureWindowBuilder,
    SanitizedPoseFeature,
    SanitizedPoseFeatureWindow,
)
from .fusion import DualViewPoseFuser, DualViewPoseSynchronizer, PoseFusionConfig
from .models import (
    POSE_ANGLE_SCHEMA_ID,
    SEATED_KNEE_EXTENSION_ANGLE_NAMES,
    CameraView,
    FusedPoseSummary,
    FusionIssue,
    FusionResult,
    PoseAngleName,
    PoseAngles,
    PoseAngleSchema,
    PoseViewSummary,
)

__all__ = [
    "POSE_ANGLE_SCHEMA_ID",
    "POSE_FEATURE_SCHEMA_VERSION",
    "SEATED_KNEE_EXTENSION_ANGLE_NAMES",
    "CameraView",
    "DualViewPoseFuser",
    "DualViewPoseSynchronizer",
    "FusedPoseSummary",
    "FusionIssue",
    "FusionResult",
    "PoseAngleName",
    "PoseAngleSchema",
    "PoseAngles",
    "PoseFeatureWindowBuilder",
    "PoseFusionConfig",
    "PoseViewSummary",
    "SanitizedPoseFeature",
    "SanitizedPoseFeatureWindow",
]
