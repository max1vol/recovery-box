"""Federated learning primitives for sanitized rehabilitation features.

The Flower application modules intentionally are not imported here.  This keeps the
data and model code usable in local tests and tooling that do not install Flower.
"""

from .data import SanitizedDataset, load_sanitized_jsonl, split_dataset
from .model import (
    ModelParameters,
    evaluate,
    initial_parameters,
    train,
    validate_parameters,
    weighted_average_parameters,
)
from .pose_adapter import PostSessionRepSummary, pose_window_to_post_session_record
from .schema import (
    EXERCISE_ID,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    LABEL_DEFINITION_VERSION,
    LABEL_MEANINGS,
    MODEL_PARAMETER_SHAPES,
    MODEL_SCHEMA_SIGNATURE,
)

__all__ = [
    "EXERCISE_ID",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "LABEL_DEFINITION_VERSION",
    "LABEL_MEANINGS",
    "MODEL_PARAMETER_SHAPES",
    "MODEL_SCHEMA_SIGNATURE",
    "ModelParameters",
    "PostSessionRepSummary",
    "SanitizedDataset",
    "evaluate",
    "initial_parameters",
    "load_sanitized_jsonl",
    "pose_window_to_post_session_record",
    "split_dataset",
    "train",
    "validate_parameters",
    "weighted_average_parameters",
]
