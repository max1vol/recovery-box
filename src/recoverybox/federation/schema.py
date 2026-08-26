"""Closed schema for model-ready, pose-derived rehabilitation features."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import SanitizedDataError

FEATURE_SCHEMA_VERSION = "rehab-quality-features/v1"
"""Version of the only feature payload accepted by the first hackathon model."""

EXERCISE_ID = "seated-knee-extension"
"""The sole exercise represented by the first Flower application bundle."""

LABEL_DEFINITION_VERSION = "seated-knee-extension-rep-quality/v1"
"""Version of the fixed binary quality-label semantics."""

LABEL_MEANINGS = (
    "rep-needs-coaching-review",
    "rep-meets-prescribed-form-criteria",
)
"""Meaning of labels 0 and 1, respectively; neither label is a safety decision."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One bounded, scalar feature and its fixed normalization range."""

    name: str
    minimum: float
    maximum: float

    def normalize(self, value: float) -> float:
        """Map the approved physical range to ``[-1, 1]``."""

        return (2.0 * (value - self.minimum) / (self.maximum - self.minimum)) - 1.0


FEATURE_SPECS = (
    FeatureSpec("joint_angle_deg", 0.0, 180.0),
    FeatureSpec("angular_velocity_deg_s", -720.0, 720.0),
    FeatureSpec("pose_confidence", 0.0, 1.0),
    FeatureSpec("camera_disagreement_deg", 0.0, 180.0),
    FeatureSpec("range_progress", 0.0, 1.0),
    FeatureSpec("rep_duration_s", 0.1, 30.0),
    FeatureSpec("stability_score", 0.0, 1.0),
    FeatureSpec("symmetry_score", 0.0, 1.0),
)

FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_COUNT = len(FEATURE_NAMES)
MODEL_PARAMETER_SHAPES = ((FEATURE_COUNT,), (1,))

_MODEL_SCHEMA_CANONICAL = json.dumps(
    {
        "exercise_id": EXERCISE_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": [
            {
                "name": spec.name,
                "minimum": spec.minimum,
                "maximum": spec.maximum,
                "normalization": {
                    "method": "affine-min-max/v1",
                    "output_minimum": -1.0,
                    "output_maximum": 1.0,
                    "out_of_range": "reject",
                },
            }
            for spec in FEATURE_SPECS
        ],
        "label_definition": {
            "version": LABEL_DEFINITION_VERSION,
            "meanings": [
                {"label": label, "meaning": meaning} for label, meaning in enumerate(LABEL_MEANINGS)
            ],
        },
        "model": "binary-logistic-regression/v1",
        "parameter_shapes": [list(shape) for shape in MODEL_PARAMETER_SHAPES],
    },
    separators=(",", ":"),
    sort_keys=True,
)
MODEL_SCHEMA_SIGNATURE = hashlib.sha256(_MODEL_SCHEMA_CANONICAL.encode()).hexdigest()

ALLOWED_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "exercise_id",
        "label_definition_version",
        "model_schema_signature",
        "features",
        "label",
        "sample_id",
    }
)
REQUIRED_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "exercise_id",
        "label_definition_version",
        "model_schema_signature",
        "features",
        "label",
    }
)

# These fields carry raw or replayable media and must never enter the model store.
RAW_MEDIA_FIELD_NAMES = frozenset(
    {
        "audio",
        "audio_bytes",
        "audio_path",
        "frame",
        "frame_bytes",
        "frame_path",
        "frames",
        "image",
        "image_bytes",
        "image_path",
        "media",
        "media_bytes",
        "pcm",
        "raw_audio",
        "raw_frame",
        "raw_media",
        "transcript",
        "video",
        "video_bytes",
        "video_path",
        "waveform",
    }
)


def _path(parent: str, child: object) -> str:
    return f"{parent}.{child}"


def reject_raw_media_fields(value: Any, *, path: str = "$") -> None:
    """Reject raw-media keys at any nesting level before parsing a record."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in RAW_MEDIA_FIELD_NAMES:
                raise SanitizedDataError(f"raw media field {_path(path, key)!r} is forbidden")
            reject_raw_media_fields(item, path=_path(path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_raw_media_fields(item, path=f"{path}[{index}]")


def validate_and_normalize_record(record: Any) -> tuple[list[float], int]:
    """Validate one JSON object and return normalized features plus its label.

    Only a closed allowlist is accepted.  Identifiers are discarded by the caller and
    never become model features.
    """

    if not isinstance(record, Mapping):
        raise SanitizedDataError("each JSONL row must be an object")

    reject_raw_media_fields(record)

    keys = frozenset(record.keys())
    missing = REQUIRED_RECORD_FIELDS - keys
    unknown = keys - ALLOWED_RECORD_FIELDS
    if missing:
        raise SanitizedDataError(f"record is missing fields: {sorted(missing)}")
    if unknown:
        raise SanitizedDataError(f"record has unapproved fields: {sorted(unknown)}")

    version = record["schema_version"]
    if version != FEATURE_SCHEMA_VERSION:
        raise SanitizedDataError(
            f"feature schema {version!r} is incompatible; expected {FEATURE_SCHEMA_VERSION!r}"
        )

    exercise_id = record["exercise_id"]
    if exercise_id != EXERCISE_ID:
        raise SanitizedDataError(
            f"exercise {exercise_id!r} is incompatible; expected {EXERCISE_ID!r}"
        )

    label_definition_version = record["label_definition_version"]
    if label_definition_version != LABEL_DEFINITION_VERSION:
        raise SanitizedDataError(
            f"label definition {label_definition_version!r} is incompatible; expected "
            f"{LABEL_DEFINITION_VERSION!r}"
        )

    model_schema_signature = record["model_schema_signature"]
    if model_schema_signature != MODEL_SCHEMA_SIGNATURE:
        raise SanitizedDataError("model schema signature is incompatible")

    if "sample_id" in record:
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or len(sample_id) > 128:
            raise SanitizedDataError("sample_id must be a non-empty string up to 128 chars")

    features = record["features"]
    if not isinstance(features, Mapping):
        raise SanitizedDataError("features must be an object")
    feature_keys = frozenset(features.keys())
    expected_keys = frozenset(FEATURE_NAMES)
    if feature_keys != expected_keys:
        missing_features = expected_keys - feature_keys
        unknown_features = feature_keys - expected_keys
        details: list[str] = []
        if missing_features:
            details.append(f"missing={sorted(missing_features)}")
        if unknown_features:
            details.append(f"unapproved={sorted(unknown_features)}")
        raise SanitizedDataError("feature fields do not match schema: " + ", ".join(details))

    normalized: list[float] = []
    for spec in FEATURE_SPECS:
        raw_value = features[spec.name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise SanitizedDataError(f"feature {spec.name!r} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise SanitizedDataError(f"feature {spec.name!r} must be finite")
        if not spec.minimum <= value <= spec.maximum:
            raise SanitizedDataError(
                f"feature {spec.name!r}={value} is outside [{spec.minimum}, {spec.maximum}]"
            )
        normalized.append(spec.normalize(value))

    label = record["label"]
    if (
        isinstance(label, bool)
        or not isinstance(label, int)
        or not 0 <= label < len(LABEL_MEANINGS)
    ):
        raise SanitizedDataError(
            f"label must be an integer defined by {LABEL_DEFINITION_VERSION!r}"
        )

    return normalized, label
