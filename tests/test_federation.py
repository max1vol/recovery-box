from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from recoverybox.federation.data import load_sanitized_jsonl, split_dataset
from recoverybox.federation.errors import (
    FederationConfigError,
    ModelSchemaError,
    SanitizedDataError,
)
from recoverybox.federation.local_client import LocalQualityClient
from recoverybox.federation.model import (
    evaluate,
    initial_parameters,
    train,
    validate_parameters,
    weighted_average_parameters,
)
from recoverybox.federation.paths import resolve_sanitized_feature_path
from recoverybox.federation.pose_adapter import (
    PostSessionRepSummary,
    pose_window_to_post_session_record,
)
from recoverybox.federation.schema import (
    EXERCISE_ID,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SPECS,
    LABEL_DEFINITION_VERSION,
    LABEL_MEANINGS,
    MODEL_PARAMETER_SHAPES,
    MODEL_SCHEMA_SIGNATURE,
    reject_raw_media_fields,
)
from recoverybox.pose import (
    POSE_FEATURE_SCHEMA_VERSION,
    PoseAngles,
    SanitizedPoseFeature,
    SanitizedPoseFeatureWindow,
)


def feature_record(label: int, *, offset: float = 0.0) -> dict[str, object]:
    if label == 1:
        features = {
            "joint_angle_deg": 120.0 + offset,
            "angular_velocity_deg_s": 100.0 + offset,
            "pose_confidence": 0.95,
            "camera_disagreement_deg": 2.0,
            "range_progress": 0.9,
            "rep_duration_s": 3.0,
            "stability_score": 0.9,
            "symmetry_score": 0.9,
        }
    else:
        features = {
            "joint_angle_deg": 35.0 + offset,
            "angular_velocity_deg_s": -100.0 + offset,
            "pose_confidence": 0.35,
            "camera_disagreement_deg": 55.0,
            "range_progress": 0.15,
            "rep_duration_s": 12.0,
            "stability_score": 0.2,
            "symmetry_score": 0.25,
        }
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "exercise_id": EXERCISE_ID,
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "model_schema_signature": MODEL_SCHEMA_SIGNATURE,
        "sample_id": f"sample-{label}-{offset}",
        "features": features,
        "label": label,
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_loads_closed_sanitized_schema_and_discards_metadata(tmp_path: Path) -> None:
    source = tmp_path / "client-0.jsonl"
    write_jsonl(source, [feature_record(0), feature_record(1)])

    dataset = load_sanitized_jsonl(source)

    assert dataset.features.shape == (2, len(FEATURE_NAMES))
    assert dataset.features.dtype == np.float32
    assert dataset.labels.tolist() == [0, 1]
    assert np.all(dataset.features >= -1.0)
    assert np.all(dataset.features <= 1.0)
    assert not hasattr(dataset, "sample_ids")


def test_pose_window_transitions_to_identifier_free_post_session_record(
    tmp_path: Path,
) -> None:
    window = SanitizedPoseFeatureWindow(
        schema_version=POSE_FEATURE_SCHEMA_VERSION,
        rows=(
            SanitizedPoseFeature(
                angles_degrees=PoseAngles.seated_knee_extension((80.0, 120.0, 5.0)),
                angular_velocity_degrees_per_second=(0.0, 0.0, 0.0),
                confidence=0.9,
                camera_disagreement_degrees=2.0,
            ),
            SanitizedPoseFeature(
                angles_degrees=PoseAngles.seated_knee_extension((100.0, 115.0, 4.0)),
                angular_velocity_degrees_per_second=(40.0, -10.0, -2.0),
                confidence=0.8,
                camera_disagreement_degrees=4.0,
            ),
        ),
    )
    summary = PostSessionRepSummary(
        range_progress=0.8,
        rep_duration_s=3.2,
        stability_score=0.75,
        symmetry_score=0.85,
        label=1,
    )

    record = pose_window_to_post_session_record(window, summary, joint_index=0)

    assert set(record) == {
        "schema_version",
        "exercise_id",
        "label_definition_version",
        "model_schema_signature",
        "features",
        "label",
    }
    assert record["schema_version"] == FEATURE_SCHEMA_VERSION
    assert record["exercise_id"] == EXERCISE_ID
    assert record["label_definition_version"] == LABEL_DEFINITION_VERSION
    assert record["model_schema_signature"] == MODEL_SCHEMA_SIGNATURE
    assert set(record["features"]) == set(FEATURE_NAMES)
    assert record["features"]["joint_angle_deg"] == pytest.approx(90.0)
    assert record["features"]["angular_velocity_deg_s"] == pytest.approx(20.0)
    assert record["features"]["camera_disagreement_deg"] == pytest.approx(4.0)
    serialized = json.dumps(record)
    for forbidden in (
        "timestamp",
        "session_id",
        "device_id",
        "camera_id",
        "frame",
        "audio",
    ):
        assert forbidden not in serialized

    destination = tmp_path / "client-0.jsonl"
    write_jsonl(destination, [record])
    dataset = load_sanitized_jsonl(destination)
    assert dataset.features.shape == (1, len(FEATURE_NAMES))


def test_pose_adapter_rejects_empty_window_and_unknown_joint() -> None:
    summary = PostSessionRepSummary(0.8, 3.0, 0.8, 0.8, 1)
    empty = SanitizedPoseFeatureWindow(
        schema_version=POSE_FEATURE_SCHEMA_VERSION,
        rows=(),
    )
    with pytest.raises(SanitizedDataError, match="at least one row"):
        pose_window_to_post_session_record(empty, summary, joint_index=0)

    pose_row = SanitizedPoseFeatureWindow(
        schema_version=POSE_FEATURE_SCHEMA_VERSION,
        rows=(
            SanitizedPoseFeature(
                PoseAngles.seated_knee_extension((90.0, 100.0, 5.0)),
                (0.0, 0.0, 0.0),
                0.9,
                1.0,
            ),
        ),
    )
    with pytest.raises(SanitizedDataError, match="outside"):
        pose_window_to_post_session_record(pose_row, summary, joint_index=3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_path", "/private/frame.jpg"),
        ("audio_bytes", "base64-data"),
        ("transcript", "spoken health information"),
    ],
)
def test_rejects_raw_or_replayable_media_fields(tmp_path: Path, field: str, value: str) -> None:
    source = tmp_path / "client-0.jsonl"
    record = feature_record(1)
    record[field] = value
    write_jsonl(source, [record])

    with pytest.raises(SanitizedDataError, match="raw media field"):
        load_sanitized_jsonl(source)


def test_recursive_raw_media_guard_descends_immutable_sequences() -> None:
    with pytest.raises(SanitizedDataError, match="raw media field"):
        reject_raw_media_fields({"rows": ({"image": b"raw-pixels"},)})


def test_rejects_feature_schema_drift_and_unknown_features(tmp_path: Path) -> None:
    source = tmp_path / "client-0.jsonl"
    wrong_version = feature_record(1)
    wrong_version["schema_version"] = "rehab-quality-features/v2"
    write_jsonl(source, [wrong_version])
    with pytest.raises(SanitizedDataError, match="incompatible"):
        load_sanitized_jsonl(source)

    extra_feature = feature_record(1)
    assert isinstance(extra_feature["features"], dict)
    extra_feature["features"]["image_embedding"] = 0.3
    write_jsonl(source, [extra_feature])
    with pytest.raises(SanitizedDataError, match="unapproved"):
        load_sanitized_jsonl(source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exercise_id", "standing-knee-extension", "exercise"),
        ("label_definition_version", "some-other-labels/v1", "label definition"),
        ("model_schema_signature", "bad-signature", "model schema signature"),
    ],
)
def test_rejects_cross_exercise_or_model_identity(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    source = tmp_path / "client-0.jsonl"
    record = feature_record(1)
    record[field] = value
    write_jsonl(source, [record])

    with pytest.raises(SanitizedDataError, match=message):
        load_sanitized_jsonl(source)


def test_model_signature_contract_has_exact_identity_ranges_labels_and_shapes() -> None:
    assert EXERCISE_ID == "seated-knee-extension"
    assert LABEL_DEFINITION_VERSION == "seated-knee-extension-rep-quality/v1"
    assert LABEL_MEANINGS == (
        "rep-needs-coaching-review",
        "rep-meets-prescribed-form-criteria",
    )
    assert [(spec.name, spec.minimum, spec.maximum) for spec in FEATURE_SPECS] == [
        ("joint_angle_deg", 0.0, 180.0),
        ("angular_velocity_deg_s", -720.0, 720.0),
        ("pose_confidence", 0.0, 1.0),
        ("camera_disagreement_deg", 0.0, 180.0),
        ("range_progress", 0.0, 1.0),
        ("rep_duration_s", 0.1, 30.0),
        ("stability_score", 0.0, 1.0),
        ("symmetry_score", 0.0, 1.0),
    ]
    assert MODEL_PARAMETER_SHAPES == ((len(FEATURE_NAMES),), (1,))
    # This digest changes if any identity, meaning, normalization rule/range, model,
    # or parameter shape changes, forcing an intentional protocol migration.
    assert MODEL_SCHEMA_SIGNATURE == (
        "cbaca87b8a73b3b5274a4bbaab6fe07be1add2122b51912c8341361816953182"
    )


def test_local_training_is_deterministic_and_reduces_loss(tmp_path: Path) -> None:
    source = tmp_path / "client-0.jsonl"
    records = [feature_record(label=index % 2, offset=(index % 3) * 0.5) for index in range(30)]
    write_jsonl(source, records)
    dataset = load_sanitized_jsonl(source)
    parameters = initial_parameters(seed=8)

    before = evaluate(parameters, dataset)
    first = train(parameters, dataset, epochs=80, learning_rate=0.2)
    second = train(parameters, dataset, epochs=80, learning_rate=0.2)

    assert first.loss < before.loss
    assert first.accuracy >= 0.95
    for first_array, second_array in zip(first.parameters, second.parameters, strict=True):
        np.testing.assert_array_equal(first_array, second_array)


def test_three_client_weighted_aggregate_and_shape_rejection(tmp_path: Path) -> None:
    updates = []
    counts = []
    for client_id in range(3):
        source = tmp_path / f"client-{client_id}.jsonl"
        records = [
            feature_record(label=index % 2, offset=client_id + index * 0.1) for index in range(12)
        ]
        write_jsonl(source, records)
        dataset = load_sanitized_jsonl(source)
        result = train(
            initial_parameters(seed=3),
            dataset,
            epochs=10,
            learning_rate=0.1,
        )
        updates.append(result.parameters)
        counts.append(len(dataset))

    aggregate = weighted_average_parameters(updates, counts)
    assert [array.shape for array in aggregate] == [(len(FEATURE_NAMES),), (1,)]
    expected = np.mean([update[0] for update in updates], axis=0)
    np.testing.assert_allclose(aggregate[0], expected, rtol=1e-6)

    incompatible = copy.deepcopy(updates)
    incompatible[2][0] = np.zeros((len(FEATURE_NAMES) + 1,), dtype=np.float32)
    with pytest.raises(ModelSchemaError, match="shape"):
        weighted_average_parameters(incompatible, counts)


def test_local_client_fails_closed_on_protocol_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "client-0.jsonl"
    write_jsonl(
        source,
        [feature_record(index % 2, offset=index * 0.1) for index in range(10)],
    )
    train_set, validation_set = split_dataset(load_sanitized_jsonl(source), seed=9)
    client = LocalQualityClient(train_set, validation_set, 2, 0.1)
    parameters = initial_parameters()
    valid_config = {
        "exercise-id": EXERCISE_ID,
        "feature-schema-version": FEATURE_SCHEMA_VERSION,
        "label-definition-version": LABEL_DEFINITION_VERSION,
        "model-schema-signature": MODEL_SCHEMA_SIGNATURE,
    }

    result = client.fit(parameters, valid_config)
    assert result.epochs == 2

    incompatible_config = dict(valid_config)
    incompatible_config["model-schema-signature"] = "bad-signature"
    with pytest.raises(ModelSchemaError, match="signature"):
        client.fit(parameters, incompatible_config)

    wrong_exercise = dict(valid_config)
    wrong_exercise["exercise-id"] = "standing-knee-extension"
    with pytest.raises(ModelSchemaError, match="server exercise"):
        client.fit(parameters, wrong_exercise)

    wrong_labels = dict(valid_config)
    wrong_labels["label-definition-version"] = "some-other-labels/v1"
    with pytest.raises(ModelSchemaError, match="label definition"):
        client.fit(parameters, wrong_labels)

    with pytest.raises(ModelSchemaError, match="feature schema"):
        client.evaluate(parameters, {})


def test_local_client_enforces_secagg_max_weight_boundary(tmp_path: Path) -> None:
    source = tmp_path / "client-0.jsonl"
    write_jsonl(
        source,
        [feature_record(index % 2, offset=index * 0.1) for index in range(6)],
    )
    train_set, validation_set = split_dataset(
        load_sanitized_jsonl(source), validation_fraction=1 / 3, seed=2
    )
    assert len(train_set) == 4
    protocol = {
        "exercise-id": EXERCISE_ID,
        "feature-schema-version": FEATURE_SCHEMA_VERSION,
        "label-definition-version": LABEL_DEFINITION_VERSION,
        "model-schema-signature": MODEL_SCHEMA_SIGNATURE,
    }

    boundary_client = LocalQualityClient(
        train_set,
        validation_set,
        1,
        0.1,
        max_weight=float(len(train_set)),
    )
    boundary_result = boundary_client.fit(initial_parameters(), protocol)
    assert boundary_result.epochs == 1

    oversized_client = LocalQualityClient(
        train_set,
        validation_set,
        1,
        0.1,
        max_weight=float(len(train_set) - 1),
    )
    with pytest.raises(FederationConfigError, match="exceeds SecAgg\\+ max-weight"):
        oversized_client.fit(initial_parameters(), protocol)


def test_parameter_validator_rejects_shape_and_non_finite_values() -> None:
    parameters = initial_parameters()
    validate_parameters(parameters)

    with pytest.raises(ModelSchemaError, match="shape"):
        validate_parameters([np.zeros((2,), dtype=np.float32), parameters[1]])
    parameters[0][0] = np.nan
    with pytest.raises(ModelSchemaError, match="non-finite"):
        validate_parameters(parameters)


def test_feature_path_must_be_explicit_and_template_is_closed(tmp_path: Path) -> None:
    direct = tmp_path / "local.jsonl"
    assert (
        resolve_sanitized_feature_path(
            {"partition-id": 2, "sanitized-feature-path": str(direct)}, {}
        )
        == direct
    )

    resolved = resolve_sanitized_feature_path(
        {"partition-id": 2},
        {"sanitized-feature-path-template": str(tmp_path / "client-{partition_id}.jsonl")},
    )
    assert resolved == tmp_path / "client-2.jsonl"

    with pytest.raises(FederationConfigError, match="configure"):
        resolve_sanitized_feature_path({"partition-id": 0}, {})
    with pytest.raises(FederationConfigError, match="only permits"):
        resolve_sanitized_feature_path(
            {"partition-id": 0},
            {"sanitized-feature-path-template": "{partition_id}-{secret}.jsonl"},
        )
