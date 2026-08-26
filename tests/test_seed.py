import json

from recoverybox.federation import load_sanitized_jsonl
from recoverybox.federation.schema import (
    EXERCISE_ID,
    LABEL_DEFINITION_VERSION,
    MODEL_SCHEMA_SIGNATURE,
)
from recoverybox.seed import seed_flower_demo


def test_seed_writes_three_valid_balanced_synthetic_stores(tmp_path) -> None:
    paths = seed_flower_demo(tmp_path, rows_per_client=8)

    assert [path.name for path in paths] == [
        "client-0.jsonl",
        "client-1.jsonl",
        "client-2.jsonl",
    ]
    for path in paths:
        dataset = load_sanitized_jsonl(path)
        assert len(dataset) == 8
        assert set(dataset.labels.tolist()) == {0, 1}
        first_record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert first_record["exercise_id"] == EXERCISE_ID
        assert first_record["label_definition_version"] == LABEL_DEFINITION_VERSION
        assert first_record["model_schema_signature"] == MODEL_SCHEMA_SIGNATURE
