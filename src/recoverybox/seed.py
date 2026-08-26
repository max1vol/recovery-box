"""Synthetic, privacy-safe feature stores for the local Flower demonstration."""

from __future__ import annotations

import json
import random
from pathlib import Path

from recoverybox.federation.schema import (
    EXERCISE_ID,
    FEATURE_SCHEMA_VERSION,
    LABEL_DEFINITION_VERSION,
    MODEL_SCHEMA_SIGNATURE,
)


def seed_flower_demo(
    output_directory: str | Path,
    *,
    clients: int = 3,
    rows_per_client: int = 30,
    seed: int = 2026,
) -> tuple[Path, ...]:
    """Write deterministic synthetic JSONL stores for local SuperNodes."""

    if clients != 3:
        raise ValueError("the SecAgg+ hackathon workflow requires exactly 3 clients")
    if rows_per_client < 4:
        raise ValueError("each demo client needs at least 4 rows")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for client_id in range(clients):
        rng = random.Random(seed + client_id)
        target = destination / f"client-{client_id}.jsonl"
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row_index in range(rows_per_client):
                label = (row_index + client_id) % 2
                record = _synthetic_record(rng, label=label)
                handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
                handle.write("\n")
        temporary.replace(target)
        written.append(target)
    return tuple(written)


def _synthetic_record(rng: random.Random, *, label: int) -> dict[str, object]:
    if label == 1:
        ranges = {
            "joint_angle_deg": (112.0, 154.0),
            "angular_velocity_deg_s": (60.0, 220.0),
            "pose_confidence": (0.84, 0.99),
            "camera_disagreement_deg": (1.0, 8.0),
            "range_progress": (0.72, 0.98),
            "rep_duration_s": (2.0, 4.5),
            "stability_score": (0.76, 0.98),
            "symmetry_score": (0.79, 0.99),
        }
    else:
        ranges = {
            "joint_angle_deg": (65.0, 108.0),
            "angular_velocity_deg_s": (260.0, 610.0),
            "pose_confidence": (0.62, 0.86),
            "camera_disagreement_deg": (10.0, 38.0),
            "range_progress": (0.25, 0.64),
            "rep_duration_s": (0.7, 1.9),
            "stability_score": (0.28, 0.68),
            "symmetry_score": (0.38, 0.74),
        }
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "exercise_id": EXERCISE_ID,
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "model_schema_signature": MODEL_SCHEMA_SIGNATURE,
        "features": {
            name: round(rng.uniform(minimum, maximum), 6)
            for name, (minimum, maximum) in ranges.items()
        },
        "label": label,
    }
