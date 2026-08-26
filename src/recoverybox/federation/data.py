"""Read and partition a local store of sanitized, pose-derived features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .errors import SanitizedDataError
from .schema import FEATURE_COUNT, validate_and_normalize_record

FloatMatrix = NDArray[np.float32]
IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SanitizedDataset:
    """A model-ready matrix with no retained media, text, or record metadata."""

    features: FloatMatrix
    labels: IntVector

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.int64)
        if features.ndim != 2 or features.shape[1:] != (FEATURE_COUNT,):
            raise SanitizedDataError(
                f"feature matrix shape {features.shape} is incompatible; "
                f"expected (samples, {FEATURE_COUNT})"
            )
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise SanitizedDataError("labels must contain one value per feature row")
        if features.shape[0] == 0:
            raise SanitizedDataError("dataset must contain at least one sample")
        if not np.isfinite(features).all():
            raise SanitizedDataError("feature matrix contains non-finite values")
        if not np.isin(labels, (0, 1)).all():
            raise SanitizedDataError("labels must contain only 0 and 1")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def load_sanitized_jsonl(
    path: str | Path,
    *,
    max_bytes: int = 32 * 1024 * 1024,
) -> SanitizedDataset:
    """Load the exact configured JSONL file through the closed feature schema."""

    source = Path(path)
    if source.suffix.lower() != ".jsonl":
        raise SanitizedDataError("sanitized feature store must use a .jsonl file")
    if not source.exists():
        raise SanitizedDataError(f"sanitized feature store does not exist: {source}")
    if not source.is_file():
        raise SanitizedDataError(f"sanitized feature store is not a file: {source}")
    size = source.stat().st_size
    if size > max_bytes:
        raise SanitizedDataError(f"sanitized feature store is {size} bytes; limit is {max_bytes}")

    feature_rows: list[list[float]] = []
    labels: list[int] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SanitizedDataError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            try:
                normalized, label = validate_and_normalize_record(record)
            except SanitizedDataError as exc:
                raise SanitizedDataError(f"line {line_number}: {exc}") from exc
            feature_rows.append(normalized)
            labels.append(label)

    if not feature_rows:
        raise SanitizedDataError("sanitized feature store has no records")
    return SanitizedDataset(
        features=np.asarray(feature_rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
    )


def split_dataset(
    dataset: SanitizedDataset,
    *,
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> tuple[SanitizedDataset, SanitizedDataset]:
    """Create a deterministic, disjoint local train/validation split."""

    if len(dataset) < 2:
        raise SanitizedDataError("at least two samples are required for a local split")
    if not 0.0 < validation_fraction < 1.0:
        raise SanitizedDataError("validation_fraction must be between 0 and 1")

    validation_size = max(1, round(len(dataset) * validation_fraction))
    validation_size = min(validation_size, len(dataset) - 1)
    indices = np.random.default_rng(seed).permutation(len(dataset))
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    train_set = SanitizedDataset(
        dataset.features[train_indices].copy(), dataset.labels[train_indices].copy()
    )
    validation_set = SanitizedDataset(
        dataset.features[validation_indices].copy(),
        dataset.labels[validation_indices].copy(),
    )
    return train_set, validation_set
