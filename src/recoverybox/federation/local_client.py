"""Flower-independent local client behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .data import SanitizedDataset
from .errors import FederationConfigError, ModelSchemaError
from .model import Evaluation, TrainingResult, evaluate, train
from .schema import (
    EXERCISE_ID,
    FEATURE_SCHEMA_VERSION,
    LABEL_DEFINITION_VERSION,
    MODEL_SCHEMA_SIGNATURE,
)


def validate_round_protocol(config: Mapping[str, object]) -> None:
    """Fail closed when a server and client disagree about schemas."""

    feature_version = config.get("feature-schema-version")
    if feature_version != FEATURE_SCHEMA_VERSION:
        raise ModelSchemaError(
            f"server feature schema {feature_version!r} does not match {FEATURE_SCHEMA_VERSION!r}"
        )
    exercise_id = config.get("exercise-id")
    if exercise_id != EXERCISE_ID:
        raise ModelSchemaError(f"server exercise {exercise_id!r} does not match {EXERCISE_ID!r}")
    label_definition_version = config.get("label-definition-version")
    if label_definition_version != LABEL_DEFINITION_VERSION:
        raise ModelSchemaError(
            f"server label definition {label_definition_version!r} does not match "
            f"{LABEL_DEFINITION_VERSION!r}"
        )
    signature = config.get("model-schema-signature")
    if signature != MODEL_SCHEMA_SIGNATURE:
        raise ModelSchemaError("server model schema signature is incompatible")


@dataclass(slots=True)
class LocalQualityClient:
    """Deterministic train/evaluate engine owned by one SuperNode."""

    train_set: SanitizedDataset
    validation_set: SanitizedDataset
    local_epochs: int
    learning_rate: float
    l2: float = 1e-4
    max_weight: float = 10_000.0

    def __post_init__(self) -> None:
        if isinstance(self.local_epochs, bool) or not isinstance(self.local_epochs, int):
            raise FederationConfigError("local_epochs must be an integer")
        if self.local_epochs <= 0:
            raise FederationConfigError("local_epochs must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise FederationConfigError("learning_rate must be finite and positive")
        if not np.isfinite(self.l2) or self.l2 < 0.0:
            raise FederationConfigError("l2 must be finite and non-negative")
        if not np.isfinite(self.max_weight) or self.max_weight <= 0.0:
            raise FederationConfigError("max_weight must be finite and positive")

    def fit(
        self,
        parameters: Sequence[np.ndarray],
        config: Mapping[str, object],
    ) -> TrainingResult:
        validate_round_protocol(config)
        if self.train_examples > self.max_weight:
            raise FederationConfigError(
                f"local train example count {self.train_examples} exceeds SecAgg+ "
                f"max-weight {self.max_weight}"
            )
        return train(
            parameters,
            self.train_set,
            epochs=self.local_epochs,
            learning_rate=self.learning_rate,
            l2=self.l2,
        )

    def evaluate(
        self,
        parameters: Sequence[np.ndarray],
        config: Mapping[str, object],
    ) -> Evaluation:
        validate_round_protocol(config)
        return evaluate(parameters, self.validation_set, l2=self.l2)

    @property
    def train_examples(self) -> int:
        return len(self.train_set)

    @property
    def validation_examples(self) -> int:
        return len(self.validation_set)
