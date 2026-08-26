"""A small deterministic classifier for already-sanitized movement features.

This model estimates the quality label represented in the feature store.  It does not
make clinical safety decisions; those remain in the device's deterministic guardian.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .data import SanitizedDataset
from .errors import ModelSchemaError
from .schema import FEATURE_COUNT, MODEL_PARAMETER_SHAPES

FloatArray = NDArray[np.float32]
ModelParameters: TypeAlias = list[FloatArray]
EXPECTED_PARAMETER_SHAPES = MODEL_PARAMETER_SHAPES


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Scalar metrics for one local dataset."""

    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Updated parameters and final training metrics."""

    parameters: ModelParameters
    loss: float
    accuracy: float
    epochs: int


def initial_parameters(*, seed: int = 2026) -> ModelParameters:
    """Create a deterministic, near-zero model with the canonical shapes."""

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, FEATURE_COUNT).astype(np.float32)
    bias = np.zeros((1,), dtype=np.float32)
    return [weights, bias]


def validate_parameters(parameters: Sequence[np.ndarray]) -> ModelParameters:
    """Validate and defensively copy parameters at every trust boundary."""

    if len(parameters) != len(EXPECTED_PARAMETER_SHAPES):
        raise ModelSchemaError(
            f"expected {len(EXPECTED_PARAMETER_SHAPES)} parameter arrays; "
            f"received {len(parameters)}"
        )

    validated: ModelParameters = []
    for index, (parameter, expected_shape) in enumerate(
        zip(parameters, EXPECTED_PARAMETER_SHAPES, strict=True)
    ):
        try:
            array = np.asarray(parameter, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ModelSchemaError(f"parameter {index} is not a numeric array") from exc
        if array.shape != expected_shape:
            raise ModelSchemaError(
                f"parameter {index} shape {array.shape} is incompatible; expected {expected_shape}"
            )
        if not np.isfinite(array).all():
            raise ModelSchemaError(f"parameter {index} contains non-finite values")
        validated.append(array.copy())
    return validated


def _probabilities(parameters: Sequence[np.ndarray], features: FloatArray) -> FloatArray:
    weights, bias = validate_parameters(parameters)
    logits = features @ weights + bias[0]
    # Clipping avoids overflow while preserving all useful probability resolution.
    clipped = np.clip(logits, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def predict_probabilities(
    parameters: Sequence[np.ndarray], dataset: SanitizedDataset
) -> FloatArray:
    """Return the probability of label ``1`` for every local feature row."""

    return _probabilities(parameters, dataset.features)


def evaluate(
    parameters: Sequence[np.ndarray],
    dataset: SanitizedDataset,
    *,
    l2: float = 0.0,
) -> Evaluation:
    """Compute deterministic binary cross-entropy and accuracy."""

    if l2 < 0.0:
        raise ModelSchemaError("l2 must be non-negative")
    weights, _ = validate_parameters(parameters)
    probabilities = _probabilities(parameters, dataset.features).astype(np.float64)
    labels = dataset.labels.astype(np.float64)
    epsilon = 1e-7
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    loss = -np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
    loss += 0.5 * l2 * float(np.dot(weights, weights))
    predictions = (probabilities >= 0.5).astype(np.int64)
    accuracy = float(np.mean(predictions == dataset.labels))
    return Evaluation(loss=float(loss), accuracy=accuracy)


def train(
    parameters: Sequence[np.ndarray],
    dataset: SanitizedDataset,
    *,
    epochs: int,
    learning_rate: float,
    l2: float = 1e-4,
) -> TrainingResult:
    """Run deterministic full-batch gradient descent on a local client."""

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ModelSchemaError("epochs must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ModelSchemaError("learning_rate must be finite and positive")
    if not np.isfinite(l2) or l2 < 0.0:
        raise ModelSchemaError("l2 must be finite and non-negative")

    weights, bias = validate_parameters(parameters)
    features = dataset.features.astype(np.float64)
    labels = dataset.labels.astype(np.float64)
    weights64 = weights.astype(np.float64)
    bias64 = float(bias[0])

    for _ in range(epochs):
        logits = features @ weights64 + bias64
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        errors = probabilities - labels
        weight_gradient = (features.T @ errors) / len(dataset) + l2 * weights64
        bias_gradient = float(np.mean(errors))
        weights64 -= learning_rate * weight_gradient
        bias64 -= learning_rate * bias_gradient
        if not np.isfinite(weights64).all() or not np.isfinite(bias64):
            raise ModelSchemaError("training produced non-finite parameters")

    updated = [
        weights64.astype(np.float32),
        np.asarray([bias64], dtype=np.float32),
    ]
    metrics = evaluate(updated, dataset, l2=l2)
    return TrainingResult(
        parameters=updated,
        loss=metrics.loss,
        accuracy=metrics.accuracy,
        epochs=epochs,
    )


def weighted_average_parameters(
    client_parameters: Iterable[Sequence[np.ndarray]],
    example_counts: Iterable[int],
) -> ModelParameters:
    """Reference weighted aggregation for local tests and candidate validation.

    Flower's SecAgg+ workflow performs the live aggregate.  This pure function mirrors
    its weighted-average result so model/schema behavior can be tested without Flower.
    """

    parameter_sets = list(client_parameters)
    counts = list(example_counts)
    if not parameter_sets:
        raise ModelSchemaError("at least one client update is required")
    if len(parameter_sets) != len(counts):
        raise ModelSchemaError("one example count is required per client update")
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in counts):
        raise ModelSchemaError("client example counts must be positive integers")

    validated_sets = [validate_parameters(parameters) for parameters in parameter_sets]
    total = float(sum(counts))
    aggregate: ModelParameters = []
    for parameter_index in range(len(EXPECTED_PARAMETER_SHAPES)):
        weighted = np.zeros(EXPECTED_PARAMETER_SHAPES[parameter_index], dtype=np.float64)
        for parameters, count in zip(validated_sets, counts, strict=True):
            weighted += parameters[parameter_index].astype(np.float64) * count
        aggregate.append((weighted / total).astype(np.float32))
    return aggregate
