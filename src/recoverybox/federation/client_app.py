"""Flower ClientApp for local, sanitized movement-quality features."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from .data import load_sanitized_jsonl, split_dataset
from .errors import FederationConfigError
from .local_client import LocalQualityClient
from .paths import resolve_sanitized_feature_path


def _integer(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FederationConfigError(f"{key} must be a positive integer")
    return value


def _number(config: Mapping[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FederationConfigError(f"{key} must be numeric")
    return float(value)


def _positive_number(config: Mapping[str, object], key: str, default: float) -> float:
    number = _number(config, key, default)
    if not isfinite(number) or number <= 0.0:
        raise FederationConfigError(f"{key} must be finite and positive")
    return number


class RehabilitationQualityNumPyClient(NumPyClient):
    """Thin Flower adapter around deterministic, independently testable logic."""

    def __init__(self, local: LocalQualityClient) -> None:
        self.local = local

    def fit(self, parameters, config):
        """Validate protocol metadata, train locally, and return only parameters."""

        result = self.local.fit(parameters, config)
        # Per-client metrics are intentionally omitted from the live workflow.  SecAgg+
        # protects the parameter aggregate; this demo does not imply that arbitrary
        # metric metadata is private.
        return result.parameters, self.local.train_examples, {}

    def evaluate(self, parameters, config):
        """Evaluate locally when explicitly invoked outside the default server flow."""

        result = self.local.evaluate(parameters, config)
        return result.loss, self.local.validation_examples, {"accuracy": result.accuracy}


def client_fn(context: Context):
    """Construct a client that reads exactly one configured local JSONL store."""

    path = resolve_sanitized_feature_path(context.node_config, context.run_config)
    dataset = load_sanitized_jsonl(path)
    partition_id = context.node_config.get("partition-id", 0)
    if isinstance(partition_id, bool) or not isinstance(partition_id, int):
        raise FederationConfigError("partition-id must be an integer")

    validation_fraction = _number(context.run_config, "validation-fraction", 0.2)
    train_set, validation_set = split_dataset(
        dataset,
        validation_fraction=validation_fraction,
        seed=2026 + partition_id,
    )
    local = LocalQualityClient(
        train_set=train_set,
        validation_set=validation_set,
        local_epochs=_integer(context.run_config, "local-epochs", 5),
        learning_rate=_number(context.run_config, "learning-rate", 0.1),
        l2=_number(context.run_config, "l2", 1e-4),
        max_weight=_positive_number(context.run_config, "max-weight", 10_000.0),
    )
    return RehabilitationQualityNumPyClient(local).to_client()


app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])
