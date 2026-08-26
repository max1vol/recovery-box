"""Three-client Flower ServerApp using the documented SecAgg+ compatibility lane.

This is a hackathon deployment shape.  SecAgg+ protects aggregation of model updates,
but this module alone does not establish production privacy, identity, transport, model
governance, or clinical safety guarantees.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from flwr.common import (
    Context,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from .errors import FederationConfigError
from .model import initial_parameters, validate_parameters
from .schema import (
    EXERCISE_ID,
    FEATURE_SCHEMA_VERSION,
    LABEL_DEFINITION_VERSION,
    MODEL_SCHEMA_SIGNATURE,
)

HACKATHON_CLIENT_COUNT = 3
HACKATHON_NUM_SHARES = 3
HACKATHON_RECONSTRUCTION_THRESHOLD = 2


def _positive_int(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FederationConfigError(f"{key} must be a positive integer")
    return value


def _positive_number(config: Mapping[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FederationConfigError(f"{key} must be numeric")
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise FederationConfigError(f"{key} must be finite and positive")
    return number


def round_protocol_config(server_round: int) -> dict[str, str | int]:
    """Attach schema identities to every model update request."""

    return {
        "server-round": server_round,
        "exercise-id": EXERCISE_ID,
        "feature-schema-version": FEATURE_SCHEMA_VERSION,
        "label-definition-version": LABEL_DEFINITION_VERSION,
        "model-schema-signature": MODEL_SCHEMA_SIGNATURE,
    }


class ShapeValidatingFedAvg(FedAvg):
    """Reject an aggregate that does not match the fixed model schema."""

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None
        parameters, _ = aggregated
        validate_parameters(parameters_to_ndarrays(parameters))
        return aggregated


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run a fixed three-client, two-share-threshold SecAgg+ demonstration."""

    run_config = context.run_config
    configured_schema = run_config.get("feature-schema-version", FEATURE_SCHEMA_VERSION)
    if configured_schema != FEATURE_SCHEMA_VERSION:
        raise FederationConfigError(
            f"configured feature schema {configured_schema!r} does not match "
            f"{FEATURE_SCHEMA_VERSION!r}"
        )
    configured_exercise = run_config.get("exercise-id", EXERCISE_ID)
    if configured_exercise != EXERCISE_ID:
        raise FederationConfigError(
            f"configured exercise {configured_exercise!r} does not match {EXERCISE_ID!r}"
        )
    configured_labels = run_config.get("label-definition-version", LABEL_DEFINITION_VERSION)
    if configured_labels != LABEL_DEFINITION_VERSION:
        raise FederationConfigError(
            f"configured label definition {configured_labels!r} does not match "
            f"{LABEL_DEFINITION_VERSION!r}"
        )
    client_count = _positive_int(run_config, "min-available-clients", HACKATHON_CLIENT_COUNT)
    num_shares = _positive_int(run_config, "num-shares", HACKATHON_NUM_SHARES)
    threshold = _positive_int(
        run_config,
        "reconstruction-threshold",
        HACKATHON_RECONSTRUCTION_THRESHOLD,
    )
    if client_count != HACKATHON_CLIENT_COUNT:
        raise FederationConfigError(
            f"this hackathon workflow requires exactly {HACKATHON_CLIENT_COUNT} clients"
        )
    if num_shares != HACKATHON_NUM_SHARES or threshold != HACKATHON_RECONSTRUCTION_THRESHOLD:
        raise FederationConfigError(
            "this hackathon workflow requires num-shares=3 and reconstruction-threshold=2"
        )

    strategy = ShapeValidatingFedAvg(
        fraction_fit=1.0,
        min_fit_clients=client_count,
        min_available_clients=client_count,
        # Evaluation results are client-specific metadata, so the default demo does
        # not request them through the unprotected legacy evaluation path.
        fraction_evaluate=0.0,
        on_fit_config_fn=round_protocol_config,
        initial_parameters=ndarrays_to_parameters(initial_parameters()),
    )
    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=_positive_int(run_config, "num-server-rounds", 3)),
        strategy=strategy,
    )
    fit_workflow = SecAggPlusWorkflow(
        num_shares=num_shares,
        reconstruction_threshold=threshold,
        max_weight=_positive_number(run_config, "max-weight", 10_000.0),
        clipping_range=_positive_number(run_config, "clipping-range", 8.0),
    )
    DefaultWorkflow(fit_workflow=fit_workflow)(grid, legacy_context)
