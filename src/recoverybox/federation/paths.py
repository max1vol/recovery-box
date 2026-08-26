"""Strict resolution of a SuperNode's configured local feature store."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from string import Formatter

from .errors import FederationConfigError


def _partition_id(node_config: Mapping[str, object]) -> int:
    value = node_config.get("partition-id")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FederationConfigError("node_config partition-id must be a non-negative integer")
    return value


def _render_path_template(template: str, partition_id: int) -> str:
    fields: list[str] = []
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise FederationConfigError("invalid sanitized feature path template") from exc
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        fields.append(field_name)
        if field_name != "partition_id" or format_spec or conversion:
            raise FederationConfigError(
                "sanitized feature path template only permits {partition_id}"
            )
    if fields != ["partition_id"]:
        raise FederationConfigError(
            "sanitized feature path template must contain {partition_id} exactly once"
        )
    return template.format(partition_id=partition_id)


def resolve_sanitized_feature_path(
    node_config: Mapping[str, object],
    run_config: Mapping[str, object],
) -> Path:
    """Resolve one explicitly configured JSONL path, with no discovery or fallback."""

    configured = node_config.get("sanitized-feature-path")
    if configured is not None:
        if not isinstance(configured, str) or not configured.strip():
            raise FederationConfigError(
                "node_config sanitized-feature-path must be a non-empty string"
            )
        rendered = configured
    else:
        template = run_config.get("sanitized-feature-path-template")
        if not isinstance(template, str) or not template.strip():
            raise FederationConfigError(
                "configure sanitized-feature-path on the node or "
                "sanitized-feature-path-template for the run"
            )
        rendered = _render_path_template(template, _partition_id(node_config))

    path = Path(rendered).expanduser()
    if path.suffix.lower() != ".jsonl":
        raise FederationConfigError("sanitized feature path must end in .jsonl")
    return path
