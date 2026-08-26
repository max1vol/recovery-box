"""Locally validated function tools for Realtime model output.

Realtime function calls are model output, not trusted commands.  This module
accepts only a deliberately small JSON Schema subset and rejects malformed,
duplicate, oversized, unknown, or wrongly typed arguments before callers can
observe a tool call as executable.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MAX_ARGUMENT_JSON_BYTES = 16_384
MAX_VALIDATION_DEPTH = 12


class ToolValidationError(ValueError):
    """A tool definition or model-supplied call failed local validation."""


@dataclass(frozen=True, slots=True)
class FunctionTool:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ToolValidationError("tool name must not be blank")
        if not self.description.strip():
            raise ToolValidationError("tool description must not be blank")
        _validate_schema_definition(self.parameters)
        if self.parameters.get("type") != "object":
            raise ToolValidationError("tool parameter schema must be an object")
        if self.parameters.get("additionalProperties") is not False:
            raise ToolValidationError(
                "tool parameter schema must set additionalProperties to false"
            )

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class ValidatedToolCall:
    name: str
    call_id: str
    arguments: Mapping[str, Any]


class ToolRegistry:
    """Immutable registry that is the only route to validated tool calls."""

    def __init__(self, tools: Iterable[FunctionTool]) -> None:
        by_name: dict[str, FunctionTool] = {}
        for tool in tools:
            if tool.name in by_name:
                raise ToolValidationError(f"duplicate tool name: {tool.name}")
            by_name[tool.name] = tool
        self._by_name = by_name

    @property
    def wire_tools(self) -> tuple[dict[str, Any], ...]:
        return tuple(tool.to_wire() for tool in self._by_name.values())

    def validate_call(self, *, name: str, call_id: str, arguments_json: str) -> ValidatedToolCall:
        if name not in self._by_name:
            raise ToolValidationError(f"unknown tool: {name}")
        if not call_id.strip():
            raise ToolValidationError("tool call id must not be blank")
        if len(arguments_json.encode("utf-8")) > MAX_ARGUMENT_JSON_BYTES:
            raise ToolValidationError("tool arguments exceed the local size limit")

        try:
            arguments = json.loads(arguments_json, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ToolValidationError("tool arguments are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ToolValidationError("tool arguments must be a JSON object")

        _validate_value(
            arguments,
            self._by_name[name].parameters,
            path="$",
            depth=0,
        )
        return ValidatedToolCall(name=name, call_id=call_id, arguments=arguments)


def extract_validated_tool_calls(
    event: Mapping[str, Any], registry: ToolRegistry
) -> tuple[ValidatedToolCall, ...]:
    """Extract and validate calls from either supported completion event.

    The Realtime API can surface completed arguments as their own event and in
    the completed response output.  Both routes converge on the same registry
    validation; partial argument deltas are intentionally ignored.
    """

    event_type = event.get("type")
    raw_calls: list[tuple[str, str, str]] = []
    if event_type == "response.function_call_arguments.done":
        raw_calls.append(
            (
                _event_string(event, "name"),
                _event_string(event, "call_id"),
                _event_string(event, "arguments", allow_blank=True),
            )
        )
    elif event_type == "response.done":
        response = event.get("response")
        if not isinstance(response, Mapping):
            raise ToolValidationError("response.done.response must be an object")
        if response.get("status") != "completed":
            return ()
        output = response.get("output", [])
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
            raise ToolValidationError("response.done output must be a list")
        for item in output:
            if not isinstance(item, Mapping):
                raise ToolValidationError("response output item must be an object")
            if item.get("type") != "function_call":
                continue
            raw_calls.append(
                (
                    _event_string(item, "name"),
                    _event_string(item, "call_id"),
                    _event_string(item, "arguments", allow_blank=True),
                )
            )
    else:
        return ()

    return tuple(
        registry.validate_call(name=name, call_id=call_id, arguments_json=arguments)
        for name, call_id, arguments in raw_calls
    )


def deduplicate_validated_tool_calls(
    calls: Iterable[ValidatedToolCall],
    seen_call_ids: set[str],
) -> tuple[ValidatedToolCall, ...]:
    """Return each locally validated call ID at most once per session.

    Realtime can report the same completed function call first through
    ``response.function_call_arguments.done`` and again in ``response.done``.
    The caller owns ``seen_call_ids`` so reconnect/session lifetime remains an
    explicit orchestration decision.  Only already-validated calls enter the
    set; a malformed first event cannot suppress a later valid completion.
    """

    unique: list[ValidatedToolCall] = []
    for call in calls:
        if call.call_id in seen_call_ids:
            continue
        seen_call_ids.add(call.call_id)
        unique.append(call)
    return tuple(unique)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ToolValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_schema_definition(schema: Mapping[str, Any], *, depth: int = 0) -> None:
    if depth > MAX_VALIDATION_DEPTH:
        raise ToolValidationError("tool schema is too deeply nested")
    if not isinstance(schema, Mapping):
        raise ToolValidationError("tool schema nodes must be objects")
    schema_type = schema.get("type")
    if schema_type not in {"object", "array", "string", "integer", "number", "boolean"}:
        raise ToolValidationError(f"unsupported tool schema type: {schema_type!r}")

    if schema_type == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ToolValidationError("object properties must be an object")
        required = schema.get("required", [])
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            raise ToolValidationError("object required must be a list")
        if any(not isinstance(name, str) for name in required):
            raise ToolValidationError("required property names must be strings")
        unknown_required = set(required) - set(properties)
        if unknown_required:
            raise ToolValidationError("required properties must be declared")
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, bool):
            raise ToolValidationError("additionalProperties must be boolean")
        for subschema in properties.values():
            _validate_schema_definition(subschema, depth=depth + 1)
    elif schema_type == "array":
        if "items" not in schema:
            raise ToolValidationError("array schema needs items")
        _validate_schema_definition(schema["items"], depth=depth + 1)

    enum = schema.get("enum")
    if enum is not None and (
        not isinstance(enum, Sequence) or isinstance(enum, (str, bytes)) or not enum
    ):
        raise ToolValidationError("enum must be a non-empty list")


def _validate_value(value: Any, schema: Mapping[str, Any], *, path: str, depth: int) -> None:
    if depth > MAX_VALIDATION_DEPTH:
        raise ToolValidationError(f"{path} exceeds maximum nesting")

    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{path} is outside the allowed enum")

    schema_type = schema["type"]
    if schema_type == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"{path} must be an object")
        properties: Mapping[str, Any] = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise ToolValidationError(f"{path} is missing required properties")
        unknown = set(value) - set(properties)
        if unknown and schema.get("additionalProperties", True) is False:
            raise ToolValidationError(f"{path} contains unknown properties")
        for name, child in value.items():
            if name in properties:
                _validate_value(
                    child,
                    properties[name],
                    path=f"{path}.{name}",
                    depth=depth + 1,
                )
        return

    if schema_type == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{path} must be an array")
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ToolValidationError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ToolValidationError(f"{path} has too many items")
        for index, child in enumerate(value):
            _validate_value(
                child,
                schema["items"],
                path=f"{path}[{index}]",
                depth=depth + 1,
            )
        return

    if schema_type == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"{path} must be a string")
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolValidationError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolValidationError(f"{path} is too long")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ToolValidationError(f"{path} does not match its pattern")
        return

    if schema_type == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(f"{path} must be a boolean")
        return

    if schema_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolValidationError(f"{path} must be an integer")
        _validate_number_bounds(value, schema, path)
        return

    if schema_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolValidationError(f"{path} must be a number")
        if not math.isfinite(value):
            raise ToolValidationError(f"{path} must be finite")
        _validate_number_bounds(value, schema, path)
        return

    raise ToolValidationError(f"unsupported schema type at {path}")


def _validate_number_bounds(value: int | float, schema: Mapping[str, Any], path: str) -> None:
    if "minimum" in schema and value < schema["minimum"]:
        raise ToolValidationError(f"{path} is below its minimum")
    if "maximum" in schema and value > schema["maximum"]:
        raise ToolValidationError(f"{path} is above its maximum")


def _event_string(value: Mapping[str, Any], key: str, *, allow_blank: bool = False) -> str:
    found = value.get(key)
    if not isinstance(found, str) or (not allow_blank and not found.strip()):
        raise ToolValidationError(f"{key} must be a string")
    return found
