"""Bounded model-role orchestration for Recovery Swarm."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .planner import BlockerCode, RecoveryPlan, build_deterministic_plan
from .prompts import ROLE_SPECS, RoleName, RoleSpec
from .schema import Day, PlanRequest


@dataclass(frozen=True, slots=True)
class RoleResult:
    """One validated specialist result or a deterministic safe fallback."""

    role: RoleName
    decision: str
    reason_codes: tuple[str, ...]
    payload: dict[str, Any]
    used_model: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "used_model": self.used_model,
        }


@dataclass(frozen=True, slots=True)
class SwarmResult:
    """Validated swarm findings plus the deterministic draft plan."""

    plan: RecoveryPlan
    role_results: tuple[RoleResult, ...]
    model_calls: int
    fallback_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "role_results": [result.as_dict() for result in self.role_results],
            "model_calls": self.model_calls,
            "fallback_count": self.fallback_count,
        }


def _records(context: Any) -> Any:
    records = getattr(context.state, "config_records", None)
    return records if records is not None else context.state


def _locked(context: Any) -> Any:
    locked = getattr(context, "locked", None)
    return locked() if callable(locked) else nullcontext()


def _private_response(agent: Any, context: Any, request: dict[str, Any]) -> Mapping[str, Any]:
    """Call a model without retaining its private specialist draft."""

    records = _records(context)
    with _locked(context):
        items_record = records.get("items")
        existed = items_record is not None
        previous = list(items_record.get("json", ())) if existed else []
    try:
        response = agent.responses.create(request)
        if not isinstance(response, Mapping):
            raise TypeError("model response must be an object")
        return response
    finally:
        with _locked(context):
            current = records.get("items")
            if existed:
                if current is None:
                    records["items"] = items_record
                    current = items_record
                current["json"] = previous
            elif current is not None:
                del records["items"]


def _response_text(response: Mapping[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    parts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("model response has no output")
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    if not parts:
        raise ValueError("model response has no output text")
    return "".join(parts)


def envelope_fingerprint(request: PlanRequest) -> str:
    encoded = json.dumps(
        request.clinician_diagnosis.treatment_envelope.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _model_request(spec: RoleSpec, request: PlanRequest, model: str) -> dict[str, Any]:
    payload = {
        "plan_request": request.as_dict(),
        "clinician_envelope_fingerprint": envelope_fingerprint(request),
    }
    return {
        "model": model,
        "instructions": spec.instructions,
        "input": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        "stream": False,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 700,
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": spec.output_schema_name,
                "strict": True,
                "schema": spec.output_schema,
            },
        },
    }


def _fallback_payload(role: RoleName, request: PlanRequest, plan: RecoveryPlan) -> dict[str, Any]:
    envelope = request.clinician_diagnosis.treatment_envelope
    preferences = request.user_preferences
    if role is RoleName.PRESCRIPTION_BOUNDARY_KEEPER:
        return {
            "decision": "PASS",
            "envelope_fingerprint": envelope_fingerprint(request),
            "exercise_ids": [item.exercise_id for item in envelope.approved_exercises],
            "contraindication_codes": list(envelope.contraindication_codes),
            "frequency_minimum": envelope.allowed_weekly_frequency.minimum,
            "frequency_maximum": envelope.allowed_weekly_frequency.maximum,
            "reason_codes": ["BOUNDARY_PRESERVED"],
        }
    if role is RoleName.PREFERENCE_MAPPER:
        return {
            "decision": "READY",
            "ordered_days": [item.value for item in preferences.available_days],
            "preferred_time": preferences.preferred_time.value,
            "coaching_style": preferences.coaching_style.value,
            "language": preferences.language,
            "accessibility": [item.value for item in preferences.accessibility],
            "reason_codes": ["PREFERENCES_MAPPED"],
        }
    return {
        "decision": "FEASIBLE" if not plan.blocker_codes else "BLOCK",
        "selected_weekly_frequency": plan.selected_weekly_frequency,
        "scheduled_days": [item.day.value for item in plan.sessions],
        "reason_codes": [
            "FIT_CONFIRMED" if not plan.blocker_codes else plan.blocker_codes[0].value
        ],
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("specialist result contains an invalid list")
    return value


def _validate_payload(role: RoleName, payload: Any, request: PlanRequest) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("specialist result must be an object")
    envelope = request.clinician_diagnosis.treatment_envelope
    preferences = request.user_preferences
    reasons = _string_list(payload.get("reason_codes"))
    if role is RoleName.PRESCRIPTION_BOUNDARY_KEEPER:
        expected_ids = [item.exercise_id for item in envelope.approved_exercises]
        if payload.get("envelope_fingerprint") != envelope_fingerprint(request):
            raise ValueError("specialist changed the envelope fingerprint")
        if _string_list(payload.get("exercise_ids")) != expected_ids:
            raise ValueError("specialist changed the exercise list")
        if _string_list(payload.get("contraindication_codes")) != list(
            envelope.contraindication_codes
        ):
            raise ValueError("specialist changed contraindications")
        if payload.get("frequency_minimum") != envelope.allowed_weekly_frequency.minimum:
            raise ValueError("specialist changed the minimum frequency")
        if payload.get("frequency_maximum") != envelope.allowed_weekly_frequency.maximum:
            raise ValueError("specialist changed the maximum frequency")
        if payload.get("decision") not in {"PASS", "BLOCK"}:
            raise ValueError("specialist returned an invalid decision")
    elif role is RoleName.PREFERENCE_MAPPER:
        days = _string_list(payload.get("ordered_days"))
        expected_days = [item.value for item in preferences.available_days]
        if len(days) != len(expected_days) or set(days) != set(expected_days):
            raise ValueError("specialist added or removed an available day")
        exact = {
            "preferred_time": preferences.preferred_time.value,
            "coaching_style": preferences.coaching_style.value,
            "language": preferences.language,
        }
        if any(payload.get(key) != value for key, value in exact.items()):
            raise ValueError("specialist changed a presentation preference")
        if _string_list(payload.get("accessibility")) != [
            item.value for item in preferences.accessibility
        ]:
            raise ValueError("specialist changed accessibility preferences")
        if payload.get("decision") not in {"READY", "BLOCK"}:
            raise ValueError("specialist returned an invalid decision")
    else:
        days = _string_list(payload.get("scheduled_days"))
        available = {item.value for item in preferences.available_days}
        if not set(days).issubset(available):
            raise ValueError("specialist scheduled an unavailable day")
        frequency = payload.get("selected_weekly_frequency")
        if frequency is not None and (
            isinstance(frequency, bool)
            or not isinstance(frequency, int)
            or not envelope.allowed_weekly_frequency.minimum
            <= frequency
            <= envelope.allowed_weekly_frequency.maximum
        ):
            raise ValueError("specialist selected an invalid frequency")
        if payload.get("decision") not in {"FEASIBLE", "BLOCK"}:
            raise ValueError("specialist returned an invalid decision")
    payload["reason_codes"] = reasons
    return payload


def _role_result(
    agent: Any,
    context: Any,
    spec: RoleSpec,
    request: PlanRequest,
    model: str,
    base_plan: RecoveryPlan,
) -> RoleResult:
    used_model = True
    try:
        response = _private_response(agent, context, _model_request(spec, request, model))
        payload = _validate_payload(spec.name, json.loads(_response_text(response)), request)
    except (KeyError, TypeError, ValueError, RuntimeError, TimeoutError, json.JSONDecodeError):
        used_model = False
        payload = _fallback_payload(spec.name, request, base_plan)
    return RoleResult(
        role=spec.name,
        decision=str(payload["decision"]),
        reason_codes=tuple(payload["reason_codes"]),
        payload=payload,
        used_model=used_model,
    )


def run_recovery_swarm(
    agent: Any,
    context: Any,
    request: PlanRequest,
    *,
    model: str,
) -> SwarmResult:
    """Run three private advisory roles, then build one bounded draft plan."""

    if not isinstance(model, str) or not model.strip():
        raise ValueError("agent.model must be a non-empty string")
    base_plan = build_deterministic_plan(request)
    results = tuple(
        _role_result(agent, context, spec, request, model.strip(), base_plan)
        for spec in ROLE_SPECS
    )
    extra: list[BlockerCode] = []
    by_role = {item.role: item for item in results}
    if by_role[RoleName.PRESCRIPTION_BOUNDARY_KEEPER].decision == "BLOCK":
        extra.append(BlockerCode.PRESCRIPTION_BOUNDARY_REVIEW)
    if by_role[RoleName.PREFERENCE_MAPPER].decision == "BLOCK":
        extra.append(BlockerCode.PREFERENCE_MAPPING_REVIEW)
    if by_role[RoleName.FEASIBILITY_REVIEWER].decision == "BLOCK":
        extra.append(BlockerCode.FEASIBILITY_REVIEW)
    preferred_days: Sequence[Day] | None = None
    mapped_days = by_role[RoleName.PREFERENCE_MAPPER].payload.get("ordered_days")
    if isinstance(mapped_days, list):
        preferred_days = tuple(Day(value) for value in mapped_days)
    plan = build_deterministic_plan(
        request,
        preferred_days=preferred_days,
        additional_blockers=extra,
    )
    fallback_count = sum(not item.used_model for item in results)
    return SwarmResult(
        plan=plan,
        role_results=results,
        model_calls=3,
        fallback_count=fallback_count,
    )
