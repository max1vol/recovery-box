"""Flower AgentApp entry point for the bounded Recovery Swarm demo."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import ConfigRecord, Context

from .planner import BlockerCode, build_deterministic_plan, render_plan_markdown
from .prompts import RoleName
from .schema import DEMO_REQUEST, Day, PlanRequest, PlanRequestValidationError, parse_plan_request
from .swarm import SwarmResult, run_recovery_swarm

STATE_KEY = "recovery-swarm-state"
STATE_SCHEMA = "recovery-swarm-state.v1"
MAX_STATE_BYTES = 100_000

_FOCUS = {
    "1": "schedule fit inside the unchanged clinician envelope",
    "2": "coaching, language, accessibility, and equipment preferences",
    "3": "clinician approval checklist and unresolved blockers",
}
_LEADS = {
    "1": RoleName.FEASIBILITY_REVIEWER,
    "2": RoleName.PREFERENCE_MAPPER,
    "3": RoleName.PRESCRIPTION_BOUNDARY_KEEPER,
}


def _records(context: Any) -> Any:
    records = getattr(context.state, "config_records", None)
    return records if records is not None else context.state


def _locked(context: Any) -> Any:
    locked = getattr(context, "locked", None)
    return locked() if callable(locked) else nullcontext()


def _state_record(value: Mapping[str, Any]) -> ConfigRecord:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_STATE_BYTES:
        raise ValueError("Recovery Swarm state is too large")
    return ConfigRecord({"json": encoded})


def _load_state(context: Any) -> dict[str, Any] | None:
    record = _records(context).get(STATE_KEY)
    if record is None:
        return None
    encoded = record.get("json")
    if not isinstance(encoded, str) or len(encoded.encode("utf-8")) > MAX_STATE_BYTES:
        return None
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
        return None
    return value


def _append_answer(
    agent: Any,
    context: Any,
    markdown: str,
    *,
    next_state: Mapping[str, Any] | None = None,
    clear_state: bool = False,
) -> None:
    assistant_json = json.dumps(
        {"type": "message", "role": "assistant", "content": markdown},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prepared_state = _state_record(next_state) if next_state is not None else None
    records = _records(context)
    with _locked(context):
        if clear_state:
            records.pop(STATE_KEY, None)
        elif prepared_state is not None:
            records[STATE_KEY] = prepared_state
        items = records.setdefault("items", ConfigRecord({"json": []}))
        encoded_items = items.get("json")
        if not isinstance(encoded_items, list):
            raise TypeError("Flower conversation items must be a list")
        encoded_items.append(assistant_json)

    events = getattr(agent, "events", None)
    if events is None:
        return
    try:
        events.emit({"type": "response.output_text.delta", "delta": markdown})
        events.emit({"type": "response.completed"})
    except RuntimeError:
        # The durable assistant item remains available when the chat is reopened.
        return


def _role_rows(role_results: list[dict[str, Any]], lead: RoleName) -> list[str]:
    rows = ["| Role | Assignment | Result |", "|---|---|---|"]
    for role in RoleName:
        stored = next((item for item in role_results if item.get("role") == role.value), {})
        assignment = "Lead" if role is lead else "Support"
        decision = stored.get("decision", "SAFE_FALLBACK")
        mode = "model" if stored.get("used_model") is True else "deterministic fallback"
        rows.append(f"| {role.value} | {assignment} | `{decision}` ({mode}) |")
    return rows


def _render_response(
    request: PlanRequest,
    plan: Any,
    *,
    role_results: list[dict[str, Any]],
    organisation_version: int,
    decision_history: list[str],
    fallback_count: int,
) -> str:
    latest = decision_history[-1] if decision_history else None
    focus = _FOCUS.get(latest or "", "clinician envelope and preference fit")
    lead = _LEADS.get(latest or "", RoleName.PRESCRIPTION_BOUNDARY_KEEPER)
    base = render_plan_markdown(
        plan,
        organisation_version=f"recovery-swarm.v{organisation_version}",
        focus=focus,
    )
    lines = [base, "", "## Swarm organisation", "", *_role_rows(role_results, lead)]
    lines.extend(
        [
            "",
            "## Activity",
            "",
            "1. Prescription Boundary Keeper checked the immutable clinician envelope.",
            "2. Preference Mapper mapped only coded schedule and presentation preferences.",
            "3. Feasibility Reviewer checked day count, duration, and required equipment.",
            "4. The deterministic composer copied every exercise and dose unchanged.",
        ]
    )
    if latest is not None:
        lines.extend(
            [
                "",
                f"Reorganised from Decision Gate choice **{latest}**; "
                f"{lead.value} is now lead.",
            ]
        )
    if fallback_count:
        lines.extend(
            [
                "",
                f"{fallback_count} specialist call(s) were unavailable or invalid; "
                "the protected deterministic fallback was used.",
            ]
        )
    if latest == "3":
        envelope = request.clinician_diagnosis.treatment_envelope
        lines.extend(
            [
                "",
                "## Clinician approval checklist",
                "",
                "- Confirm diagnosis code and review date.",
                "- Confirm every exercise, set, rep, duration, and frequency bound.",
                "- Confirm all contraindication codes remain present.",
                "- Approve, revise, or reject the draft; the swarm cannot activate it.",
                f"- Current review date: `{envelope.review_date.isoformat()}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            "Reply **1** to inspect schedule fit.",
            "Reply **2** to inspect preference and accessibility fit.",
            "Reply **3** to prepare the clinician approval checklist.",
            "Use **/reset** to clear Recovery Swarm's app-managed draft state.",
            "",
            "> The plan is not active. The clinician owns the prescription, and the local "
            "Guardian remains the sole authority for live pause, stop, escalation, and cue "
            "selection.",
        ]
    )
    return "\n".join(lines)


def _initial_state(request: PlanRequest, result: SwarmResult) -> dict[str, Any]:
    preference_result = next(
        item for item in result.role_results if item.role is RoleName.PREFERENCE_MAPPER
    )
    raw_preferred_days = preference_result.payload.get("ordered_days")
    preferred_days = (
        raw_preferred_days
        if isinstance(raw_preferred_days, list)
        else [item.value for item in request.user_preferences.available_days]
    )
    return {
        "schema": STATE_SCHEMA,
        "request": request.as_dict(),
        "plan": result.plan.as_dict(),
        "role_results": [item.as_dict() for item in result.role_results],
        "organisation_version": 1,
        "decision_history": [],
        "fallback_count": result.fallback_count,
        "preferred_days": preferred_days,
    }


def _help(markdown: str) -> str:
    return (
        markdown
        + "\n\nRun **/demo** for the bundled synthetic example. Do not paste names, contact "
        "details, clinical notes, images, audio, or real patient records into this public demo."
    )


def run_agent_task(agent: Any, context: Any) -> str:
    """Run one safe turn and return the exact visible Markdown."""

    prompt = context.run_config.get("agent.input")
    if not isinstance(prompt, str) or not prompt.strip():
        markdown = _help("Recovery Swarm needs a command.")
        _append_answer(agent, context, markdown)
        return markdown
    command = prompt.strip().lower()

    if command == "/reset":
        markdown = (
            "Recovery Swarm's app-managed draft state is cleared. Start a new Flower chat "
            "to create a separate platform conversation, or run **/demo** again."
        )
        _append_answer(agent, context, markdown, clear_state=True)
        return markdown

    state = _load_state(context)
    if command in _FOCUS:
        if state is None:
            markdown = _help("There is no recoverable draft in this conversation.")
            _append_answer(agent, context, markdown)
            return markdown
        try:
            request = parse_plan_request(state["request"])
            raw_blockers = state["plan"].get("blocker_codes", [])
            if not isinstance(raw_blockers, list):
                raise ValueError("invalid blocker state")
            retained_blockers = tuple(BlockerCode(value) for value in raw_blockers)
            raw_preferred_days = state.get(
                "preferred_days",
                [item.value for item in request.user_preferences.available_days],
            )
            if not isinstance(raw_preferred_days, list):
                raise ValueError("invalid preferred-day state")
            preferred_days = tuple(Day(value) for value in raw_preferred_days)
            plan = build_deterministic_plan(
                request,
                preferred_days=preferred_days,
                additional_blockers=retained_blockers,
            )
            version = int(state["organisation_version"]) + 1
            history = [str(item) for item in state.get("decision_history", [])][-7:]
            history.append(command)
            roles = state.get("role_results", [])
            if not isinstance(roles, list):
                raise ValueError("invalid role state")
            fallback_count = int(state.get("fallback_count", 3))
        except (KeyError, TypeError, ValueError, PlanRequestValidationError):
            markdown = _help("The saved draft is unavailable; restart with **/demo**.")
            _append_answer(agent, context, markdown, clear_state=True)
            return markdown
        next_state = dict(state)
        next_state["plan"] = plan.as_dict()
        next_state["organisation_version"] = version
        next_state["decision_history"] = history
        markdown = _render_response(
            request,
            plan,
            role_results=roles,
            organisation_version=version,
            decision_history=history,
            fallback_count=fallback_count,
        )
        _append_answer(agent, context, markdown, next_state=next_state)
        return markdown

    configured = context.run_config.get("agent.plan-request-json", "")
    if command != "/demo" and not (isinstance(configured, str) and configured.strip()):
        markdown = _help(
            "Recovery Swarm accepts only its structured synthetic plan-request configuration."
        )
        _append_answer(agent, context, markdown)
        return markdown

    try:
        raw_request = DEMO_REQUEST if command == "/demo" else json.loads(configured)
        request = parse_plan_request(raw_request)
    except (json.JSONDecodeError, PlanRequestValidationError):
        markdown = _help("The synthetic plan request failed its strict schema check.")
        _append_answer(agent, context, markdown)
        return markdown

    model = context.run_config.get("agent.model")
    try:
        result = run_recovery_swarm(agent, context, request, model=model)
    except ValueError:
        markdown = _help("The configured model reference is unavailable.")
        _append_answer(agent, context, markdown)
        return markdown
    state = _initial_state(request, result)
    markdown = _render_response(
        request,
        result.plan,
        role_results=state["role_results"],
        organisation_version=1,
        decision_history=[],
        fallback_count=result.fallback_count,
    )
    _append_answer(agent, context, markdown, next_state=state)
    return markdown


app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Run one Recovery Swarm conversation turn."""

    run_agent_task(agent, context)
