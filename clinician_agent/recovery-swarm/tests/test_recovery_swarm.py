from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from recovery_swarm.agent_app import STATE_KEY, run_agent_task
from recovery_swarm.planner import BlockerCode, PlanStatus, build_deterministic_plan
from recovery_swarm.schema import DEMO_REQUEST, PlanRequestValidationError, parse_plan_request


class FakeEvents:
    def __init__(self) -> None:
        self.emitted: list[dict[str, object]] = []

    def emit(self, event: dict[str, object]) -> None:
        self.emitted.append(event)


class FallbackResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(request)
        return {"output": []}


def fake_agent() -> SimpleNamespace:
    return SimpleNamespace(responses=FallbackResponses(), events=FakeEvents())


def fake_context(command: str = "/demo") -> SimpleNamespace:
    return SimpleNamespace(
        state={},
        run_config={
            "agent.input": command,
            "agent.model": "openai/gpt-5.5",
            "agent.plan-request-json": "",
        },
    )


def test_schema_is_synthetic_only_and_does_not_echo_rejected_values() -> None:
    request = parse_plan_request(DEMO_REQUEST)
    assert request.synthetic is True

    invalid = copy.deepcopy(DEMO_REQUEST)
    invalid["synthetic"] = False
    invalid["patient_name"] = "SECRET-SENTINEL"
    with pytest.raises(PlanRequestValidationError) as caught:
        parse_plan_request(invalid)
    assert "SECRET-SENTINEL" not in str(caught.value)


def test_planner_preserves_the_clinician_envelope_and_preferences() -> None:
    request = parse_plan_request(DEMO_REQUEST)
    plan = build_deterministic_plan(request)

    assert plan.status is PlanStatus.DRAFT_FOR_CLINICIAN_APPROVAL
    assert plan.treatment_envelope == request.clinician_diagnosis.treatment_envelope
    assert plan.selected_weekly_frequency == 3
    assert [session.day for session in plan.sessions] == list(
        request.user_preferences.available_days
    )
    assert all(
        session.exercise_ids
        == tuple(
            exercise.exercise_id
            for exercise in request.clinician_diagnosis.treatment_envelope.approved_exercises
        )
        for session in plan.sessions
    )


def test_agentapp_streams_a_safe_draft_when_specialists_fall_back() -> None:
    agent = fake_agent()
    context = fake_context()

    markdown = run_agent_task(agent, context)

    assert "DRAFT_FOR_CLINICIAN_APPROVAL" in markdown
    assert "## Swarm organisation" in markdown
    assert "## Decision Gate" in markdown
    assert "3 specialist call(s)" in markdown
    assert len(agent.responses.calls) == 3
    assert STATE_KEY in context.state
    assert [event["type"] for event in agent.events.emitted] == [
        "response.output_text.delta",
        "response.completed",
    ]


def test_follow_up_never_drops_an_existing_safety_blocker() -> None:
    agent = fake_agent()
    context = fake_context()
    run_agent_task(agent, context)
    record = context.state[STATE_KEY]
    saved = json.loads(record["json"])
    saved["plan"]["blocker_codes"] = [BlockerCode.FEASIBILITY_REVIEW.value]
    record["json"] = json.dumps(saved)

    context.run_config["agent.input"] = "2"
    markdown = run_agent_task(agent, context)

    assert "BLOCKED" in markdown
    assert "FEASIBILITY_REVIEW" in markdown
    assert "recovery-swarm.v2" in markdown
    assert "Preference Mapper" in markdown
    assert len(agent.responses.calls) == 3


def test_follow_up_preserves_the_validated_preferred_day_order() -> None:
    agent = fake_agent()
    context = fake_context()
    run_agent_task(agent, context)
    record = context.state[STATE_KEY]
    saved = json.loads(record["json"])
    saved["preferred_days"] = ["FRIDAY", "WEDNESDAY", "MONDAY"]
    record["json"] = json.dumps(saved)

    context.run_config["agent.input"] = "2"
    markdown = run_agent_task(agent, context)

    assert markdown.index("Friday") < markdown.index("Wednesday") < markdown.index("Monday")
