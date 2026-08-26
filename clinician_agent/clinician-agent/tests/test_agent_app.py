from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_input import valid_summary

from recoverybox_clinician.agent_app import run_agent_task


class NoModelAccess:
    @property
    def responses(self) -> object:
        raise AssertionError("the deterministic AgentApp must not access a model")

    @property
    def connectors(self) -> object:
        raise AssertionError("the read-only AgentApp must not access a connector")


class RecordingEvents:
    def __init__(self) -> None:
        self.emitted: list[dict[str, object]] = []

    def emit(self, event: dict[str, object]) -> None:
        self.emitted.append(event)


def context(**extra_config: object) -> SimpleNamespace:
    run_config = {
        "agent.input": "Which sessions should I review first?",
        "agent.session-summaries-json": json.dumps([valid_summary()]),
    }
    run_config.update(extra_config)
    return SimpleNamespace(
        run_config=run_config,
        state={},
    )


def test_agent_returns_prints_and_persists_the_same_authoritative_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_context = context()

    result = run_agent_task(NoModelAccess(), fake_context)

    stored = fake_context.state["clinician-review"]
    assert stored["read-only"] is True
    assert "sess-test-001" in stored["report-json"]
    assert result == stored["report-markdown"]
    assert capsys.readouterr().out == result + "\n"
    assert "[session:sess-test-001]" in result


def test_agent_runs_without_model_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_agent_task(NoModelAccess(), context())

    assert "Clinician review support" in result
    assert capsys.readouterr().out == result + "\n"


def test_agent_emits_the_report_to_the_live_chat(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events = RecordingEvents()
    fake_context = context()

    result = run_agent_task(SimpleNamespace(events=events), fake_context)

    assert [event["type"] for event in events.emitted] == [
        "response.output_text.delta",
        "response.completed",
    ]
    assert events.emitted[0]["delta"] == result
    stored_message = json.loads(fake_context.state["items"]["json"][0])
    assert stored_message["content"] == result
    assert capsys.readouterr().out == result + "\n"


def test_exact_deidentified_prompt_returns_ranked_chat_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt = (
        "Review these de-identified knee-extension sessions and rank which needs clinician "
        "review first. S1: 2026-08-20, 5/10 reps, 188s, quality .58, confidence .91, "
        "no pain, stopped early, no flags. S2: 2026-08-21, 3/10 reps, 97s, quality .42, "
        "confidence .88, pain reported, stopped early, SHARP_PAIN_REPORTED. Cite the session "
        "and evidence for each concern; summarize shared patterns and limits. Do not diagnose, "
        "prescribe, change the plan, or control the device."
    )
    sessions = [
        valid_summary(
            session_id="sess-demo-001",
            completed_reps=5,
            duration_seconds=188,
            quality_score=0.58,
            stopped_early=True,
        ),
        valid_summary(
            session_id="sess-demo-002",
            participant_ref="anon-def456",
            session_date="2026-08-21",
            completed_reps=3,
            duration_seconds=97,
            quality_score=0.42,
            observation_confidence=0.88,
            pain_reported=True,
            stopped_early=True,
            safety_flags=["SHARP_PAIN_REPORTED"],
        ),
    ]
    fake_context = context(
        **{
            "agent.input": prompt,
            "agent.session-summaries-json": json.dumps(sessions),
        }
    )

    result = run_agent_task(SimpleNamespace(events=RecordingEvents()), fake_context)

    assert result.index("sess-demo-002") < result.index("sess-demo-001")
    assert "SHARP_PAIN_REPORTED" in result
    assert "scope" not in result.lower()
    assert capsys.readouterr().out == result + "\n"


@pytest.mark.parametrize("key", ["agent.model", "agent.use-model"])
def test_model_presentation_configuration_is_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="deterministic Markdown is the sole clinician output"):
        run_agent_task(NoModelAccess(), context(**{key: True}))
