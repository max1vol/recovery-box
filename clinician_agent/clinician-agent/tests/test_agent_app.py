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


@pytest.mark.parametrize("key", ["agent.model", "agent.use-model"])
def test_model_presentation_configuration_is_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="deterministic Markdown is the sole clinician output"):
        run_agent_task(NoModelAccess(), context(**{key: True}))
