"""Flower AgentApp entry point backed by a deterministic review engine."""

from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any

from .input import parse_question, parse_session_summaries_json
from .review import build_review_report, render_markdown

try:  # The pure review engine and CLI remain usable without Flower installed.
    from flwr.agentapp import AgentApp, AgentSession
    from flwr.app import ConfigRecord, Context
except ImportError:  # pragma: no cover - exercised by environments without Flower
    AgentApp = None  # type: ignore[assignment,misc]
    AgentSession = Any  # type: ignore[assignment,misc]
    ConfigRecord = None  # type: ignore[assignment,misc]
    Context = Any  # type: ignore[assignment,misc]


def _persist_report(context: Any, report_json: str, markdown: str) -> None:
    payload = {
        "schema-version": "1",
        "report-json": report_json,
        "report-markdown": markdown,
        "read-only": True,
    }
    if ConfigRecord is not None:
        context.state["clinician-review"] = ConfigRecord(payload)
    else:
        context.state["clinician-review"] = payload


def _locked(context: Any) -> Any:
    locked = getattr(context, "locked", None)
    return locked() if callable(locked) else nullcontext()


def _emit_report(agent: Any, context: Any, markdown: str) -> None:
    if ConfigRecord is not None:
        assistant_json = json.dumps(
            {"type": "message", "role": "assistant", "content": markdown},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with _locked(context):
            items = context.state.setdefault("items", ConfigRecord({"json": []}))
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
        return


def run_agent_task(agent: Any, context: Any) -> str:
    """Validate input, persist the authoritative report, and emit that exact report."""
    unsupported_keys = {"agent.model", "agent.use-model"}.intersection(context.run_config)
    if unsupported_keys:
        keys = ", ".join(sorted(unsupported_keys))
        raise ValueError(
            f"model presentation configuration is unsupported ({keys}); "
            "the deterministic Markdown is the sole clinician output"
        )

    question = parse_question(context.run_config.get("agent.input"))
    sessions = parse_session_summaries_json(
        context.run_config.get("agent.session-summaries-json", "[]")
    )
    report = build_review_report(question, sessions)
    markdown = render_markdown(report)
    report_json = json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":"))
    _persist_report(context, report_json, markdown)
    _emit_report(agent, context, markdown)
    print(markdown)
    return markdown


if AgentApp is not None:
    app = AgentApp()

    @app.main()
    def main(agent: AgentSession, context: Context) -> None:
        """Run one bounded clinician review task."""
        run_agent_task(agent, context)

else:
    app = None
