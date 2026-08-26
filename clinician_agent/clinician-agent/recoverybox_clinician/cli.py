"""Credential-free local entry point for the deterministic review layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .input import parse_question, parse_session_summaries_json
from .review import build_review_report, render_markdown


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only review from de-identified session summaries."
    )
    parser.add_argument("--sessions", required=True, type=Path, help="Path to a JSON array")
    parser.add_argument(
        "--question",
        default="Which sessions should I review first, and what trends are visible?",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    return parser


def main() -> int:
    args = _parser().parse_args()
    question = parse_question(args.question)
    sessions = parse_session_summaries_json(args.sessions.read_text(encoding="utf-8"))
    report = build_review_report(question, sessions)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
