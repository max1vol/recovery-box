"""Credential-free local renderer for the synthetic Recovery Swarm draft."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .planner import build_deterministic_plan, render_plan_markdown
from .schema import DEMO_REQUEST, parse_plan_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a synthetic Recovery Swarm draft")
    parser.add_argument("--request", type=Path, help="Path to a synthetic plan-request JSON")
    parser.add_argument("--json", action="store_true", help="Print structured plan JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = DEMO_REQUEST
    if args.request is not None:
        raw = json.loads(args.request.read_text(encoding="utf-8"))
    request = parse_plan_request(raw)
    plan = build_deterministic_plan(request)
    if args.json:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
    else:
        print(render_plan_markdown(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
