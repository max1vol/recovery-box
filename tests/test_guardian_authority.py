from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = PROJECT_ROOT / "src" / "recoverybox"


def _production_trees() -> tuple[tuple[Path, ast.AST], ...]:
    return tuple(
        (path, ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(PRODUCTION_ROOT.rglob("*.py"))
    )


def test_production_cannot_construct_guardian_verdicts_or_end_signals() -> None:
    forbidden: list[str] = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id in {"GuardianDecision", "SessionEndSignal"}:
                forbidden.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert forbidden == []


def test_only_guardian_implementation_uses_private_verdict_issuer() -> None:
    callers: list[str] = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if name == "_issue_guardian_decision":
                callers.append(str(path.relative_to(PROJECT_ROOT)))

    assert set(callers) == {"src/recoverybox/core/guardian.py"}


def test_production_has_no_public_arbitrary_mode_transition() -> None:
    callers: list[str] = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "transition_to":
                callers.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert callers == []


class _PhysicalStopCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_names: list[str] = []
        self._function_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "request_physical_stop":
            self.function_names.append(
                self._function_stack[-1] if self._function_stack else "<module>"
            )
        self.generic_visit(node)


def test_physical_stop_calls_exist_only_at_physical_input_boundaries() -> None:
    expected = {
        "src/recoverybox/device/remote_pose_service.py": {
            "request_local_stop",
            "_accept_start_serialized",
        },
        "src/recoverybox/laptop/squat_launcher.py": {"_request_physical_stop"},
        "src/recoverybox/laptop/squat_session.py": {"request_physical_stop"},
    }
    observed: dict[str, set[str]] = {}
    for path, tree in _production_trees():
        visitor = _PhysicalStopCallVisitor()
        visitor.visit(tree)
        if visitor.function_names:
            observed[str(path.relative_to(PROJECT_ROOT))] = set(visitor.function_names)

    assert observed == expected
