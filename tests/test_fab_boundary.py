from __future__ import annotations

import tomllib
from pathlib import Path


def test_training_fab_has_an_explicit_federation_only_allowlist() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    includes = project["tool"]["flwr"]["app"]["fab-include"]

    assert includes == [
        "LICENSE",
        "src/recoverybox/__init__.py",
        "src/recoverybox/federation/**/*.py",
    ]
    assert all("clinician" not in pattern for pattern in includes)
    assert all("realtime" not in pattern for pattern in includes)
    assert all("device" not in pattern for pattern in includes)
