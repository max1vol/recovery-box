from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "recoverybox_pi_power_gate.py"


@pytest.fixture(scope="module")
def power_gate() -> ModuleType:
    name = "_recoverybox_test_pi_power_gate"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("report", "history"),
    [
        ("throttled=0x0\n", 0),
        ("throttled=0x00000000\n", 0),
        ("throttled=0xD0000\n", 0xD),
        ("throttled=0xd0000", 0xD),
    ],
)
def test_parser_accepts_clear_current_flags_and_separates_history(
    power_gate: ModuleType,
    report: str,
    history: int,
) -> None:
    status = power_gate.parse_throttled_report(report)

    assert status.current_flags == 0
    assert status.historical_flags == history


@pytest.mark.parametrize("current", [1, 2, 4, 8])
def test_clear_gate_rejects_each_current_flag(power_gate: ModuleType, current: int) -> None:
    runner = lambda *args, **kwargs: SimpleNamespace(  # noqa: E731
        returncode=0,
        stdout=f"throttled=0x{current:x}\n",
    )

    with pytest.raises(power_gate.PiPowerGateError, match="current throttle flags"):
        power_gate.require_clear_current_power(runner=runner, sleeper=lambda _: None)


def test_clear_gate_reports_current_separately_from_sticky_history(
    power_gate: ModuleType,
) -> None:
    runner = lambda *args, **kwargs: SimpleNamespace(  # noqa: E731
        returncode=0,
        stdout="throttled=0xd0005\n",
    )

    with pytest.raises(power_gate.PiPowerGateError, match=r"current=0x5, history=0xd"):
        power_gate.require_clear_current_power(runner=runner, sleeper=lambda _: None)


@pytest.mark.parametrize(
    "report",
    [
        "",
        "throttled=0x",
        " throttled=0x0\n",
        "throttled=0x0\nextra\n",
        "throttled=-0x1\n",
        "throttled=0x000000000\n",
        "throttled=0x10\n",
        "throttled=0x10000000\n",
    ],
)
def test_parser_rejects_malformed_or_unknown_reports(
    power_gate: ModuleType,
    report: str,
) -> None:
    with pytest.raises(power_gate.PiPowerGateError):
        power_gate.parse_throttled_report(report)


def test_reader_uses_absolute_bounded_command(power_gate: ModuleType) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def runner(command: object, **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="throttled=0x0\n")

    status = power_gate.read_power_status(runner=runner)

    assert status.current_flags == 0
    assert calls == [
        (
            ["/usr/bin/vcgencmd", "get_throttled"],
            {
                "stdin": subprocess.DEVNULL,
                "capture_output": True,
                "text": True,
                "timeout": 2.0,
                "check": False,
            },
        )
    ]


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (subprocess.TimeoutExpired("vcgencmd", 2), "timed out"),
        (FileNotFoundError(), "unavailable"),
    ],
)
def test_reader_fails_closed_on_command_error(
    power_gate: ModuleType,
    exception: BaseException,
    message: str,
) -> None:
    def runner(*args: object, **kwargs: object) -> SimpleNamespace:
        raise exception

    with pytest.raises(power_gate.PiPowerGateError, match=message):
        power_gate.read_power_status(runner=runner)


def test_reader_fails_closed_on_nonzero_command(power_gate: ModuleType) -> None:
    runner = lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="")  # noqa: E731

    with pytest.raises(power_gate.PiPowerGateError, match="vcgencmd failed"):
        power_gate.read_power_status(runner=runner)


def test_gate_requires_three_consecutive_clear_samples(power_gate: ModuleType) -> None:
    reports = iter(("throttled=0x10000\n", "throttled=0x40000\n", "throttled=0x80000\n"))
    sleeps: list[float] = []

    def runner(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=next(reports))

    status = power_gate.require_clear_current_power(runner=runner, sleeper=sleeps.append)

    assert status.value == 0xD0000
    assert status.current_flags == 0
    assert status.historical_flags == 0xD
    assert sleeps == [0.25, 0.25]


@pytest.mark.parametrize("fault_index", [1, 2])
def test_gate_rejects_a_later_sample_and_stops_sampling(
    power_gate: ModuleType,
    fault_index: int,
) -> None:
    reports = ["throttled=0x0\n"] * 3
    reports[fault_index] = "throttled=0x1\n"
    calls = 0
    sleeps: list[float] = []

    def runner(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        report = reports[calls]
        calls += 1
        return SimpleNamespace(returncode=0, stdout=report)

    with pytest.raises(power_gate.PiPowerGateError, match="current throttle flags"):
        power_gate.require_clear_current_power(runner=runner, sleeper=sleeps.append)

    assert calls == fault_index + 1
    assert sleeps == [0.25] * fault_index


def test_main_emits_only_stderr(power_gate: ModuleType, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        power_gate,
        "require_clear_current_power",
        lambda: power_gate.PiPowerStatus(value=0xD0000, current_flags=0, historical_flags=0xD),
    )

    assert power_gate.main([]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "current=0x0 history=0xd" in captured.err


def test_main_scrubs_command_exception_detail(power_gate: ModuleType, monkeypatch, capsys) -> None:
    def fail() -> None:
        raise power_gate.PiPowerGateError("vcgencmd is unavailable")

    monkeypatch.setattr(power_gate, "require_clear_current_power", fail)

    assert power_gate.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "RecoveryBox Pi power gate failed: vcgencmd is unavailable\n"
