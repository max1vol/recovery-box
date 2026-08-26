"""Fail-closed Raspberry Pi current-throttle gate for deployment actions."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

_VCGENCMD: Final = "/usr/bin/vcgencmd"
_CURRENT_MASK: Final = 0x0000000F
_HISTORICAL_MASK: Final = 0x000F0000
_KNOWN_MASK: Final = _CURRENT_MASK | _HISTORICAL_MASK
_SAMPLE_COUNT: Final = 3
_SAMPLE_INTERVAL_SECONDS: Final = 0.25
_COMMAND_TIMEOUT_SECONDS: Final = 2.0
_REPORT_PATTERN: Final = re.compile(r"throttled=0x([0-9A-Fa-f]{1,8})\n?")


class _CompletedProcess(Protocol):
    returncode: int
    stdout: str


Runner = Callable[..., _CompletedProcess]
Sleeper = Callable[[float], None]


class PiPowerGateError(RuntimeError):
    """The Pi power state cannot safely authorize a protected action."""


@dataclass(frozen=True, slots=True)
class PiPowerStatus:
    value: int
    current_flags: int
    historical_flags: int


def parse_throttled_report(report: str) -> PiPowerStatus:
    """Parse the exact single-line ``vcgencmd get_throttled`` response."""

    if not isinstance(report, str):
        raise TypeError("report must be text")
    match = _REPORT_PATTERN.fullmatch(report)
    if match is None:
        raise PiPowerGateError("vcgencmd returned a malformed throttle report")
    value = int(match.group(1), 16)
    if value & ~_KNOWN_MASK:
        raise PiPowerGateError("vcgencmd returned unknown throttle flags")
    return PiPowerStatus(
        value=value,
        current_flags=value & _CURRENT_MASK,
        historical_flags=(value & _HISTORICAL_MASK) >> 16,
    )


def read_power_status(*, runner: Runner = subprocess.run) -> PiPowerStatus:
    """Read one bounded power status sample using the absolute Pi command."""

    try:
        completed = runner(
            [_VCGENCMD, "get_throttled"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PiPowerGateError("vcgencmd timed out") from exc
    except OSError as exc:
        raise PiPowerGateError("vcgencmd is unavailable") from exc
    if completed.returncode != 0:
        raise PiPowerGateError("vcgencmd failed")
    return parse_throttled_report(completed.stdout)


def require_clear_current_power(
    *,
    runner: Runner = subprocess.run,
    sleeper: Sleeper = time.sleep,
    samples: int = _SAMPLE_COUNT,
) -> PiPowerStatus:
    """Require consecutive clear current samples; sticky history is diagnostic."""

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    historical_flags = 0
    final_status: PiPowerStatus | None = None
    for index in range(samples):
        status = read_power_status(runner=runner)
        historical_flags |= status.historical_flags
        if status.current_flags:
            raise PiPowerGateError(
                "current throttle flags are active "
                f"(current=0x{status.current_flags:x}, history=0x{historical_flags:x})"
            )
        final_status = status
        if index + 1 < samples:
            sleeper(_SAMPLE_INTERVAL_SECONDS)
    assert final_status is not None
    return PiPowerStatus(
        value=historical_flags << 16,
        current_flags=0,
        historical_flags=historical_flags,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        print("RecoveryBox Pi power gate failed: arguments are not accepted", file=sys.stderr)
        return 2
    try:
        status = require_clear_current_power()
    except (PiPowerGateError, ValueError) as exc:
        print(f"RecoveryBox Pi power gate failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"RecoveryBox Pi power gate passed: current=0x0 history=0x{status.historical_flags:x}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PiPowerGateError",
    "PiPowerStatus",
    "main",
    "parse_throttled_report",
    "read_power_status",
    "require_clear_current_power",
]
