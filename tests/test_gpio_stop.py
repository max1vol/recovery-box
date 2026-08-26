from __future__ import annotations

import errno
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from recoverybox.device.gpio_stop import (
    DebouncedStopLatch,
    GpioBias,
    GpiodStopLine,
    GpioStopConfig,
    PhysicalStopMonitor,
    StopInputSnapshot,
    StopInputState,
    StopInputTransition,
    StopInputTrigger,
    open_gpiod_stop_line,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeRequest:
    def __init__(self, *, value: object = "inactive") -> None:
        self.value = value
        self.get_calls: list[int] = []
        self.wait_calls: list[float] = []
        self.read_calls: list[int] = []
        self.release_calls = 0
        self.events_ready = True

    def get_value(self, line_offset: int) -> object:
        self.get_calls.append(line_offset)
        return self.value

    def wait_edge_events(self, timeout_seconds: float) -> bool:
        self.wait_calls.append(timeout_seconds)
        return self.events_ready

    def read_edge_events(self, max_events: int) -> list[object]:
        self.read_calls.append(max_events)
        return []

    def release(self) -> None:
        self.release_calls += 1


class FakeGpiodRuntime:
    def __init__(self, request: FakeRequest) -> None:
        self.request = request
        self.settings: dict[str, object] | None = None
        self.settings_calls: list[dict[str, object]] = []
        self.request_call: tuple[str, dict[str, object]] | None = None
        self.request_errors: list[OSError] = []
        self.line = SimpleNamespace(
            Bias=SimpleNamespace(
                PULL_UP="bias-up",
                PULL_DOWN="bias-down",
                DISABLED="bias-disabled",
            ),
            Clock=SimpleNamespace(MONOTONIC="clock-monotonic"),
            Direction=SimpleNamespace(INPUT="direction-input"),
            Edge=SimpleNamespace(BOTH="edge-both"),
            Value=SimpleNamespace(ACTIVE="active"),
        )

    def LineSettings(self, **kwargs: object) -> object:
        self.settings = dict(kwargs)
        self.settings_calls.append(self.settings)
        return f"line-settings-{len(self.settings_calls)}"

    def request_lines(self, path: str, **kwargs: object) -> FakeRequest:
        self.request_call = (path, dict(kwargs))
        if self.request_errors:
            raise self.request_errors.pop(0)
        return self.request


class ScriptedLine:
    def __init__(
        self,
        states: list[bool],
        *,
        clock: FakeClock,
        terminal_error: Exception | None = None,
    ) -> None:
        self._states = states
        self._clock = clock
        self._terminal_error = terminal_error
        self._index = 0
        self.closed = False

    def read_active(self) -> bool:
        return self._states[self._index]

    def wait_for_change(self, timeout_seconds: float) -> bool:
        previous = self._states[self._index]
        self._clock.advance(timeout_seconds)
        if self._index + 1 >= len(self._states):
            if self._terminal_error is not None:
                raise self._terminal_error
            return False
        self._index += 1
        return self._states[self._index] is not previous

    def close(self) -> None:
        self.closed = True


def test_verified_pi3_configuration_requires_explicit_line_offset() -> None:
    config = GpioStopConfig(line_offset=23)

    assert config.line_offset == 23
    assert config.chip_path == Path("/dev/gpiochip0")
    assert config.active_low is True
    assert config.bias is GpioBias.PULL_UP
    assert config.debounce_seconds == pytest.approx(0.10)

    with pytest.raises(TypeError, match="line_offset"):
        GpioStopConfig()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="absolute"):
        GpioStopConfig(line_offset=23, chip_path=Path("gpiochip0"))
    with pytest.raises(ValueError, match="between 0 and 1023"):
        GpioStopConfig(line_offset=-1)
    with pytest.raises(ValueError, match="debounce_seconds"):
        GpioStopConfig(line_offset=23, debounce_seconds=0.01)


def test_debounced_latch_rejects_bounce_and_rearms_only_after_stable_release() -> None:
    latch = DebouncedStopLatch(0.10)

    assert latch.observe(True, observed_at=0.00) is None
    assert latch.observe(False, observed_at=0.04) is None
    assert latch.observe(True, observed_at=0.05) is None
    assert latch.observe(True, observed_at=0.14) is None
    assert latch.observe(True, observed_at=0.151) is StopInputTransition.PRESSED
    assert latch.observe(True, observed_at=1.00) is None

    assert latch.observe(False, observed_at=1.01) is None
    assert latch.observe(True, observed_at=1.05) is None
    assert latch.observe(False, observed_at=1.10) is None
    assert latch.observe(False, observed_at=1.201) is StopInputTransition.RELEASED

    assert latch.observe(True, observed_at=2.00) is None
    assert latch.observe(True, observed_at=2.101) is StopInputTransition.PRESSED


def test_debounced_latch_rejects_regressing_or_nonfinite_clock() -> None:
    latch = DebouncedStopLatch(0.10)
    assert latch.observe(False, observed_at=1.0) is None

    with pytest.raises(ValueError, match="backwards"):
        latch.observe(True, observed_at=0.9)
    with pytest.raises(ValueError, match="finite"):
        DebouncedStopLatch(0.10).observe(True, observed_at=float("nan"))


def test_gpiod_factory_requests_active_low_input_with_kernel_debounce() -> None:
    request = FakeRequest(value="active")
    runtime = FakeGpiodRuntime(request)
    config = GpioStopConfig(line_offset=23, debounce_seconds=0.125)

    line = open_gpiod_stop_line(config, runtime=runtime)

    assert runtime.settings is not None
    assert runtime.settings["direction"] == "direction-input"
    assert runtime.settings["edge_detection"] == "edge-both"
    assert runtime.settings["bias"] == "bias-up"
    assert runtime.settings["active_low"] is True
    assert runtime.settings["debounce_period"].total_seconds() == pytest.approx(0.125)  # type: ignore[union-attr]
    assert runtime.settings["event_clock"] == "clock-monotonic"
    assert runtime.request_call == (
        "/dev/gpiochip0",
        {
            "consumer": "recoverybox-physical-stop",
            "config": {23: "line-settings-1"},
            "event_buffer_size": 16,
        },
    )
    assert line.read_active() is True
    assert line.wait_for_change(0.025) is True
    assert request.get_calls == [23]
    assert request.read_calls == [64]

    line.close()
    line.close()
    assert request.release_calls == 1


def test_gpiod_factory_falls_back_only_when_kernel_debounce_is_unsupported() -> None:
    runtime = FakeGpiodRuntime(FakeRequest())
    runtime.request_errors.append(OSError(errno.ENOTSUP, "debounce unsupported"))

    open_gpiod_stop_line(GpioStopConfig(line_offset=23), runtime=runtime)

    assert len(runtime.settings_calls) == 2
    assert runtime.settings_calls[0]["debounce_period"].total_seconds() == pytest.approx(0.10)  # type: ignore[union-attr]
    assert runtime.settings_calls[1]["debounce_period"].total_seconds() == 0  # type: ignore[union-attr]

    denied = FakeGpiodRuntime(FakeRequest())
    denied.request_errors.append(PermissionError(errno.EACCES, "line unavailable"))
    with pytest.raises(PermissionError):
        open_gpiod_stop_line(GpioStopConfig(line_offset=23), runtime=denied)
    assert len(denied.settings_calls) == 1


def test_gpiod_line_does_not_drain_events_on_timeout() -> None:
    request = FakeRequest()
    request.events_ready = False
    line = GpiodStopLine(request, line_offset=23, active_value="active")

    assert line.wait_for_change(0.05) is False
    assert request.read_calls == []


def test_monitor_confirms_stable_initial_release_before_available() -> None:
    clock = FakeClock()
    line = ScriptedLine(
        [False, False, False],
        clock=clock,
        terminal_error=OSError("end deterministic startup probe"),
    )
    statuses: list[tuple[StopInputSnapshot, float]] = []
    monitor = PhysicalStopMonitor(
        GpioStopConfig(
            line_offset=23,
            debounce_seconds=0.05,
            poll_seconds=0.025,
        ),
        on_stop=lambda _trigger: None,
        on_status=lambda snapshot: statuses.append((snapshot, clock.now)),
        line_factory=lambda _config: line,
        clock=clock,
    )

    monitor.run()

    available = [
        observed_at
        for snapshot, observed_at in statuses
        if snapshot.state is StopInputState.AVAILABLE
    ]
    assert available == [pytest.approx(0.05)]
    assert statuses[-1][0].state is StopInputState.UNAVAILABLE


def test_monitor_held_at_boot_stops_without_any_available_window() -> None:
    clock = FakeClock()
    line = ScriptedLine(
        [True, True, True],
        clock=clock,
        terminal_error=OSError("end deterministic startup probe"),
    )
    events: list[tuple[str, object]] = []
    monitor = PhysicalStopMonitor(
        GpioStopConfig(
            line_offset=23,
            debounce_seconds=0.05,
            poll_seconds=0.025,
        ),
        on_stop=lambda trigger: events.append(("stop", trigger)),
        on_status=lambda snapshot: events.append(("status", snapshot.state)),
        line_factory=lambda _config: line,
        clock=clock,
    )

    monitor.run()

    assert StopInputState.AVAILABLE not in [value for kind, value in events if kind == "status"]
    assert events[:2] == [
        ("status", StopInputState.PRESSED),
        ("stop", StopInputTrigger.BUTTON_PRESSED),
    ]
    assert monitor.snapshot == StopInputSnapshot(
        state=StopInputState.UNAVAILABLE,
        stop_count=2,
        failure_kind="GPIOMonitorError",
    )


def test_monitor_invokes_once_per_debounced_press_and_once_for_input_loss() -> None:
    clock = FakeClock()
    line = ScriptedLine(
        [
            False,
            True,
            False,  # first contact bounce
            True,
            True,
            True,  # first stable press
            True,
            False,
            False,
            False,
            False,  # stable release rearms
            True,
            True,
            True,
            True,  # second stable press
        ],
        clock=clock,
        terminal_error=OSError("GPIO request disappeared"),
    )
    triggers: list[StopInputTrigger] = []
    statuses: list[StopInputSnapshot] = []
    monitor = PhysicalStopMonitor(
        GpioStopConfig(
            line_offset=23,
            debounce_seconds=0.05,
            poll_seconds=0.025,
        ),
        on_stop=triggers.append,
        on_status=statuses.append,
        line_factory=lambda _config: line,
        clock=clock,
    )

    monitor.run()

    assert triggers == [
        StopInputTrigger.BUTTON_PRESSED,
        StopInputTrigger.BUTTON_PRESSED,
        StopInputTrigger.INPUT_UNAVAILABLE,
    ]
    assert monitor.snapshot == StopInputSnapshot(
        state=StopInputState.UNAVAILABLE,
        stop_count=3,
        failure_kind="GPIOMonitorError",
    )
    assert [status.state for status in statuses] == [
        StopInputState.PRESSED,
        StopInputState.AVAILABLE,
        StopInputState.PRESSED,
        StopInputState.UNAVAILABLE,
    ]
    assert line.closed is True


def test_monitor_open_failure_fails_closed_once_without_exception_content() -> None:
    triggers: list[StopInputTrigger] = []
    statuses: list[StopInputSnapshot] = []

    def unavailable(_config: GpioStopConfig) -> ScriptedLine:
        raise PermissionError("sensitive path details")

    monitor = PhysicalStopMonitor(
        GpioStopConfig(line_offset=23),
        on_stop=triggers.append,
        on_status=statuses.append,
        line_factory=unavailable,
    )

    monitor.run()

    assert triggers == [StopInputTrigger.INPUT_UNAVAILABLE]
    assert monitor.snapshot == StopInputSnapshot(
        state=StopInputState.UNAVAILABLE,
        stop_count=1,
        failure_kind="GPIOOpenError",
    )
    assert statuses[-1].failure_kind == "GPIOOpenError"
    assert "sensitive" not in repr(statuses)


def test_stop_callback_failure_is_not_retried_as_an_input_loss() -> None:
    clock = FakeClock()
    line = ScriptedLine(
        [True, True, True],
        clock=clock,
        terminal_error=AssertionError("monitor should stop before another wait"),
    )
    triggers: list[StopInputTrigger] = []

    def failing_stop(trigger: StopInputTrigger) -> None:
        triggers.append(trigger)
        raise RuntimeError("service stop callback failed")

    monitor = PhysicalStopMonitor(
        GpioStopConfig(
            line_offset=23,
            debounce_seconds=0.05,
            poll_seconds=0.025,
        ),
        on_stop=failing_stop,
        line_factory=lambda _config: line,
        clock=clock,
    )

    monitor.run()

    assert triggers == [StopInputTrigger.BUTTON_PRESSED]
    assert monitor.snapshot == StopInputSnapshot(
        state=StopInputState.UNAVAILABLE,
        stop_count=1,
        failure_kind="StopCallbackError",
    )
    assert line.closed is True


def test_background_monitor_closes_without_manufacturing_input_loss() -> None:
    clock = FakeClock()
    opened = threading.Event()

    class ClosingLine:
        def __init__(self) -> None:
            self.closed = False

        def read_active(self) -> bool:
            return False

        def wait_for_change(self, timeout_seconds: float) -> bool:
            clock.advance(timeout_seconds)
            return False

        def close(self) -> None:
            self.closed = True

    line = ClosingLine()
    triggers: list[StopInputTrigger] = []

    def open_line(_config: GpioStopConfig) -> ClosingLine:
        opened.set()
        return line

    monitor = PhysicalStopMonitor(
        GpioStopConfig(line_offset=23),
        on_stop=triggers.append,
        line_factory=open_line,
        clock=clock,
    )

    monitor.start()
    assert opened.wait(0.5)
    monitor.close(timeout_seconds=1.0)

    assert triggers == []
    assert monitor.snapshot.state is StopInputState.CLOSED
    assert line.closed is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GpioStopConfig(line_offset=True),
        lambda: GpioStopConfig(line_offset=23, active_low=1),
        lambda: GpioStopConfig(line_offset=23, bias="pull_up"),
        lambda: GpioStopConfig(line_offset=23, poll_seconds=0.2),
    ],
)
def test_gpio_config_rejects_ambiguous_values(factory: Callable[[], object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()
