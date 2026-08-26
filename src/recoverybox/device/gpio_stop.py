"""Fail-closed, Pi-local physical stop input.

The module deliberately imports ``gpiod`` only when the explicit hardware
factory is called.  Ordinary tests and imports therefore stay hardware-free.
The monitor treats loss of the GPIO request as a stop condition: a missing,
busy, or failed button is never interpreted as permission to continue.
"""

from __future__ import annotations

import errno
import importlib
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class GpioBias(StrEnum):
    """Bias requested from the GPIO character-device API."""

    PULL_UP = "pull_up"
    PULL_DOWN = "pull_down"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class GpioStopConfig:
    """Explicit wiring and timing for one physical stop button.

    ``line_offset`` intentionally has no default.  Deployments must name the
    wired BCM GPIO rather than silently enabling an assumed pin.  The verified
    RecoveryBox Pi3 wiring uses ``line_offset=23`` on ``/dev/gpiochip0``, with
    an active-low button and pull-up bias.
    """

    line_offset: int
    chip_path: Path = Path("/dev/gpiochip0")
    active_low: bool = True
    bias: GpioBias = GpioBias.PULL_UP
    debounce_seconds: float = 0.10
    poll_seconds: float = 0.025

    def __post_init__(self) -> None:
        if isinstance(self.line_offset, bool) or not isinstance(self.line_offset, int):
            raise TypeError("line_offset must be an integer")
        if not 0 <= self.line_offset <= 1_023:
            raise ValueError("line_offset must be between 0 and 1023")
        if not isinstance(self.chip_path, Path):
            raise TypeError("chip_path must be a pathlib.Path")
        if not self.chip_path.is_absolute():
            raise ValueError("chip_path must be absolute")
        if not isinstance(self.active_low, bool):
            raise TypeError("active_low must be a boolean")
        if not isinstance(self.bias, GpioBias):
            raise TypeError("bias must be a GpioBias")
        _require_finite_duration(
            "debounce_seconds",
            self.debounce_seconds,
            minimum=0.05,
            maximum=1.0,
        )
        _require_finite_duration(
            "poll_seconds",
            self.poll_seconds,
            minimum=0.005,
            maximum=0.10,
        )
        if self.poll_seconds > self.debounce_seconds:
            raise ValueError("poll_seconds cannot exceed debounce_seconds")


class StopInputTrigger(StrEnum):
    """A local condition that must request the deterministic STOP boundary."""

    BUTTON_PRESSED = "button_pressed"
    INPUT_UNAVAILABLE = "input_unavailable"


class StopInputState(StrEnum):
    """Sanitized physical-input state suitable for local status output."""

    STARTING = "starting"
    AVAILABLE = "available"
    PRESSED = "pressed"
    UNAVAILABLE = "unavailable"
    CLOSED = "closed"


class StopInputTransition(StrEnum):
    """A completed, software-debounced state transition."""

    PRESSED = "pressed"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class StopInputSnapshot:
    """Content-free observable state for the service status endpoint."""

    state: StopInputState
    stop_count: int
    failure_kind: str | None


class StopLine(Protocol):
    """Small injectable boundary around one requested GPIO input line."""

    def read_active(self) -> bool:
        """Return the logical button state after active-low handling."""

    def wait_for_change(self, timeout_seconds: float) -> bool:
        """Wait for an edge, returning false on an ordinary timeout."""

    def close(self) -> None:
        """Release the GPIO line."""


class DebouncedStopLatch:
    """Require a state to remain stable before emitting one transition.

    The latch begins in the safe electrical expectation of an unpressed
    pulled-up button.  An already-held button at startup must remain active for
    the configured debounce interval and then emits exactly one press.
    """

    def __init__(self, debounce_seconds: float) -> None:
        _require_finite_duration(
            "debounce_seconds",
            debounce_seconds,
            minimum=0.05,
            maximum=1.0,
        )
        self._debounce_seconds = float(debounce_seconds)
        self._stable_active = False
        self._candidate_active: bool | None = None
        self._candidate_since: float | None = None
        self._last_observed_at: float | None = None

    @property
    def stable_active(self) -> bool:
        return self._stable_active

    def observe(
        self,
        active: bool,
        *,
        observed_at: float,
    ) -> StopInputTransition | None:
        """Observe one raw state without interpreting bounce as another press."""

        if not isinstance(active, bool):
            raise TypeError("active must be a boolean")
        timestamp = _finite_time(observed_at)
        if self._last_observed_at is not None and timestamp < self._last_observed_at:
            raise ValueError("observed_at cannot move backwards")
        self._last_observed_at = timestamp

        if active is self._stable_active:
            self._candidate_active = None
            self._candidate_since = None
            return None

        if active is not self._candidate_active:
            self._candidate_active = active
            self._candidate_since = timestamp
            return None

        assert self._candidate_since is not None
        if timestamp - self._candidate_since < self._debounce_seconds:
            return None

        self._stable_active = active
        self._candidate_active = None
        self._candidate_since = None
        if active:
            return StopInputTransition.PRESSED
        return StopInputTransition.RELEASED


class GpiodStopLine:
    """Adapter for a libgpiod 2.x ``LineRequest``."""

    def __init__(self, request: object, *, line_offset: int, active_value: object) -> None:
        self._request = request
        self._line_offset = line_offset
        self._active_value = active_value
        self._closed = False

    def read_active(self) -> bool:
        if self._closed:
            raise RuntimeError("GPIO stop line is closed")
        getter = self._request.get_value
        return bool(getter(self._line_offset) == self._active_value)

    def wait_for_change(self, timeout_seconds: float) -> bool:
        if self._closed:
            raise RuntimeError("GPIO stop line is closed")
        ready = bool(self._request.wait_edge_events(timeout_seconds))
        if ready:
            # State is read after the event. Event metadata is intentionally
            # discarded so event-clock quirks cannot weaken the local latch.
            self._request.read_edge_events(64)
        return ready

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._request.release()


def open_gpiod_stop_line(
    config: GpioStopConfig,
    *,
    runtime: object | None = None,
) -> GpiodStopLine:
    """Request the configured input through libgpiod 2.x.

    ``runtime`` is an injection point for hardware-free tests.  Production
    callers omit it, which performs the first and only import of ``gpiod``.
    Kernel debounce is requested first. If the GPIO driver explicitly reports
    that setting unsupported, the same line is requested without it and the
    software :class:`DebouncedStopLatch` remains authoritative.
    """

    if not isinstance(config, GpioStopConfig):
        raise TypeError("config must be a GpioStopConfig")
    selected_runtime = importlib.import_module("gpiod") if runtime is None else runtime
    line = selected_runtime.line
    bias_by_config = {
        GpioBias.PULL_UP: line.Bias.PULL_UP,
        GpioBias.PULL_DOWN: line.Bias.PULL_DOWN,
        GpioBias.DISABLED: line.Bias.DISABLED,
    }

    def request(*, debounce_period: timedelta) -> object:
        settings = selected_runtime.LineSettings(
            direction=line.Direction.INPUT,
            edge_detection=line.Edge.BOTH,
            bias=bias_by_config[config.bias],
            active_low=config.active_low,
            debounce_period=debounce_period,
            event_clock=line.Clock.MONOTONIC,
        )
        return selected_runtime.request_lines(
            str(config.chip_path),
            consumer="recoverybox-physical-stop",
            config={config.line_offset: settings},
            event_buffer_size=16,
        )

    try:
        requested_line = request(debounce_period=timedelta(seconds=config.debounce_seconds))
    except OSError as exc:
        unsupported_errors = {errno.EINVAL, errno.ENOTSUP}
        if hasattr(errno, "EOPNOTSUPP"):
            unsupported_errors.add(errno.EOPNOTSUPP)
        if exc.errno not in unsupported_errors:
            raise
        requested_line = request(debounce_period=timedelta(0))
    return GpiodStopLine(
        requested_line,
        line_offset=config.line_offset,
        active_value=line.Value.ACTIVE,
    )


LineFactory = Callable[[GpioStopConfig], StopLine]
StopCallback = Callable[[StopInputTrigger], None]
StatusCallback = Callable[[StopInputSnapshot], None]


class PhysicalStopMonitor:
    """Monitor one local button and fail closed if its input disappears."""

    def __init__(
        self,
        config: GpioStopConfig,
        *,
        on_stop: StopCallback,
        on_status: StatusCallback | None = None,
        line_factory: LineFactory = open_gpiod_stop_line,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, GpioStopConfig):
            raise TypeError("config must be a GpioStopConfig")
        if not callable(on_stop):
            raise TypeError("on_stop must be callable")
        if on_status is not None and not callable(on_status):
            raise TypeError("on_status must be callable or None")
        if not callable(line_factory):
            raise TypeError("line_factory must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.config = config
        self._on_stop = on_stop
        self._on_status = on_status
        self._line_factory = line_factory
        self._clock = clock

        self._lock = threading.RLock()
        self._close_requested = threading.Event()
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._state = StopInputState.STARTING
        self._stop_count = 0
        self._failure_kind: str | None = None
        self._loss_notified = False

    @property
    def snapshot(self) -> StopInputSnapshot:
        with self._lock:
            return StopInputSnapshot(
                state=self._state,
                stop_count=self._stop_count,
                failure_kind=self._failure_kind,
            )

    def start(self) -> None:
        """Start a daemon monitor exactly once."""

        with self._lock:
            self._claim_start_locked()
            thread = threading.Thread(
                target=self._run_claimed,
                name="recoverybox-physical-stop",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def run(self) -> None:
        """Run synchronously; useful for a service-owned worker or tests."""

        with self._lock:
            self._claim_start_locked()
        self._run_claimed()

    def close(self, *, timeout_seconds: float = 1.0) -> None:
        """Stop monitoring without manufacturing an input-loss episode."""

        _require_finite_duration(
            "timeout_seconds",
            timeout_seconds,
            minimum=0.01,
            maximum=10.0,
        )
        self._close_requested.set()
        with self._lock:
            thread = self._thread
            started = self._started
        if not started:
            self._set_status(StopInputState.CLOSED, failure_kind=None)
            self._finished.set()
            return
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_seconds)
            if thread.is_alive():
                raise TimeoutError("physical stop monitor did not close")
        elif thread is None:
            if not self._finished.wait(timeout_seconds):
                raise TimeoutError("physical stop monitor did not close")

    def wait_finished(self, timeout_seconds: float) -> bool:
        """Wait for shutdown or a terminal unavailable state."""

        _require_finite_duration(
            "timeout_seconds",
            timeout_seconds,
            minimum=0.0,
            maximum=60.0,
        )
        return self._finished.wait(timeout_seconds)

    def _claim_start_locked(self) -> None:
        if self._started:
            raise RuntimeError("physical stop monitor can only be started once")
        if self._close_requested.is_set():
            raise RuntimeError("physical stop monitor is already closed")
        self._started = True

    def _run_claimed(self) -> None:
        line: StopLine | None = None
        try:
            if self._close_requested.is_set():
                return
            try:
                line = self._line_factory(self.config)
            except Exception:
                self._mark_unavailable("GPIOOpenError")
                return

            latch = DebouncedStopLatch(self.config.debounce_seconds)
            # STARTING is a real safety gate, not merely a cosmetic status.
            # The line must be observed continuously released for one software
            # debounce interval before a caller may bind a listener. A button
            # already held at boot therefore reaches PRESSED first and can
            # never create a transient AVAILABLE window.
            initial_release_since: float | None = None
            initial_release_confirmed = False
            while not self._close_requested.is_set():
                try:
                    active = line.read_active()
                    observed_at = _finite_time(self._clock())
                    transition = latch.observe(active, observed_at=observed_at)
                except Exception:
                    self._mark_unavailable("GPIOReadError")
                    return

                if not initial_release_confirmed:
                    if active:
                        initial_release_since = None
                    elif transition is StopInputTransition.RELEASED:
                        initial_release_confirmed = True
                    elif initial_release_since is None:
                        initial_release_since = observed_at
                    elif observed_at - initial_release_since >= self.config.debounce_seconds:
                        initial_release_confirmed = True

                if transition is StopInputTransition.PRESSED:
                    if not self._emit_stop(
                        StopInputTrigger.BUTTON_PRESSED,
                        state=StopInputState.PRESSED,
                        failure_kind=None,
                    ):
                        return
                elif transition is StopInputTransition.RELEASED:
                    self._set_status(StopInputState.AVAILABLE, failure_kind=None)
                elif initial_release_confirmed and self.snapshot.state is StopInputState.STARTING:
                    self._set_status(StopInputState.AVAILABLE, failure_kind=None)

                if self._close_requested.is_set():
                    break
                try:
                    line.wait_for_change(self.config.poll_seconds)
                except Exception:
                    self._mark_unavailable("GPIOMonitorError")
                    return
        finally:
            if line is not None:
                try:
                    line.close()
                except Exception:
                    # A release failure cannot generate another STOP after a
                    # prior loss, and normal service shutdown is already safe.
                    pass
            with self._lock:
                terminal_unavailable = self._state is StopInputState.UNAVAILABLE
            if not terminal_unavailable:
                self._set_status(StopInputState.CLOSED, failure_kind=None)
            self._finished.set()

    def _mark_unavailable(self, failure_kind: str) -> None:
        with self._lock:
            if self._loss_notified:
                return
            self._loss_notified = True
        self._emit_stop(
            StopInputTrigger.INPUT_UNAVAILABLE,
            state=StopInputState.UNAVAILABLE,
            failure_kind=failure_kind,
        )

    def _emit_stop(
        self,
        trigger: StopInputTrigger,
        *,
        state: StopInputState,
        failure_kind: str | None,
    ) -> bool:
        with self._lock:
            self._stop_count += 1
            self._state = state
            self._failure_kind = failure_kind
            snapshot = StopInputSnapshot(
                state=state,
                stop_count=self._stop_count,
                failure_kind=failure_kind,
            )
        self._report_status(snapshot)
        try:
            self._on_stop(trigger)
        except Exception:
            # Retrying a possibly committed STOP would violate the one-shot
            # boundary. Report the callback path unavailable and terminate.
            self._set_status(StopInputState.UNAVAILABLE, failure_kind="StopCallbackError")
            return False
        return True

    def _set_status(
        self,
        state: StopInputState,
        *,
        failure_kind: str | None,
    ) -> None:
        with self._lock:
            self._state = state
            self._failure_kind = failure_kind
            snapshot = StopInputSnapshot(
                state=state,
                stop_count=self._stop_count,
                failure_kind=failure_kind,
            )
        self._report_status(snapshot)

    def _report_status(self, snapshot: StopInputSnapshot) -> None:
        if self._on_status is not None:
            try:
                self._on_status(snapshot)
            except Exception:
                # Status reporting is observational and cannot weaken STOP.
                pass


def _finite_time(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("clock must return a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("clock must return a finite number")
    return result


def _require_finite_duration(
    name: str,
    value: float,
    *,
    minimum: float,
    maximum: float,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if not minimum <= converted <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


__all__ = [
    "DebouncedStopLatch",
    "GpioBias",
    "GpioStopConfig",
    "GpiodStopLine",
    "PhysicalStopMonitor",
    "StopInputSnapshot",
    "StopInputState",
    "StopInputTransition",
    "StopInputTrigger",
    "StopLine",
    "open_gpiod_stop_line",
]
