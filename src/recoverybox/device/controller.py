"""Deterministic button, LED, capture, playback, and conversation controller."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from .ports import (
    PCM_S16LE_24K_MONO,
    AudioFormat,
    ConversationPort,
    DeviceState,
    LedMode,
    LedPort,
    PlaybackPort,
    RecorderPort,
)


def _require_finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Timing and audio constraints for the physical interaction loop."""

    min_capture_seconds: float = 0.20
    max_capture_seconds: float = 30.0
    audio_format: AudioFormat = PCM_S16LE_24K_MONO

    def __post_init__(self) -> None:
        _require_finite_number("min_capture_seconds", self.min_capture_seconds)
        _require_finite_number("max_capture_seconds", self.max_capture_seconds)
        if self.min_capture_seconds < 0:
            raise ValueError("min_capture_seconds cannot be negative")
        if self.max_capture_seconds <= 0:
            raise ValueError("max_capture_seconds must be positive")
        if self.max_capture_seconds < self.min_capture_seconds:
            raise ValueError("max_capture_seconds must be at least min_capture_seconds")


class _ResponseEventKind(StrEnum):
    STARTED = "started"
    AUDIO = "audio"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class _ResponseEvent:
    kind: _ResponseEventKind
    turn_id: str
    response_id: str | None
    item_id: str | None = None
    pcm: bytes = b""
    error: Exception | None = None


class DeviceController:
    """Own the physical push-to-talk lifecycle.

    The state lock is used only for short, in-memory transitions. Hardware,
    subprocess, clock, and conversation calls always run after it is released.
    This is important because the Realtime receive pump invokes controller
    callbacks while holding its own correlation lock.

    Realtime callbacks carry both a locally assigned ``turn_id`` and the
    provider's ``response_id``. Events from superseded turns are ignored, so a
    late packet cannot restart playback after a barge-in or session clear.
    """

    def __init__(
        self,
        *,
        led: LedPort,
        recorder: RecorderPort,
        playback: PlaybackPort,
        conversation: ConversationPort,
        config: ControllerConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._led = led
        self._recorder = recorder
        self._playback = playback
        self._conversation = conversation
        self._config = config or ControllerConfig()
        self._clock = clock

        # Never invoke a port while holding _lock. _ports_lock serializes the
        # physical half of a transition, but is released before conversation
        # calls so it cannot form a cycle with the Realtime pump lock.
        self._lock = threading.RLock()
        self._ports_lock = threading.RLock()

        self._state = DeviceState.IDLE
        self._epoch = 0
        self._pressed_at: float | None = None
        self._turn_id: str | None = None
        self._response_id: str | None = None
        self._assistant_item_id: str | None = None
        self._last_error: str | None = None

        self._capture_started = False
        self._capture_stopping = False
        self._pending_capture_finish_at: float | None = None
        self._defer_response_events = False
        self._deferred_response_events: deque[_ResponseEvent] = deque()

        # These two fields are protected by _ports_lock. Keys prevent a late
        # completion from finishing a newer physical stream.
        self._recorder_epoch: int | None = None
        self._playback_key: tuple[str, str] | None = None

        self._led.set_mode(LedMode.OFF)

    @property
    def state(self) -> DeviceState:
        with self._lock:
            return self._state

    @property
    def active_turn_id(self) -> str | None:
        with self._lock:
            return self._turn_id

    @property
    def active_response_id(self) -> str | None:
        with self._lock:
            return self._response_id

    @property
    def active_assistant_item_id(self) -> str | None:
        with self._lock:
            return self._assistant_item_id

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def on_button_pressed(self) -> None:
        """Begin capture, preempting a pending or audible response if needed."""

        try:
            pressed_at = self._clock()
        except Exception as exc:
            self._enter_error(exc)
            return

        with self._lock:
            if self._state in {DeviceState.ENDED, DeviceState.ERROR, DeviceState.RECORDING}:
                return

            prior_state = self._state
            prior_turn_id = self._turn_id
            prior_response_id = self._response_id
            prior_item_id = self._assistant_item_id

            self._epoch += 1
            epoch = self._epoch
            # Invalidate the previous turn before any blocking work. A pump
            # callback that races with cancellation now fails correlation.
            self._turn_id = None
            self._response_id = None
            self._assistant_item_id = None
            self._pressed_at = pressed_at
            self._capture_started = False
            self._capture_stopping = False
            self._pending_capture_finish_at = None
            self._clear_deferred_locked()
            self._state = DeviceState.RECORDING

        self._start_capture(
            epoch=epoch,
            prior_state=prior_state,
            prior_turn_id=prior_turn_id,
            prior_response_id=prior_response_id,
            prior_item_id=prior_item_id,
        )

    def on_button_released(self) -> None:
        """Commit a sufficiently long manual turn."""

        try:
            now = self._clock()
        except Exception as exc:
            self._enter_error(exc)
            return
        self._finish_capture(now=now)

    def on_tick(self) -> None:
        """Bound a held button so a failed release event cannot record forever."""

        try:
            now = self._clock()
        except Exception as exc:
            self._enter_error(exc)
            return
        with self._lock:
            if self._state is not DeviceState.RECORDING or self._pressed_at is None:
                return
            should_finish = now - self._pressed_at >= self._config.max_capture_seconds
        if should_finish:
            self._finish_capture(now=now)

    def on_double_click(self) -> None:
        """Stop all audio, clear remote context, and end this device session."""

        with self._lock:
            if self._state is DeviceState.ENDED:
                return
            prior_turn_id = self._turn_id
            prior_response_id = self._response_id
            self._epoch += 1
            epoch = self._epoch
            self._pressed_at = None
            self._turn_id = None
            self._response_id = None
            self._assistant_item_id = None
            self._capture_started = False
            self._capture_stopping = False
            self._pending_capture_finish_at = None
            self._clear_deferred_locked()
            self._state = DeviceState.ENDED

        first_error = self._stop_physical_ports()
        if prior_turn_id is not None:
            try:
                self._conversation.cancel_response(prior_turn_id, prior_response_id)
            except Exception as exc:
                first_error = first_error or exc
        try:
            self._conversation.clear_and_end()
        except Exception as exc:
            first_error = first_error or exc
        try:
            with self._ports_lock:
                self._led.set_mode(LedMode.OFF)
        except Exception as exc:
            first_error = first_error or exc

        if first_error is not None:
            self._enter_error(first_error, expected_epoch=epoch)

    def recover_after_connectivity_restored(self, *, conversation: ConversationPort) -> None:
        """Explicitly recover a contained error after connectivity is restored.

        Supplying ``conversation`` is an explicit caller assertion that the
        Realtime connection has been restored or replaced. Recovery also
        verifies that recorder/playback cleanup and the quiet LED state can be
        reached. Any cleanup failure leaves the controller in ``ERROR`` and is
        raised to the caller; hardware failures are never silently cleared.
        """

        with self._lock:
            if self._state is not DeviceState.ERROR:
                raise RuntimeError("controller recovery requires ERROR state")
            self._epoch += 1
            epoch = self._epoch

        failures: list[Exception] = []
        # Recovery deliberately calls both idempotent cleanup ports even if a
        # prior best-effort cleanup cleared the controller's ownership keys.
        with self._ports_lock:
            try:
                self._recorder.abort()
            except Exception as exc:
                failures.append(exc)
            finally:
                self._recorder_epoch = None
            try:
                self._playback.stop()
            except Exception as exc:
                failures.append(exc)
            finally:
                self._playback_key = None
            try:
                self._led.set_mode(LedMode.OFF)
            except Exception as exc:
                failures.append(exc)

        if failures:
            failure = failures[0]
            with self._lock:
                if self._epoch == epoch and self._state is DeviceState.ERROR:
                    self._last_error = f"Recovery failed: {type(failure).__name__}: {failure}"
            try:
                with self._ports_lock:
                    self._led.set_mode(LedMode.FAST_BLINK)
            except Exception:
                pass
            raise RuntimeError("controller recovery cleanup failed") from failure

        with self._lock:
            if self._epoch != epoch or self._state is not DeviceState.ERROR:
                return
            self._conversation = conversation
            self._last_error = None
            self._state = DeviceState.IDLE
            self._epoch += 1

    def on_response_started(self, *, turn_id: str, response_id: str) -> None:
        """Start speaker output for the current turn only."""

        event = _ResponseEvent(_ResponseEventKind.STARTED, turn_id, response_id)
        self._handle_response_started(event, allow_defer=True)

    def on_response_audio(
        self,
        *,
        turn_id: str,
        response_id: str,
        item_id: str,
        pcm: bytes,
    ) -> None:
        """Play a correlated output chunk; discard stale or empty chunks."""

        if not pcm:
            return
        event = _ResponseEvent(
            _ResponseEventKind.AUDIO,
            turn_id,
            response_id,
            item_id=item_id,
            pcm=pcm,
        )
        self._handle_response_audio(event, allow_defer=True)

    def on_response_done(self, *, turn_id: str, response_id: str | None) -> None:
        """Finish the current response and return to quiet idle."""

        event = _ResponseEvent(_ResponseEventKind.DONE, turn_id, response_id)
        self._handle_response_done(event, allow_defer=True)

    def on_response_error(
        self,
        *,
        turn_id: str,
        response_id: str | None,
        error: Exception,
    ) -> None:
        """Contain a failure for the current response; ignore stale failures."""

        event = _ResponseEvent(_ResponseEventKind.ERROR, turn_id, response_id, error=error)
        self._handle_response_error(event, allow_defer=True)

    def _start_capture(
        self,
        *,
        epoch: int,
        prior_state: DeviceState,
        prior_turn_id: str | None,
        prior_response_id: str | None,
        prior_item_id: str | None,
    ) -> None:
        try:
            played_ms = 0
            with self._ports_lock:
                if not self._capture_is_current(epoch):
                    return
                if self._playback_key is not None:
                    played_ms = self._playback.stop()
                    self._playback_key = None

            if prior_turn_id is not None and prior_state in {
                DeviceState.WAITING,
                DeviceState.SPEAKING,
            }:
                if not self._capture_is_current(epoch):
                    return
                self._conversation.cancel_response(prior_turn_id, prior_response_id)
                if prior_state is DeviceState.SPEAKING and prior_item_id is not None:
                    if not self._capture_is_current(epoch):
                        return
                    self._conversation.truncate_assistant(prior_item_id, max(0, played_ms))

            with self._ports_lock:
                if not self._capture_is_current(epoch):
                    return
                self._recorder.start()
                self._recorder_epoch = epoch

                with self._lock:
                    if self._capture_is_current_locked(epoch):
                        self._capture_started = True
                        pending_finish_at = self._pending_capture_finish_at
                        self._pending_capture_finish_at = None
                    else:
                        pending_finish_at = None

                if not self._capture_is_current(epoch):
                    try:
                        self._recorder.abort()
                    finally:
                        self._recorder_epoch = None
                    return
                self._led.set_mode(LedMode.SOLID)

            if pending_finish_at is not None:
                self._finish_capture(now=pending_finish_at)
        except Exception as exc:
            self._enter_error(exc, expected_epoch=epoch)

    def _finish_capture(self, *, now: float) -> None:
        with self._lock:
            if self._state is not DeviceState.RECORDING or self._capture_stopping:
                return
            if not self._capture_started:
                if self._pending_capture_finish_at is None:
                    self._pending_capture_finish_at = now
                else:
                    self._pending_capture_finish_at = min(
                        self._pending_capture_finish_at,
                        now,
                    )
                return
            epoch = self._epoch
            pressed_at = self._pressed_at
            self._capture_started = False
            self._capture_stopping = True
            self._pending_capture_finish_at = None
            self._defer_response_events = True
            self._deferred_response_events.clear()

        try:
            with self._ports_lock:
                if not self._capture_is_stopping(epoch):
                    return
                if self._recorder_epoch != epoch:
                    raise RuntimeError("recording state has no active recorder")
                pcm = self._recorder.stop()
                self._recorder_epoch = None

            duration = 0.0 if pressed_at is None else max(0.0, now - pressed_at)
            if duration < self._config.min_capture_seconds or not pcm:
                self._discard_capture(epoch)
                return

            turn_id = self._conversation.send_audio_turn(
                pcm,
                audio_format=self._config.audio_format,
            )
            if not turn_id:
                raise RuntimeError("conversation returned an empty turn ID")

            with self._lock:
                if not self._capture_is_stopping_locked(epoch):
                    accepted = False
                else:
                    self._turn_id = turn_id
                    self._response_id = None
                    self._assistant_item_id = None
                    self._pressed_at = None
                    self._capture_stopping = False
                    self._state = DeviceState.WAITING
                    accepted = True

            if not accepted:
                try:
                    self._conversation.cancel_response(turn_id, None)
                except Exception:
                    pass
                return

            with self._ports_lock:
                if self._waiting_is_current(epoch, turn_id):
                    self._led.set_mode(LedMode.BLINK)
            self._drain_deferred_events()
        except Exception as exc:
            self._enter_error(exc, expected_epoch=epoch)

    def _discard_capture(self, epoch: int) -> None:
        with self._lock:
            if not self._capture_is_stopping_locked(epoch):
                return
            self._pressed_at = None
            self._capture_stopping = False
            self._clear_deferred_locked()
            self._state = DeviceState.IDLE
            self._epoch += 1
            idle_epoch = self._epoch
        try:
            with self._ports_lock:
                if self._idle_is_current(idle_epoch):
                    self._led.set_mode(LedMode.OFF)
        except Exception as exc:
            self._enter_error(exc, expected_epoch=idle_epoch)

    def _handle_response_started(self, event: _ResponseEvent, *, allow_defer: bool) -> None:
        response_id = event.response_id
        if response_id is None:
            return
        with self._lock:
            if allow_defer and self._defer_response_events:
                self._deferred_response_events.append(event)
                return
            if self._state is not DeviceState.WAITING or event.turn_id != self._turn_id:
                return
            epoch = self._epoch
            self._response_id = response_id
            self._assistant_item_id = None
            self._state = DeviceState.SPEAKING

        key = (event.turn_id, response_id)
        try:
            with self._ports_lock:
                if not self._speaking_is_current(epoch, key):
                    return
                self._playback.start(response_id)
                self._playback_key = key
                if self._speaking_is_current(epoch, key):
                    self._led.set_mode(LedMode.BLINK)
        except Exception as exc:
            self._enter_error(exc, expected_epoch=epoch)

    def _handle_response_audio(self, event: _ResponseEvent, *, allow_defer: bool) -> None:
        response_id = event.response_id
        item_id = event.item_id
        if response_id is None or item_id is None or not event.pcm:
            return
        with self._lock:
            if allow_defer and self._defer_response_events:
                self._deferred_response_events.append(event)
                return
            if (
                self._state is not DeviceState.SPEAKING
                or event.turn_id != self._turn_id
                or response_id != self._response_id
            ):
                return
            if self._assistant_item_id is None:
                self._assistant_item_id = item_id
            elif item_id != self._assistant_item_id:
                return
            epoch = self._epoch

        key = (event.turn_id, response_id)
        try:
            # Do not hold physical-port serialization around pipe I/O. A full
            # speaker pipe may block; PlaybackPort.stop is required to be safe
            # concurrently and must remain able to preempt that write.
            with self._ports_lock:
                if not self._speaking_is_current(epoch, key) or self._playback_key != key:
                    return
            self._playback.write(response_id, event.pcm)
        except Exception as exc:
            self._enter_error(exc, expected_epoch=epoch)

    def _handle_response_done(self, event: _ResponseEvent, *, allow_defer: bool) -> None:
        with self._lock:
            if allow_defer and self._defer_response_events:
                self._deferred_response_events.append(event)
                return
            if event.turn_id != self._turn_id:
                return
            if self._state is DeviceState.SPEAKING:
                if event.response_id != self._response_id or event.response_id is None:
                    return
                playback_key = (event.turn_id, event.response_id)
            elif self._state is DeviceState.WAITING:
                playback_key = None
            else:
                return

            self._turn_id = None
            self._response_id = None
            self._assistant_item_id = None
            self._clear_deferred_locked()
            self._state = DeviceState.IDLE
            self._epoch += 1
            idle_epoch = self._epoch

        try:
            with self._ports_lock:
                if playback_key is not None and self._playback_key == playback_key:
                    self._playback.finish()
                    self._playback_key = None
                if self._idle_is_current(idle_epoch):
                    self._led.set_mode(LedMode.OFF)
        except Exception as exc:
            self._enter_error(exc, expected_epoch=idle_epoch)

    def _handle_response_error(self, event: _ResponseEvent, *, allow_defer: bool) -> None:
        with self._lock:
            if allow_defer and self._defer_response_events:
                self._deferred_response_events.append(event)
                return
            if event.turn_id != self._turn_id:
                return
            if self._response_id is not None and event.response_id != self._response_id:
                return
            epoch = self._epoch
        self._enter_error(
            event.error or RuntimeError("Realtime response failed"),
            expected_epoch=epoch,
        )

    def _drain_deferred_events(self) -> None:
        while True:
            with self._lock:
                if not self._defer_response_events:
                    return
                if not self._deferred_response_events:
                    self._defer_response_events = False
                    return
                event = self._deferred_response_events.popleft()

            if event.kind is _ResponseEventKind.STARTED:
                self._handle_response_started(event, allow_defer=False)
            elif event.kind is _ResponseEventKind.AUDIO:
                self._handle_response_audio(event, allow_defer=False)
            elif event.kind is _ResponseEventKind.DONE:
                self._handle_response_done(event, allow_defer=False)
            else:
                self._handle_response_error(event, allow_defer=False)

    def _enter_error(self, error: Exception, *, expected_epoch: int | None = None) -> None:
        """Enter fail-safe state, then clean up ports without the state lock."""

        with self._lock:
            if expected_epoch is not None and expected_epoch != self._epoch:
                return
            if self._state is DeviceState.ERROR and expected_epoch is None:
                return
            turn_id = self._turn_id
            response_id = self._response_id
            self._epoch += 1
            self._last_error = f"{type(error).__name__}: {error}"
            self._state = DeviceState.ERROR
            self._pressed_at = None
            self._turn_id = None
            self._response_id = None
            self._assistant_item_id = None
            self._capture_started = False
            self._capture_stopping = False
            self._pending_capture_finish_at = None
            self._clear_deferred_locked()

        self._stop_physical_ports()
        if turn_id is not None:
            try:
                self._conversation.cancel_response(turn_id, response_id)
            except Exception:
                pass
        try:
            with self._ports_lock:
                self._led.set_mode(LedMode.FAST_BLINK)
        except Exception:
            pass

    def _stop_physical_ports(self) -> Exception | None:
        first_error: Exception | None = None
        with self._ports_lock:
            if self._recorder_epoch is not None:
                try:
                    self._recorder.abort()
                except Exception as exc:
                    first_error = first_error or exc
                finally:
                    self._recorder_epoch = None
            if self._playback_key is not None:
                try:
                    self._playback.stop()
                except Exception as exc:
                    first_error = first_error or exc
                finally:
                    self._playback_key = None
        return first_error

    def _clear_deferred_locked(self) -> None:
        self._defer_response_events = False
        self._deferred_response_events.clear()

    def _capture_is_current(self, epoch: int) -> bool:
        with self._lock:
            return self._capture_is_current_locked(epoch)

    def _capture_is_current_locked(self, epoch: int) -> bool:
        return (
            self._epoch == epoch
            and self._state is DeviceState.RECORDING
            and not self._capture_stopping
        )

    def _capture_is_stopping(self, epoch: int) -> bool:
        with self._lock:
            return self._capture_is_stopping_locked(epoch)

    def _capture_is_stopping_locked(self, epoch: int) -> bool:
        return (
            self._epoch == epoch and self._state is DeviceState.RECORDING and self._capture_stopping
        )

    def _waiting_is_current(self, epoch: int, turn_id: str) -> bool:
        with self._lock:
            return (
                self._epoch == epoch
                and self._state is DeviceState.WAITING
                and self._turn_id == turn_id
            )

    def _speaking_is_current(self, epoch: int, key: tuple[str, str]) -> bool:
        with self._lock:
            return (
                self._epoch == epoch
                and self._state is DeviceState.SPEAKING
                and (self._turn_id, self._response_id) == key
            )

    def _idle_is_current(self, epoch: int) -> bool:
        with self._lock:
            return self._epoch == epoch and self._state is DeviceState.IDLE
