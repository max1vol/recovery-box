from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field

import pytest

from recoverybox.device import (
    AlsaCommandConfig,
    AudioFormat,
    ControllerConfig,
    DeviceController,
    DeviceState,
    LedMode,
    SubprocessPlayback,
    SubprocessRecorder,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeLed:
    modes: list[LedMode] = field(default_factory=list)

    def set_mode(self, mode: LedMode) -> None:
        self.modes.append(mode)


class FakeRecorder:
    def __init__(self, calls: list[tuple], playback: FakePlayback) -> None:
        self.calls = calls
        self.playback = playback
        self.active = False
        self.pcm = b"\x01\x00" * 12_000
        self.fail_on_abort = False
        self.fail_on_stop = False

    def start(self) -> None:
        assert not self.playback.active, "capture overlapped playback"
        assert not self.active
        self.calls.append(("recorder.start",))
        self.active = True

    def stop(self) -> bytes:
        self.calls.append(("recorder.stop",))
        if self.fail_on_stop:
            raise OSError("capture failed")
        assert self.active
        self.active = False
        return self.pcm

    def abort(self) -> None:
        self.calls.append(("recorder.abort",))
        if self.fail_on_abort:
            raise OSError("capture cleanup failed")
        self.active = False


class FakePlayback:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls
        self.active = False
        self.stream_id: str | None = None
        self.played_ms = 375
        self.data = bytearray()
        self.fail_on_write = False

    def start(self, stream_id: str) -> None:
        assert not self.active
        self.calls.append(("playback.start", stream_id))
        self.active = True
        self.stream_id = stream_id

    def write(self, stream_id: str, pcm: bytes) -> None:
        self.calls.append(("playback.write", stream_id, pcm))
        if self.fail_on_write:
            raise OSError("speaker disconnected")
        assert self.active
        assert stream_id == self.stream_id
        self.data.extend(pcm)

    def finish(self) -> None:
        self.calls.append(("playback.finish",))
        assert self.active
        self.active = False
        self.stream_id = None

    def stop(self) -> int:
        self.calls.append(("playback.stop",))
        self.active = False
        self.stream_id = None
        return self.played_ms


class FakeConversation:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls
        self.turn_number = 0
        self.fail_on_send = False

    def send_audio_turn(self, pcm: bytes, *, audio_format: AudioFormat) -> str:
        self.calls.append(("conversation.send", pcm, audio_format))
        if self.fail_on_send:
            raise ConnectionError("realtime unavailable")
        self.turn_number += 1
        return f"turn-{self.turn_number}"

    def cancel_response(self, turn_id: str, response_id: str | None) -> None:
        self.calls.append(("conversation.cancel", turn_id, response_id))

    def truncate_assistant(self, item_id: str, audio_end_ms: int) -> None:
        self.calls.append(("conversation.truncate", item_id, audio_end_ms))

    def clear_and_end(self) -> None:
        self.calls.append(("conversation.clear_and_end",))


@dataclass
class Rig:
    controller: DeviceController
    clock: FakeClock
    led: FakeLed
    recorder: FakeRecorder
    playback: FakePlayback
    conversation: FakeConversation
    calls: list[tuple]


@pytest.fixture
def rig() -> Rig:
    calls: list[tuple] = []
    clock = FakeClock()
    led = FakeLed()
    playback = FakePlayback(calls)
    recorder = FakeRecorder(calls, playback)
    conversation = FakeConversation(calls)
    controller = DeviceController(
        led=led,
        recorder=recorder,
        playback=playback,
        conversation=conversation,
        config=ControllerConfig(min_capture_seconds=0.2, max_capture_seconds=2.0),
        clock=clock,
    )
    return Rig(controller, clock, led, recorder, playback, conversation, calls)


def commit_turn(rig: Rig) -> str:
    rig.controller.on_button_pressed()
    rig.clock.advance(0.5)
    rig.controller.on_button_released()
    assert rig.controller.state is DeviceState.WAITING
    turn_id = rig.controller.active_turn_id
    assert turn_id is not None
    return turn_id


def test_manual_capture_led_and_response_lifecycle(rig: Rig) -> None:
    assert rig.controller.state is DeviceState.IDLE
    assert rig.led.modes == [LedMode.OFF]

    turn_id = commit_turn(rig)
    assert rig.led.modes[-2:] == [LedMode.SOLID, LedMode.BLINK]

    rig.controller.on_response_started(turn_id=turn_id, response_id="response-1")
    rig.controller.on_response_audio(
        turn_id=turn_id,
        response_id="response-1",
        item_id="item-1",
        pcm=b"audio",
    )
    assert rig.controller.state is DeviceState.SPEAKING
    assert bytes(rig.playback.data) == b"audio"
    assert rig.led.modes[-1] is LedMode.BLINK

    rig.controller.on_response_done(turn_id=turn_id, response_id="response-1")
    assert rig.controller.state is DeviceState.IDLE
    assert rig.led.modes[-1] is LedMode.OFF
    assert not rig.recorder.active
    assert not rig.playback.active


def test_short_or_empty_capture_is_discarded(rig: Rig) -> None:
    rig.controller.on_button_pressed()
    rig.clock.advance(0.05)
    rig.controller.on_button_released()
    assert rig.controller.state is DeviceState.IDLE
    assert not any(call[0] == "conversation.send" for call in rig.calls)

    rig.recorder.pcm = b""
    rig.controller.on_button_pressed()
    rig.clock.advance(0.5)
    rig.controller.on_button_released()
    assert rig.controller.state is DeviceState.IDLE
    assert not any(call[0] == "conversation.send" for call in rig.calls)


def test_stale_realtime_events_never_reach_playback(rig: Rig) -> None:
    turn_id = commit_turn(rig)
    rig.controller.on_response_started(turn_id="old-turn", response_id="stale")
    rig.controller.on_response_audio(
        turn_id="old-turn",
        response_id="stale",
        item_id="stale-item",
        pcm=b"must not play",
    )
    rig.controller.on_response_done(turn_id="old-turn", response_id="stale")
    assert rig.controller.state is DeviceState.WAITING
    assert not rig.playback.active

    rig.controller.on_response_started(turn_id=turn_id, response_id="current")
    rig.controller.on_response_audio(
        turn_id=turn_id,
        response_id="different",
        item_id="other-item",
        pcm=b"must not play",
    )
    assert bytes(rig.playback.data) == b""


def test_barge_in_stops_cancels_truncates_then_records(rig: Rig) -> None:
    first_turn = commit_turn(rig)
    rig.controller.on_response_started(turn_id=first_turn, response_id="response-1")
    rig.controller.on_response_audio(
        turn_id=first_turn,
        response_id="response-1",
        item_id="item-1",
        pcm=b"first audio",
    )

    before = len(rig.calls)
    rig.controller.on_button_pressed()
    barge_calls = rig.calls[before:]
    assert barge_calls == [
        ("playback.stop",),
        ("conversation.cancel", first_turn, "response-1"),
        ("conversation.truncate", "item-1", rig.playback.played_ms),
        ("recorder.start",),
    ]
    assert rig.controller.state is DeviceState.RECORDING
    assert rig.controller.active_turn_id is None
    assert rig.recorder.active
    assert not rig.playback.active

    # Packets already in flight for the canceled response stay inaudible.
    rig.controller.on_response_audio(
        turn_id=first_turn,
        response_id="response-1",
        item_id="item-1",
        pcm=b"late audio",
    )
    assert bytes(rig.playback.data) == b"first audio"


def test_failure_enters_error_and_stops_all_audio(rig: Rig) -> None:
    turn_id = commit_turn(rig)
    rig.controller.on_response_started(turn_id=turn_id, response_id="response-1")
    rig.playback.fail_on_write = True

    rig.controller.on_response_audio(
        turn_id=turn_id,
        response_id="response-1",
        item_id="item-1",
        pcm=b"audio",
    )

    assert rig.controller.state is DeviceState.ERROR
    assert rig.controller.last_error == "OSError: speaker disconnected"
    assert rig.led.modes[-1] is LedMode.FAST_BLINK
    assert not rig.recorder.active
    assert not rig.playback.active
    assert ("conversation.cancel", turn_id, "response-1") in rig.calls


def test_capture_failure_is_contained(rig: Rig) -> None:
    rig.recorder.fail_on_stop = True
    rig.controller.on_button_pressed()
    rig.clock.advance(0.5)
    rig.controller.on_button_released()

    assert rig.controller.state is DeviceState.ERROR
    assert not rig.recorder.active
    assert not rig.playback.active
    assert rig.led.modes[-1] is LedMode.FAST_BLINK


def test_long_hold_auto_commits_exactly_once(rig: Rig) -> None:
    rig.controller.on_button_pressed()
    rig.clock.advance(2.1)
    rig.controller.on_tick()
    assert rig.controller.state is DeviceState.WAITING

    rig.controller.on_button_released()
    sends = [call for call in rig.calls if call[0] == "conversation.send"]
    stops = [call for call in rig.calls if call[0] == "recorder.stop"]
    assert len(sends) == 1
    assert len(stops) == 1


def test_response_cannot_overlap_new_capture(rig: Rig) -> None:
    first_turn = commit_turn(rig)
    rig.controller.on_button_pressed()  # cancels WAITING, then starts capture
    assert rig.controller.state is DeviceState.RECORDING

    rig.controller.on_response_started(turn_id=first_turn, response_id="late")
    rig.controller.on_response_audio(
        turn_id=first_turn,
        response_id="late",
        item_id="late-item",
        pcm=b"late audio",
    )
    assert rig.controller.state is DeviceState.RECORDING
    assert rig.recorder.active
    assert not rig.playback.active


@pytest.mark.parametrize("initial", ["recording", "waiting", "speaking"])
def test_double_click_stops_and_ends_session(rig: Rig, initial: str) -> None:
    if initial == "recording":
        rig.controller.on_button_pressed()
    else:
        turn_id = commit_turn(rig)
        if initial == "speaking":
            rig.controller.on_response_started(turn_id=turn_id, response_id="response-1")

    rig.controller.on_double_click()

    assert rig.controller.state is DeviceState.ENDED
    assert rig.led.modes[-1] is LedMode.OFF
    assert not rig.recorder.active
    assert not rig.playback.active
    assert rig.calls[-1] == ("conversation.clear_and_end",)


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -float("inf")])
def test_controller_config_rejects_non_finite_and_boolean_timings(value: float) -> None:
    with pytest.raises(ValueError):
        ControllerConfig(min_capture_seconds=value)
    with pytest.raises(ValueError):
        ControllerConfig(max_capture_seconds=value)


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -float("inf")])
def test_alsa_config_rejects_non_finite_and_boolean_timeout(value: float) -> None:
    with pytest.raises(ValueError):
        AlsaCommandConfig(process_timeout_seconds=value)


def test_explicit_recovery_requires_restored_conversation(rig: Rig) -> None:
    rig.conversation.fail_on_send = True
    rig.controller.on_button_pressed()
    rig.clock.advance(0.5)
    rig.controller.on_button_released()
    assert rig.controller.state is DeviceState.ERROR

    restored = FakeConversation(rig.calls)
    rig.controller.recover_after_connectivity_restored(conversation=restored)

    assert rig.controller.state is DeviceState.IDLE
    assert rig.controller.last_error is None
    rig.controller.on_button_pressed()
    rig.clock.advance(0.5)
    rig.controller.on_button_released()
    assert rig.controller.state is DeviceState.WAITING
    assert restored.turn_number == 1


def test_recovery_cleanup_failure_keeps_hardware_contained(rig: Rig) -> None:
    rig.conversation.fail_on_send = True
    rig.controller.on_button_pressed()
    rig.clock.advance(0.5)
    rig.controller.on_button_released()
    rig.recorder.fail_on_abort = True

    with pytest.raises(RuntimeError, match="recovery cleanup failed"):
        rig.controller.recover_after_connectivity_restored(conversation=FakeConversation(rig.calls))

    assert rig.controller.state is DeviceState.ERROR
    assert rig.controller.last_error == "Recovery failed: OSError: capture cleanup failed"
    assert rig.led.modes[-1] is LedMode.FAST_BLINK


class BlockingPlayback(FakePlayback):
    def __init__(self, calls: list[tuple]) -> None:
        super().__init__(calls)
        self.stop_entered = threading.Event()
        self.allow_stop = threading.Event()

    def stop(self) -> int:
        self.calls.append(("playback.stop",))
        self.stop_entered.set()
        if not self.allow_stop.wait(timeout=1):
            raise TimeoutError("test did not release playback stop")
        self.active = False
        return self.played_ms


class LockingConversation(FakeConversation):
    def __init__(self, calls: list[tuple]) -> None:
        super().__init__(calls)
        self.adapter_lock = threading.Lock()

    def cancel_response(self, turn_id: str, response_id: str | None) -> None:
        with self.adapter_lock:
            super().cancel_response(turn_id, response_id)


def test_barge_in_cannot_deadlock_with_pump_callback() -> None:
    calls: list[tuple] = []
    clock = FakeClock()
    led = FakeLed()
    playback = BlockingPlayback(calls)
    recorder = FakeRecorder(calls, playback)
    conversation = LockingConversation(calls)
    controller = DeviceController(
        led=led,
        recorder=recorder,
        playback=playback,
        conversation=conversation,
        config=ControllerConfig(min_capture_seconds=0.2, max_capture_seconds=2.0),
        clock=clock,
    )
    local_rig = Rig(controller, clock, led, recorder, playback, conversation, calls)
    turn_id = commit_turn(local_rig)
    controller.on_response_started(turn_id=turn_id, response_id="response-1")
    controller.on_response_audio(
        turn_id=turn_id,
        response_id="response-1",
        item_id="item-1",
        pcm=b"started",
    )

    pump_has_lock = threading.Event()
    invoke_callback = threading.Event()

    def pump_callback() -> None:
        with conversation.adapter_lock:
            pump_has_lock.set()
            assert invoke_callback.wait(timeout=1)
            controller.on_response_audio(
                turn_id=turn_id,
                response_id="response-1",
                item_id="item-1",
                pcm=b"stale",
            )

    pump_thread = threading.Thread(target=pump_callback, daemon=True)
    button_thread = threading.Thread(target=controller.on_button_pressed, daemon=True)
    pump_thread.start()
    assert pump_has_lock.wait(timeout=1)
    button_thread.start()
    assert playback.stop_entered.wait(timeout=1)

    invoke_callback.set()
    pump_thread.join(timeout=1)
    assert not pump_thread.is_alive(), "pump callback waited on the controller state lock"
    playback.allow_stop.set()
    button_thread.join(timeout=1)
    assert not button_thread.is_alive(), "button thread waited forever on the pump lock"
    assert controller.state is DeviceState.RECORDING
    assert b"stale" not in playback.data
    controller.on_double_click()


class BlockingRecorder(FakeRecorder):
    def __init__(self, calls: list[tuple], playback: FakePlayback) -> None:
        super().__init__(calls, playback)
        self.start_entered = threading.Event()
        self.allow_start = threading.Event()

    def start(self) -> None:
        assert not self.playback.active
        assert not self.active
        self.calls.append(("recorder.start",))
        self.active = True
        self.start_entered.set()
        if not self.allow_start.wait(timeout=1):
            raise TimeoutError("test did not release recorder start")


def test_release_during_recorder_start_is_committed_once() -> None:
    calls: list[tuple] = []
    clock = FakeClock()
    led = FakeLed()
    playback = FakePlayback(calls)
    recorder = BlockingRecorder(calls, playback)
    conversation = FakeConversation(calls)
    controller = DeviceController(
        led=led,
        recorder=recorder,
        playback=playback,
        conversation=conversation,
        config=ControllerConfig(min_capture_seconds=0.2, max_capture_seconds=2.0),
        clock=clock,
    )

    press_thread = threading.Thread(target=controller.on_button_pressed, daemon=True)
    press_thread.start()
    assert recorder.start_entered.wait(timeout=1)
    clock.advance(0.5)
    controller.on_button_released()
    recorder.allow_start.set()
    press_thread.join(timeout=1)

    assert not press_thread.is_alive()
    assert controller.state is DeviceState.WAITING
    assert [call[0] for call in calls].count("recorder.stop") == 1
    assert [call[0] for call in calls].count("conversation.send") == 1


class BlockingWritePlayback(FakePlayback):
    def __init__(self, calls: list[tuple]) -> None:
        super().__init__(calls)
        self.write_entered = threading.Event()
        self.stop_requested = threading.Event()

    def write(self, stream_id: str, pcm: bytes) -> None:
        self.calls.append(("playback.write", stream_id, pcm))
        assert self.active
        assert stream_id == self.stream_id
        self.write_entered.set()
        if not self.stop_requested.wait(timeout=1):
            raise TimeoutError("test playback write was not preempted")
        raise BrokenPipeError("speaker pipe closed by stop")

    def stop(self) -> int:
        self.calls.append(("playback.stop",))
        self.active = False
        self.stream_id = None
        self.stop_requested.set()
        return self.played_ms


def test_barge_in_preempts_blocked_playback_write() -> None:
    calls: list[tuple] = []
    clock = FakeClock()
    led = FakeLed()
    playback = BlockingWritePlayback(calls)
    recorder = FakeRecorder(calls, playback)
    conversation = FakeConversation(calls)
    controller = DeviceController(
        led=led,
        recorder=recorder,
        playback=playback,
        conversation=conversation,
        config=ControllerConfig(min_capture_seconds=0.2, max_capture_seconds=2.0),
        clock=clock,
    )
    local_rig = Rig(controller, clock, led, recorder, playback, conversation, calls)
    turn_id = commit_turn(local_rig)
    controller.on_response_started(turn_id=turn_id, response_id="response-1")

    write_thread = threading.Thread(
        target=controller.on_response_audio,
        kwargs={
            "turn_id": turn_id,
            "response_id": "response-1",
            "item_id": "item-1",
            "pcm": b"blocked audio",
        },
        daemon=True,
    )
    write_thread.start()
    assert playback.write_entered.wait(timeout=1)

    button_thread = threading.Thread(target=controller.on_button_pressed, daemon=True)
    button_thread.start()
    button_thread.join(timeout=1)
    write_thread.join(timeout=1)

    assert not button_thread.is_alive(), "barge-in waited for the blocked speaker pipe"
    assert not write_thread.is_alive()
    assert controller.state is DeviceState.RECORDING
    assert recorder.active
    assert controller.last_error is None
    controller.on_double_click()


class DelayedTokenPlayback(FakePlayback):
    def __init__(self, calls: list[tuple]) -> None:
        super().__init__(calls)
        self._stream_lock = threading.Lock()
        self.old_write_entered = threading.Event()
        self.allow_old_validation = threading.Event()

    def start(self, stream_id: str) -> None:
        with self._stream_lock:
            super().start(stream_id)

    def write(self, stream_id: str, pcm: bytes) -> None:
        self.calls.append(("playback.write", stream_id, pcm))
        if stream_id == "response-1":
            self.old_write_entered.set()
            if not self.allow_old_validation.wait(timeout=1):
                raise TimeoutError("test did not release stale write")
        with self._stream_lock:
            if not self.active or stream_id != self.stream_id:
                raise RuntimeError("stale playback stream")
            self.data.extend(pcm)

    def stop(self) -> int:
        with self._stream_lock:
            return super().stop()


def test_delayed_old_audio_cannot_write_to_replacement_stream() -> None:
    calls: list[tuple] = []
    clock = FakeClock()
    led = FakeLed()
    playback = DelayedTokenPlayback(calls)
    recorder = FakeRecorder(calls, playback)
    conversation = FakeConversation(calls)
    controller = DeviceController(
        led=led,
        recorder=recorder,
        playback=playback,
        conversation=conversation,
        config=ControllerConfig(min_capture_seconds=0.2, max_capture_seconds=2.0),
        clock=clock,
    )
    local_rig = Rig(controller, clock, led, recorder, playback, conversation, calls)
    first_turn = commit_turn(local_rig)
    controller.on_response_started(turn_id=first_turn, response_id="response-1")

    stale_thread = threading.Thread(
        target=controller.on_response_audio,
        kwargs={
            "turn_id": first_turn,
            "response_id": "response-1",
            "item_id": "item-1",
            "pcm": b"stale audio",
        },
        daemon=True,
    )
    stale_thread.start()
    assert playback.old_write_entered.wait(timeout=1)

    controller.on_button_pressed()
    clock.advance(0.5)
    controller.on_button_released()
    second_turn = controller.active_turn_id
    assert second_turn is not None
    controller.on_response_started(turn_id=second_turn, response_id="response-2")

    playback.allow_old_validation.set()
    stale_thread.join(timeout=1)
    assert not stale_thread.is_alive()
    assert bytes(playback.data) == b""
    assert controller.state is DeviceState.SPEAKING
    assert controller.last_error is None

    controller.on_response_audio(
        turn_id=second_turn,
        response_id="response-2",
        item_id="item-2",
        pcm=b"current audio",
    )
    assert bytes(playback.data) == b"current audio"
    controller.on_double_click()


class FakePipe:
    def __init__(self, *, broken_on_close: bool = False) -> None:
        self.broken_on_close = broken_on_close
        self.close_calls = 0
        self.writes: list[bytes] = []

    def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)

    def flush(self) -> None:
        return

    def close(self) -> None:
        self.close_calls += 1
        if self.broken_on_close:
            raise BrokenPipeError("child already closed")


class FakeProcess:
    def __init__(
        self,
        *,
        broken_on_close: bool = False,
        timeout_once: bool = False,
        terminate_failures: int = 0,
        kill_failures: int = 0,
    ) -> None:
        self.stdin = FakePipe(broken_on_close=broken_on_close)
        self.returncode: int | None = None
        self.timeout_once = timeout_once
        self.terminate_failures = terminate_failures
        self.kill_failures = kill_failures
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.terminate_failures:
            self.terminate_failures -= 1
            raise OSError("terminate failed")

    def kill(self) -> None:
        self.kill_calls += 1
        if self.kill_failures:
            self.kill_failures -= 1
            raise OSError("kill failed")
        self.returncode = -9

    def wait(self, timeout: float) -> int:
        self.wait_calls += 1
        if self.timeout_once and self.wait_calls == 1:
            raise subprocess.TimeoutExpired("aplay", timeout)
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class BlockingPipe(FakePipe):
    def __init__(self) -> None:
        super().__init__()
        self.flush_entered = threading.Event()
        self.preempted = threading.Event()

    def flush(self) -> None:
        self.flush_entered.set()
        if not self.preempted.wait(timeout=1):
            raise TimeoutError("test pipe was not preempted")
        raise BrokenPipeError("playback was stopped")


class PreemptibleProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__()
        self.stdin = BlockingPipe()

    def terminate(self) -> None:
        super().terminate()
        self.stdin.preempted.set()


def test_playback_stop_reaps_before_suppressing_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(broken_on_close=True)
    monkeypatch.setattr(
        "recoverybox.device.audio_process.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    playback = SubprocessPlayback(clock=FakeClock())
    playback.start("response-1")

    assert playback.stop() == 0
    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    assert process.stdin.close_calls == 1


def test_playback_stop_kills_and_reaps_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess(timeout_once=True)
    monkeypatch.setattr(
        "recoverybox.device.audio_process.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    playback = SubprocessPlayback(
        config=AlsaCommandConfig(process_timeout_seconds=0.01),
        clock=FakeClock(),
    )
    playback.start("response-1")

    playback.stop()
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 2
    assert process.stdin.close_calls == 1


def test_subprocess_playback_stop_preempts_blocked_pipe_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = PreemptibleProcess()
    monkeypatch.setattr(
        "recoverybox.device.audio_process.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    playback = SubprocessPlayback(clock=FakeClock())
    playback.start("response-1")
    write_errors: list[Exception] = []

    def blocked_write() -> None:
        try:
            playback.write("response-1", b"audio")
        except Exception as exc:
            write_errors.append(exc)

    write_thread = threading.Thread(target=blocked_write, daemon=True)
    write_thread.start()
    assert process.stdin.flush_entered.wait(timeout=1)

    playback.stop()
    write_thread.join(timeout=1)

    assert not write_thread.is_alive()
    assert len(write_errors) == 1
    assert isinstance(write_errors[0], BrokenPipeError)
    assert process.terminate_calls == 1
    assert process.wait_calls == 1


def test_playback_retains_handle_when_terminate_and_kill_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(terminate_failures=1, kill_failures=1)
    monkeypatch.setattr(
        "recoverybox.device.audio_process.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    playback = SubprocessPlayback(clock=FakeClock())
    playback.start("response-1")

    with pytest.raises(OSError, match="kill failed"):
        playback.stop()
    playback.stop()

    assert process.terminate_calls == 2
    assert process.kill_calls == 1
    assert process.wait_calls == 1


def test_subprocess_playback_rejects_old_token_after_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_process = FakeProcess()
    new_process = FakeProcess()
    processes = iter([old_process, new_process])
    monkeypatch.setattr(
        "recoverybox.device.audio_process.subprocess.Popen",
        lambda *args, **kwargs: next(processes),
    )
    playback = SubprocessPlayback(clock=FakeClock())
    playback.start("response-1")
    playback.stop()
    playback.start("response-2")

    with pytest.raises(RuntimeError, match="playback is not active"):
        playback.write("response-1", b"stale audio")
    assert new_process.stdin.writes == []

    playback.write("response-2", b"current audio")
    assert new_process.stdin.writes == [b"current audio"]
    playback.stop()


def test_recorder_terminate_failure_falls_back_to_kill_and_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(terminate_failures=1)
    monkeypatch.setattr(
        "recoverybox.device.audio_process.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    recorder = SubprocessRecorder()
    recorder.start()

    with pytest.raises(OSError, match="terminate failed"):
        recorder.stop()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 1
    recorder.abort()
    assert process.terminate_calls == 1, "reaped process was still marked active"


def test_recorder_retains_handle_when_terminate_and_kill_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(terminate_failures=1, kill_failures=1)
    monkeypatch.setattr(
        "recoverybox.device.audio_process.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    recorder = SubprocessRecorder()
    recorder.start()

    with pytest.raises(OSError, match="kill failed"):
        recorder.stop()
    assert process.wait_calls == 0

    # The failed stop retained both process and spool, so fail-safe cleanup can
    # retry and reap the same child instead of orphaning it.
    recorder.abort()
    assert process.terminate_calls == 2
    assert process.kill_calls == 1
    assert process.wait_calls == 1
