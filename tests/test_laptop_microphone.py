from __future__ import annotations

import builtins
from collections.abc import Callable

import pytest

from recoverybox.device import PCM_S16LE_24K_MONO, AudioFormat, RecorderPort
from recoverybox.laptop import (
    LaptopMicrophoneConfig,
    LaptopMicrophoneError,
    LaptopMicrophoneRecorder,
)


class FakeRawInputStream:
    def __init__(
        self,
        callback: Callable[[object, int, object, object], None],
        *,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.callback = callback
        self.start_error = start_error
        self.stop_error = stop_error
        self.close_error = close_error
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.stopped += 1
        if self.stop_error is not None:
            raise self.stop_error

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error

    def emit(
        self,
        pcm: object,
        *,
        frames: int,
        status: object = False,
    ) -> None:
        self.callback(pcm, frames, object(), status)


class FakeStreamFactory:
    def __init__(
        self,
        *,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.start_error = start_error
        self.stop_error = stop_error
        self.close_error = close_error
        self.calls: list[dict[str, object]] = []
        self.streams: list[FakeRawInputStream] = []

    def __call__(self, **kwargs: object) -> FakeRawInputStream:
        self.calls.append(dict(kwargs))
        callback = kwargs["callback"]
        assert callable(callback)
        stream = FakeRawInputStream(
            callback,
            start_error=self.start_error,
            stop_error=self.stop_error,
            close_error=self.close_error,
        )
        self.streams.append(stream)
        return stream


def test_start_lazily_constructs_exact_raw_pcm_stream_and_stop_returns_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = FakeStreamFactory()
    recorder = LaptopMicrophoneRecorder(
        LaptopMicrophoneConfig(blocksize_frames=480, device="test input"),
        stream_factory=factory,
    )
    assert isinstance(recorder, RecorderPort)
    assert factory.calls == []

    original_import = builtins.__import__

    def reject_sounddevice(name: str, *args: object, **kwargs: object):
        if name == "sounddevice":
            raise AssertionError("injected tests must never import the native audio runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_sounddevice)
    recorder.start()

    assert recorder.active
    assert len(factory.calls) == 1
    stream_options = factory.calls[0]
    assert stream_options == {
        "samplerate": 24_000,
        "channels": 1,
        "dtype": "int16",
        "blocksize": 480,
        "device": "test input",
        "callback": stream_options["callback"],
    }

    stream = factory.streams[0]
    stream.emit(b"\x01\x00\x02\x00", frames=2)
    stream.emit(b"\x03\x00", frames=1)
    assert recorder.stop() == b"\x01\x00\x02\x00\x03\x00"
    assert not recorder.active
    assert stream.started == 1
    assert stream.stopped == 1
    assert stream.closed == 1
    assert recorder._pcm == bytearray()

    # A late PortAudio callback cannot repopulate an ended turn.
    stream.emit(b"\x04\x00", frames=1)
    assert recorder._pcm == bytearray()


def test_abort_discards_audio_is_idempotent_and_next_turn_starts_empty() -> None:
    factory = FakeStreamFactory()
    recorder = LaptopMicrophoneRecorder(stream_factory=factory)

    recorder.start()
    first_stream = factory.streams[0]
    first_stream.emit(b"\x01\x00" * 4, frames=4)
    recorder.abort()
    recorder.abort()

    assert not recorder.active
    assert first_stream.stopped == 1
    assert first_stream.closed == 1
    assert recorder._pcm == bytearray()

    recorder.start()
    second_stream = factory.streams[1]
    second_stream.emit(b"\x02\x00", frames=1)
    assert recorder.stop() == b"\x02\x00"


@pytest.mark.parametrize(
    ("pcm", "frames", "status", "message"),
    [
        (b"\x00\x00", 1, object(), "status error"),
        (b"\x00\x00", 2, False, "malformed PCM"),
        (b"", -1, False, "invalid frame count"),
        (b"", True, False, "invalid frame count"),
    ],
)
def test_callback_status_or_malformed_block_discards_whole_turn(
    pcm: object,
    frames: int,
    status: object,
    message: str,
) -> None:
    factory = FakeStreamFactory()
    recorder = LaptopMicrophoneRecorder(stream_factory=factory)
    recorder.start()
    stream = factory.streams[0]
    stream.emit(b"\x01\x00", frames=1)

    # The callback itself must not raise into the native audio thread.
    stream.emit(pcm, frames=frames, status=status)
    with pytest.raises(LaptopMicrophoneError, match=message):
        recorder.stop()

    assert stream.stopped == 1
    assert stream.closed == 1
    assert recorder._pcm == bytearray()


def test_callback_conversion_failure_is_contained_and_fails_closed() -> None:
    class BadBuffer:
        def __bytes__(self) -> bytes:
            raise RuntimeError("must not escape the callback")

    factory = FakeStreamFactory()
    recorder = LaptopMicrophoneRecorder(stream_factory=factory)
    recorder.start()

    factory.streams[0].emit(BadBuffer(), frames=1)
    with pytest.raises(LaptopMicrophoneError, match="callback failed"):
        recorder.stop()
    assert recorder._pcm == bytearray()


def test_capture_limit_discards_existing_and_excess_audio() -> None:
    config = LaptopMicrophoneConfig(max_capture_seconds=0.001)
    assert config.max_capture_bytes == 48
    factory = FakeStreamFactory()
    recorder = LaptopMicrophoneRecorder(config, stream_factory=factory)
    recorder.start()
    stream = factory.streams[0]

    stream.emit(b"\x01\x00" * 24, frames=24)
    stream.emit(b"\x02\x00", frames=1)

    with pytest.raises(LaptopMicrophoneError, match="exceeded its configured limit"):
        recorder.stop()
    assert recorder._pcm == bytearray()


def test_start_failure_cleans_up_and_does_not_retain_audio() -> None:
    factory = FakeStreamFactory(start_error=RuntimeError("native start failed"))
    recorder = LaptopMicrophoneRecorder(stream_factory=factory)

    with pytest.raises(LaptopMicrophoneError, match="could not start"):
        recorder.start()

    stream = factory.streams[0]
    assert not recorder.active
    assert stream.stopped == 1
    assert stream.closed == 1
    assert recorder._pcm == bytearray()


@pytest.mark.parametrize("failure_point", ["stop", "close"])
def test_shutdown_failure_closes_stream_discards_pcm_and_resets_state(
    failure_point: str,
) -> None:
    factory = FakeStreamFactory(
        stop_error=RuntimeError("stop failed") if failure_point == "stop" else None,
        close_error=RuntimeError("close failed") if failure_point == "close" else None,
    )
    recorder = LaptopMicrophoneRecorder(stream_factory=factory)
    recorder.start()
    stream = factory.streams[0]
    stream.emit(b"\x01\x00", frames=1)

    with pytest.raises(LaptopMicrophoneError, match="could not stop"):
        recorder.stop()

    assert not recorder.active
    assert stream.stopped == 1
    assert stream.closed == 1
    assert recorder._pcm == bytearray()
    recorder.abort()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"max_capture_seconds": 0}, ValueError, "positive finite"),
        ({"max_capture_seconds": float("inf")}, ValueError, "positive finite"),
        ({"max_capture_seconds": True}, ValueError, "positive finite"),
        ({"max_capture_seconds": 0.000001}, ValueError, "one complete frame"),
        ({"blocksize_frames": -1}, ValueError, "non-negative integer"),
        ({"blocksize_frames": True}, ValueError, "non-negative integer"),
        ({"device": "  "}, ValueError, "must not be blank"),
        ({"device": False}, TypeError, "integer, string, or None"),
        (
            {"audio_format": AudioFormat(sample_rate_hz=16_000)},
            ValueError,
            "24 kHz mono signed S16LE",
        ),
    ],
)
def test_config_rejects_unbounded_or_nonconforming_capture(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        LaptopMicrophoneConfig(**kwargs)  # type: ignore[arg-type]


def test_stop_requires_an_active_turn() -> None:
    recorder = LaptopMicrophoneRecorder(stream_factory=FakeStreamFactory())
    with pytest.raises(RuntimeError, match="not active"):
        recorder.stop()
    recorder.abort()


def test_fixed_audio_format_is_exposed_for_realtime_turns() -> None:
    recorder = LaptopMicrophoneRecorder(stream_factory=FakeStreamFactory())
    assert recorder.audio_format == PCM_S16LE_24K_MONO
