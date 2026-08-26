from __future__ import annotations

import io
import subprocess
import tempfile
import threading
import time
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest

from recoverybox.device import PCM_S16LE_24K_MONO, AudioFormat
from recoverybox.laptop import (
    LaptopAudioError,
    MacOSAudioConfig,
    MacOSAudioPlayer,
    PlaybackCancelledError,
)

PCM_A = b"\x01\x00\x02\x00" * 200
PCM_B = b"\x03\x00\x04\x00" * 100
PCM_C = b"\x05\x00\x06\x00" * 50


class FakeProcess:
    def __init__(self, *, finish_immediately: bool = False, returncode: int = 0) -> None:
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._done = threading.Event()
        self.terminated = False
        self.killed = False
        if finish_immediately:
            self.finish()

    def finish(self) -> None:
        self.returncode = self._final_returncode
        self._done.set()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if not self._done.wait(timeout):
            raise subprocess.TimeoutExpired("afplay", timeout)
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self._done.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._done.set()


class RecordingProcessFactory:
    def __init__(self, *, auto_finish: bool = False, returncode: int = 0) -> None:
        self.auto_finish = auto_finish
        self.returncode = returncode
        self.commands: list[tuple[str, ...]] = []
        self.wav_files: list[bytes] = []
        self.paths: list[Path] = []
        self.processes: list[FakeProcess] = []
        self.started = threading.Event()

    def __call__(self, command: Sequence[str]) -> FakeProcess:
        path = Path(command[1])
        self.commands.append(tuple(command))
        self.paths.append(path)
        self.wav_files.append(path.read_bytes())
        process = FakeProcess(
            finish_immediately=self.auto_finish,
            returncode=self.returncode,
        )
        self.processes.append(process)
        self.started.set()
        return process


def temp_factory(tmp_path: Path):
    def create():
        return tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=".wav",
            dir=tmp_path,
            delete=False,
        )

    return create


def wait_for_count(items: list[object], count: int, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while len(items) < count and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(items) == count


def test_wraps_exact_pcm_as_private_temporary_wav_then_deletes_it(tmp_path: Path) -> None:
    factory = RecordingProcessFactory(auto_finish=True)
    with MacOSAudioPlayer(
        process_factory=factory,
        temp_file_factory=temp_factory(tmp_path),
    ) as player:
        ticket = player.play(PCM_A)
        ticket.result(timeout=1)

    assert factory.commands[0][0] == "/usr/bin/afplay"
    with wave.open(io.BytesIO(factory.wav_files[0]), "rb") as wav_file:
        assert wav_file.getframerate() == 24_000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(wav_file.getnframes()) == PCM_A
    assert not factory.paths[0].exists()


@pytest.mark.parametrize(
    ("pcm", "audio_format", "error", "message"),
    [
        (b"", PCM_S16LE_24K_MONO, ValueError, "at least one"),
        (b"\x00", PCM_S16LE_24K_MONO, ValueError, "complete 2-byte"),
        (bytearray(b"\x00\x00"), PCM_S16LE_24K_MONO, TypeError, "must be bytes"),
        (
            b"\x00\x00",
            AudioFormat(sample_rate_hz=16_000),
            ValueError,
            "24 kHz mono signed S16LE",
        ),
        (
            b"\x00\x00",
            AudioFormat(channels=2),
            ValueError,
            "24 kHz mono signed S16LE",
        ),
        (
            b"\x00\x00",
            AudioFormat(little_endian=False),
            ValueError,
            "24 kHz mono signed S16LE",
        ),
    ],
)
def test_rejects_nonconforming_pcm_before_process_start(
    tmp_path: Path,
    pcm: object,
    audio_format: AudioFormat,
    error: type[Exception],
    message: str,
) -> None:
    factory = RecordingProcessFactory(auto_finish=True)
    with MacOSAudioPlayer(
        process_factory=factory,
        temp_file_factory=temp_factory(tmp_path),
    ) as player:
        with pytest.raises(error, match=message):
            player.play(pcm, audio_format=audio_format)  # type: ignore[arg-type]
    assert factory.commands == []


def test_serializes_clips_on_one_background_worker(tmp_path: Path) -> None:
    factory = RecordingProcessFactory()
    with MacOSAudioPlayer(
        process_factory=factory,
        temp_file_factory=temp_factory(tmp_path),
    ) as player:
        first = player.play(PCM_A)
        second = player.play(PCM_B)
        assert factory.started.wait(1)
        assert len(factory.processes) == 1
        assert not first.done
        assert not second.done

        factory.processes[0].finish()
        wait_for_count(factory.processes, 2)
        first.result(timeout=1)
        factory.processes[1].finish()
        second.result(timeout=1)

    assert [
        wav_data[-len(pcm) :]
        for wav_data, pcm in zip(factory.wav_files, (PCM_A, PCM_B), strict=True)
    ] == [
        PCM_A,
        PCM_B,
    ]
    assert all(not path.exists() for path in factory.paths)


def test_preempt_cancels_active_and_queued_before_replacement(tmp_path: Path) -> None:
    factory = RecordingProcessFactory()
    with MacOSAudioPlayer(
        process_factory=factory,
        temp_file_factory=temp_factory(tmp_path),
    ) as player:
        active = player.play(PCM_A)
        assert factory.started.wait(1)
        queued = player.play(PCM_B)

        replacement = player.preempt(PCM_C)
        with pytest.raises(PlaybackCancelledError):
            queued.result(timeout=1)
        with pytest.raises(PlaybackCancelledError):
            active.result(timeout=1)
        wait_for_count(factory.processes, 2)
        factory.processes[1].finish()
        replacement.result(timeout=1)

    assert factory.processes[0].terminated
    assert len(factory.processes) == 2
    assert factory.wav_files[1].endswith(PCM_C)
    assert all(not path.exists() for path in factory.paths)


def test_stop_synchronously_cancels_active_and_pending_work(tmp_path: Path) -> None:
    factory = RecordingProcessFactory()
    with MacOSAudioPlayer(
        process_factory=factory,
        temp_file_factory=temp_factory(tmp_path),
    ) as player:
        active = player.play(PCM_A)
        assert factory.started.wait(1)
        queued = player.play(PCM_B)

        player.stop(timeout=1)

        assert active.cancelled
        assert queued.cancelled
        assert not player.active
        assert player.queued_clip_count == 0
        assert factory.processes[0].terminated
        assert len(factory.processes) == 1
    assert all(not path.exists() for path in factory.paths)


def test_process_failure_is_reported_on_ticket_and_wav_is_deleted(tmp_path: Path) -> None:
    factory = RecordingProcessFactory(auto_finish=True, returncode=7)
    with MacOSAudioPlayer(
        process_factory=factory,
        temp_file_factory=temp_factory(tmp_path),
    ) as player:
        ticket = player.play(PCM_A)
        with pytest.raises(LaptopAudioError, match="audio playback failed") as caught:
            ticket.result(timeout=1)

    assert isinstance(caught.value.__cause__, LaptopAudioError)
    assert "status 7" in str(caught.value.__cause__)
    assert ticket.error is caught.value.__cause__
    assert all(not path.exists() for path in factory.paths)


def test_factory_failure_still_deletes_temporary_wav(tmp_path: Path) -> None:
    paths: list[Path] = []

    def failing_factory(command: Sequence[str]):
        paths.append(Path(command[1]))
        raise OSError("speaker unavailable")

    with MacOSAudioPlayer(
        process_factory=failing_factory,
        temp_file_factory=temp_factory(tmp_path),
    ) as player:
        ticket = player.play(PCM_A)
        with pytest.raises(LaptopAudioError):
            ticket.result(timeout=1)

    assert paths
    assert all(not path.exists() for path in paths)


def test_close_is_idempotent_and_rejects_new_work(tmp_path: Path) -> None:
    factory = RecordingProcessFactory()
    player = MacOSAudioPlayer(
        MacOSAudioConfig(close_timeout_seconds=1),
        process_factory=factory,
        temp_file_factory=temp_factory(tmp_path),
    )
    ticket = player.play(PCM_A)
    assert factory.started.wait(1)

    player.close()
    player.close()

    assert ticket.cancelled
    assert factory.processes[0].terminated
    with pytest.raises(RuntimeError, match="closed"):
        player.play(PCM_B)
    with pytest.raises(RuntimeError, match="closed"):
        player.preempt(PCM_B)
    assert all(not path.exists() for path in factory.paths)
