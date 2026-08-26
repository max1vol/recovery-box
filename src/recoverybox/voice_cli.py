"""Developer-only ALSA and Realtime voice check-in command.

This module deliberately composes the existing device controller and Realtime
adapter without cameras or exercise-state inputs.  It is an integration aid for
manual check-in conversations; it is not an active-exercise runtime or physical
Raspberry Pi acceptance test.
"""

from __future__ import annotations

import os
import selectors
import socket
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TextIO

from recoverybox.config import ConfigurationError, Settings
from recoverybox.core import SessionMode
from recoverybox.device import (
    AlsaCommandConfig,
    ControllerConfig,
    DeviceController,
    DeviceState,
    LedMode,
    PlaybackPort,
    RecorderPort,
    SubprocessPlayback,
    SubprocessRecorder,
)
from recoverybox.realtime import (
    RealtimeConversationAdapter,
    RealtimeEventSink,
    RealtimeSession,
    RealtimeTransport,
    WebSocketJsonTransport,
)

SOCKET_TIMEOUT_SECONDS = 5.0
TERMINAL_POLL_SECONDS = 0.10
PUMP_JOIN_SECONDS = SOCKET_TIMEOUT_SECONDS + 1.0

CHECKIN_INSTRUCTIONS = """\
You are the voice assistant for a developer check-in demonstration.
Keep responses brief, calm, and conversational. This lane is only for a general
check-in before or after rehabilitation. Never coach an exercise in progress,
assess movement, change a prescription, make a safety decision, or control
hardware. If the user asks for active-exercise guidance, explain that the
device's deterministic local safety runtime must handle that workflow.
"""

STARTUP_MESSAGE = (
    "Developer voice check-in only. This exercises ALSA audio and OpenAI "
    "Realtime; it does not establish Pi hardware acceptance and must not be "
    "used during an active exercise."
)


class VoiceCheckinConfigurationError(ValueError):
    """Raised before opening audio or a socket when check-in setup is unsafe."""


@dataclass(frozen=True, slots=True)
class TerminalLine:
    """One polled terminal line; ``text=None`` represents end-of-input."""

    text: str | None


class LineReader(Protocol):
    """Non-blocking line source used by the terminal event loop."""

    def poll(self, timeout_seconds: float) -> TerminalLine | None: ...

    def close(self) -> None: ...


class SelectorLineReader:
    """Poll a terminal stream without blocking Realtime or capture ticks."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._selector = selectors.DefaultSelector()
        self._selector.register(stream, selectors.EVENT_READ)

    def poll(self, timeout_seconds: float) -> TerminalLine | None:
        if timeout_seconds < 0:
            raise ValueError("terminal poll timeout cannot be negative")
        if not self._selector.select(timeout_seconds):
            return None
        raw = self._stream.readline()
        if raw == "":
            return TerminalLine(text=None)
        return TerminalLine(text=raw.rstrip("\r\n"))

    def close(self) -> None:
        self._selector.close()


class TerminalStatus:
    """A minimal terminal representation of the device LED and safe status."""

    def __init__(self, output: TextIO) -> None:
        self._output = output
        self._lock = threading.Lock()

    def set_mode(self, mode: LedMode) -> None:
        self.write_line(f"[LED] {mode.value}")

    def write_line(self, message: str) -> None:
        with self._lock:
            self._output.write(f"{message}\n")
            self._output.flush()


class _ControllerSink(RealtimeEventSink):
    """Break the controller/adapter construction cycle without buffering events."""

    def __init__(self) -> None:
        self._controller: DeviceController | None = None

    def bind(self, controller: DeviceController) -> None:
        if self._controller is not None:
            raise RuntimeError("Realtime event sink is already bound")
        self._controller = controller

    def on_response_started(self, *, turn_id: str, response_id: str) -> None:
        self._require_controller().on_response_started(
            turn_id=turn_id,
            response_id=response_id,
        )

    def on_response_audio(
        self,
        *,
        turn_id: str,
        response_id: str,
        item_id: str,
        pcm: bytes,
    ) -> None:
        self._require_controller().on_response_audio(
            turn_id=turn_id,
            response_id=response_id,
            item_id=item_id,
            pcm=pcm,
        )

    def on_response_done(self, *, turn_id: str, response_id: str | None) -> None:
        self._require_controller().on_response_done(
            turn_id=turn_id,
            response_id=response_id,
        )

    def on_response_error(
        self,
        *,
        turn_id: str,
        response_id: str | None,
        error: Exception,
    ) -> None:
        self._require_controller().on_response_error(
            turn_id=turn_id,
            response_id=response_id,
            error=error,
        )

    def _require_controller(self) -> DeviceController:
        if self._controller is None:
            raise RuntimeError("Realtime event sink has not been bound")
        return self._controller


class _CheckinModeProvider:
    """Immutable product-mode view for this check-in-only composition."""

    @property
    def current_mode(self) -> SessionMode:
        return SessionMode.CHECK_IN


class DeveloperVoiceCheckinApplication:
    """Run terminal button events alongside bounded Realtime receives."""

    def __init__(
        self,
        *,
        controller: DeviceController,
        adapter: RealtimeConversationAdapter,
        status: TerminalStatus,
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
    ) -> None:
        self._controller = controller
        self._adapter = adapter
        self._status = status
        self._thread_factory = thread_factory
        self._stop = threading.Event()
        self._failure_lock = threading.Lock()
        self._pump_failure_kind: str | None = None

    @property
    def pump_failure_kind(self) -> str | None:
        with self._failure_lock:
            return self._pump_failure_kind

    def handle_line(self, line: str) -> bool:
        """Handle Enter as a press/release toggle and ``q`` as shutdown."""

        command = line.strip().lower()
        if command == "q":
            self.shutdown()
            return False
        if command:
            self._status.write_line("Use Enter to toggle recording, or q then Enter to stop.")
            return True

        if self._controller.state is DeviceState.RECORDING:
            self._controller.on_button_released()
        elif self._controller.state in {
            DeviceState.IDLE,
            DeviceState.WAITING,
            DeviceState.SPEAKING,
        }:
            self._controller.on_button_pressed()
        elif self._controller.state is DeviceState.ENDED:
            return False

        state = self._controller.state
        if state is DeviceState.RECORDING:
            self._status.write_line("Recording. Press Enter again to release and send.")
        elif state is DeviceState.WAITING:
            self._status.write_line("Turn sent. Waiting for the check-in response.")
        elif state is DeviceState.ERROR:
            self._status.write_line(
                "The check-in entered a safe error state. Restart the command to try again."
            )
            return False
        return state is not DeviceState.ENDED

    def pump_once(self) -> bool:
        """Pump one bounded receive, retrying only interruption/timeout errors."""

        try:
            self._adapter.pump_once()
        except Exception as exc:  # socket/protocol boundary
            if self._stop.is_set():
                return False
            if _is_recoverable_receive_timeout(exc):
                return True
            with self._failure_lock:
                # Retain only the exception class. Exception text can contain
                # provider payloads and is intentionally excluded from output.
                self._pump_failure_kind = type(exc).__name__
            return False
        return True

    def run(self, reader: LineReader) -> int:
        """Run until q, terminal EOF, a controller error, or connection loss."""

        self._status.write_line(STARTUP_MESSAGE)
        self._status.write_line(
            "Press Enter to start recording; press Enter again to release/send. "
            "Type q then Enter to stop."
        )
        pump_thread = self._thread_factory(
            target=self._pump_loop,
            name="recoverybox-realtime-pump",
            daemon=True,
        )
        pump_thread.start()
        exit_code = 0
        try:
            while not self._stop.is_set():
                failure_kind = self.pump_failure_kind
                if failure_kind is not None:
                    self._status.write_line(
                        f"Realtime connection ended ({failure_kind}). Automatic reconnect is "
                        "not available in this developer command; restart it to reconnect."
                    )
                    exit_code = 1
                    break
                if self._controller.state is DeviceState.ERROR:
                    self._status.write_line(
                        "The check-in entered a safe error state. Restart the command to try again."
                    )
                    exit_code = 1
                    break

                self._controller.on_tick()
                line = reader.poll(TERMINAL_POLL_SECONDS)
                if line is None:
                    continue
                if line.text is None or not self.handle_line(line.text):
                    break
        except KeyboardInterrupt:
            self._status.write_line("Stopping developer voice check-in.")
        finally:
            self.shutdown()
            reader.close()
            pump_thread.join(PUMP_JOIN_SECONDS)
            if pump_thread.is_alive():
                self._status.write_line("Realtime receive did not stop before the shutdown bound.")
                exit_code = 1
        return exit_code

    def shutdown(self) -> None:
        """Close audio and conversation state; safe to call more than once."""

        self._stop.set()
        self._controller.on_double_click()

    def _pump_loop(self) -> None:
        while not self._stop.is_set() and self.pump_once():
            pass


class _TransportFactory(Protocol):
    def __call__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> RealtimeTransport: ...


def _connect_transport(*, api_key: str, timeout_seconds: float) -> RealtimeTransport:
    return WebSocketJsonTransport.connect(
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )


def build_voice_checkin_application(
    *,
    environment: Mapping[str, str] | None = None,
    output: TextIO | None = None,
    transport_factory: _TransportFactory = _connect_transport,
    recorder_factory: Callable[[AlsaCommandConfig], RecorderPort] = SubprocessRecorder,
    playback_factory: Callable[[AlsaCommandConfig], PlaybackPort] = SubprocessPlayback,
    socket_timeout_seconds: float = SOCKET_TIMEOUT_SECONDS,
    thread_factory: Callable[..., threading.Thread] = threading.Thread,
) -> DeveloperVoiceCheckinApplication:
    """Compose production adapters while keeping every external edge injectable."""

    env = os.environ if environment is None else environment
    settings = Settings.from_environment(env)
    api_key = env.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise VoiceCheckinConfigurationError(
            "OPENAI_API_KEY must be set in the environment for voice-checkin"
        )
    if socket_timeout_seconds <= 0:
        raise VoiceCheckinConfigurationError("socket timeout must be positive")

    stream = sys.stdout if output is None else output
    status = TerminalStatus(stream)
    audio_config = AlsaCommandConfig(
        capture_device=settings.audio.capture_device,
        playback_device=settings.audio.playback_device,
    )
    transport: RealtimeTransport | None = None
    try:
        transport = transport_factory(
            api_key=api_key,
            timeout_seconds=socket_timeout_seconds,
        )
        session = RealtimeSession(transport=transport)
        session.configure(
            instructions=CHECKIN_INSTRUCTIONS,
            voice=settings.realtime.voice,
        )
        sink = _ControllerSink()
        adapter = RealtimeConversationAdapter(
            session=session,
            sink=sink,
            mode_provider=_CheckinModeProvider(),
        )
        controller = DeviceController(
            led=status,
            recorder=recorder_factory(audio_config),
            playback=playback_factory(audio_config),
            conversation=adapter,
            config=ControllerConfig(
                min_capture_seconds=settings.hardware.minimum_hold_ms / 1_000,
            ),
        )
        sink.bind(controller)
        return DeveloperVoiceCheckinApplication(
            controller=controller,
            adapter=adapter,
            status=status,
            thread_factory=thread_factory,
        )
    except Exception:
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
        raise
    finally:
        # Do not retain a second local reference after connection setup.
        api_key = ""


def run_voice_checkin(
    *,
    environment: Mapping[str, str] | None = None,
    input_stream: TextIO | None = None,
    output: TextIO | None = None,
    line_reader: LineReader | None = None,
    transport_factory: _TransportFactory = _connect_transport,
    recorder_factory: Callable[[AlsaCommandConfig], RecorderPort] = SubprocessRecorder,
    playback_factory: Callable[[AlsaCommandConfig], PlaybackPort] = SubprocessPlayback,
) -> int:
    """Run the real developer check-in without accepting an API key argument."""

    stream = sys.stdout if output is None else output
    status = TerminalStatus(stream)
    try:
        application = build_voice_checkin_application(
            environment=environment,
            output=stream,
            transport_factory=transport_factory,
            recorder_factory=recorder_factory,
            playback_factory=playback_factory,
        )
    except (ConfigurationError, VoiceCheckinConfigurationError) as exc:
        status.write_line(f"voice-checkin configuration error: {exc}")
        return 2
    except Exception as exc:
        # Only the exception class is safe to expose. In particular, never
        # print a WebSocket exception that could echo request headers.
        status.write_line(
            f"voice-checkin could not start ({type(exc).__name__}). "
            "Check network and ALSA prerequisites, then retry."
        )
        return 1

    reader = line_reader
    if reader is None:
        try:
            reader = SelectorLineReader(sys.stdin if input_stream is None else input_stream)
        except (OSError, ValueError) as exc:
            application.shutdown()
            status.write_line(
                f"voice-checkin terminal input is unavailable ({type(exc).__name__})."
            )
            return 2
    return application.run(reader)


def _is_recoverable_receive_timeout(error: Exception) -> bool:
    """Recognize bounded receive wakeups without importing websocket-client."""

    return isinstance(error, (TimeoutError, socket.timeout, InterruptedError)) or type(
        error
    ).__name__ in {"WebSocketTimeoutException", "WebSocketWouldBlockException"}


__all__ = [
    "CHECKIN_INSTRUCTIONS",
    "SOCKET_TIMEOUT_SECONDS",
    "STARTUP_MESSAGE",
    "DeveloperVoiceCheckinApplication",
    "LineReader",
    "SelectorLineReader",
    "TerminalLine",
    "TerminalStatus",
    "VoiceCheckinConfigurationError",
    "build_voice_checkin_application",
    "run_voice_checkin",
]
