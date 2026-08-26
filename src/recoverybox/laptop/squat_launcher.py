"""Repository-owned laptop launcher for the single-camera squat demo.

The module is safe to import in the ordinary test sandbox.  Native MediaPipe,
OpenCV, camera, microphone, and speaker resources are acquired only when
``LaptopSquatDemo.run`` is invoked with the production factories.  Tests
inject fakes for every one of those boundaries.

One :class:`~recoverybox.laptop.squat_session.LaptopSquatSession` owns one
Realtime transport for the whole demo.  Its sole receiver runs beside the
camera loop so Guardian-authorized cues never block pose processing.  A cloud
failure pauses coaching but does not stop the deterministic squat tracker.
When explicitly configured, the same loop may enqueue only the derived numeric
``SquatAnalysis`` records to a RecoveryBox peer; camera frames stay local.
"""

from __future__ import annotations

import json
import os
import select
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from time import monotonic_ns
from typing import Protocol, TextIO

from recoverybox.config import ConfigurationError, Settings
from recoverybox.core import CueId, SessionMode
from recoverybox.exercise import SquatAnalysis, SquatPhase, SquatTracker
from recoverybox.realtime import (
    BoundedOrderedTransport,
    RealtimeTransport,
    ReleasedCueAudio,
    RuntimeAbortReason,
    SessionEndSignal,
    SessionEndSource,
    WebSocketJsonTransport,
)
from recoverybox.remote_pose import (
    MAX_REMOTE_POSE_EVIDENCE_AGE_MS,
    RemotePosePublisher,
    RemotePoseRequest,
    load_remote_pose_token,
)
from recoverybox.vision import WebcamPoseConfig, WebcamPoseSample, WebcamPoseSource

from .audio import MacOSAudioPlayer, PlaybackCancelledError
from .doctor import MEDIAPIPE_EXPECTED_VERSION, OPENCV_EXPECTED_VERSION
from .microphone import LaptopMicrophoneRecorder
from .pose_model import PoseModelError, validate_pose_model
from .squat_session import LaptopSquatSession, build_single_camera_squat_plan

SQUAT_DEMO_INSTRUCTIONS = """
You are the voice interface for one locally supervised squat session.
The deterministic local Guardian, never the model, decides exercise cues.
During the exercise, do not produce ordinary spoken conversation. The local
application may request an isolated exact cue phrase, or a no-audio response
to a push-to-talk turn. Call finish_session with an empty object only after the
user explicitly says goodbye or clearly asks to end the session. A completed
set, silence, a network error, or any ordinary response must not end it.
""".strip()

_REQUIRED_VISION_DISTRIBUTIONS = {
    "mediapipe": MEDIAPIPE_EXPECTED_VERSION,
    "opencv-contrib-python": OPENCV_EXPECTED_VERSION,
}
_CONFLICTING_OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)


class LaptopRuntimePinError(RuntimeError):
    """The installed native laptop runtime is not the reviewed exact pair."""


def validate_laptop_runtime_pins(
    *,
    version_provider: Callable[[str], str] = metadata.version,
) -> dict[str, str]:
    """Check distribution metadata only, without importing native modules."""

    versions: dict[str, str] = {}
    for distribution, expected in _REQUIRED_VISION_DISTRIBUTIONS.items():
        try:
            installed = version_provider(distribution)
        except metadata.PackageNotFoundError as exc:
            raise LaptopRuntimePinError("required laptop runtime distribution is missing") from exc
        except Exception as exc:
            raise LaptopRuntimePinError("could not read laptop runtime metadata") from exc
        if installed != expected:
            raise LaptopRuntimePinError("installed laptop runtime does not match exact pins")
        versions[distribution] = installed

    for distribution in _CONFLICTING_OPENCV_DISTRIBUTIONS:
        try:
            version_provider(distribution)
        except metadata.PackageNotFoundError:
            continue
        except Exception as exc:
            raise LaptopRuntimePinError("could not audit OpenCV distribution metadata") from exc
        raise LaptopRuntimePinError("a second distribution exporting cv2 is installed")
    return versions


class SquatDemoCommand(StrEnum):
    """Small terminal-control vocabulary for the laptop stand-in button."""

    TOGGLE_MICROPHONE = "toggle_microphone"
    RESUME = "resume"
    STOP = "stop"
    HELP = "help"


class SquatDemoEndReason(StrEnum):
    """Observable reason the launcher left its camera loop."""

    PHYSICAL_STOP = "physical_stop"
    VALIDATED_TOOL_CALL = "validated_tool_call"
    MAX_FRAMES = "max_frames"
    RUNTIME_ABORT = "runtime_abort"


class SquatDemoCommandSource(Protocol):
    """Non-blocking source for terminal/button commands."""

    def poll(self) -> tuple[SquatDemoCommand, ...]: ...

    def close(self) -> None: ...


class _PoseSource(Protocol):
    def open(self) -> object: ...

    def read(self, *, preview_lines: Sequence[str] = ()) -> WebcamPoseSample: ...

    def close(self) -> None: ...


class _SquatTracker(Protocol):
    @property
    def rep_count(self) -> int: ...

    def update(self, frame: object) -> SquatAnalysis: ...

    def update_missing(self, timestamp_ms: int) -> SquatAnalysis: ...


class _AudioPlayer(Protocol):
    def play(self, pcm: bytes) -> _PlaybackTicket: ...

    def stop(self, timeout: float | None = None) -> None: ...

    def close(self) -> None: ...


class _PlaybackTicket(Protocol):
    def result(self, timeout: float | None = None) -> None: ...


class _Microphone(Protocol):
    @property
    def active(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> bytes: ...

    def abort(self) -> None: ...


class _RemotePosePublisher(Protocol):
    """Non-blocking numeric-pose edge owned by the laptop launcher."""

    @property
    def connected(self) -> bool: ...

    @property
    def failure_kind(self) -> str | None: ...

    @property
    def messages_sent(self) -> int: ...

    def start(self) -> None: ...

    def wait_for_request(
        self, timeout_seconds: float | None = None
    ) -> RemotePoseRequest | None: ...

    def submit(
        self,
        analysis: SquatAnalysis,
        *,
        request: RemotePoseRequest,
        evidence_age_ms: int,
    ) -> None: ...

    def request_resume(self) -> None: ...

    def close(self) -> None: ...


class TerminalCommandSource:
    """Poll newline-delimited terminal controls without blocking the camera."""

    def __init__(self, input_stream: TextIO = sys.stdin) -> None:
        self._input = input_stream
        self._closed = False

    def poll(self) -> tuple[SquatDemoCommand, ...]:
        if self._closed:
            return ()
        commands: list[SquatDemoCommand] = []
        while True:
            try:
                readable, _, _ = select.select((self._input,), (), (), 0)
            except (OSError, TypeError, ValueError):
                # A non-selectable or detached stdin simply disables terminal
                # controls; q/Escape in the OpenCV preview remains available.
                self._closed = True
                break
            if not readable:
                break
            line = self._input.readline()
            if line == "":
                self._closed = True
                break
            command = _parse_terminal_command(line)
            if command is not None:
                commands.append(command)
        return tuple(commands)

    def close(self) -> None:
        # stdin is caller-owned and must never be closed here.
        self._closed = True


def _parse_terminal_command(line: str) -> SquatDemoCommand | None:
    normalized = line.strip().lower()
    if normalized == "":
        return SquatDemoCommand.TOGGLE_MICROPHONE
    if normalized in {"q", "quit", "exit"}:
        return SquatDemoCommand.STOP
    if normalized in {"r", "resume"}:
        return SquatDemoCommand.RESUME
    if normalized in {"h", "help", "?"}:
        return SquatDemoCommand.HELP
    return None


@dataclass(frozen=True, slots=True)
class SquatDemoConfig:
    """Explicit launcher settings for one laptop run."""

    model_asset_path: Path
    camera_index: int = 0
    preview: bool = True
    voice: str = "marin"
    voice_enabled: bool = True
    microphone_enabled: bool = True
    max_frames: int | None = None
    target_reps: int = 3
    pose_peer: str | None = None
    pose_token_file: str | Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.camera_index, bool) or not isinstance(self.camera_index, int):
            raise TypeError("camera_index must be an integer")
        if self.camera_index < 0:
            raise ValueError("camera_index must be non-negative")
        if type(self.preview) is not bool:
            raise TypeError("preview must be a boolean")
        if type(self.voice_enabled) is not bool or type(self.microphone_enabled) is not bool:
            raise TypeError("voice and microphone options must be booleans")
        if self.microphone_enabled and not self.voice_enabled:
            raise ValueError("microphone requires Realtime voice")
        if not isinstance(self.voice, str) or not self.voice.strip():
            raise ValueError("voice must not be blank")
        if self.max_frames is not None and (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, int)
            or self.max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer when provided")
        if isinstance(self.target_reps, bool) or not isinstance(self.target_reps, int):
            raise TypeError("target_reps must be an integer")
        if self.target_reps != 3:
            raise ValueError("the reviewed laptop squat script requires exactly 3 reps")
        pose_peer = self.pose_peer
        if pose_peer is not None:
            if not isinstance(pose_peer, str):
                raise TypeError("pose_peer must be a HOST:PORT string")
            if not pose_peer.strip():
                raise ValueError("pose_peer must be a non-blank HOST:PORT value")
            pose_peer = pose_peer.strip()
        pose_token_file = self.pose_token_file
        if pose_token_file is not None:
            if isinstance(pose_token_file, str):
                pose_token_file = pose_token_file.strip()
                if not pose_token_file:
                    raise ValueError("pose_token_file must not be blank")
            try:
                pose_token_file = Path(pose_token_file).expanduser()
            except TypeError as exc:
                raise TypeError("pose_token_file must be a filesystem path") from exc
            if not str(pose_token_file).strip():
                raise ValueError("pose_token_file must not be blank")
        if (pose_peer is None) != (pose_token_file is None):
            raise ValueError("pose_peer and pose_token_file must be configured together")
        object.__setattr__(self, "model_asset_path", Path(self.model_asset_path).expanduser())
        object.__setattr__(self, "voice", self.voice.strip())
        object.__setattr__(self, "pose_peer", pose_peer)
        object.__setattr__(self, "pose_token_file", pose_token_file)


@dataclass(frozen=True, slots=True)
class SquatDemoResult:
    """Content-free evidence from a completed launcher run."""

    frames_processed: int
    pose_frames: int
    assessable_frames: int
    rep_count: int
    end_reason: SquatDemoEndReason
    final_mode: SessionMode
    voice_enabled: bool
    voice_connected: bool
    runtime_versions: Mapping[str, str]
    realtime_failure_kind: str | None
    microphone_failure_kind: str | None
    cue_failure_reason: str | None
    remote_pose_enabled: bool
    remote_pose_connected: bool
    remote_pose_failure_kind: str | None
    remote_pose_messages_sent: int

    def as_dict(self) -> dict[str, object]:
        return {
            "frames_processed": self.frames_processed,
            "pose_frames": self.pose_frames,
            "assessable_frames": self.assessable_frames,
            "rep_count": self.rep_count,
            "end_reason": self.end_reason.value,
            "final_mode": self.final_mode.value,
            "voice_enabled": self.voice_enabled,
            "voice_connected": self.voice_connected,
            "runtime_versions": dict(self.runtime_versions),
            "realtime_failure_kind": self.realtime_failure_kind,
            "microphone_failure_kind": self.microphone_failure_kind,
            "cue_failure_reason": self.cue_failure_reason,
            "remote_pose_enabled": self.remote_pose_enabled,
            "remote_pose_connected": self.remote_pose_connected,
            "remote_pose_failure_kind": self.remote_pose_failure_kind,
            "remote_pose_messages_sent": self.remote_pose_messages_sent,
        }


ValidateRuntime = Callable[[], Mapping[str, str]]
ValidatePoseModel = Callable[[str | Path], Path]
TransportFactory = Callable[..., RealtimeTransport]
PoseSourceFactory = Callable[[WebcamPoseConfig], _PoseSource]
TrackerFactory = Callable[[], _SquatTracker]
AudioPlayerFactory = Callable[[], _AudioPlayer]
MicrophoneFactory = Callable[[], _Microphone]
CommandSourceFactory = Callable[[], SquatDemoCommandSource]
RemotePosePublisherFactory = Callable[[str, bytes], _RemotePosePublisher]
RemotePoseTokenLoader = Callable[[str | Path], bytes]
MonotonicNanoseconds = Callable[[], int]


def _default_transport_factory(*, api_key: str) -> RealtimeTransport:
    # No finite receive timeout is installed: the one receiver is intentionally
    # long-lived and is unblocked by transport.close during explicit shutdown.
    connection = WebSocketJsonTransport.connect(api_key=api_key)
    return BoundedOrderedTransport(connection)


@dataclass(frozen=True, slots=True)
class SquatDemoDependencies:
    """Replaceable native/network edges; production defaults are lazy."""

    validate_runtime: ValidateRuntime = validate_laptop_runtime_pins
    validate_model: ValidatePoseModel = validate_pose_model
    transport_factory: TransportFactory = _default_transport_factory
    pose_source_factory: PoseSourceFactory = WebcamPoseSource
    tracker_factory: TrackerFactory = SquatTracker
    audio_player_factory: AudioPlayerFactory = MacOSAudioPlayer
    microphone_factory: MicrophoneFactory = LaptopMicrophoneRecorder
    command_source_factory: CommandSourceFactory = TerminalCommandSource
    remote_pose_publisher_factory: RemotePosePublisherFactory = RemotePosePublisher
    remote_pose_token_loader: RemotePoseTokenLoader = load_remote_pose_token
    monotonic_ns: MonotonicNanoseconds = monotonic_ns


class _LocalOnlyTransport:
    """No-network transport used only by explicit camera-only operation."""

    def __init__(self) -> None:
        self.closed = False
        self.sent_event_types: list[str] = []

    def send_event(self, event: Mapping[str, object]) -> None:
        if self.closed:
            raise RuntimeError("local-only transport is closed")
        event_type = event.get("type")
        if isinstance(event_type, str):
            # Retain event types only, never prompts, audio, or transcripts.
            self.sent_event_types.append(event_type)

    def receive_event(self) -> Mapping[str, object]:
        raise EOFError("local-only mode has no Realtime receiver")

    def close(self) -> None:
        self.closed = True


class _RemotePoseBridge:
    """Contain every optional publisher failure outside the local safety loop."""

    def __init__(
        self,
        *,
        enabled: bool,
        publisher: _RemotePosePublisher | None = None,
        failure_kind: str | None = None,
    ) -> None:
        self.enabled = enabled
        self._publisher = publisher
        self._connected = False
        self._failure_kind = failure_kind
        self._messages_sent = 0

    @property
    def connected(self) -> bool:
        self._observe_status()
        return self._connected

    @property
    def failure_kind(self) -> str | None:
        self._observe_status()
        return self._failure_kind

    @property
    def messages_sent(self) -> int:
        self._observe_status()
        return self._messages_sent

    def start(self) -> None:
        publisher = self._publisher
        if publisher is None:
            return
        try:
            publisher.start()
        except Exception as exc:
            self._record_failure(type(exc).__name__)
        self._observe_status()

    def wait_for_request(self, timeout_seconds: float) -> RemotePoseRequest | None:
        publisher = self._publisher
        if publisher is None:
            return None
        try:
            return publisher.wait_for_request(timeout_seconds)
        except Exception as exc:
            self._record_failure(type(exc).__name__)
            self._observe_status()
            return None

    def submit(
        self,
        analysis: SquatAnalysis,
        *,
        request: RemotePoseRequest,
        evidence_age_ms: int,
    ) -> None:
        publisher = self._publisher
        if publisher is None:
            return
        try:
            publisher.submit(
                analysis,
                request=request,
                evidence_age_ms=evidence_age_ms,
            )
        except Exception as exc:
            self._record_failure(type(exc).__name__)
        self._observe_status()

    def request_resume(self) -> None:
        publisher = self._publisher
        if publisher is None:
            return
        try:
            publisher.request_resume()
        except Exception as exc:
            self._record_failure(type(exc).__name__)
        self._observe_status()

    def close(self) -> None:
        publisher = self._publisher
        if publisher is None:
            return
        # Capture the useful connected state and sent count before close tears
        # down the socket.  The result describes whether this run connected,
        # matching the existing voice_connected field.
        self._observe_status()
        try:
            publisher.close()
        except Exception as exc:
            self._record_failure(type(exc).__name__)
        self._observe_status()

    def _observe_status(self) -> None:
        publisher = self._publisher
        if publisher is None:
            return
        try:
            self._connected = self._connected or bool(publisher.connected)
        except Exception as exc:
            self._record_failure(type(exc).__name__)
        try:
            messages_sent = publisher.messages_sent
            if (
                not isinstance(messages_sent, bool)
                and isinstance(messages_sent, int)
                and messages_sent >= 0
            ):
                self._messages_sent = max(self._messages_sent, messages_sent)
            else:
                self._record_failure("InvalidRemotePoseStatus")
        except Exception as exc:
            self._record_failure(type(exc).__name__)
        try:
            failure_kind = publisher.failure_kind
        except Exception as exc:
            self._record_failure(type(exc).__name__)
        else:
            if failure_kind:
                self._record_failure(
                    failure_kind
                    if isinstance(failure_kind, str)
                    and failure_kind.isascii()
                    and failure_kind.isidentifier()
                    else "RemotePosePublisherError"
                )

    def _record_failure(self, failure_kind: str) -> None:
        if self._failure_kind is None:
            self._failure_kind = failure_kind


class _CueSpeakerBridge:
    """Quick speaker handoff plus asynchronous playback-failure reporting."""

    def __init__(self, player: _AudioPlayer | None) -> None:
        self.player = player
        self._session: LaptopSquatSession | None = None
        self._failure_callback: Callable[[str], None] | None = None

    def bind(self, session: LaptopSquatSession) -> None:
        if self._session is not None:
            raise RuntimeError("speaker bridge is already bound")
        self._session = session

    def bind_failure_callback(self, callback: Callable[[str], None]) -> None:
        if self._failure_callback is not None:
            raise RuntimeError("speaker failure callback is already bound")
        self._failure_callback = callback

    def enqueue(self, clip: ReleasedCueAudio) -> None:
        player = self.player
        if player is None:
            return
        ticket = player.play(clip.pcm16_mono_24khz)
        threading.Thread(
            target=self._watch_ticket,
            args=(ticket, clip.authorization.cue_id),
            name=f"recoverybox-cue-playback-{clip.ticket_id}",
            daemon=True,
        ).start()

    def preempt(self) -> None:
        if self.player is not None:
            self.player.stop()

    def _watch_ticket(self, ticket: _PlaybackTicket, cue_id: CueId) -> None:
        try:
            ticket.result()
        except PlaybackCancelledError:
            # Guardian pause/stop preemption intentionally cancels a ticket;
            # successful cancellation is not a speaker failure.
            return
        except Exception:
            callback = self._failure_callback
            if callback is not None:
                callback("SpeakerPlaybackError")
            session = self._session
            if session is None or session.ended:
                return
            try:
                session.report_speaker_failure()
            except Exception:
                pass
            return

        if cue_id is not CueId.SQUAT_PERSON_DETECTED:
            return
        session = self._session
        if session is None or session.ended:
            return
        try:
            session.notify_cue_playback_succeeded(cue_id)
        except Exception:
            # Failure at this boundary cannot be allowed to arm exercise.
            callback = self._failure_callback
            if callback is not None:
                callback("SpeakerPlaybackError")
            try:
                session.report_speaker_failure()
            except Exception:
                pass


class LaptopSquatDemo:
    """Run local pose tracking beside one persistent Realtime receiver."""

    def __init__(
        self,
        *,
        config: SquatDemoConfig,
        session: LaptopSquatSession,
        pose_source: _PoseSource,
        tracker: _SquatTracker,
        command_source: SquatDemoCommandSource,
        output: TextIO,
        voice_connected: bool,
        voice_failure_kind: str | None = None,
        audio_player: _AudioPlayer | None = None,
        microphone: _Microphone | None = None,
        microphone_failure_kind: str | None = None,
        runtime_versions: Mapping[str, str] | None = None,
        remote_pose: _RemotePoseBridge | None = None,
        monotonic_ns_clock: MonotonicNanoseconds = monotonic_ns,
    ) -> None:
        self.config = config
        self.session = session
        self.pose_source = pose_source
        self.tracker = tracker
        self.command_source = command_source
        self.output = output
        self.audio_player = audio_player
        self.microphone = microphone
        self._voice_connected = voice_connected
        self._realtime_failure_kind = voice_failure_kind
        self._microphone_failure_kind = microphone_failure_kind
        self._runtime_versions = dict(runtime_versions or {})
        self._remote_pose = remote_pose or _RemotePoseBridge(enabled=False)
        if not callable(monotonic_ns_clock):
            raise TypeError("monotonic_ns_clock must be callable")
        self._monotonic_ns = monotonic_ns_clock
        self._receiver_stop = threading.Event()
        self._receiver_thread: threading.Thread | None = None
        self._receiver_lock = threading.Lock()
        self._resume_requested = False
        self._last_analysis: SquatAnalysis | None = None
        self._last_reported_rep_count = 0
        self._last_reported_mode = SessionMode.IDLE

    @property
    def realtime_failure_kind(self) -> str | None:
        with self._receiver_lock:
            return self._realtime_failure_kind

    def run(self) -> SquatDemoResult:
        """Open the camera once and run until an authorized end boundary."""

        frames_processed = 0
        pose_frames = 0
        assessable_frames = 0
        end_reason: SquatDemoEndReason | None = None
        opened = False
        try:
            self.pose_source.open()
            opened = True
            self._remote_pose.start()
            self._start_receiver()
            self._print_controls()

            while not self.session.ended:
                command_end = self._handle_commands()
                if command_end is not None:
                    end_reason = command_end
                    break
                if self.session.ended:
                    break

                request: RemotePoseRequest | None = None
                if self._remote_pose.enabled:
                    # The Pi owns cadence and freshness.  A verified request
                    # must arrive before this process acquires the next frame.
                    request = self._remote_pose.wait_for_request(0.05)
                    if request is None:
                        continue

                sample = self.pose_source.read(preview_lines=self._preview_lines())
                frames_processed += 1
                if sample.pose is not None:
                    pose_frames += 1
                if sample.quit_requested:
                    # q/Escape is a local physical-stop edge.  Honor it before
                    # the tracker or Guardian can count or enqueue another cue.
                    signal = self._request_physical_stop()
                    end_reason = _end_reason_from_signal(
                        self.session.end_controller.end_signal or signal
                    )
                    break
                if self.session.ended:
                    break
                analysis = (
                    self.tracker.update(sample.pose)
                    if sample.pose is not None
                    else self.tracker.update_missing(sample.timestamp_ms)
                )
                if analysis.assessable:
                    assessable_frames += 1
                self._last_analysis = analysis
                if request is not None:
                    # submit() is a non-blocking, bounded handoff of numeric
                    # analysis only. Any remote failure stays outside Guardian.
                    self._remote_pose.submit(
                        analysis,
                        request=request,
                        evidence_age_ms=self._capture_to_submit_age_ms(sample.timestamp_ms),
                    )

                self._activate_or_resume(analysis)
                if not self.session.ended:
                    result = self.session.process_analysis(analysis)
                    if result.cue_delivery_failed and self._realtime_failure_kind is None:
                        failure = self.session.last_cue_failure
                        self._set_realtime_failure(
                            failure.reason.value if failure is not None else "CueDeliveryError"
                        )
                    self.session.tick()
                    self._sync_session_failure()
                    self._report_progress(result.mode, analysis)

                if (
                    self.config.max_frames is not None
                    and frames_processed >= self.config.max_frames
                ):
                    signal = self.session.abort_runtime(RuntimeAbortReason.MAX_FRAMES_REACHED)
                    winning_signal = self.session.end_controller.end_signal or signal
                    end_reason = _end_reason_from_signal(winning_signal)
                    break

            if end_reason is None:
                signal = self.session.end_controller.end_signal
                end_reason = _end_reason_from_signal(signal)
        finally:
            self._shutdown(opened=opened)

        self._sync_session_failure()
        cue_failure = self.session.last_cue_failure
        return SquatDemoResult(
            frames_processed=frames_processed,
            pose_frames=pose_frames,
            assessable_frames=assessable_frames,
            rep_count=self.session.completed_rep_count,
            end_reason=end_reason,
            final_mode=self.session.coordinator.current_mode,
            voice_enabled=self.config.voice_enabled,
            voice_connected=self._voice_connected,
            runtime_versions=self._runtime_versions,
            realtime_failure_kind=self.realtime_failure_kind,
            microphone_failure_kind=self._microphone_failure_kind,
            cue_failure_reason=cue_failure.reason.value if cue_failure is not None else None,
            remote_pose_enabled=self._remote_pose.enabled,
            remote_pose_connected=self._remote_pose.connected,
            remote_pose_failure_kind=self._remote_pose.failure_kind,
            remote_pose_messages_sent=self._remote_pose.messages_sent,
        )

    def _start_receiver(self) -> None:
        if not self._voice_connected:
            return
        self._receiver_thread = threading.Thread(
            target=self._receive_realtime,
            name="recoverybox-realtime-receiver",
            daemon=True,
        )
        self._receiver_thread.start()

    def _receive_realtime(self) -> None:
        while not self._receiver_stop.is_set() and not self.session.ended:
            dispatched = self.session.pump_once()
            if dispatched.end_signal is not None:
                return
            if dispatched.failure_kind is not None:
                if not self._receiver_stop.is_set() and not self.session.ended:
                    self._set_realtime_failure(dispatched.failure_kind)
                return

    def _handle_commands(self) -> SquatDemoEndReason | None:
        commands = self.command_source.poll()
        if SquatDemoCommand.STOP in commands:
            # Physical stop outranks other controls in the same poll batch, so
            # an Enter event cannot commit captured audio before shutdown.
            signal = self._request_physical_stop()
            return _end_reason_from_signal(self.session.end_controller.end_signal or signal)
        self._sync_session_failure()
        for command in commands:
            if command is SquatDemoCommand.RESUME:
                self._remote_pose.request_resume()
                self._resume_requested = True
            elif command is SquatDemoCommand.TOGGLE_MICROPHONE:
                self._toggle_microphone()
            elif command is SquatDemoCommand.HELP:
                self._print_controls()
        return None

    def _toggle_microphone(self) -> None:
        microphone = self.microphone
        if (
            microphone is None
            or not self._voice_connected
            or self.realtime_failure_kind is not None
        ):
            self._write("[mic] unavailable")
            return
        try:
            if not microphone.active:
                microphone.start()
                self._write("[mic] recording; press Enter to send")
                return
            pcm = microphone.stop()
            submitted = self.session.submit_user_audio_turn(pcm)
            if submitted.submitted:
                self._write("[mic] turn sent")
            else:
                self._microphone_failure_kind = submitted.failure_kind
                self._write(f"[mic] turn discarded ({submitted.failure_kind})")
        except Exception as exc:
            # Never include provider, device, transcript, or captured-audio
            # details in terminal diagnostics.
            self._microphone_failure_kind = type(exc).__name__
            try:
                microphone.abort()
            except Exception:
                pass
            self._write(f"[mic] capture failed ({self._microphone_failure_kind})")

    def _activate_or_resume(self, analysis: SquatAnalysis) -> None:
        mode = self.session.coordinator.current_mode
        if mode in {SessionMode.IDLE, SessionMode.CHECK_IN}:
            if self.session.activate_exercise(analysis):
                self._write("[exercise] active")
            return

        if mode is not SessionMode.PAUSED or not self._resume_requested:
            return
        self._resume_requested = False
        if self.config.voice_enabled and self.realtime_failure_kind is not None:
            self._write("[exercise] cannot resume while Realtime is unavailable")
            return
        if self.session.resume_after_assessable_pose(analysis):
            self._write("[exercise] resumed")
        else:
            self._write("[exercise] resume requires a fresh assessable standing pose")

    def _preview_lines(self) -> tuple[str, ...]:
        analysis = self._last_analysis
        rep_count = self.session.completed_rep_count
        phase = analysis.phase.value if analysis is not None else SquatPhase.UNKNOWN.value
        arms = "unknown"
        if analysis is not None and analysis.arms_in_t is not None:
            arms = "T" if analysis.arms_in_t else "adjust"
        if not self.config.voice_enabled:
            voice = "off"
        elif self.realtime_failure_kind is not None:
            voice = f"paused ({self.realtime_failure_kind})"
        else:
            voice = "connected"
        controls = "q/Esc stop"
        if self.config.microphone_enabled:
            controls += " | Enter mic"
        controls += " | r+Enter resume"
        return (
            f"Squats: {rep_count}/{self.config.target_reps}",
            f"Phase: {phase} | Arms: {arms}",
            f"Mode: {self.session.coordinator.current_mode.value} | Voice: {voice}",
            controls,
        )

    def _capture_to_submit_age_ms(self, capture_timestamp_ms: int) -> int:
        """Return a conservative rounded-up same-host evidence age.

        A malformed or backwards clock becomes exactly the stale boundary so
        the publisher withholds the response and the Pi request watchdog fails
        closed. The Pi's measured request round trip remains authoritative.
        """

        try:
            now_ns = self._monotonic_ns()
        except Exception:
            return MAX_REMOTE_POSE_EVIDENCE_AGE_MS
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            return MAX_REMOTE_POSE_EVIDENCE_AGE_MS
        now_ceiling_ms = (now_ns + 999_999) // 1_000_000
        if now_ceiling_ms < capture_timestamp_ms:
            return MAX_REMOTE_POSE_EVIDENCE_AGE_MS
        return min(
            now_ceiling_ms - capture_timestamp_ms,
            MAX_REMOTE_POSE_EVIDENCE_AGE_MS,
        )

    def _report_progress(self, mode: SessionMode, analysis: SquatAnalysis) -> None:
        del analysis
        rep_count = self.session.completed_rep_count
        if rep_count > self._last_reported_rep_count:
            self._last_reported_rep_count = rep_count
            self._write(f"[exercise] rep {rep_count}")
        if mode is not self._last_reported_mode:
            self._last_reported_mode = mode
            self._write(f"[exercise] mode {mode.value}")

    def _set_realtime_failure(self, failure_kind: str) -> None:
        with self._receiver_lock:
            if self._realtime_failure_kind is None:
                self._realtime_failure_kind = failure_kind
                self._write(f"[realtime] unavailable ({failure_kind}); local tracking continues")

    def _sync_session_failure(self) -> None:
        failure_kind = self.session.realtime_failure_kind
        if failure_kind is None and self.session.last_cue_failure is not None:
            failure_kind = self.session.last_cue_failure.reason.value
        if failure_kind is not None:
            self._set_realtime_failure(failure_kind)

    def _print_controls(self) -> None:
        if self.config.microphone_enabled:
            self._write("[controls] q/Escape stops; Enter toggles push-to-talk; r+Enter resumes")
        else:
            self._write("[controls] q/Escape stops; r+Enter resumes")

    def _request_physical_stop(self) -> SessionEndSignal | None:
        try:
            return self.session.request_physical_stop()
        except Exception:
            # The end controller commits the one-shot local signal before its
            # cleanup callback. Cleanup itself uses finally blocks to close the
            # cue gate and transport, so a speaker-stop error must not reopen
            # or delay the physical-stop boundary.
            return self.session.end_controller.end_signal

    def _write(self, message: str) -> None:
        print(message, file=self.output, flush=True)

    def _shutdown(self, *, opened: bool) -> None:
        try:
            self._shutdown_local(opened=opened)
        finally:
            # The publisher owns the remote STOP/control teardown.  It must be
            # attempted even if an unrelated local cleanup edge misbehaves.
            self._remote_pose.close()

    def _shutdown_local(self, *, opened: bool) -> None:
        microphone = self.microphone
        if microphone is not None and microphone.active:
            try:
                microphone.abort()
            except Exception:
                pass

        self._receiver_stop.set()
        if not self.session.ended:
            try:
                self.session.abort_runtime(RuntimeAbortReason.LAUNCHER_CLEANUP)
            except Exception:
                pass

        receiver = self._receiver_thread
        if receiver is not None and receiver is not threading.current_thread():
            receiver.join(timeout=1.0)

        if opened:
            try:
                self.pose_source.close()
            except Exception:
                pass
        try:
            self.command_source.close()
        except Exception:
            pass
        if self.audio_player is not None:
            try:
                self.audio_player.close()
            except Exception:
                pass


def _end_reason_from_signal(signal: SessionEndSignal | None) -> SquatDemoEndReason:
    if signal is not None and signal.source is SessionEndSource.VALIDATED_TOOL_CALL:
        return SquatDemoEndReason.VALIDATED_TOOL_CALL
    if signal is not None and signal.source is SessionEndSource.RUNTIME_ABORT:
        if signal.abort_reason is RuntimeAbortReason.MAX_FRAMES_REACHED:
            return SquatDemoEndReason.MAX_FRAMES
        return SquatDemoEndReason.RUNTIME_ABORT
    return SquatDemoEndReason.PHYSICAL_STOP


def _new_session(
    *,
    transport: RealtimeTransport,
    speaker: _CueSpeakerBridge,
    config: SquatDemoConfig,
    cue_delivery_enabled: bool,
) -> LaptopSquatSession:
    session = LaptopSquatSession(
        transport=transport,
        on_cue_audio=speaker.enqueue,
        on_audio_preempt=speaker.preempt,
        plan=build_single_camera_squat_plan(target_reps=config.target_reps),
        cue_delivery_enabled=cue_delivery_enabled,
    )
    speaker.bind(session)
    session.start(instructions=SQUAT_DEMO_INSTRUCTIONS, voice=config.voice)
    return session


def _optional_distribution_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "missing"
    except Exception:
        return "metadata_error"


def _optional_environment_value(environment: Mapping[str, str], name: str) -> str | None:
    value = environment.get(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _build_remote_pose_bridge(
    config: SquatDemoConfig,
    dependencies: SquatDemoDependencies,
) -> _RemotePoseBridge:
    if config.pose_peer is None:
        return _RemotePoseBridge(enabled=False)

    # SquatDemoConfig enforces the pair.  Keep both assertions local so a
    # future refactor cannot accidentally treat a missing credential as an
    # anonymous publisher connection.
    assert config.pose_token_file is not None
    token = b""
    try:
        token = dependencies.remote_pose_token_loader(config.pose_token_file)
        publisher = dependencies.remote_pose_publisher_factory(config.pose_peer, token)
    except Exception as exc:
        return _RemotePoseBridge(enabled=True, failure_kind=type(exc).__name__)
    finally:
        # Do not retain the loader's credential on the launcher or bridge.
        token = b""
    return _RemotePoseBridge(enabled=True, publisher=publisher)


def build_squat_demo(
    config: SquatDemoConfig,
    *,
    environment: Mapping[str, str] | None = None,
    output: TextIO = sys.stdout,
    dependencies: SquatDemoDependencies | None = None,
) -> LaptopSquatDemo:
    """Validate assets and compose native edges without opening the camera."""

    if not isinstance(config, SquatDemoConfig):
        raise TypeError("config must be a SquatDemoConfig")
    selected = dependencies or SquatDemoDependencies()
    env = os.environ if environment is None else environment

    # Exact distribution metadata and model integrity checks happen before
    # source construction, WebcamPoseSource.open, or any native runtime import.
    runtime_versions = dict(selected.validate_runtime())
    runtime_versions["sounddevice"] = _optional_distribution_version("sounddevice")
    validated_model = selected.validate_model(config.model_asset_path)
    pose_source = selected.pose_source_factory(
        WebcamPoseConfig(
            model_asset_path=validated_model,
            camera_index=config.camera_index,
            preview=config.preview,
        )
    )
    tracker = selected.tracker_factory()
    command_source = selected.command_source_factory()

    voice_connected = False
    voice_failure_kind: str | None = None
    microphone_failure_kind: str | None = None
    player: _AudioPlayer | None = None
    microphone: _Microphone | None = None
    transport: RealtimeTransport = _LocalOnlyTransport()

    if config.voice_enabled:
        api_key = env.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            voice_failure_kind = "MissingOpenAIAPIKey"
        else:
            try:
                transport = selected.transport_factory(api_key=api_key)
                voice_connected = True
            except Exception as exc:
                voice_failure_kind = type(exc).__name__
            finally:
                # Do not retain a credential on the application object.
                api_key = ""

    if voice_connected:
        try:
            player = selected.audio_player_factory()
        except Exception as exc:
            voice_failure_kind = type(exc).__name__
            voice_connected = False
            try:
                transport.close()
            except Exception:
                pass
            transport = _LocalOnlyTransport()

    if voice_connected and config.microphone_enabled:
        try:
            microphone = selected.microphone_factory()
        except Exception as exc:
            microphone_failure_kind = type(exc).__name__

    speaker = _CueSpeakerBridge(player)
    try:
        session = _new_session(
            transport=transport,
            speaker=speaker,
            config=config,
            cue_delivery_enabled=voice_connected,
        )
    except Exception as exc:
        # A failed initial session.update invalidates the only live connection.
        # Continue camera/Guardian work on an explicit local-only session; do
        # not create a second network connection.
        voice_failure_kind = type(exc).__name__
        voice_connected = False
        try:
            transport.close()
        except Exception:
            pass
        if player is not None:
            try:
                player.close()
            except Exception:
                pass
            player = None
        if microphone is not None:
            try:
                microphone.abort()
            except Exception:
                pass
            microphone = None
        speaker = _CueSpeakerBridge(None)
        session = _new_session(
            transport=_LocalOnlyTransport(),
            speaker=speaker,
            config=config,
            cue_delivery_enabled=False,
        )

    remote_pose = _build_remote_pose_bridge(config, selected)
    try:
        demo = LaptopSquatDemo(
            config=config,
            session=session,
            pose_source=pose_source,
            tracker=tracker,
            command_source=command_source,
            output=output,
            voice_connected=voice_connected,
            voice_failure_kind=voice_failure_kind,
            audio_player=player,
            microphone=microphone,
            microphone_failure_kind=microphone_failure_kind,
            runtime_versions=runtime_versions,
            remote_pose=remote_pose,
            monotonic_ns_clock=selected.monotonic_ns,
        )
    except Exception:
        remote_pose.close()
        raise
    speaker.bind_failure_callback(demo._set_realtime_failure)
    return demo


def run_squat_demo(
    *,
    environment: Mapping[str, str] | None = None,
    output: TextIO = sys.stdout,
    camera_index: int | None = None,
    model_path: str | Path | None = None,
    voice: str | None = None,
    no_preview: bool = False,
    no_voice: bool = False,
    no_mic: bool = False,
    max_frames: int | None = None,
    pose_peer: str | None = None,
    pose_token_file: str | Path | None = None,
    dependencies: SquatDemoDependencies | None = None,
) -> int:
    """CLI-safe entrypoint; diagnostics never include provider payloads."""

    env = os.environ if environment is None else environment
    try:
        settings = Settings.from_environment(env)
        resolved_pose_peer = (
            _optional_environment_value(env, "RECOVERYBOX_POSE_PEER")
            if pose_peer is None
            else pose_peer
        )
        resolved_pose_token_file = (
            _optional_environment_value(env, "RECOVERYBOX_POSE_TOKEN_FILE")
            if pose_token_file is None
            else pose_token_file
        )
        config = SquatDemoConfig(
            model_asset_path=settings.pose.model_path if model_path is None else Path(model_path),
            camera_index=settings.pose.camera_index if camera_index is None else camera_index,
            preview=settings.pose.preview and not no_preview,
            voice=settings.realtime.voice if voice is None else voice,
            voice_enabled=not no_voice,
            microphone_enabled=not no_voice and not no_mic,
            max_frames=max_frames,
            pose_peer=resolved_pose_peer,
            pose_token_file=resolved_pose_token_file,
        )
        demo = build_squat_demo(
            config,
            environment=env,
            output=output,
            dependencies=dependencies,
        )
        result = demo.run()
    except (ConfigurationError, LaptopRuntimePinError, PoseModelError):
        print(
            "error: laptop prerequisites are invalid; run "
            "`recoverybox doctor` and `recoverybox download-pose-model`",
            file=output,
        )
        return 2
    except KeyboardInterrupt:
        print("stopped", file=output)
        return 130
    except Exception as exc:
        print(f"error: squat demo failed ({type(exc).__name__})", file=output)
        return 2

    print(json.dumps(result.as_dict(), indent=2, sort_keys=True), file=output)
    return 0


__all__ = [
    "SQUAT_DEMO_INSTRUCTIONS",
    "LaptopRuntimePinError",
    "LaptopSquatDemo",
    "SquatDemoCommand",
    "SquatDemoCommandSource",
    "SquatDemoConfig",
    "SquatDemoDependencies",
    "SquatDemoEndReason",
    "SquatDemoResult",
    "TerminalCommandSource",
    "build_squat_demo",
    "run_squat_demo",
    "validate_laptop_runtime_pins",
]
