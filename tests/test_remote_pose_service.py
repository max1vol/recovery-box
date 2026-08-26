from __future__ import annotations

import json
import threading
import time
from collections import deque
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import recoverybox.device.remote_pose_service as remote_pose_service
from recoverybox.core import (
    CueId,
    CueKind,
    GuardianReason,
    GuardianRuntimeFault,
    SessionMode,
)
from recoverybox.device.gpio_stop import (
    GpioStopConfig,
    PhysicalStopMonitor,
    StopInputSnapshot,
    StopInputState,
    StopInputTrigger,
)
from recoverybox.device.remote_pose_service import (
    DEFAULT_REMOTE_POSE_PORT,
    PoseSourceMode,
    RemotePoseService,
    RemotePoseServiceConfig,
    RemotePoseServiceConfigurationError,
    RemotePoseServiceDependencies,
    _CoalescingStatusWriter,
    _SubprocessCueSpeaker,
    load_openai_api_key_credential,
    run_remote_pose_service,
)
from recoverybox.exercise import (
    SquatAnalysis,
    SquatAssessmentIssue,
    SquatPhase,
)
from recoverybox.realtime import ReleasedCueAudio, RuntimeAbortReason
from recoverybox.remote_pose import (
    RemotePoseKind,
    RemotePoseMessage,
    decode_remote_pose_challenge,
    decode_remote_pose_request,
    encode_remote_pose_message,
)
from recoverybox.session import ApprovedCuePlaybackAuthorization

TOKEN = b"r" * 32
SESSION_ONE = "1" * 32
SESSION_TWO = "2" * 32
SERVICE_EPOCH = "e" * 64
SERVER_NONCE = "c" * 64
REQUEST_NONCE = "d" * 64


def analysis(
    timestamp_ms: int,
    *,
    phase: SquatPhase = SquatPhase.STANDING,
    assessable: bool = True,
    rep_count: int = 0,
) -> SquatAnalysis:
    return SquatAnalysis(
        timestamp_ms=timestamp_ms,
        assessable=assessable,
        phase=phase if assessable else SquatPhase.UNKNOWN,
        rep_count=rep_count,
        events=(),
        issues=() if assessable else (SquatAssessmentIssue.NO_POSE,),
        confidence=0.95 if assessable else 0.0,
        knee_angle_degrees=170.0 if assessable else None,
        arms_in_t=True if assessable else None,
    )


def encoded(
    kind: RemotePoseKind,
    *,
    session_id: str = SESSION_ONE,
    sequence: int | None = None,
    pose: SquatAnalysis | None = None,
    evidence_age_ms: int = 0,
    request_sequence: int | None = None,
    request_nonce: str = REQUEST_NONCE,
) -> bytes:
    if kind is RemotePoseKind.ANALYSIS and request_sequence is None:
        request_sequence = sequence
    return encode_remote_pose_message(
        RemotePoseMessage(
            kind=kind,
            session_id=session_id,
            sequence=sequence,
            analysis=pose,
            service_epoch=SERVICE_EPOCH,
            server_nonce=SERVER_NONCE,
            evidence_age_ms=(evidence_age_ms if kind is RemotePoseKind.ANALYSIS else None),
            request_sequence=(request_sequence if kind is RemotePoseKind.ANALYSIS else None),
            request_nonce=(request_nonce if kind is RemotePoseKind.ANALYSIS else None),
        ),
        TOKEN,
    )


def service_dependencies(**overrides) -> RemotePoseServiceDependencies:
    return RemotePoseServiceDependencies(
        service_epoch_factory=lambda: SERVICE_EPOCH,
        server_nonce_factory=lambda: SERVER_NONCE,
        request_nonce_factory=lambda: REQUEST_NONCE,
        **overrides,
    )


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class IncrementingClock:
    def __init__(self, *, step: float = 0.1) -> None:
        self.value = 0.0
        self.step = step
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            self.value += self.step
            return self.value


class FakeLocalPoseSource:
    def __init__(self, read) -> None:
        self._read = read
        self.opened = False
        self.closed = False
        self.read_count = 0

    def open(self) -> None:
        self.opened = True

    def read(self):
        self.read_count += 1
        result = self._read(self.read_count)
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(analysis=result)

    def close(self) -> None:
        self.closed = True


def wait_until(predicate, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class FakeConnection:
    def __init__(self, chunks: list[bytes | BaseException | object]) -> None:
        self.chunks = deque(chunks)
        self.timeout: float | None = None
        self.closed = False
        self.close_event = threading.Event()
        self.sent: list[bytes] = []

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def recv(self, size: int) -> bytes:
        assert size == 4096
        if not self.chunks:
            return b""
        item = self.chunks.popleft()
        if callable(item):
            return item()  # type: ignore[no-any-return,operator]
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, bytes)
        return item

    def sendall(self, data: bytes) -> None:
        assert isinstance(data, bytes)
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True
        self.close_event.set()


class ExitBarrierLock:
    """Pause the first service-lock exit after a candidate session starts."""

    def __init__(self, delegate, *, candidate_started: threading.Event) -> None:
        self._delegate = delegate
        self._candidate_started = candidate_started
        self._guard = threading.Lock()
        self._triggered = False
        self.after_candidate_check = threading.Event()
        self.release_receiver = threading.Event()

    def __enter__(self):
        self._delegate.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self._delegate.release()
        should_pause = False
        with self._guard:
            if self._candidate_started.is_set() and not self._triggered:
                self._triggered = True
                should_pause = True
        if should_pause:
            self.after_candidate_check.set()
            assert self.release_receiver.wait(1)


class FailingAcceptListener:
    def __init__(self) -> None:
        self.bound: tuple[str, int] | None = None
        self.closed = False

    def setsockopt(self, level: int, option: int, value: int) -> None:
        del level, option, value

    def bind(self, address: tuple[str, int]) -> None:
        self.bound = address

    def listen(self, backlog: int) -> None:
        assert backlog == 1

    def settimeout(self, timeout: float) -> None:
        assert timeout == 0.1

    def accept(self):
        raise OSError("persistent listener failure")

    def close(self) -> None:
        self.closed = True


class FakeCoordinator:
    def __init__(self) -> None:
        self.current_mode = SessionMode.IDLE

    def transition_to(self, mode: SessionMode) -> None:
        self.current_mode = mode


class FakeSession:
    def __init__(self, *, preempt) -> None:
        self.coordinator = FakeCoordinator()
        self.plan = SimpleNamespace(max_pose_age_ms=500)
        self.ended = False
        self.preempt = preempt
        self.starts: list[tuple[str, str]] = []
        self.activations: list[tuple[SquatAnalysis, int]] = []
        self.resumes: list[tuple[SquatAnalysis, int]] = []
        self.processed: list[tuple[SquatAnalysis, int]] = []
        self.ticks = 0
        self.stop_requests = 0
        self.physical_stop_requests = 0
        self.abort_requests: list[RuntimeAbortReason] = []
        self.runtime_faults: list[GuardianRuntimeFault] = []
        self.realtime_failure_kind: str | None = None

    def start(self, *, instructions: str, voice: str) -> None:
        self.starts.append((instructions, voice))

    def activate_exercise(
        self,
        pose: SquatAnalysis,
        *,
        pose_age_ms: int = 0,
    ) -> bool:
        self.activations.append((pose, pose_age_ms))
        if (
            pose.assessable
            and pose.phase is SquatPhase.STANDING
            and pose_age_ms <= self.plan.max_pose_age_ms
        ):
            self.coordinator.current_mode = SessionMode.ACTIVE_EXERCISE
            return True
        return False

    def resume_after_assessable_pose(
        self,
        pose: SquatAnalysis,
        *,
        pose_age_ms: int = 0,
    ) -> bool:
        self.resumes.append((pose, pose_age_ms))
        if (
            self.coordinator.current_mode is SessionMode.PAUSED
            and pose.assessable
            and pose.phase is SquatPhase.STANDING
            and pose_age_ms <= self.plan.max_pose_age_ms
        ):
            self.coordinator.current_mode = SessionMode.ACTIVE_EXERCISE
            return True
        return False

    def process_analysis(
        self,
        pose: SquatAnalysis,
        *,
        pose_age_ms: int = 0,
    ) -> object:
        self.processed.append((pose, pose_age_ms))
        if self.coordinator.current_mode is SessionMode.ACTIVE_EXERCISE and (
            not pose.assessable or pose_age_ms > self.plan.max_pose_age_ms
        ):
            self.coordinator.current_mode = SessionMode.PAUSED
            self.preempt()
        return object()

    def tick(self) -> int:
        self.ticks += 1
        return 0

    def request_physical_stop(self) -> object:
        self.stop_requests += 1
        self.physical_stop_requests += 1
        self.ended = True
        self.coordinator.current_mode = SessionMode.STOPPED
        self.preempt()
        return object()

    def abort_runtime(self, reason: RuntimeAbortReason) -> object:
        self.stop_requests += 1
        self.abort_requests.append(reason)
        self.ended = True
        self.coordinator.current_mode = SessionMode.STOPPED
        self.preempt()
        return object()

    def apply_runtime_fault(self, fault: GuardianRuntimeFault) -> object:
        self.runtime_faults.append(fault)
        self.coordinator.current_mode = (
            SessionMode.STOPPED
            if fault is GuardianRuntimeFault.SAFETY_ENFORCEMENT_FAILURE
            else SessionMode.PAUSED
        )
        self.preempt()
        return object()


class FakeSpeaker:
    def __init__(self) -> None:
        self.enqueued: list[ReleasedCueAudio] = []
        self.enqueue_attempts: list[ReleasedCueAudio] = []
        self.quiesced = False
        self.quiescences = 0
        self.preemptions = 0
        self.closed = False

    def enqueue(self, clip: ReleasedCueAudio) -> None:
        self.enqueue_attempts.append(clip)
        if self.quiesced:
            raise RuntimeError("cue speaker is quiesced")
        self.enqueued.append(clip)

    def quiesce(self) -> None:
        if not self.quiesced:
            self.quiesced = True
            self.quiescences += 1

    def preempt(self) -> None:
        self.preemptions += 1

    def close(self) -> None:
        self.quiesce()
        self.closed = True


class FakeStopMonitor:
    def __init__(
        self,
        *,
        config: GpioStopConfig,
        on_stop,
        on_status,
        close_event: list[str] | None = None,
    ) -> None:
        self.config = config
        self.on_stop = on_stop
        self.on_status = on_status
        self.close_event = close_event
        self.started = False
        self.closed = False
        self._snapshot = StopInputSnapshot(
            state=StopInputState.STARTING,
            stop_count=0,
            failure_kind=None,
        )

    @property
    def snapshot(self) -> StopInputSnapshot:
        return self._snapshot

    def start(self) -> None:
        self.started = True
        self.set_status(StopInputState.AVAILABLE)

    def close(self, *, timeout_seconds: float = 1.0) -> None:
        assert timeout_seconds == 1.0
        self.closed = True
        if self.close_event is not None:
            self.close_event.append("monitor-close")
        self.set_status(StopInputState.CLOSED)

    def set_status(
        self,
        state: StopInputState,
        *,
        failure_kind: str | None = None,
    ) -> None:
        self._snapshot = StopInputSnapshot(
            state=state,
            stop_count=self._snapshot.stop_count,
            failure_kind=failure_kind,
        )
        self.on_status(self._snapshot)

    def press(self) -> None:
        self._snapshot = StopInputSnapshot(
            state=StopInputState.PRESSED,
            stop_count=self._snapshot.stop_count + 1,
            failure_kind=None,
        )
        self.on_status(self._snapshot)
        self.on_stop(StopInputTrigger.BUTTON_PRESSED)

    def lose_input(self, failure_kind: str = "GPIOReadError") -> None:
        self._snapshot = StopInputSnapshot(
            state=StopInputState.UNAVAILABLE,
            stop_count=self._snapshot.stop_count + 1,
            failure_kind=failure_kind,
        )
        self.on_status(self._snapshot)
        self.on_stop(StopInputTrigger.INPUT_UNAVAILABLE)


def available_stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
    return FakeStopMonitor(
        config=config,
        on_stop=on_stop,
        on_status=on_status,
    )


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        *,
        clock: FakeClock | None = None,
        api_key: str | None = None,
    ) -> None:
        self.clock = clock or FakeClock()
        self.sessions: list[FakeSession] = []
        self.speakers: list[FakeSpeaker] = []
        self.session_api_keys: list[str | None] = []
        self.speaker_configs: list[object] = []
        self.stop_monitors: list[FakeStopMonitor] = []

        def speaker_factory(config) -> FakeSpeaker:
            self.speaker_configs.append(config)
            speaker = FakeSpeaker()
            self.speakers.append(speaker)
            return speaker

        def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
            del on_cue_audio
            self.session_api_keys.append(api_key)
            session = FakeSession(preempt=on_audio_preempt)
            self.sessions.append(session)
            return session

        def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
            monitor = FakeStopMonitor(
                config=config,
                on_stop=on_stop,
                on_status=on_status,
            )
            self.stop_monitors.append(monitor)
            return monitor

        self.config = RemotePoseServiceConfig(
            bind_host="100.106.237.106",
            allowed_peer="100.70.100.93",
            token_file=tmp_path / "pose-token",
            status_path=tmp_path / "status.json",
            audio_enabled=api_key is not None,
        )
        self.service = RemotePoseService(
            self.config,
            token=TOKEN,
            dependencies=service_dependencies(
                listener_factory=lambda: None,  # type: ignore[arg-type,return-value]
                clock=self.clock,
                session_factory=session_factory,
                speaker_factory=speaker_factory,
                credential_provider=lambda: api_key,
                token_loader=lambda _: TOKEN,
                stop_monitor_factory=stop_monitor_factory,
            ),
        )
        self.service.start_local_stop_monitor()

    def handle(self, chunks: list[bytes | BaseException | object]) -> FakeConnection:
        connection = FakeConnection(chunks)
        self.service.handle_connection(connection, peer_host=self.config.allowed_peer)
        assert self.service.wait_for_status()
        return connection


def test_config_uses_file_only_secret_and_rejects_wildcard_or_slow_watchdog(
    tmp_path: Path,
) -> None:
    config = RemotePoseServiceConfig.from_environment(
        {
            "RECOVERYBOX_POSE_BIND_HOST": "100.106.237.106",
            "RECOVERYBOX_POSE_ALLOWED_PEER": "100.70.100.93",
            "RECOVERYBOX_POSE_TOKEN_FILE": str(tmp_path / "token"),
            "RECOVERYBOX_STATUS_PATH": str(tmp_path / "status.json"),
            "RECOVERYBOX_BUTTON_GPIO": "23",
        }
    )

    assert config.port == DEFAULT_REMOTE_POSE_PORT
    assert config.pose_source is PoseSourceMode.REMOTE
    assert config.audio_enabled is False
    assert config.button_gpio == 23
    assert config.token_file == tmp_path / "token"
    assert not hasattr(config, "token")
    assert "secret" not in repr(config)

    local = RemotePoseServiceConfig.from_environment(
        {
            "RECOVERYBOX_POSE_SOURCE": "local",
            "RECOVERYBOX_POSE_BIND_HOST": "100.106.237.106",
            "RECOVERYBOX_POSE_ALLOWED_PEER": "100.70.100.93",
            "RECOVERYBOX_POSE_TOKEN_FILE": str(tmp_path / "unused-token"),
            "RECOVERYBOX_STATUS_PATH": str(tmp_path / "local-status.json"),
        }
    )
    assert local.pose_source is PoseSourceMode.LOCAL

    with pytest.raises(RemotePoseServiceConfigurationError, match="remote or local"):
        RemotePoseServiceConfig.from_environment(
            {
                "RECOVERYBOX_POSE_SOURCE": "automatic",
                "RECOVERYBOX_POSE_BIND_HOST": "100.106.237.106",
                "RECOVERYBOX_POSE_ALLOWED_PEER": "100.70.100.93",
                "RECOVERYBOX_POSE_TOKEN_FILE": str(tmp_path / "token"),
            }
        )

    with pytest.raises(RemotePoseServiceConfigurationError, match="tailnet"):
        RemotePoseServiceConfig(
            bind_host="0.0.0.0",
            allowed_peer="100.70.100.93",
            token_file=tmp_path / "token",
        )
    with pytest.raises(RemotePoseServiceConfigurationError, match="pose-age"):
        RemotePoseServiceConfig(
            bind_host="100.106.237.106",
            allowed_peer="100.70.100.93",
            token_file=tmp_path / "token",
            watchdog_seconds=0.501,
        )
    with pytest.raises(RemotePoseServiceConfigurationError, match="different"):
        RemotePoseServiceConfig(
            bind_host="100.106.237.106",
            allowed_peer="100.70.100.93",
            token_file=tmp_path / "same",
            status_path=tmp_path / "same",
        )
    with pytest.raises(RemotePoseServiceConfigurationError, match="different"):
        RemotePoseServiceConfig(
            bind_host="100.106.237.106",
            allowed_peer="100.70.100.93",
            token_file=tmp_path / "same",
            status_path=tmp_path / "nested" / ".." / "same",
        )
    with pytest.raises(RemotePoseServiceConfigurationError, match="button_gpio"):
        RemotePoseServiceConfig(
            bind_host="100.106.237.106",
            allowed_peer="100.70.100.93",
            token_file=tmp_path / "token",
            button_gpio=-1,
        )


def test_openai_credential_provider_reads_only_an_explicit_private_file(
    tmp_path: Path,
) -> None:
    credential = tmp_path / "openai-api-key"
    credential.write_text("sk-test-file-only", encoding="ascii")
    credential.chmod(0o600)

    assert load_openai_api_key_credential(credential) == "sk-test-file-only"
    assert (
        remote_pose_service._credential_provider_from_environment(
            {
                "OPENAI_API_KEY": "must-not-be-read",
                "RECOVERYBOX_OPENAI_CREDENTIAL_FILE": str(credential),
            }
        )
        == "sk-test-file-only"
    )
    assert (
        remote_pose_service._credential_provider_from_environment(
            {"OPENAI_API_KEY": "must-not-be-read"}
        )
        is None
    )


@pytest.mark.parametrize("value", ["", "sk-test\n", "sk test", "sk-test\t"])
def test_openai_credential_rejects_blank_or_whitespace_tokens(
    tmp_path: Path,
    value: str,
) -> None:
    credential = tmp_path / "openai-api-key"
    credential.write_text(value, encoding="ascii")
    credential.chmod(0o600)

    with pytest.raises(RemotePoseServiceConfigurationError, match="OpenAI credential"):
        load_openai_api_key_credential(credential)


def test_openai_credential_rejects_links_and_permissive_files(tmp_path: Path) -> None:
    credential = tmp_path / "openai-api-key"
    credential.write_text("sk-private", encoding="ascii")
    credential.chmod(0o644)
    with pytest.raises(RemotePoseServiceConfigurationError, match="OpenAI credential"):
        load_openai_api_key_credential(credential)

    credential.chmod(0o600)
    link = tmp_path / "credential-link"
    link.symlink_to(credential)
    with pytest.raises(RemotePoseServiceConfigurationError, match="OpenAI credential"):
        load_openai_api_key_credential(link)


def test_explicit_local_mode_preserves_start_arm_until_assessable_standing(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    release_source = threading.Event()
    monitors: list[FakeStopMonitor] = []
    listener_calls: list[str] = []

    def read_pose(read_count: int) -> SquatAnalysis:
        if read_count == 1:
            clock.value = 0.1
            return analysis(100, assessable=False)
        if read_count == 2:
            clock.value = 0.2
            return analysis(200, phase=SquatPhase.DOWN)
        if read_count == 3:
            clock.value = 0.3
            return analysis(300)
        assert release_source.wait(5)
        return analysis(301)

    source = FakeLocalPoseSource(read_pose)

    def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        monitor = FakeStopMonitor(
            config=config,
            on_stop=on_stop,
            on_status=on_status,
        )
        monitors.append(monitor)
        return monitor

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source=PoseSourceMode.LOCAL,
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=None,
        dependencies=service_dependencies(
            listener_factory=lambda: listener_calls.append("listener"),  # type: ignore[arg-type,return-value]
            clock=clock,
            session_factory=remote_pose_service._default_session_factory,
            credential_provider=lambda: pytest.fail("audio-disabled mode read credentials"),
            token_loader=lambda _: pytest.fail("local mode loaded the remote token"),
            stop_monitor_factory=stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            service.serve_forever()
        except BaseException as exc:  # pragma: no branch - diagnostic capture
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert wait_until(lambda: service.current_mode is SessionMode.ACTIVE_EXERCISE), (
        service.current_mode,
        service.last_failure_kind,
        errors,
    )
    assert monitors
    monitors[0].press()
    release_source.set()
    thread.join(1)

    assert not thread.is_alive()
    assert errors == []
    assert listener_calls == []
    assert source.opened and source.closed
    assert service.current_mode is SessionMode.STOPPED
    status = json.loads(config.status_path.read_text())
    assert status["voice"] == "silent"
    assert status["peer"] is None
    assert status["mode"] == "stopped"
    assert status["failure"] == "PhysicalStop"
    assert "landmark" not in config.status_path.read_text()
    assert "camera" not in config.status_path.read_text()


def test_local_button_stop_allows_bounded_native_read_cleanup(tmp_path: Path) -> None:
    clock = FakeClock()
    read_blocked = threading.Event()
    release_read = threading.Event()
    monitors: list[FakeStopMonitor] = []

    def read_pose(read_count: int) -> SquatAnalysis:
        if read_count == 1:
            clock.value = 0.1
            return analysis(100)
        read_blocked.set()
        assert release_read.wait(2)
        return analysis(101)

    source = FakeLocalPoseSource(read_pose)

    def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        monitor = FakeStopMonitor(config=config, on_stop=on_stop, on_status=on_status)
        monitors.append(monitor)
        return monitor

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source=PoseSourceMode.LOCAL,
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=None,
        dependencies=service_dependencies(
            clock=clock,
            session_factory=remote_pose_service._default_session_factory,
            credential_provider=lambda: None,
            stop_monitor_factory=stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            service.serve_forever()
        except BaseException as exc:  # pragma: no branch - diagnostic capture
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert wait_until(lambda: service.current_mode is SessionMode.ACTIVE_EXERCISE)
    assert read_blocked.wait(1)

    monitors[0].press()
    time.sleep(0.6)
    release_read.set()
    thread.join(2)

    assert not thread.is_alive()
    assert errors == []
    assert source.closed
    status = json.loads(config.status_path.read_text())
    assert status["failure"] == "PhysicalStop"
    assert status["service"] == "stopped"


def test_local_button_during_blocked_source_open_latches_across_release(
    tmp_path: Path,
) -> None:
    open_started = threading.Event()
    release_open = threading.Event()
    monitors: list[FakeStopMonitor] = []
    sessions: list[FakeSession] = []

    class BlockingOpenSource(FakeLocalPoseSource):
        def open(self) -> None:
            open_started.set()
            assert release_open.wait(2)
            super().open()

    source = BlockingOpenSource(lambda _: analysis(100))

    def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        monitor = FakeStopMonitor(config=config, on_stop=on_stop, on_status=on_status)
        monitors.append(monitor)
        return monitor

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del api_key, on_cue_audio
        session = FakeSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source=PoseSourceMode.LOCAL,
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=None,
        dependencies=service_dependencies(
            session_factory=session_factory,
            credential_provider=lambda: None,
            stop_monitor_factory=stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            service.serve_forever()
        except BaseException as exc:  # pragma: no branch - diagnostic capture
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert open_started.wait(1)
    assert monitors

    monitors[0].press()
    monitors[0].set_status(StopInputState.AVAILABLE)
    release_open.set()
    thread.join(2)

    assert not thread.is_alive()
    assert errors == []
    assert sessions == []
    assert source.opened and source.closed
    assert monitors[0].closed
    assert service.session_id is None
    status = json.loads(config.status_path.read_text())
    assert status["service"] == "stopped"
    assert status["failure"] == "PhysicalStop"


def test_local_button_during_session_start_prevents_candidate_install(
    tmp_path: Path,
) -> None:
    session_start_blocked = threading.Event()
    release_session_start = threading.Event()
    source_read_blocked = threading.Event()
    release_source_read = threading.Event()
    monitors: list[FakeStopMonitor] = []
    sessions: list[FakeSession] = []
    speakers: list[FakeSpeaker] = []

    def read_pose(_: int) -> SquatAnalysis:
        source_read_blocked.set()
        assert release_source_read.wait(2)
        return analysis(100)

    source = FakeLocalPoseSource(read_pose)

    class BlockingStartSession(FakeSession):
        def start(self, *, instructions: str, voice: str) -> None:
            super().start(instructions=instructions, voice=voice)
            session_start_blocked.set()
            assert release_session_start.wait(2)

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del api_key, on_cue_audio
        session = BlockingStartSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    def speaker_factory(config) -> FakeSpeaker:
        del config
        speaker = FakeSpeaker()
        speakers.append(speaker)
        return speaker

    def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        monitor = FakeStopMonitor(config=config, on_stop=on_stop, on_status=on_status)
        monitors.append(monitor)
        return monitor

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source=PoseSourceMode.LOCAL,
        status_path=tmp_path / "status.json",
        audio_enabled=True,
    )
    service = RemotePoseService(
        config,
        token=None,
        dependencies=service_dependencies(
            session_factory=session_factory,
            speaker_factory=speaker_factory,
            credential_provider=lambda: "test-api-key",
            stop_monitor_factory=stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            service.serve_forever()
        except BaseException as exc:  # pragma: no branch - diagnostic capture
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert session_start_blocked.wait(1)
    assert source_read_blocked.wait(1)
    assert monitors

    press_thread = threading.Thread(target=monitors[0].press)
    press_thread.start()
    try:
        assert wait_until(lambda: service.last_failure_kind == "PhysicalStop")
        monitors[0].set_status(StopInputState.AVAILABLE)
    finally:
        release_session_start.set()
        release_source_read.set()
    press_thread.join(2)
    thread.join(2)

    assert not press_thread.is_alive()
    assert not thread.is_alive()
    assert errors == []
    assert len(sessions) == 1
    assert sessions[0].stop_requests == 1
    assert len(speakers) == 1
    assert speakers[0].quiesced and speakers[0].closed
    assert speakers[0].preemptions >= 1
    assert source.closed
    assert service.session_id is None
    status = json.loads(config.status_path.read_text())
    assert status["service"] == "stopped"
    assert status["failure"] == "PhysicalStop"


def test_local_session_failure_is_scrubbed_and_marked_fatal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_detail = "private-patient-detail"
    release_read = threading.Event()
    source = FakeLocalPoseSource(
        lambda read_count: (
            analysis(100)
            if read_count == 1
            else (
                RuntimeError(private_detail)
                if release_read.wait(1)
                else RuntimeError("bounded-test-timeout")
            )
        )
    )

    class FailingActivationSession(FakeSession):
        def activate_exercise(
            self,
            pose: SquatAnalysis,
            *,
            pose_age_ms: int = 0,
        ) -> bool:
            del pose, pose_age_ms
            raise RuntimeError(private_detail)

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source=PoseSourceMode.LOCAL,
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=None,
        dependencies=service_dependencies(
            session_factory=lambda **kwargs: FailingActivationSession(
                preempt=kwargs["on_audio_preempt"]
            ),
            credential_provider=lambda: None,
            stop_monitor_factory=available_stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )

    release_timer = threading.Timer(0.2, release_read.set)
    release_timer.start()
    try:
        with pytest.raises(RuntimeError, match="local Guardian processing failed") as caught:
            service.serve_forever()
    finally:
        release_timer.cancel()

    assert private_detail not in str(caught.value)
    assert private_detail not in capsys.readouterr().out
    status = json.loads(config.status_path.read_text())
    assert status["service"] == "failed"
    assert status["failure"] == "LocalSessionProcessingError"


def test_local_voice_failure_during_detection_restores_one_silent_activation_arm(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    allow_fresh_pose = threading.Event()
    release_source = threading.Event()
    detection_requested = threading.Event()
    failed_session_stopped = threading.Event()
    sessions: list[FakeSession] = []
    speakers: list[FakeSpeaker] = []
    monitors: list[FakeStopMonitor] = []

    first = analysis(100)
    fresh = analysis(101)

    def read_pose(read_count: int) -> SquatAnalysis:
        if read_count == 1:
            clock.value = 0.1
            return first
        if read_count == 2:
            assert allow_fresh_pose.wait(5)
            clock.value = 0.2
            return fresh
        assert release_source.wait(5)
        return analysis(102)

    source = FakeLocalPoseSource(read_pose)

    class VoiceFailsAfterDetectionSession(FakeSession):
        def start(self, *, instructions: str, voice: str) -> None:
            super().start(instructions=instructions, voice=voice)
            self.coordinator.current_mode = SessionMode.CHECK_IN

        def activate_exercise(
            self,
            pose: SquatAnalysis,
            *,
            pose_age_ms: int = 0,
        ) -> bool:
            self.activations.append((pose, pose_age_ms))
            detection_requested.set()
            return False

        def pump_once(self):
            assert detection_requested.wait(1)
            return SimpleNamespace(failure_kind="ProviderUnavailable", end_signal=None)

        def abort_runtime(self, reason: RuntimeAbortReason) -> object:
            result = super().abort_runtime(reason)
            failed_session_stopped.set()
            return result

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del on_cue_audio
        session = (
            VoiceFailsAfterDetectionSession(preempt=on_audio_preempt)
            if api_key is not None
            else FakeSession(preempt=on_audio_preempt)
        )
        sessions.append(session)
        return session

    def speaker_factory(config) -> FakeSpeaker:
        del config
        speaker = FakeSpeaker()
        speakers.append(speaker)
        return speaker

    def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        monitor = FakeStopMonitor(
            config=config,
            on_stop=on_stop,
            on_status=on_status,
        )
        monitors.append(monitor)
        return monitor

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source=PoseSourceMode.LOCAL,
        status_path=tmp_path / "status.json",
        audio_enabled=True,
    )
    service = RemotePoseService(
        config,
        token=None,
        dependencies=service_dependencies(
            clock=clock,
            session_factory=session_factory,
            speaker_factory=speaker_factory,
            credential_provider=lambda: "test-api-key",
            stop_monitor_factory=stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            service.serve_forever()
        except BaseException as exc:  # pragma: no branch - diagnostic capture
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert detection_requested.wait(1)
    assert failed_session_stopped.wait(1)
    assert len(sessions) == 2
    assert sessions[0].activations == [(first, 100)]
    assert sessions[1].activations == []

    allow_fresh_pose.set()
    assert wait_until(lambda: service.current_mode is SessionMode.ACTIVE_EXERCISE), (
        service.current_mode,
        service.last_failure_kind,
        errors,
    )
    assert sessions[1].activations == [(fresh, 100)]
    assert sessions[1].processed == [(fresh, 100)]

    monitors[0].press()
    release_source.set()
    thread.join(1)

    assert not thread.is_alive()
    assert errors == []
    assert sessions[0].ended
    assert speakers[0].quiesced and speakers[0].closed
    assert sessions[1].stop_requests == 1
    assert service.current_mode is SessionMode.STOPPED


def test_local_pose_at_exact_watchdog_age_is_never_processed_and_exits_fatal(
    tmp_path: Path,
) -> None:
    clock = IncrementingClock(step=0.25)
    sessions: list[FakeSession] = []

    source = FakeLocalPoseSource(lambda _: analysis(100))

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del api_key, on_cue_audio
        session = FakeSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source="local",
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=None,
        dependencies=service_dependencies(
            clock=clock,
            session_factory=session_factory,
            credential_provider=lambda: None,
            stop_monitor_factory=available_stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )

    with pytest.raises(RuntimeError, match="age limit"):
        service.serve_forever()

    assert source.closed
    assert len(sessions) == 1
    assert sessions[0].activations == []
    assert sessions[0].processed == []
    assert sessions[0].stop_requests == 1
    status = json.loads(config.status_path.read_text())
    assert status["service"] == "failed"
    assert status["failure"] == "LocalPoseStale"
    assert status["age"] >= 500


def test_local_pose_no_result_pauses_then_stops_and_does_not_auto_resume(
    tmp_path: Path,
) -> None:
    clock = IncrementingClock(step=0.05)
    release_source = threading.Event()
    sessions: list[FakeSession] = []

    def read_pose(read_count: int) -> SquatAnalysis:
        if read_count == 1:
            return analysis(100)
        assert release_source.wait(5)
        return analysis(101)

    source = FakeLocalPoseSource(read_pose)

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del api_key, on_cue_audio
        session = FakeSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source=PoseSourceMode.LOCAL,
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=None,
        dependencies=service_dependencies(
            clock=clock,
            session_factory=session_factory,
            credential_provider=lambda: None,
            stop_monitor_factory=available_stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            service.serve_forever()
        except BaseException as exc:  # pragma: no branch - diagnostic capture
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert wait_until(
        lambda: service.last_failure_kind == "LocalPoseTimeout",
        timeout=2.0,
    ), (service.last_failure_kind, service.current_mode, errors, clock.value)
    release_source.set()
    thread.join(1)

    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    session = sessions[0]
    assert session.activations
    timeout_analyses = [
        pose
        for pose, _ in session.processed
        if pose.issues == (SquatAssessmentIssue.CAMERA_TIMEOUT,)
    ]
    assert len(timeout_analyses) == 1
    assert session.resumes == []
    assert session.stop_requests == 1
    assert session.coordinator.current_mode is SessionMode.STOPPED
    status = json.loads(config.status_path.read_text())
    assert status["service"] == "failed"
    assert status["failure"] == "LocalPoseTimeout"


def test_local_button_stop_precedes_a_currently_processing_pose(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    processing_started = threading.Event()
    release_processing = threading.Event()
    release_source = threading.Event()
    sessions: list[FakeSession] = []
    monitors: list[FakeStopMonitor] = []

    def read_pose(read_count: int) -> SquatAnalysis:
        if read_count == 1:
            clock.value = 0.1
            return analysis(100)
        assert release_source.wait(5)
        return analysis(101)

    source = FakeLocalPoseSource(read_pose)

    class BlockingSession(FakeSession):
        def process_analysis(
            self,
            pose: SquatAnalysis,
            *,
            pose_age_ms: int = 0,
        ) -> object:
            processing_started.set()
            assert release_processing.wait(1)
            return super().process_analysis(pose, pose_age_ms=pose_age_ms)

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del api_key, on_cue_audio
        session = BlockingSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        monitor = FakeStopMonitor(
            config=config,
            on_stop=on_stop,
            on_status=on_status,
        )
        monitors.append(monitor)
        return monitor

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source="local",
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=None,
        dependencies=service_dependencies(
            clock=clock,
            session_factory=session_factory,
            credential_provider=lambda: None,
            stop_monitor_factory=stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )
    service_errors: list[BaseException] = []

    def run_service() -> None:
        try:
            service.serve_forever()
        except BaseException as exc:  # pragma: no branch - diagnostic capture
            service_errors.append(exc)

    service_thread = threading.Thread(target=run_service)
    service_thread.start()
    assert processing_started.wait(1)

    stop_thread = threading.Thread(target=monitors[0].press)
    stop_thread.start()
    assert wait_until(lambda: service.last_failure_kind == "PhysicalStop")
    assert service.current_mode is None
    release_processing.set()
    release_source.set()
    stop_thread.join(1)
    service_thread.join(1)

    assert not stop_thread.is_alive()
    assert not service_thread.is_alive()
    assert service_errors == []
    assert service.current_mode is SessionMode.STOPPED
    session = sessions[0]
    assert len(session.processed) == 1
    assert session.stop_requests == 1
    assert session.coordinator.current_mode is SessionMode.STOPPED
    assert json.loads(config.status_path.read_text())["failure"] == "PhysicalStop"


def test_run_returns_nonzero_when_gpio_is_unavailable_before_listener(
    tmp_path: Path,
) -> None:
    listener_calls: list[str] = []

    class UnavailableMonitor(FakeStopMonitor):
        def start(self) -> None:
            self.started = True
            self.set_status(
                StopInputState.UNAVAILABLE,
                failure_kind="GPIOOpenError",
            )

    def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        return UnavailableMonitor(
            config=config,
            on_stop=on_stop,
            on_status=on_status,
        )

    output = StringIO()
    result = run_remote_pose_service(
        environment={
            "RECOVERYBOX_POSE_BIND_HOST": "100.106.237.106",
            "RECOVERYBOX_POSE_ALLOWED_PEER": "100.70.100.93",
            "RECOVERYBOX_POSE_TOKEN_FILE": str(tmp_path / "token"),
            "RECOVERYBOX_STATUS_PATH": str(tmp_path / "status.json"),
        },
        output=output,
        dependencies=service_dependencies(
            listener_factory=lambda: listener_calls.append("listener"),  # type: ignore[arg-type,return-value]
            token_loader=lambda _: TOKEN,
            stop_monitor_factory=stop_monitor_factory,
        ),
    )

    assert result == 2
    assert listener_calls == []
    assert "RuntimeError" in output.getvalue()
    assert "GPIOOpenError" not in output.getvalue()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["service"] == "failed"
    assert status["failure"] == "GPIOOpenError"


def test_button_held_at_boot_stops_before_listener_is_constructed(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    listener_calls: list[str] = []

    class HeldLine:
        def __init__(self) -> None:
            self.closed = False

        def read_active(self) -> bool:
            return True

        def wait_for_change(self, timeout_seconds: float) -> bool:
            clock.value += timeout_seconds
            return False

        def close(self) -> None:
            self.closed = True

    line = HeldLine()

    def stop_monitor_factory(*, config, on_stop, on_status) -> PhysicalStopMonitor:
        return PhysicalStopMonitor(
            config,
            on_stop=on_stop,
            on_status=on_status,
            line_factory=lambda _config: line,
            clock=clock,
        )

    result = run_remote_pose_service(
        environment={
            "RECOVERYBOX_POSE_BIND_HOST": "100.106.237.106",
            "RECOVERYBOX_POSE_ALLOWED_PEER": "100.70.100.93",
            "RECOVERYBOX_POSE_TOKEN_FILE": str(tmp_path / "token"),
            "RECOVERYBOX_STATUS_PATH": str(tmp_path / "status.json"),
        },
        output=StringIO(),
        dependencies=service_dependencies(
            listener_factory=lambda: listener_calls.append("listener"),  # type: ignore[arg-type,return-value]
            token_loader=lambda _: TOKEN,
            stop_monitor_factory=stop_monitor_factory,
        ),
    )

    assert result == 2
    assert listener_calls == []
    assert line.closed
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["service"] == "failed"


def test_run_returns_nonzero_and_sanitizes_local_source_open_failure(
    tmp_path: Path,
) -> None:
    class FailingLocalSource(FakeLocalPoseSource):
        def open(self) -> None:
            raise RuntimeError("camera path with patient details")

    source = FailingLocalSource(lambda _: analysis(100))
    output = StringIO()
    result = run_remote_pose_service(
        environment={
            "RECOVERYBOX_POSE_SOURCE": "local",
            "RECOVERYBOX_POSE_BIND_HOST": "100.106.237.106",
            "RECOVERYBOX_POSE_ALLOWED_PEER": "100.70.100.93",
            "RECOVERYBOX_POSE_TOKEN_FILE": str(tmp_path / "unused-token"),
            "RECOVERYBOX_STATUS_PATH": str(tmp_path / "status.json"),
        },
        output=output,
        dependencies=service_dependencies(
            stop_monitor_factory=available_stop_monitor_factory,
            local_pose_source_factory=lambda: source,
            token_loader=lambda _: pytest.fail("local mode loaded a token"),
        ),
    )

    assert result == 2
    assert "camera path" not in output.getvalue()
    assert "LocalPoseOpenError" not in output.getvalue()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["service"] == "failed"
    assert status["failure"] == "LocalPoseOpenError"


def test_fragmented_and_coalesced_lines_activate_with_pi_receipt_age(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    start = encoded(RemotePoseKind.START)
    mac_future_pose = analysis(9_999_999_999)
    update = encoded(
        RemotePoseKind.ANALYSIS,
        sequence=2,
        pose=mac_future_pose,
        request_sequence=1,
    )
    authorize = encoded(RemotePoseKind.RESUME, sequence=1)
    stop = encoded(RemotePoseKind.STOP, sequence=3)

    connection = harness.handle([start[:7], start[7:] + authorize + update + stop])

    assert connection.closed
    assert len(harness.sessions) == 1
    session = harness.sessions[0]
    assert session.activations == [(mac_future_pose, 0)]
    assert session.processed == [(mac_future_pose, 0)]
    challenge = decode_remote_pose_challenge(connection.sent[0], TOKEN)
    request = decode_remote_pose_request(connection.sent[1], TOKEN)
    assert challenge.service_epoch == SERVICE_EPOCH
    assert challenge.server_nonce == SERVER_NONCE
    assert request.request_sequence == 1
    assert request.request_nonce == REQUEST_NONCE
    assert session.stop_requests == 1
    assert session.coordinator.current_mode is SessionMode.STOPPED
    status = json.loads(harness.config.status_path.read_text())
    assert set(status) == {
        "service",
        "peer",
        "session",
        "mode",
        "rep",
        "age",
        "voice",
        "failure",
        "button",
    }
    assert status["session"] == SESSION_ONE
    assert status["mode"] == "stopped"
    assert status["age"] == 0
    assert status["peer"] is None
    assert status["button"] == "available"
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def test_authorized_remote_pose_activates_voice_session_from_check_in(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path, api_key="test-key")
    standing = analysis(100)

    def enter_check_in_then_send_pose() -> bytes:
        session = harness.sessions[0]
        session.coordinator.current_mode = SessionMode.CHECK_IN
        return (
            encoded(RemotePoseKind.RESUME, sequence=1)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=standing,
                request_sequence=1,
            )
            + encoded(RemotePoseKind.STOP, sequence=3)
        )

    harness.handle(
        [
            encoded(RemotePoseKind.START),
            enter_check_in_then_send_pose,
        ]
    )

    assert harness.sessions[0].activations == [(standing, 0)]


def test_detection_playback_completion_continues_initial_resume_to_fresh_pose(
    tmp_path: Path,
) -> None:
    sessions: list[FakeSession] = []
    speakers: list[FakeSpeaker] = []

    class PlaybackGatedSession(FakeSession):
        def __init__(self, *, preempt) -> None:
            super().__init__(preempt=preempt)
            self.detection_requested = False
            self.detection_played = False
            self.notifications: list[CueId] = []

        def start(self, *, instructions: str, voice: str) -> None:
            super().start(instructions=instructions, voice=voice)
            self.coordinator.current_mode = SessionMode.CHECK_IN

        def activate_exercise(
            self,
            pose: SquatAnalysis,
            *,
            pose_age_ms: int = 0,
        ) -> bool:
            self.activations.append((pose, pose_age_ms))
            if not self.detection_requested:
                self.detection_requested = True
                return False
            if not self.detection_played:
                return False
            self.coordinator.current_mode = SessionMode.ACTIVE_EXERCISE
            return True

        def notify_cue_playback_succeeded(self, cue_id: CueId) -> bool:
            self.notifications.append(cue_id)
            if cue_id is not CueId.SQUAT_PERSON_DETECTED or not self.detection_requested:
                return False
            self.detection_played = True
            return True

    class PlaybackReportingSpeaker(FakeSpeaker):
        def __init__(self) -> None:
            super().__init__()
            self.success_callback = None

        def bind_playback_succeeded_callback(self, callback) -> None:
            assert self.success_callback is None
            self.success_callback = callback

        def complete(self, cue_id: CueId) -> None:
            assert self.success_callback is not None
            self.success_callback(cue_id)

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del on_cue_audio
        assert api_key == "test-api-key"
        session = PlaybackGatedSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    def speaker_factory(config) -> FakeSpeaker:
        del config
        speaker = PlaybackReportingSpeaker()
        speakers.append(speaker)
        return speaker

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
        audio_enabled=True,
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            clock=FakeClock(),
            session_factory=session_factory,
            speaker_factory=speaker_factory,
            credential_provider=lambda: "test-api-key",
            stop_monitor_factory=available_stop_monitor_factory,
        ),
    )
    service.start_local_stop_monitor()
    first = analysis(100)
    fresh = analysis(101)

    def complete_detection_then_send_fresh_pose() -> bytes:
        speaker = speakers[0]
        assert isinstance(speaker, PlaybackReportingSpeaker)
        speaker.complete(CueId.SQUAT_PERSON_DETECTED)
        return encoded(
            RemotePoseKind.ANALYSIS,
            sequence=3,
            pose=fresh,
            request_sequence=2,
        ) + encoded(RemotePoseKind.STOP, sequence=4)

    service.handle_connection(
        FakeConnection(
            [
                encoded(RemotePoseKind.START)
                + encoded(RemotePoseKind.RESUME, sequence=1)
                + encoded(
                    RemotePoseKind.ANALYSIS,
                    sequence=2,
                    pose=first,
                    request_sequence=1,
                ),
                complete_detection_then_send_fresh_pose,
            ]
        ),
        peer_host=config.allowed_peer,
    )

    session = sessions[0]
    assert isinstance(session, PlaybackGatedSession)
    assert session.notifications == [CueId.SQUAT_PERSON_DETECTED]
    assert session.activations == [(first, 0), (fresh, 0)]
    assert session.processed == [(first, 0), (fresh, 0)]
    assert session.stop_requests == 1
    service.shutdown()


def test_plain_fresh_start_and_standing_pose_cannot_activate_without_resume(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    first = analysis(100)
    authorized = analysis(101)

    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=1,
                pose=first,
                request_sequence=1,
            )
            + encoded(RemotePoseKind.RESUME, sequence=2)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=3,
                pose=authorized,
                request_sequence=2,
            )
            + encoded(RemotePoseKind.STOP, sequence=4)
        ]
    )

    session = harness.sessions[0]
    assert session.activations == [(authorized, 0)]
    assert session.processed[0] == (first, 0)
    assert session.stop_requests == 1


def test_request_round_trip_at_exact_watchdog_limit_is_withheld(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    harness = Harness(tmp_path, clock=clock)
    standing = analysis(100)

    def delayed_response() -> bytes:
        clock.value = 0.5
        return encoded(RemotePoseKind.RESUME, sequence=1) + encoded(
            RemotePoseKind.ANALYSIS,
            sequence=2,
            pose=standing,
            request_sequence=1,
        )

    harness.handle([encoded(RemotePoseKind.START), delayed_response, b""])

    session = harness.sessions[0]
    assert session.activations == []
    assert session.processed == []
    assert session.resumes == []
    status = json.loads(harness.config.status_path.read_text())
    assert status["age"] >= 500


def test_request_round_trip_one_millisecond_below_watchdog_is_accepted(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    harness = Harness(tmp_path, clock=clock)
    standing = analysis(100)

    def fresh_response() -> bytes:
        clock.value = 0.499
        return (
            encoded(RemotePoseKind.RESUME, sequence=1)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=standing,
                request_sequence=1,
            )
            + encoded(RemotePoseKind.STOP, sequence=3)
        )

    harness.handle([encoded(RemotePoseKind.START), fresh_response])

    session = harness.sessions[0]
    assert session.activations == [(standing, 499)]
    assert session.processed == [(standing, 499)]
    assert session.stop_requests == 1


def test_stale_response_consumes_resume_and_next_fresh_pose_cannot_resume(
    tmp_path: Path,
) -> None:
    clock = IncrementingClock(step=0.001)
    harness = Harness(tmp_path, clock=clock)
    initial = analysis(100)
    missing = analysis(101, assessable=False)
    stale = analysis(102)
    later = analysis(103)
    recovered = analysis(104)

    harness.handle(
        [
            encoded(RemotePoseKind.START) + encoded(RemotePoseKind.RESUME, sequence=1),
            encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=initial,
                request_sequence=1,
            ),
            encoded(
                RemotePoseKind.ANALYSIS,
                sequence=3,
                pose=missing,
                request_sequence=2,
            ),
            encoded(RemotePoseKind.RESUME, sequence=4)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=5,
                pose=stale,
                evidence_age_ms=499,
                request_sequence=3,
            ),
            encoded(
                RemotePoseKind.ANALYSIS,
                sequence=6,
                pose=later,
                request_sequence=4,
            )
            + encoded(RemotePoseKind.RESUME, sequence=7),
            encoded(
                RemotePoseKind.ANALYSIS,
                sequence=8,
                pose=recovered,
                request_sequence=5,
            )
            + encoded(RemotePoseKind.STOP, sequence=9),
        ]
    )

    session = harness.sessions[0]
    assert len(session.activations) == 1
    assert session.activations[0][0] == initial
    assert session.activations[0][1] < 500
    assert len(session.resumes) == 1
    assert session.resumes[0][0].timestamp_ms == recovered.timestamp_ms
    assert session.resumes[0][1] < 500
    assert all(pose.timestamp_ms != stale.timestamp_ms for pose, _ in session.processed)
    assert any(pose.timestamp_ms == later.timestamp_ms for pose, _ in session.processed)
    assert session.coordinator.current_mode is SessionMode.STOPPED


def test_new_session_id_inherits_pause_and_requires_its_own_resume(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path, api_key="test-api-key")
    initial = analysis(100)
    missing = analysis(101, assessable=False)
    unauthorized = analysis(200)
    authorized = analysis(201)

    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.RESUME, sequence=1)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=initial,
                request_sequence=1,
            )
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=3,
                pose=missing,
                request_sequence=2,
            ),
            b"",
        ]
    )
    assert harness.sessions[0].coordinator.current_mode is SessionMode.PAUSED
    assert not harness.speakers[0].quiesced

    harness.handle(
        [
            encoded(RemotePoseKind.START, session_id=SESSION_TWO)
            + encoded(
                RemotePoseKind.ANALYSIS,
                session_id=SESSION_TWO,
                sequence=1,
                pose=unauthorized,
                request_sequence=4,
            )
            + encoded(
                RemotePoseKind.RESUME,
                session_id=SESSION_TWO,
                sequence=2,
            )
            + encoded(
                RemotePoseKind.ANALYSIS,
                session_id=SESSION_TWO,
                sequence=3,
                pose=authorized,
                request_sequence=5,
            )
            + encoded(
                RemotePoseKind.STOP,
                session_id=SESSION_TWO,
                sequence=4,
            )
        ]
    )

    replacement = harness.sessions[1]
    assert harness.speakers[0].quiesced
    assert harness.speakers[0].closed
    assert replacement.activations == []
    assert replacement.resumes == [(authorized, 0)]
    assert replacement.processed[0] == (unauthorized, 0)
    assert replacement.stop_requests == 1


def test_signed_remote_path_drives_real_guardian_pause_and_explicit_resume(
    tmp_path: Path,
) -> None:
    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            clock=FakeClock(),
            session_factory=remote_pose_service._default_session_factory,
            speaker_factory=lambda _: pytest.fail(
                "audio-disabled remote mode constructed a speaker"
            ),
            credential_provider=lambda: pytest.fail("audio-disabled remote mode read credentials"),
            stop_monitor_factory=available_stop_monitor_factory,
        ),
    )
    service.start_local_stop_monitor()
    initial = analysis(100)
    missing = analysis(101, assessable=False)
    unauthorized = analysis(102)
    recovered = analysis(103)

    def pause_after_active() -> bytes:
        assert service.current_mode is SessionMode.ACTIVE_EXERCISE
        return encoded(
            RemotePoseKind.ANALYSIS,
            sequence=3,
            pose=missing,
            request_sequence=2,
        )

    def fresh_without_resume() -> bytes:
        assert service.current_mode is SessionMode.PAUSED
        return encoded(
            RemotePoseKind.ANALYSIS,
            sequence=4,
            pose=unauthorized,
            request_sequence=3,
        )

    def explicitly_resume() -> bytes:
        assert service.current_mode is SessionMode.PAUSED
        return encoded(RemotePoseKind.RESUME, sequence=5) + encoded(
            RemotePoseKind.ANALYSIS,
            sequence=6,
            pose=recovered,
            request_sequence=4,
        )

    def stop_after_resume() -> bytes:
        assert service.current_mode is SessionMode.ACTIVE_EXERCISE
        return encoded(RemotePoseKind.STOP, sequence=7)

    connection = FakeConnection(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.RESUME, sequence=1)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=initial,
                request_sequence=1,
            ),
            pause_after_active,
            fresh_without_resume,
            explicitly_resume,
            stop_after_resume,
        ]
    )
    service.handle_connection(connection, peer_host=config.allowed_peer)
    assert service.wait_for_status()

    assert connection.closed
    assert service.current_mode is SessionMode.STOPPED
    status = json.loads(config.status_path.read_text())
    assert status["mode"] == "stopped"
    assert status["voice"] == "silent"
    service.shutdown()


def test_watchdog_injects_one_nonassessable_timeout_and_preempts(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    harness = Harness(tmp_path, clock=clock, api_key="test-api-key")

    def expire() -> bytes:
        clock.value = 0.5
        raise TimeoutError

    standing = analysis(123)
    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.RESUME, sequence=1)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=standing,
                request_sequence=1,
            ),
            expire,
            TimeoutError(),
            b"",
        ]
    )

    session = harness.sessions[0]
    assert session.coordinator.current_mode is SessionMode.PAUSED
    assert len(session.processed) == 2
    timeout_pose, timeout_age = session.processed[-1]
    assert timeout_pose.assessable is False
    assert timeout_pose.phase is SquatPhase.UNKNOWN
    assert timeout_pose.issues == (SquatAssessmentIssue.CAMERA_TIMEOUT,)
    assert timeout_pose.events == ()
    assert timeout_pose.timestamp_ms == standing.timestamp_ms
    assert timeout_age == 501
    assert harness.speakers[0].preemptions == 1
    assert harness.session_api_keys == ["test-api-key"]
    assert harness.speaker_configs[0].playback_device == "default"


def test_duplicate_traffic_cannot_starve_the_pose_watchdog(tmp_path: Path) -> None:
    clock = FakeClock()
    harness = Harness(tmp_path, clock=clock)
    pose_line = encoded(
        RemotePoseKind.ANALYSIS,
        sequence=2,
        pose=analysis(100),
        request_sequence=1,
    )

    def stale_duplicate() -> bytes:
        clock.value = 0.5
        return pose_line

    harness.handle(
        [
            encoded(RemotePoseKind.START) + encoded(RemotePoseKind.RESUME, sequence=1) + pose_line,
            stale_duplicate,
            b"",
        ]
    )

    session = harness.sessions[0]
    assert session.coordinator.current_mode is SessionMode.PAUSED
    assert len(session.processed) == 2
    assert session.processed[-1][0].issues == (SquatAssessmentIssue.CAMERA_TIMEOUT,)
    assert session.processed[-1][1] == 501


def test_unterminated_byte_drip_cannot_starve_the_pose_watchdog(tmp_path: Path) -> None:
    clock = FakeClock()
    harness = Harness(tmp_path, clock=clock)

    def stale_partial_line() -> bytes:
        clock.value = 0.5
        return b'{"unterminated"'

    def later_drip() -> bytes:
        clock.value = 1.0
        return b" "

    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.RESUME, sequence=1)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=analysis(100),
                request_sequence=1,
            ),
            stale_partial_line,
            later_drip,
        ]
    )

    session = harness.sessions[0]
    assert session.coordinator.current_mode is SessionMode.PAUSED
    assert len(session.processed) == 2
    assert session.processed[-1][0].issues == (SquatAssessmentIssue.CAMERA_TIMEOUT,)
    assert session.processed[-1][1] == 501
    assert harness.service.last_failure_kind == "RemotePoseProtocolError"


def test_slow_session_start_cannot_activate_a_stale_coalesced_pose(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    sessions: list[FakeSession] = []

    class SlowStartSession(FakeSession):
        def start(self, *, instructions: str, voice: str) -> None:
            super().start(instructions=instructions, voice=voice)
            clock.value = 0.6

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del api_key, on_cue_audio
        session = SlowStartSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            listener_factory=lambda: None,  # type: ignore[arg-type,return-value]
            clock=clock,
            session_factory=session_factory,
            speaker_factory=lambda _: FakeSpeaker(),
            credential_provider=lambda: None,
            token_loader=lambda _: TOKEN,
            stop_monitor_factory=available_stop_monitor_factory,
        ),
    )
    service.start_local_stop_monitor()
    stale = analysis(100)

    service.handle_connection(
        FakeConnection(
            [
                encoded(RemotePoseKind.START)
                + encoded(RemotePoseKind.ANALYSIS, sequence=1, pose=stale)
                + encoded(RemotePoseKind.STOP, sequence=2)
            ]
        ),
        peer_host=config.allowed_peer,
    )

    session = sessions[0]
    assert session.activations == []
    assert session.processed == []
    assert session.stop_requests == 0
    assert service.last_failure_kind == "PoseResponseBeforeRequest"


def test_blocked_status_storage_cannot_delay_guardian_timeout(tmp_path: Path) -> None:
    clock = FakeClock()
    writer_entered = threading.Event()
    writer_release = threading.Event()
    sessions: list[FakeSession] = []

    def blocked_writer(path: Path, status) -> None:
        del path, status
        writer_entered.set()
        assert writer_release.wait(1)

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del api_key, on_cue_audio
        session = FakeSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            listener_factory=lambda: None,  # type: ignore[arg-type,return-value]
            clock=clock,
            session_factory=session_factory,
            speaker_factory=lambda _: FakeSpeaker(),
            credential_provider=lambda: None,
            token_loader=lambda _: TOKEN,
            status_file_writer=blocked_writer,
            stop_monitor_factory=available_stop_monitor_factory,
        ),
    )
    service.start_local_stop_monitor()

    def expire() -> bytes:
        clock.value = 0.5
        raise TimeoutError

    service.handle_connection(
        FakeConnection(
            [
                encoded(RemotePoseKind.START)
                + encoded(RemotePoseKind.RESUME, sequence=1)
                + encoded(
                    RemotePoseKind.ANALYSIS,
                    sequence=2,
                    pose=analysis(100),
                    request_sequence=1,
                ),
                expire,
                b"",
            ]
        ),
        peer_host=config.allowed_peer,
    )

    assert writer_entered.wait(1)
    assert sessions[0].coordinator.current_mode is SessionMode.PAUSED
    writer_release.set()
    assert service.wait_for_status()


def test_forced_status_heartbeat_rewrites_an_identical_snapshot(tmp_path: Path) -> None:
    writes: list[dict[str, object]] = []
    writer = _CoalescingStatusWriter(
        tmp_path / "status.json",
        lambda path, status: writes.append(dict(status)),
    )
    snapshot = {"service": "listening", "voice": "silent"}

    writer.publish(snapshot)
    assert writer.wait_idle(1)
    writer.publish(snapshot)
    assert writer.wait_idle(1)
    writer.publish(snapshot, force=True)
    assert writer.wait_idle(1)
    writer.close()

    assert writes == [snapshot, snapshot]


def test_connected_stream_forces_heartbeat_while_status_is_unchanged(
    tmp_path: Path,
) -> None:
    status_clock = FakeClock()

    class RecordingService(RemotePoseService):
        def __init__(self, *args, **kwargs) -> None:
            self.forced_publications = 0
            super().__init__(*args, **kwargs)

        def _publish_status(self, *, force: bool = False) -> None:
            if force:
                self.forced_publications += 1
            super()._publish_status(force=force)

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
    )
    service = RecordingService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            clock=FakeClock(),
            status_clock=status_clock,
            session_factory=lambda **_: FakeSession(preempt=lambda: None),
            credential_provider=lambda: None,
            stop_monitor_factory=available_stop_monitor_factory,
        ),
    )
    service.start_local_stop_monitor()

    def heartbeat(at: float):
        def expire() -> bytes:
            status_clock.value = at
            raise TimeoutError

        return expire

    service.handle_connection(
        FakeConnection(
            [
                encoded(RemotePoseKind.START),
                heartbeat(1.0),
                heartbeat(2.0),
                b"",
            ]
        ),
        peer_host=config.allowed_peer,
    )
    assert service.wait_for_status()

    assert service.forced_publications == 2
    service.shutdown()


def test_local_pose_forces_heartbeat_while_status_is_unchanged(tmp_path: Path) -> None:
    status_clock = FakeClock()
    allow_first_read = threading.Event()
    first_heartbeat = threading.Event()
    second_heartbeat = threading.Event()
    release_source = threading.Event()
    monitors: list[FakeStopMonitor] = []

    class RecordingService(RemotePoseService):
        def __init__(self, *args, **kwargs) -> None:
            self.forced_publications = 0
            super().__init__(*args, **kwargs)

        def _publish_status(self, *, force: bool = False) -> None:
            super()._publish_status(force=force)
            if force:
                self.forced_publications += 1
                (first_heartbeat if self.forced_publications == 1 else second_heartbeat).set()
            elif getattr(self, "_service_state", None) == "local":
                allow_first_read.set()

    def read_pose(read_count: int) -> SquatAnalysis:
        if read_count == 1:
            assert allow_first_read.wait(1)
            status_clock.value = 1.0
            return analysis(100, assessable=False)
        if read_count == 2:
            assert first_heartbeat.wait(1)
            status_clock.value = 2.0
            return analysis(101, assessable=False)
        assert release_source.wait(2)
        return analysis(102, assessable=False)

    source = FakeLocalPoseSource(read_pose)

    def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        monitor = FakeStopMonitor(config=config, on_stop=on_stop, on_status=on_status)
        monitors.append(monitor)
        return monitor

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source=PoseSourceMode.LOCAL,
        status_path=tmp_path / "status.json",
    )
    service = RecordingService(
        config,
        token=None,
        dependencies=service_dependencies(
            clock=FakeClock(),
            status_clock=status_clock,
            session_factory=remote_pose_service._default_session_factory,
            credential_provider=lambda: None,
            stop_monitor_factory=stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            service.serve_forever()
        except BaseException as exc:  # pragma: no branch - diagnostic capture
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert second_heartbeat.wait(2)

    monitors[0].press()
    release_source.set()
    thread.join(2)

    assert not thread.is_alive()
    assert errors == []
    assert service.forced_publications == 2


def test_timeout_processing_failure_uses_guardian_escalation_then_runtime_abort(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    sessions: list[FakeSession] = []

    class FailingPauseSession(FakeSession):
        def process_analysis(self, pose: SquatAnalysis, *, pose_age_ms: int = 0) -> object:
            if pose.issues == (SquatAssessmentIssue.CAMERA_TIMEOUT,):
                self.processed.append((pose, pose_age_ms))
                raise RuntimeError("secret processing detail")
            return super().process_analysis(pose, pose_age_ms=pose_age_ms)

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del api_key, on_cue_audio
        session = FailingPauseSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            listener_factory=lambda: None,  # type: ignore[arg-type,return-value]
            clock=clock,
            session_factory=session_factory,
            speaker_factory=lambda _: FakeSpeaker(),
            credential_provider=lambda: None,
            token_loader=lambda _: TOKEN,
            stop_monitor_factory=available_stop_monitor_factory,
        ),
    )
    service.start_local_stop_monitor()

    def expire() -> bytes:
        clock.value = 0.5
        raise TimeoutError

    service.handle_connection(
        FakeConnection(
            [
                encoded(RemotePoseKind.START)
                + encoded(RemotePoseKind.RESUME, sequence=1)
                + encoded(
                    RemotePoseKind.ANALYSIS,
                    sequence=2,
                    pose=analysis(100),
                    request_sequence=1,
                ),
                expire,
                b"",
            ]
        ),
        peer_host=config.allowed_peer,
    )
    assert service.wait_for_status()

    assert sessions[0].ended
    assert sessions[0].coordinator.current_mode is SessionMode.STOPPED
    assert sessions[0].runtime_faults == [GuardianRuntimeFault.SAFETY_ENFORCEMENT_FAILURE]
    assert sessions[0].abort_requests == [RuntimeAbortReason.SAFETY_ENFORCEMENT_FAILURE]
    assert sessions[0].physical_stop_requests == 0
    status_text = config.status_path.read_text()
    assert "SessionProcessingError" in status_text
    assert "secret processing detail" not in status_text


def test_resume_is_one_shot_and_requires_the_next_pose_to_be_standing(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    harness = Harness(tmp_path, clock=clock)

    def expire() -> bytes:
        clock.value = 0.5
        raise TimeoutError

    down = analysis(200, phase=SquatPhase.DOWN)
    standing = analysis(201)
    later_standing = analysis(202)
    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.RESUME, sequence=1)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=analysis(100),
                request_sequence=1,
            ),
            expire,
        ]
    )
    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.RESUME, sequence=3)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=4,
                pose=down,
                request_sequence=3,
            )
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=5,
                pose=standing,
                request_sequence=4,
            )
            + encoded(RemotePoseKind.RESUME, sequence=6)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=7,
                pose=later_standing,
                request_sequence=5,
            )
            + encoded(RemotePoseKind.STOP, sequence=8),
        ]
    )

    session = harness.sessions[0]
    assert session.resumes == [(down, 0), (later_standing, 0)]
    assert session.stop_requests == 1
    assert session.coordinator.current_mode is SessionMode.STOPPED


def test_reconnect_preserves_session_but_rejects_old_request_response(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    first_pose = analysis(10)
    first_line = encoded(
        RemotePoseKind.ANALYSIS,
        sequence=2,
        pose=first_pose,
        request_sequence=1,
    )
    harness.handle(
        [
            encoded(RemotePoseKind.START) + encoded(RemotePoseKind.RESUME, sequence=1) + first_line,
            b"",
        ]
    )

    session = harness.sessions[0]
    assert session.coordinator.current_mode is SessionMode.PAUSED
    assert len(session.processed) == 2

    harness.handle([encoded(RemotePoseKind.START) + first_line])
    assert harness.service.last_failure_kind == "PoseRequestMismatch"

    fresh = analysis(20)
    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.RESUME, sequence=3)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=4,
                pose=fresh,
                request_sequence=4,
            )
            + encoded(RemotePoseKind.STOP, sequence=5)
        ]
    )

    assert len(harness.sessions) == 1
    assert session.resumes == [(fresh, 0)]
    assert sum(pose.timestamp_ms == first_pose.timestamp_ms for pose, _ in session.processed) == 2
    assert session.stop_requests == 1


def test_conflicting_duplicate_is_rejected_and_cannot_reprocess_pose(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    first = analysis(10)
    conflicting = analysis(11)
    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.RESUME, sequence=1)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=first,
                request_sequence=1,
            )
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=conflicting,
                request_sequence=2,
            )
        ]
    )

    session = harness.sessions[0]
    assert all(pose is not conflicting for pose, _ in session.processed)
    assert session.coordinator.current_mode is SessionMode.PAUSED
    status_text = harness.config.status_path.read_text()
    assert "SequenceConflict" in status_text
    assert "timestamp_ms" not in status_text


def test_stop_then_new_session_id_constructs_a_fresh_idle_lane(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.handle(
        [
            encoded(RemotePoseKind.START, session_id=SESSION_ONE)
            + encoded(RemotePoseKind.STOP, session_id=SESSION_ONE, sequence=1)
        ]
    )
    first = harness.sessions[0]
    harness.handle(
        [
            encoded(RemotePoseKind.START, session_id=SESSION_TWO)
            + encoded(RemotePoseKind.STOP, session_id=SESSION_TWO, sequence=1)
        ]
    )

    assert len(harness.sessions) == 2
    assert first.ended
    assert harness.sessions[1] is not first
    assert harness.service.session_id == SESSION_TWO


def test_bad_authenticator_and_wrong_peer_create_no_session_or_speaker(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path, api_key="must-not-leak")
    signed = bytearray(encoded(RemotePoseKind.START))
    signature_index = signed.find(b'"hmac":"') + len(b'"hmac":"')
    signed[signature_index] = ord("0") if signed[signature_index] != ord("0") else ord("1")
    bad = FakeConnection([bytes(signed)])
    harness.service.handle_connection(bad, peer_host=harness.config.allowed_peer)

    wrong_peer = FakeConnection([encoded(RemotePoseKind.START)])
    harness.service.handle_connection(wrong_peer, peer_host="100.70.100.94")
    assert harness.service.wait_for_status()

    assert harness.sessions == []
    assert harness.speakers == []
    status_text = harness.config.status_path.read_text()
    assert "must-not-leak" not in status_text
    assert "hmac" not in status_text
    assert "landmark" not in status_text


def test_no_api_key_never_constructs_native_speaker(tmp_path: Path) -> None:
    harness = Harness(tmp_path, api_key=None)
    harness.handle([encoded(RemotePoseKind.START) + encoded(RemotePoseKind.STOP, sequence=1)])

    assert harness.speakers == []
    assert harness.session_api_keys == [None]
    status = json.loads(harness.config.status_path.read_text())
    assert status["voice"] == "silent"


def test_audio_disabled_never_reads_credential_connects_or_constructs_speaker(
    tmp_path: Path,
) -> None:
    sessions: list[FakeSession] = []
    calls: list[str] = []

    def credential_provider() -> str:
        calls.append("credential")
        return "must-not-be-read"

    def speaker_factory(config):
        del config
        calls.append("speaker")
        raise AssertionError("disabled audio constructed a speaker")

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del on_cue_audio
        assert api_key is None
        session = FakeSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
        audio_enabled=False,
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            listener_factory=lambda: None,  # type: ignore[arg-type,return-value]
            clock=FakeClock(),
            session_factory=session_factory,
            speaker_factory=speaker_factory,
            credential_provider=credential_provider,
            token_loader=lambda _: TOKEN,
            stop_monitor_factory=available_stop_monitor_factory,
        ),
    )
    service.start_local_stop_monitor()

    service.handle_connection(
        FakeConnection([encoded(RemotePoseKind.START) + encoded(RemotePoseKind.STOP, sequence=1)]),
        peer_host=config.allowed_peer,
    )
    assert service.wait_for_status()

    assert calls == []
    assert len(sessions) == 1
    assert sessions[0].stop_requests == 1
    status = json.loads(config.status_path.read_text())
    assert status["voice"] == "silent"


def test_activation_is_blocked_until_physical_stop_input_is_available(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    monitor = harness.stop_monitors[0]
    first = analysis(100)
    monitor.set_status(StopInputState.STARTING)

    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.RESUME, sequence=1)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=2,
                pose=first,
                request_sequence=1,
            ),
            b"",
        ]
    )

    session = harness.sessions[0]
    assert session.activations == []
    assert session.processed == []
    assert session.coordinator.current_mode is SessionMode.IDLE

    monitor.set_status(StopInputState.AVAILABLE)
    fresh = analysis(101)
    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.RESUME, sequence=3)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=4,
                pose=fresh,
                request_sequence=3,
            )
            + encoded(RemotePoseKind.STOP, sequence=5)
        ]
    )

    assert session.activations == [(fresh, 0)]
    assert session.stop_requests == 1


def test_button_press_preempts_and_retires_current_session_once(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path, api_key="test-api-key")
    monitor = harness.stop_monitors[0]

    def press_button() -> bytes:
        monitor.press()
        return b""

    harness.handle(
        [
            encoded(RemotePoseKind.START)
            + encoded(RemotePoseKind.ANALYSIS, sequence=1, pose=analysis(100)),
            press_button,
        ]
    )

    first = harness.sessions[0]
    speaker = harness.speakers[0]
    assert first.stop_requests == 1
    assert speaker.preemptions >= 1
    assert speaker.closed
    assert harness.service.current_mode is SessionMode.STOPPED
    status = json.loads(harness.config.status_path.read_text())
    assert status["button"] == "pressed"
    assert status["failure"] == "PhysicalStop"

    # The stopped workout cannot be resurrected, while release permits a new
    # session ID without retrying the first session's stop edge.
    monitor.set_status(StopInputState.AVAILABLE)
    harness.handle(
        [
            encoded(RemotePoseKind.START, session_id=SESSION_TWO)
            + encoded(
                RemotePoseKind.STOP,
                session_id=SESSION_TWO,
                sequence=1,
            )
        ]
    )
    assert first.stop_requests == 1
    assert len(harness.sessions) == 2


def test_gpio_loss_is_process_terminal_and_rejects_remote_start(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.stop_monitors[0].lose_input("GPIOReadError")

    connection = harness.handle([encoded(RemotePoseKind.START)])

    assert connection.closed
    assert harness.sessions == []
    status = json.loads(harness.config.status_path.read_text())
    assert status["service"] == "failed"
    assert status["button"] == "unavailable"
    assert status["failure"] == "GPIOReadError"


def test_gpio_failure_status_cannot_expose_exception_content(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.stop_monitors[0].lose_input("patient-specific hardware detail")
    assert harness.service.wait_for_status()

    status_text = harness.config.status_path.read_text()
    assert "patient-specific" not in status_text
    assert json.loads(status_text)["failure"] == "GPIOInputUnavailable"


def test_local_stop_failure_does_not_synthesize_stopped_mode(tmp_path: Path) -> None:
    sessions: list[FakeSession] = []
    speakers: list[FakeSpeaker] = []
    monitors: list[FakeStopMonitor] = []

    class FailingStopSession(FakeSession):
        def request_physical_stop(self) -> object:
            self.stop_requests += 1
            raise RuntimeError("private stop failure")

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del api_key, on_cue_audio
        session = FailingStopSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    def speaker_factory(config) -> FakeSpeaker:
        del config
        speaker = FakeSpeaker()
        speakers.append(speaker)
        return speaker

    def monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        monitor = FakeStopMonitor(
            config=config,
            on_stop=on_stop,
            on_status=on_status,
        )
        monitors.append(monitor)
        return monitor

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
        audio_enabled=True,
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            listener_factory=lambda: None,  # type: ignore[arg-type,return-value]
            clock=FakeClock(),
            session_factory=session_factory,
            speaker_factory=speaker_factory,
            credential_provider=lambda: "test-api-key",
            token_loader=lambda _: TOKEN,
            stop_monitor_factory=monitor_factory,
        ),
    )
    service.start_local_stop_monitor()
    service.handle_connection(
        FakeConnection([encoded(RemotePoseKind.START)]),
        peer_host=config.allowed_peer,
    )

    monitors[0].press()
    service.request_local_stop()
    assert service.wait_for_status()

    assert sessions[0].stop_requests == 1
    assert speakers[0].preemptions >= 1
    assert speakers[0].closed
    assert service.current_mode is SessionMode.IDLE
    status_text = config.status_path.read_text()
    assert "SessionStopError" in status_text
    assert '"mode":"idle"' in status_text
    assert '"mode":"stopped"' not in status_text
    assert "private stop failure" not in status_text


def test_monitor_closes_before_shutdown_requests_session_stop(tmp_path: Path) -> None:
    events: list[str] = []
    sessions: list[FakeSession] = []

    class OrderedStopSession(FakeSession):
        def abort_runtime(self, reason: RuntimeAbortReason) -> object:
            events.append("session-abort")
            return super().abort_runtime(reason)

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del api_key, on_cue_audio
        session = OrderedStopSession(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    def monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        return FakeStopMonitor(
            config=config,
            on_stop=on_stop,
            on_status=on_status,
            close_event=events,
        )

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            listener_factory=lambda: None,  # type: ignore[arg-type,return-value]
            clock=FakeClock(),
            session_factory=session_factory,
            speaker_factory=lambda _: FakeSpeaker(),
            credential_provider=lambda: None,
            token_loader=lambda _: TOKEN,
            stop_monitor_factory=monitor_factory,
        ),
    )
    service.start_local_stop_monitor()
    service.handle_connection(
        FakeConnection([encoded(RemotePoseKind.START)]),
        peer_host=config.allowed_peer,
    )

    service.shutdown()

    assert events == ["monitor-close", "session-abort"]
    assert sessions[0].stop_requests == 1
    assert sessions[0].abort_requests == [RuntimeAbortReason.SERVICE_SHUTDOWN]
    assert sessions[0].physical_stop_requests == 0


def test_persistent_listener_accept_error_is_terminal(tmp_path: Path) -> None:
    listener = FailingAcceptListener()
    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            listener_factory=lambda: listener,
            clock=FakeClock(),
            session_factory=lambda **kwargs: FakeSession(preempt=kwargs["on_audio_preempt"]),
            speaker_factory=lambda _: FakeSpeaker(),
            credential_provider=lambda: None,
            token_loader=lambda _: TOKEN,
            stop_monitor_factory=available_stop_monitor_factory,
        ),
    )

    with pytest.raises(RuntimeError, match="accept failed"):
        service.serve_forever()

    assert listener.closed
    assert service.last_failure_kind == "AcceptError"


def test_bounded_realtime_handshake_restores_blocking_idle_receive(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class Connection:
        def set_receive_timeout(self, timeout: float | None) -> None:
            calls["receive_timeout"] = timeout

        def close(self) -> None:
            calls["closed"] = True

    connection = Connection()

    def connect(*, api_key: str, timeout_seconds: float):
        calls["api_key"] = api_key
        calls["connect_timeout"] = timeout_seconds
        return connection

    def session_factory(**kwargs):
        calls["session_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        remote_pose_service.WebSocketJsonTransport,
        "connect",
        staticmethod(connect),
    )
    monkeypatch.setattr(
        remote_pose_service,
        "BoundedOrderedTransport",
        lambda selected: ("bounded", selected),
    )
    monkeypatch.setattr(remote_pose_service, "LaptopSquatSession", session_factory)

    result = remote_pose_service._default_session_factory(
        api_key="test-key",
        on_cue_audio=lambda _: None,
        on_audio_preempt=lambda: None,
    )

    assert result is not None
    assert calls["connect_timeout"] == 5.0
    assert calls["receive_timeout"] is None
    assert calls["session_kwargs"]["transport"] == ("bounded", connection)  # type: ignore[index]


def test_provider_failure_before_activation_rebuilds_a_local_idle_lane(
    tmp_path: Path,
) -> None:
    failure_reported = threading.Event()
    failed_session_stopped = threading.Event()
    sessions: list[FakeSession] = []
    speakers: list[FakeSpeaker] = []

    class FailingPumpSession(FakeSession):
        def pump_once(self):
            failure_reported.set()
            return SimpleNamespace(failure_kind="ProviderUnavailable", end_signal=None)

        def abort_runtime(self, reason: RuntimeAbortReason) -> object:
            result = super().abort_runtime(reason)
            failed_session_stopped.set()
            return result

    def speaker_factory(config) -> FakeSpeaker:
        del config
        speaker = FakeSpeaker()
        speakers.append(speaker)
        return speaker

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        del on_cue_audio
        session_type = FailingPumpSession if api_key is not None else FakeSession
        session = session_type(preempt=on_audio_preempt)
        sessions.append(session)
        return session

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
        audio_enabled=True,
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            listener_factory=lambda: None,  # type: ignore[arg-type,return-value]
            clock=FakeClock(),
            session_factory=session_factory,
            speaker_factory=speaker_factory,
            credential_provider=lambda: "test-api-key",
            token_loader=lambda _: TOKEN,
            stop_monitor_factory=available_stop_monitor_factory,
        ),
    )
    service.start_local_stop_monitor()

    def pose_after_failure() -> bytes:
        assert failure_reported.wait(1)
        assert failed_session_stopped.wait(1)
        return (
            encoded(
                RemotePoseKind.ANALYSIS,
                sequence=1,
                pose=analysis(100),
                request_sequence=1,
            )
            + encoded(RemotePoseKind.RESUME, sequence=2)
            + encoded(
                RemotePoseKind.ANALYSIS,
                sequence=3,
                pose=analysis(101),
                request_sequence=2,
            )
            + encoded(
                RemotePoseKind.STOP,
                sequence=4,
            )
        )

    connection = FakeConnection([encoded(RemotePoseKind.START), pose_after_failure])
    service.handle_connection(connection, peer_host=config.allowed_peer)
    assert service.wait_for_status()

    assert len(sessions) == 2
    assert sessions[0].ended
    assert speakers[0].quiesced
    assert speakers[0].quiescences == 1
    assert speakers[0].closed
    assert sessions[1].activations == [(analysis(101), 0)]
    assert sessions[1].processed == [(analysis(100), 0), (analysis(101), 0)]
    assert sessions[1].stop_requests == 1
    status = json.loads(config.status_path.read_text())
    assert status["voice"] == "silent"
    assert status["failure"] == "RealtimeProviderError"


class BlockingPlayback:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.finished = threading.Event()
        self.stream_ids: list[str] = []

    def start(self, stream_id: str) -> None:
        self.stream_ids.append(stream_id)
        self.started.set()

    def write(self, stream_id: str, pcm: bytes) -> None:
        assert stream_id == self.stream_ids[-1]
        assert pcm == b"\x01\x00"
        self.stopped.wait(1)

    def finish(self) -> None:
        self.finished.set()

    def stop(self) -> int:
        self.stopped.set()
        return 0


class CompletingPlayback:
    def __init__(self) -> None:
        self.finish_started = threading.Event()
        self.allow_finish = threading.Event()
        self.stopped = threading.Event()
        self.stream_id: str | None = None

    def start(self, stream_id: str) -> None:
        self.stream_id = stream_id

    def write(self, stream_id: str, pcm: bytes) -> None:
        assert stream_id == self.stream_id
        assert pcm == b"\x01\x00"

    def finish(self) -> None:
        self.finish_started.set()
        assert self.allow_finish.wait(1)

    def stop(self) -> int:
        self.stopped.set()
        return 0


def released_clip(
    cue_id: CueId = CueId.MOVE_SLOWLY,
) -> ReleasedCueAudio:
    authorization = ApprovedCuePlaybackAuthorization(
        cue_id=cue_id,
        cue_kind=(CueKind.STATUS if cue_id is CueId.SQUAT_PERSON_DETECTED else CueKind.CORRECTION),
        catalog_version="prompt-cues-v2",
        guardian_rule_version="guardian-rules-v1",
        reason_codes=(GuardianReason.LOCAL_CUE_ACCEPTED,),
    )
    return ReleasedCueAudio(
        ticket_id=1,
        authorization=authorization,
        response_id="response-1",
        item_id="item-1",
        content_index=0,
        pcm16_mono_24khz=b"\x01\x00",
        queued_at_seconds=0.0,
        requested_at_seconds=0.0,
        released_at_seconds=0.1,
    )


def blocking_realtime_harness(tmp_path: Path) -> SimpleNamespace:
    pump_started = threading.Event()
    release_pump = threading.Event()
    callback_rejected = threading.Event()
    callback_finished = threading.Event()
    sessions: list[FakeSession] = []
    speakers: list[FakeSpeaker] = []

    class BlockingPumpSession(FakeSession):
        def __init__(self, *, preempt, on_cue_audio) -> None:
            super().__init__(preempt=preempt)
            self._on_cue_audio = on_cue_audio

        def pump_once(self):
            pump_started.set()
            assert release_pump.wait(1)
            try:
                self._on_cue_audio(released_clip())
            except RuntimeError:
                callback_rejected.set()
                raise
            finally:
                callback_finished.set()
            return SimpleNamespace(failure_kind=None, end_signal=None)

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        assert api_key == "test-api-key"
        session = BlockingPumpSession(
            preempt=on_audio_preempt,
            on_cue_audio=on_cue_audio,
        )
        sessions.append(session)
        return session

    def speaker_factory(config) -> FakeSpeaker:
        del config
        speaker = FakeSpeaker()
        speakers.append(speaker)
        return speaker

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
        audio_enabled=True,
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            clock=FakeClock(),
            session_factory=session_factory,
            speaker_factory=speaker_factory,
            credential_provider=lambda: "test-api-key",
            stop_monitor_factory=available_stop_monitor_factory,
        ),
    )
    service.start_local_stop_monitor()
    return SimpleNamespace(
        service=service,
        config=config,
        pump_started=pump_started,
        release_pump=release_pump,
        callback_rejected=callback_rejected,
        callback_finished=callback_finished,
        sessions=sessions,
        speakers=speakers,
    )


def test_subprocess_speaker_handoff_is_nonblocking_and_preemptible() -> None:
    playback = BlockingPlayback()
    speaker = _SubprocessCueSpeaker(playback)  # type: ignore[arg-type]
    completed: list[CueId] = []
    speaker.bind_playback_succeeded_callback(completed.append)

    speaker.enqueue(released_clip())
    assert playback.started.wait(1)
    speaker.quiesce()
    with pytest.raises(RuntimeError, match="quiesced"):
        speaker.enqueue(released_clip())
    speaker.preempt()
    assert playback.stopped.wait(1)
    speaker.close()
    assert playback.stream_ids == ["approved-cue-1-0"]
    assert not playback.finished.is_set()
    assert completed == []


def test_subprocess_speaker_reports_detection_only_after_playback_finishes() -> None:
    playback = CompletingPlayback()
    speaker = _SubprocessCueSpeaker(playback)  # type: ignore[arg-type]
    completed: list[CueId] = []
    failures: list[str] = []
    speaker.bind_playback_succeeded_callback(completed.append)
    speaker.bind_failure_callback(lambda: failures.append("failed"))

    speaker.enqueue(released_clip(CueId.SQUAT_PERSON_DETECTED))
    assert playback.finish_started.wait(1)
    assert completed == []

    playback.allow_finish.set()
    assert wait_until(lambda: completed == [CueId.SQUAT_PERSON_DETECTED])
    speaker.close()

    assert failures == []


def test_button_quiesces_audio_before_inflight_analysis_leaves_lifecycle(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    processing_started = threading.Event()
    release_processing = threading.Event()
    callback_rejected = threading.Event()
    release_source = threading.Event()
    sessions: list[FakeSession] = []
    speakers: list[FakeSpeaker] = []
    monitors: list[FakeStopMonitor] = []

    def read_pose(read_count: int) -> SquatAnalysis:
        if read_count == 1:
            clock.value = 0.1
            return analysis(100)
        assert release_source.wait(5)
        return analysis(101)

    source = FakeLocalPoseSource(read_pose)

    class CueReturningAnalysisSession(FakeSession):
        def __init__(self, *, preempt, on_cue_audio) -> None:
            super().__init__(preempt=preempt)
            self._on_cue_audio = on_cue_audio

        def process_analysis(
            self,
            pose: SquatAnalysis,
            *,
            pose_age_ms: int = 0,
        ) -> object:
            processing_started.set()
            assert release_processing.wait(1)
            try:
                self._on_cue_audio(released_clip())
            except RuntimeError:
                callback_rejected.set()
            return super().process_analysis(pose, pose_age_ms=pose_age_ms)

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        assert api_key == "test-api-key"
        session = CueReturningAnalysisSession(
            preempt=on_audio_preempt,
            on_cue_audio=on_cue_audio,
        )
        sessions.append(session)
        return session

    def speaker_factory(config) -> FakeSpeaker:
        del config
        speaker = FakeSpeaker()
        speakers.append(speaker)
        return speaker

    def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        monitor = FakeStopMonitor(
            config=config,
            on_stop=on_stop,
            on_status=on_status,
        )
        monitors.append(monitor)
        return monitor

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "unused-token",
        pose_source=PoseSourceMode.LOCAL,
        status_path=tmp_path / "status.json",
        audio_enabled=True,
    )
    service = RemotePoseService(
        config,
        token=None,
        dependencies=service_dependencies(
            clock=clock,
            session_factory=session_factory,
            speaker_factory=speaker_factory,
            credential_provider=lambda: "test-api-key",
            stop_monitor_factory=stop_monitor_factory,
            local_pose_source_factory=lambda: source,
        ),
    )
    service_errors: list[BaseException] = []

    def run_service() -> None:
        try:
            service.serve_forever()
        except BaseException as exc:  # pragma: no branch - diagnostic capture
            service_errors.append(exc)

    service_thread = threading.Thread(target=run_service)
    service_thread.start()
    assert processing_started.wait(1)

    stop_thread = threading.Thread(target=monitors[0].press)
    stop_thread.start()
    assert wait_until(lambda: speakers[0].quiesced)
    assert stop_thread.is_alive(), "stop must still be waiting for the active lifecycle"
    release_processing.set()
    release_source.set()
    assert callback_rejected.wait(1)
    stop_thread.join(1)
    service_thread.join(1)

    assert not stop_thread.is_alive()
    assert not service_thread.is_alive()
    assert service_errors == []
    assert speakers[0].enqueue_attempts == [released_clip()]
    assert speakers[0].enqueued == []
    assert speakers[0].quiescences == 1
    assert speakers[0].preemptions >= 1
    assert speakers[0].closed
    assert sessions[0].stop_requests == 1


def test_button_quiesces_audio_before_blocking_realtime_pump_returns(
    tmp_path: Path,
) -> None:
    pump_started = threading.Event()
    release_pump = threading.Event()
    callback_rejected = threading.Event()
    callback_finished = threading.Event()
    sessions: list[FakeSession] = []
    speakers: list[FakeSpeaker] = []
    monitors: list[FakeStopMonitor] = []

    class BlockingPumpSession(FakeSession):
        def __init__(self, *, preempt, on_cue_audio) -> None:
            super().__init__(preempt=preempt)
            self._on_cue_audio = on_cue_audio

        def pump_once(self):
            pump_started.set()
            assert release_pump.wait(1)
            try:
                self._on_cue_audio(released_clip())
            except RuntimeError:
                callback_rejected.set()
                raise
            finally:
                callback_finished.set()
            return SimpleNamespace(failure_kind=None, end_signal=None)

    def session_factory(*, api_key, on_cue_audio, on_audio_preempt) -> FakeSession:
        assert api_key == "test-api-key"
        session = BlockingPumpSession(
            preempt=on_audio_preempt,
            on_cue_audio=on_cue_audio,
        )
        sessions.append(session)
        return session

    def speaker_factory(config) -> FakeSpeaker:
        del config
        speaker = FakeSpeaker()
        speakers.append(speaker)
        return speaker

    def stop_monitor_factory(*, config, on_stop, on_status) -> FakeStopMonitor:
        monitor = FakeStopMonitor(
            config=config,
            on_stop=on_stop,
            on_status=on_status,
        )
        monitors.append(monitor)
        return monitor

    config = RemotePoseServiceConfig(
        bind_host="100.106.237.106",
        allowed_peer="100.70.100.93",
        token_file=tmp_path / "token",
        status_path=tmp_path / "status.json",
        audio_enabled=True,
    )
    service = RemotePoseService(
        config,
        token=TOKEN,
        dependencies=service_dependencies(
            clock=FakeClock(),
            session_factory=session_factory,
            speaker_factory=speaker_factory,
            credential_provider=lambda: "test-api-key",
            stop_monitor_factory=stop_monitor_factory,
        ),
    )
    service.start_local_stop_monitor()

    def stop_while_pump_is_blocked() -> bytes:
        assert pump_started.wait(1)
        monitors[0].press()
        assert speakers[0].quiesced
        release_pump.set()
        assert callback_finished.wait(1)
        return b""

    service.handle_connection(
        FakeConnection([encoded(RemotePoseKind.START), stop_while_pump_is_blocked]),
        peer_host=config.allowed_peer,
    )
    assert service.wait_for_status()

    assert callback_rejected.is_set()
    assert speakers[0].enqueue_attempts == [released_clip()]
    assert speakers[0].enqueued == []
    assert speakers[0].quiescences == 1
    assert speakers[0].preemptions >= 1
    assert speakers[0].closed
    assert sessions[0].stop_requests == 1
    assert sessions[0].physical_stop_requests == 1
    assert sessions[0].abort_requests == []
    assert service.current_mode is SessionMode.STOPPED
    service.shutdown()


def test_remote_stop_quiesces_audio_before_blocking_realtime_pump_returns(
    tmp_path: Path,
) -> None:
    harness = blocking_realtime_harness(tmp_path)

    def stop_after_pump_starts() -> bytes:
        assert harness.pump_started.wait(1)
        return encoded(RemotePoseKind.STOP, sequence=1)

    harness.service.handle_connection(
        FakeConnection([encoded(RemotePoseKind.START), stop_after_pump_starts]),
        peer_host=harness.config.allowed_peer,
    )

    speaker = harness.speakers[0]
    assert speaker.quiesced
    assert speaker.enqueued == []
    harness.release_pump.set()
    assert harness.callback_finished.wait(1)

    assert harness.callback_rejected.is_set()
    assert speaker.enqueue_attempts == [released_clip()]
    assert speaker.enqueued == []
    assert speaker.quiescences == 1
    assert speaker.preemptions >= 1
    assert speaker.closed
    assert harness.sessions[0].stop_requests == 1
    assert harness.sessions[0].abort_requests == [RuntimeAbortReason.REMOTE_STOP]
    assert harness.sessions[0].physical_stop_requests == 0
    assert harness.service.current_mode is SessionMode.STOPPED
    harness.service.shutdown()


def test_shutdown_quiesces_audio_before_blocking_realtime_pump_returns(
    tmp_path: Path,
) -> None:
    harness = blocking_realtime_harness(tmp_path)
    connection: FakeConnection

    def wait_for_shutdown() -> bytes:
        assert connection.close_event.wait(1)
        return b""

    connection = FakeConnection([encoded(RemotePoseKind.START), wait_for_shutdown])
    handler = threading.Thread(
        target=harness.service.handle_connection,
        kwargs={"connection": connection, "peer_host": harness.config.allowed_peer},
    )
    handler.start()
    assert harness.pump_started.wait(1)

    harness.service.shutdown()
    speaker = harness.speakers[0]
    assert speaker.quiesced
    assert speaker.enqueued == []
    harness.release_pump.set()
    assert harness.callback_finished.wait(1)
    handler.join(1)

    assert not handler.is_alive()
    assert harness.callback_rejected.is_set()
    assert speaker.enqueue_attempts == [released_clip()]
    assert speaker.enqueued == []
    assert speaker.quiescences == 1
    assert speaker.preemptions >= 1
    assert speaker.closed
    assert harness.sessions[0].stop_requests == 1
    assert harness.sessions[0].abort_requests == [RuntimeAbortReason.SERVICE_SHUTDOWN]
    assert harness.sessions[0].physical_stop_requests == 0
    assert harness.service.current_mode is SessionMode.STOPPED
