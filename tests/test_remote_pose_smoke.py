from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

import recoverybox.remote_pose_smoke as smoke_module
from recoverybox.exercise import SquatAnalysis, SquatAssessmentIssue, SquatPhase
from recoverybox.remote_pose import RemotePoseRequest
from recoverybox.remote_pose_smoke import (
    RemotePoseSmokeDependencies,
    RemotePoseSmokeError,
    main,
    run_remote_pose_smoke,
)

PEER = "100.106.237.106:45873"
TOKEN_PATH = Path("/content-free/pose-token")
TOKEN = bytes(range(32))


def _request(sequence: int) -> RemotePoseRequest:
    return RemotePoseRequest(
        session_id="0" * 32,
        service_epoch="1" * 64,
        server_nonce="2" * 64,
        request_sequence=sequence,
        request_nonce=f"{sequence:064x}",
    )


class FakePublisher:
    def __init__(
        self,
        *,
        requests: list[RemotePoseRequest | None] | None = None,
        start_error: Exception | None = None,
        submit_error: Exception | None = None,
        close_error: Exception | None = None,
        final_failure_kind: str | None = None,
        final_messages_sent: int | None = None,
    ) -> None:
        self.requests = list(requests or [_request(1), _request(2)])
        self.start_error = start_error
        self.submit_error = submit_error
        self.close_error = close_error
        self._final_failure_kind = final_failure_kind
        self._final_messages_sent = final_messages_sent
        self.calls: list[str] = []
        self.wait_timeouts: list[float | None] = []
        self.submissions: list[tuple[SquatAnalysis, RemotePoseRequest, int]] = []
        self.resume_calls = 0
        self._messages_sent = 0

    @property
    def failure_kind(self) -> str | None:
        return self._final_failure_kind

    @property
    def messages_sent(self) -> int:
        if self._final_messages_sent is not None:
            return self._final_messages_sent
        return self._messages_sent

    def start(self) -> None:
        self.calls.append("START")
        if self.start_error is not None:
            raise self.start_error
        self._messages_sent += 1

    def wait_for_request(self, timeout_seconds: float | None = None) -> RemotePoseRequest | None:
        self.calls.append("WAIT")
        self.wait_timeouts.append(timeout_seconds)
        return self.requests.pop(0)

    def submit(
        self,
        analysis: SquatAnalysis,
        *,
        request: RemotePoseRequest,
        evidence_age_ms: int,
    ) -> None:
        self.calls.append("ANALYSIS")
        self.submissions.append((analysis, request, evidence_age_ms))
        if self.submit_error is not None:
            raise self.submit_error
        self._messages_sent += 1

    def request_resume(self) -> None:
        self.calls.append("RESUME")
        self.resume_calls += 1

    def close(self) -> None:
        self.calls.append("STOP")
        if self.close_error is not None:
            raise self.close_error
        self._messages_sent += 1


class Factory:
    def __init__(self, publisher: FakePublisher) -> None:
        self.publisher = publisher
        self.calls: list[tuple[str, bytes, bool]] = []

    def __call__(
        self,
        peer: str,
        token: bytes,
        *,
        authorize_initial_epoch: bool,
    ) -> FakePublisher:
        self.calls.append((peer, token, authorize_initial_epoch))
        return self.publisher


def _dependencies(
    publisher: FakePublisher,
    *,
    monotonic: Any = lambda: 12.345,
    token_loader: Any = None,
) -> tuple[RemotePoseSmokeDependencies, Factory, list[str | Path]]:
    loaded_paths: list[str | Path] = []

    def load_token(path: str | Path) -> bytes:
        loaded_paths.append(path)
        if token_loader is not None:
            return token_loader(path)
        return TOKEN

    factory = Factory(publisher)
    return (
        RemotePoseSmokeDependencies(
            load_token=load_token,
            publisher_factory=factory,
            monotonic=monotonic,
        ),
        factory,
        loaded_paths,
    )


def test_smoke_sends_one_exact_no_pose_analysis_and_requires_second_request() -> None:
    publisher = FakePublisher()
    dependencies, factory, loaded_paths = _dependencies(publisher)

    result = run_remote_pose_smoke(
        PEER,
        TOKEN_PATH,
        request_timeout_seconds=2.5,
        dependencies=dependencies,
    )

    assert result.as_dict() == {
        "analysis_accepted": True,
        "messages_sent": 3,
        "ok": True,
    }
    assert loaded_paths == [TOKEN_PATH]
    assert factory.calls == [(PEER, TOKEN, False)]
    assert publisher.calls == ["START", "WAIT", "ANALYSIS", "WAIT", "STOP"]
    assert publisher.wait_timeouts == [2.5, 2.5]
    assert publisher.resume_calls == 0
    assert len(publisher.submissions) == 1
    analysis, bound_request, evidence_age_ms = publisher.submissions[0]
    assert analysis == SquatAnalysis(
        timestamp_ms=12345,
        assessable=False,
        phase=SquatPhase.UNKNOWN,
        rep_count=0,
        events=(),
        issues=(SquatAssessmentIssue.NO_POSE,),
        confidence=0.0,
        knee_angle_degrees=None,
        arms_in_t=None,
    )
    assert bound_request.request_sequence == 1
    assert evidence_age_ms == 0


@pytest.mark.parametrize(
    ("publisher", "failure_kind", "expected_calls"),
    [
        (
            FakePublisher(start_error=OSError("secret peer detail")),
            "OSError",
            ["START", "STOP"],
        ),
        (
            FakePublisher(requests=[None]),
            "InitialRequestTimeout",
            ["START", "WAIT", "STOP"],
        ),
        (
            FakePublisher(submit_error=RuntimeError("secret analysis detail")),
            "RuntimeError",
            ["START", "WAIT", "ANALYSIS", "STOP"],
        ),
        (
            FakePublisher(requests=[_request(1), None]),
            "AcceptanceRequestTimeout",
            ["START", "WAIT", "ANALYSIS", "WAIT", "STOP"],
        ),
        (
            FakePublisher(close_error=TimeoutError("secret close detail")),
            "TimeoutError",
            ["START", "WAIT", "ANALYSIS", "WAIT", "STOP"],
        ),
    ],
)
def test_smoke_closes_in_finally_and_reports_only_failure_kind(
    publisher: FakePublisher,
    failure_kind: str,
    expected_calls: list[str],
) -> None:
    dependencies, _, _ = _dependencies(publisher)

    with pytest.raises(RemotePoseSmokeError) as raised:
        run_remote_pose_smoke(PEER, TOKEN_PATH, dependencies=dependencies)

    assert raised.value.failure_kind == failure_kind
    assert str(raised.value) == failure_kind
    assert publisher.calls == expected_calls
    assert publisher.resume_calls == 0


@pytest.mark.parametrize(
    ("publisher", "failure_kind"),
    [
        (FakePublisher(final_failure_kind="ConnectionResetError"), "ConnectionResetError"),
        (FakePublisher(final_failure_kind="peer=/private/secret"), "RemotePoseSmokeFailure"),
        (FakePublisher(final_messages_sent=2), "UnexpectedMessageCount"),
        (FakePublisher(final_messages_sent=4), "UnexpectedMessageCount"),
        (FakePublisher(final_messages_sent=3.0), "UnexpectedMessageCount"),  # type: ignore[arg-type]
    ],
)
def test_smoke_requires_clean_close_and_exactly_three_observed_messages(
    publisher: FakePublisher,
    failure_kind: str,
) -> None:
    dependencies, _, _ = _dependencies(publisher)

    with pytest.raises(RemotePoseSmokeError) as raised:
        run_remote_pose_smoke(PEER, TOKEN_PATH, dependencies=dependencies)

    assert raised.value.failure_kind == failure_kind
    assert publisher.calls[-1] == "STOP"
    assert publisher.resume_calls == 0


@pytest.mark.parametrize("clock_value", [-0.1, float("inf"), float("nan"), True, "1.0"])
def test_invalid_monotonic_clock_fails_closed(clock_value: object) -> None:
    publisher = FakePublisher()
    dependencies, _, _ = _dependencies(publisher, monotonic=lambda: clock_value)

    with pytest.raises(RemotePoseSmokeError, match=r"^InvalidMonotonicClock$"):
        run_remote_pose_smoke(PEER, TOKEN_PATH, dependencies=dependencies)

    assert publisher.calls == ["START", "WAIT", "STOP"]
    assert publisher.submissions == []
    assert publisher.resume_calls == 0


def test_token_loading_and_publisher_construction_errors_are_sanitized() -> None:
    publisher = FakePublisher()

    def fail_load(_path: str | Path) -> bytes:
        raise PermissionError("/private/token-path")

    load_dependencies, _, _ = _dependencies(publisher, token_loader=fail_load)
    with pytest.raises(RemotePoseSmokeError) as load_error:
        run_remote_pose_smoke(PEER, TOKEN_PATH, dependencies=load_dependencies)
    assert load_error.value.failure_kind == "PermissionError"
    assert publisher.calls == []

    class FailingFactory:
        def __call__(self, *_args: object, **_kwargs: object) -> FakePublisher:
            raise ValueError("peer and token detail")

    factory_dependencies = RemotePoseSmokeDependencies(
        load_token=lambda _path: TOKEN,
        publisher_factory=FailingFactory(),
        monotonic=lambda: 1.0,
    )
    with pytest.raises(RemotePoseSmokeError) as factory_error:
        run_remote_pose_smoke(PEER, TOKEN_PATH, dependencies=factory_dependencies)
    assert factory_error.value.failure_kind == "ValueError"


@pytest.mark.parametrize("timeout", [0, -1, 61, float("inf"), float("nan"), True, "5"])
def test_timeout_is_bounded_and_validated_before_loading_token(timeout: object) -> None:
    publisher = FakePublisher()
    dependencies, _, loaded_paths = _dependencies(publisher)

    with pytest.raises((TypeError, ValueError)):
        run_remote_pose_smoke(
            PEER,
            TOKEN_PATH,
            request_timeout_seconds=timeout,  # type: ignore[arg-type]
            dependencies=dependencies,
        )

    assert loaded_paths == []
    assert publisher.calls == []


def test_cli_success_prints_only_content_free_evidence(capsys: pytest.CaptureFixture[str]) -> None:
    publisher = FakePublisher()
    dependencies, _, _ = _dependencies(publisher)

    exit_code = main(
        [
            "--peer",
            PEER,
            "--pose-token-file",
            str(TOKEN_PATH),
        ],
        dependencies=dependencies,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {
        "analysis_accepted": True,
        "messages_sent": 3,
        "ok": True,
    }
    assert captured.err == ""
    assert PEER not in captured.out
    assert str(TOKEN_PATH) not in captured.out


def test_cli_failure_is_nonzero_and_does_not_print_exception_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    publisher = FakePublisher(start_error=OSError("private peer and token content"))
    dependencies, _, _ = _dependencies(publisher)

    exit_code = main(
        [
            "--peer",
            PEER,
            "--pose-token-file",
            str(TOKEN_PATH),
        ],
        dependencies=dependencies,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {"failure_kind": "OSError", "ok": False}
    assert PEER not in captured.err
    assert str(TOKEN_PATH) not in captured.err
    assert "private" not in captured.err


def test_module_has_no_vision_camera_or_audio_imports() -> None:
    source = Path(smoke_module.__file__).read_text(encoding="utf-8")
    imported_modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    forbidden = {
        "cv2",
        "mediapipe",
        "sounddevice",
        "recoverybox.vision",
        "recoverybox.audio",
        "recoverybox.laptop.pose_client",
        "recoverybox.laptop.squat_launcher",
    }
    assert imported_modules.isdisjoint(forbidden)
