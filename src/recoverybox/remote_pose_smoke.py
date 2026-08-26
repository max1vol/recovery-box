"""Hardware-free authenticated smoke check for a deployed remote-pose peer.

The smoke check deliberately does not acquire a camera frame or authorize a
paused session.  It answers one authenticated Pi request with a synthetic,
non-assessable numeric analysis and requires the next authenticated request as
proof that the Pi accepted it.  Closing the publisher then flushes STOP.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Protocol

from recoverybox.exercise import SquatAnalysis, SquatAssessmentIssue, SquatPhase
from recoverybox.remote_pose import (
    RemotePosePublisher,
    RemotePoseRequest,
    load_remote_pose_token,
)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 5.0
EXPECTED_MESSAGE_COUNT = 3


class _Publisher(Protocol):
    @property
    def failure_kind(self) -> str | None: ...

    @property
    def messages_sent(self) -> int: ...

    def start(self) -> None: ...

    def wait_for_request(
        self,
        timeout_seconds: float | None = None,
    ) -> RemotePoseRequest | None: ...

    def submit(
        self,
        analysis: SquatAnalysis,
        *,
        request: RemotePoseRequest,
        evidence_age_ms: int,
    ) -> None: ...

    def close(self) -> None: ...


class _PublisherFactory(Protocol):
    def __call__(
        self,
        peer: str,
        token: bytes,
        *,
        authorize_initial_epoch: bool,
    ) -> _Publisher: ...


@dataclass(frozen=True, slots=True)
class RemotePoseSmokeDependencies:
    """Hardware-free seams used by unit tests and the deployment verifier."""

    load_token: Callable[[str | Path], bytes] = load_remote_pose_token
    publisher_factory: _PublisherFactory = RemotePosePublisher
    monotonic: Callable[[], float] = time.monotonic


@dataclass(frozen=True, slots=True)
class RemotePoseSmokeResult:
    """Content-free evidence returned after the full authenticated exchange."""

    analysis_accepted: bool
    messages_sent: int

    def as_dict(self) -> dict[str, bool | int]:
        return {
            "analysis_accepted": self.analysis_accepted,
            "messages_sent": self.messages_sent,
            "ok": True,
        }


class RemotePoseSmokeError(RuntimeError):
    """A smoke failure represented only by a safe, content-free kind."""

    def __init__(self, failure_kind: str) -> None:
        self.failure_kind = _sanitized_failure_kind(failure_kind)
        super().__init__(self.failure_kind)


def _sanitized_failure_kind(value: object) -> str:
    if isinstance(value, str) and value.isascii() and value.isidentifier():
        return value
    return "RemotePoseSmokeFailure"


def _error_for_exception(exc: Exception) -> RemotePoseSmokeError:
    if isinstance(exc, RemotePoseSmokeError):
        return exc
    return RemotePoseSmokeError(type(exc).__name__)


def _positive_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("request_timeout_seconds must be a real number")
    converted = float(value)
    if not isfinite(converted) or not 0.0 < converted <= 60.0:
        raise ValueError("request_timeout_seconds must be between 0 and 60 seconds")
    return converted


def _fresh_timestamp_ms(monotonic: Callable[[], float]) -> int:
    value = monotonic()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RemotePoseSmokeError("InvalidMonotonicClock")
    converted = float(value)
    if not isfinite(converted) or converted < 0.0:
        raise RemotePoseSmokeError("InvalidMonotonicClock")
    return int(converted * 1000.0)


def _synthetic_no_pose_analysis(timestamp_ms: int) -> SquatAnalysis:
    return SquatAnalysis(
        timestamp_ms=timestamp_ms,
        assessable=False,
        phase=SquatPhase.UNKNOWN,
        rep_count=0,
        events=(),
        issues=(SquatAssessmentIssue.NO_POSE,),
        confidence=0.0,
        knee_angle_degrees=None,
        arms_in_t=None,
    )


def run_remote_pose_smoke(
    peer: str,
    pose_token_file: str | Path,
    *,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    dependencies: RemotePoseSmokeDependencies | None = None,
) -> RemotePoseSmokeResult:
    """Prove one authenticated synthetic analysis is accepted by ``peer``.

    The default publisher validates that ``peer`` is a literal Tailscale
    IPv4-and-port pair.  No resume control is sent: the deployed Guardian
    remains paused unless a separately authorized live session resumes it.
    """

    timeout = _positive_timeout(request_timeout_seconds)
    deps = dependencies or RemotePoseSmokeDependencies()
    publisher: _Publisher | None = None

    # ``bytes`` cannot be wiped in place.  Drop this helper's local reference
    # immediately after the publisher has consumed it; the publisher retains
    # only the reference it needs to authenticate its bounded connection.
    try:
        token: bytes | None = deps.load_token(pose_token_file)
    except Exception as exc:
        raise _error_for_exception(exc) from None
    try:
        try:
            publisher = deps.publisher_factory(
                peer,
                token,
                authorize_initial_epoch=False,
            )
        except Exception as exc:
            raise _error_for_exception(exc) from None
    finally:
        token = None

    failure: RemotePoseSmokeError | None = None
    try:
        publisher.start()
        first_request = publisher.wait_for_request(timeout)
        if first_request is None:
            raise RemotePoseSmokeError("InitialRequestTimeout")

        publisher.submit(
            _synthetic_no_pose_analysis(_fresh_timestamp_ms(deps.monotonic)),
            request=first_request,
            evidence_age_ms=0,
        )

        if publisher.wait_for_request(timeout) is None:
            raise RemotePoseSmokeError("AcceptanceRequestTimeout")
    except Exception as exc:
        failure = _error_for_exception(exc)
    finally:
        try:
            publisher.close()
        except Exception as exc:
            if failure is None:
                failure = _error_for_exception(exc)

    if failure is not None:
        raise failure

    try:
        publisher_failure = publisher.failure_kind
        messages_sent = publisher.messages_sent
    except Exception as exc:
        raise _error_for_exception(exc) from None
    if publisher_failure is not None:
        raise RemotePoseSmokeError(publisher_failure)
    if type(messages_sent) is not int or messages_sent != EXPECTED_MESSAGE_COUNT:
        raise RemotePoseSmokeError("UnexpectedMessageCount")
    return RemotePoseSmokeResult(analysis_accepted=True, messages_sent=messages_sent)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recoverybox.remote_pose_smoke",
        description="Run one hardware-free authenticated remote-pose deployment smoke check",
    )
    parser.add_argument("--peer", required=True, metavar="TAILSCALE_IPV4:PORT")
    parser.add_argument("--pose-token-file", required=True, type=Path, metavar="PATH")
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        metavar="SECONDS",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: RemotePoseSmokeDependencies | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_remote_pose_smoke(
            args.peer,
            args.pose_token_file,
            request_timeout_seconds=args.request_timeout_seconds,
            dependencies=dependencies,
        )
    except KeyboardInterrupt:
        print(
            json.dumps({"failure_kind": "KeyboardInterrupt", "ok": False}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        return 130
    except Exception as exc:
        failure_kind = (
            exc.failure_kind
            if isinstance(exc, RemotePoseSmokeError)
            else _sanitized_failure_kind(type(exc).__name__)
        )
        print(
            json.dumps({"failure_kind": failure_kind, "ok": False}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        return 2

    print(json.dumps(result.as_dict(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "EXPECTED_MESSAGE_COUNT",
    "RemotePoseSmokeDependencies",
    "RemotePoseSmokeError",
    "RemotePoseSmokeResult",
    "main",
    "run_remote_pose_smoke",
]
