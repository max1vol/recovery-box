"""Asynchronous, fail-closed delivery of Guardian-authorized prompt cues.

This module deliberately does not receive camera frames, construct speech text,
own a network transport, or write to a speaker.  It schedules typed
``ApprovedCuePlaybackAuthorization`` values on an existing
:class:`~recoverybox.realtime.client.RealtimeSession`, consumes already parsed
``RealtimeClientResult`` values from the application's single event dispatcher,
and hands one atomic, safety-gated PCM clip to a non-blocking callback.

Only one cue response is allowed on the wire at a time.  Pending counts may
jump ahead of repeated form reminders, but they never preempt an active cue and
never outrank a safety cue.  Repeated corrections are coalesced by cue ID, and
pending counts are coalesced to the newest count so delayed speech does not read
an obsolete backlog.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Protocol

from recoverybox.core import CueId, CueKind, SessionMode
from recoverybox.session import ApprovedCuePlaybackAuthorization, SessionModeProvider

from .client import RealtimeClientResult
from .protocol import ServerEventKind
from .safety import ModelAudioPolicy, ReleasedModelAudio


class CueDeliveryError(RuntimeError):
    """A cue could not remain inside the approved Realtime delivery lane."""


class CueQueueDisposition(StrEnum):
    """Observable result of offering one authorization to the scheduler."""

    STARTED = "started"
    QUEUED = "queued"
    COALESCED = "coalesced"
    SUPERSEDED_COUNT = "superseded_count"
    DROPPED_CAPACITY = "dropped_capacity"


class CueDeliveryFailureReason(StrEnum):
    """Safe, content-free reasons reported to the composition layer."""

    REQUEST_FAILED = "request_failed"
    RESPONSE_FAILED = "response_failed"
    QUARANTINE_REJECTED = "quarantine_rejected"
    RESPONSE_TIMEOUT = "response_timeout"
    MODE_CHANGED = "mode_changed"
    OUTPUT_CALLBACK_FAILED = "output_callback_failed"


class _CueLane(IntEnum):
    # Lower values are selected first.  A new cue never interrupts the active
    # one, so this ordering applies only to not-yet-requested work.
    SAFETY = 0
    COUNT = 1
    CORRECTION = 2
    OTHER = 3


@dataclass(frozen=True, slots=True)
class CueDeliveryConfig:
    """Timing and bounded-queue policy for one local exercise session."""

    max_pending_cues: int = 6
    safety_max_queue_age_seconds: float = 1.0
    count_max_queue_age_seconds: float = 2.0
    correction_max_queue_age_seconds: float = 4.0
    other_max_queue_age_seconds: float = 3.0
    response_timeout_seconds: float = 8.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_pending_cues, bool)
            or not isinstance(self.max_pending_cues, int)
            or self.max_pending_cues <= 0
        ):
            raise ValueError("max_pending_cues must be a positive integer")
        for field_name in (
            "safety_max_queue_age_seconds",
            "count_max_queue_age_seconds",
            "correction_max_queue_age_seconds",
            "other_max_queue_age_seconds",
            "response_timeout_seconds",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be finite and positive")


DEFAULT_CUE_DELIVERY_CONFIG = CueDeliveryConfig()


@dataclass(frozen=True, slots=True)
class CueEnqueueResult:
    """Ticket returned without exposing or accepting any speech text."""

    ticket_id: int
    cue_id: CueId
    disposition: CueQueueDisposition


@dataclass(frozen=True, slots=True)
class ReleasedCueAudio:
    """One complete approved cue ready for the exclusive speaker arbiter."""

    ticket_id: int
    authorization: ApprovedCuePlaybackAuthorization
    response_id: str
    item_id: str
    content_index: int
    pcm16_mono_24khz: bytes
    queued_at_seconds: float
    requested_at_seconds: float
    released_at_seconds: float

    @property
    def queue_delay_ms(self) -> float:
        return max(0.0, (self.requested_at_seconds - self.queued_at_seconds) * 1000.0)

    @property
    def response_latency_ms(self) -> float:
        return max(0.0, (self.released_at_seconds - self.requested_at_seconds) * 1000.0)

    @property
    def total_latency_ms(self) -> float:
        return max(0.0, (self.released_at_seconds - self.queued_at_seconds) * 1000.0)


@dataclass(frozen=True, slots=True)
class CueDeliveryFailure:
    """Content-free asynchronous failure evidence for fail-safe composition."""

    ticket_id: int
    cue_id: CueId
    reason: CueDeliveryFailureReason
    response_id: str | None


@dataclass(frozen=True, slots=True)
class CueDeliverySnapshot:
    """Small diagnostic surface suitable for deterministic latency simulation."""

    active_ticket_id: int | None
    active_cue_id: CueId | None
    pending_cue_ids: tuple[CueId, ...]
    draining_stale_response: bool
    requested_count: int
    released_count: int
    coalesced_count: int
    superseded_count: int
    stale_drop_count: int
    capacity_drop_count: int
    cancelled_count: int
    failed_count: int


class PromptCueSession(Protocol):
    """Narrow Realtime operations used by the scheduler."""

    def request_approved_prompt_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None: ...

    def cancel_response(self, *, response_id: str | None = None) -> None: ...

    def revoke_pending_response_locally(self) -> int: ...


@dataclass(slots=True)
class _CueWork:
    ticket_id: int
    authorization: ApprovedCuePlaybackAuthorization
    lane: _CueLane
    queued_at: float
    requested_at: float | None = None
    response_id: str | None = None
    released_audio: ReleasedModelAudio | None = None
    invalid_release: bool = False
    finished: bool = False


@dataclass(slots=True)
class _DrainingResponse:
    """Cancelled wire work that must become terminal before another request."""

    ticket_id: int
    response_id: str | None


class RealtimeCueDelivery:
    """Queue and correlate isolated Realtime prompt-cue responses.

    ``enqueue`` is non-blocking with respect to model generation: it sends at
    most one ``response.create`` and returns without waiting for server audio.
    The application's sole Realtime event reader must pass every parsed result
    to :meth:`handle_result` before considering another consumer.

    ``on_audio`` is invoked only for one complete ``PROMPT_CUE_QUARANTINE``
    clip, after a completed ``response.done`` and a final ACTIVE_EXERCISE mode
    check.  It must be a quick, non-blocking handoff to the exclusive speaker
    arbiter.  ``on_preempt`` lets that arbiter synchronously revoke already
    handed-off playback before network cancellation is attempted.

    ``count_cue_ids`` is composition-owned configuration, not caller input.
    It affects scheduling only; it cannot authorize text or bypass the catalog.
    """

    def __init__(
        self,
        *,
        session: PromptCueSession,
        mode_provider: SessionModeProvider,
        on_audio: Callable[[ReleasedCueAudio], None],
        count_cue_ids: Collection[CueId | str] = (),
        on_preempt: Callable[[], None] | None = None,
        on_failure: Callable[[CueDeliveryFailure], None] | None = None,
        config: CueDeliveryConfig = DEFAULT_CUE_DELIVERY_CONFIG,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(on_audio):
            raise TypeError("on_audio must be callable")
        if on_preempt is not None and not callable(on_preempt):
            raise TypeError("on_preempt must be callable when provided")
        if on_failure is not None and not callable(on_failure):
            raise TypeError("on_failure must be callable when provided")
        if not isinstance(config, CueDeliveryConfig):
            raise TypeError("config must be a CueDeliveryConfig")
        if not callable(clock):
            raise TypeError("clock must be callable")

        normalized_count_ids: set[str] = set()
        for cue_id in count_cue_ids:
            value = cue_id.value if isinstance(cue_id, CueId) else cue_id
            if not isinstance(value, str) or not value.strip():
                raise ValueError("count_cue_ids must contain nonblank cue identifiers")
            normalized_count_ids.add(value.strip())

        self._session = session
        self._mode_provider = mode_provider
        self._on_audio = on_audio
        self._on_preempt = on_preempt
        self._on_failure = on_failure
        self._count_cue_ids = frozenset(normalized_count_ids)
        self._config = config
        self._clock = clock
        self._lock = threading.RLock()
        self._next_ticket_id = 1
        self._active: _CueWork | None = None
        self._pending: list[_CueWork] = []
        self._draining: _DrainingResponse | None = None
        self._request_inflight = False
        self._closed = False

        self._requested_count = 0
        self._released_count = 0
        self._coalesced_count = 0
        self._superseded_count = 0
        self._stale_drop_count = 0
        self._capacity_drop_count = 0
        self._cancelled_count = 0
        self._failed_count = 0

    def play_approved_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None:
        """Implement ``ApprovedCuePlaybackPort`` without accepting speech text."""

        self.enqueue(authorization)

    def enqueue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> CueEnqueueResult:
        """Start, queue, or coalesce one Guardian capability."""

        if not isinstance(authorization, ApprovedCuePlaybackAuthorization):
            raise TypeError("authorization must be ApprovedCuePlaybackAuthorization")
        self._require_active_exercise()
        now = self._now()
        to_request: _CueWork | None = None

        with self._lock:
            if self._closed:
                raise CueDeliveryError("cue delivery is closed")
            self._drop_stale_pending_locked(now)
            lane = self._lane_for(authorization)

            if self._active is not None and self._same_cue(
                self._active.authorization,
                authorization,
            ):
                self._coalesced_count += 1
                return CueEnqueueResult(
                    self._active.ticket_id,
                    authorization.cue_id,
                    CueQueueDisposition.COALESCED,
                )

            for pending in self._pending:
                if self._same_cue(pending.authorization, authorization):
                    pending.authorization = authorization
                    pending.queued_at = now
                    self._coalesced_count += 1
                    return CueEnqueueResult(
                        pending.ticket_id,
                        authorization.cue_id,
                        CueQueueDisposition.COALESCED,
                    )

            if lane is _CueLane.COUNT:
                for pending in self._pending:
                    if pending.lane is _CueLane.COUNT:
                        pending.authorization = authorization
                        pending.queued_at = now
                        self._superseded_count += 1
                        return CueEnqueueResult(
                            pending.ticket_id,
                            authorization.cue_id,
                            CueQueueDisposition.SUPERSEDED_COUNT,
                        )

            work = _CueWork(
                ticket_id=self._allocate_ticket_locked(),
                authorization=authorization,
                lane=lane,
                queued_at=now,
            )
            if self._active is None and self._draining is None and not self._request_inflight:
                self._active = work
                self._request_inflight = True
                work.requested_at = now
                to_request = work
                disposition = CueQueueDisposition.STARTED
            else:
                disposition = self._queue_pending_locked(work)

        if to_request is not None:
            self._request_work(to_request)
        return CueEnqueueResult(work.ticket_id, authorization.cue_id, disposition)

    def handle_result(self, result: RealtimeClientResult) -> bool:
        """Consume a parsed result when it belongs to the active/stale cue lane.

        Returns ``True`` when the event was correlated to cue work.  A caller
        can use this as part of a single event dispatcher; it must not run a
        second receiver thread for cue events.
        """

        if not isinstance(result, RealtimeClientResult):
            raise TypeError("result must be a RealtimeClientResult")

        event = result.event
        to_request: _CueWork | None = None
        failure: CueDeliveryFailure | None = None
        callback_error: Exception | None = None
        cancel_for_mode: tuple[_CueWork, str | None] | None = None
        cancel_for_failure = False

        with self._lock:
            if self._closed:
                return False

            if self._draining is not None:
                consumed, to_request = self._handle_draining_locked(result)
                if consumed:
                    # Never let a cancelled response fall through to another
                    # audio consumer, even if a stale server delta arrives.
                    pass
                else:
                    return False
            else:
                active = self._active
                if active is None:
                    return False

                if event.kind is ServerEventKind.RESPONSE_CREATED:
                    if event.response_id is None or not result.response_authorized:
                        return False
                    active.response_id = event.response_id
                    return True

                response_id = active.response_id
                matching_audio = tuple(
                    audio
                    for audio in result.released_audio
                    if response_id is not None and audio.response_id == response_id
                )
                if matching_audio:
                    if (
                        len(matching_audio) != 1
                        or active.released_audio is not None
                        or not self._valid_released_audio(matching_audio[0])
                    ):
                        active.invalid_release = True
                    else:
                        active.released_audio = matching_audio[0]

                if event.kind is ServerEventKind.ERROR:
                    cancel_for_failure = True
                    failure = self._fail_active_locked(
                        CueDeliveryFailureReason.RESPONSE_FAILED,
                        terminal=False,
                    )
                elif event.kind is ServerEventKind.RESPONSE_DONE:
                    if response_id is None or event.response_id != response_id:
                        return False
                    mode_is_active = self._current_mode() is SessionMode.ACTIVE_EXERCISE
                    if not mode_is_active:
                        failure = self._fail_active_locked(
                            CueDeliveryFailureReason.MODE_CHANGED,
                            terminal=True,
                        )
                    elif event.response_status != "completed":
                        failure = self._fail_active_locked(
                            CueDeliveryFailureReason.RESPONSE_FAILED,
                            terminal=True,
                        )
                    elif active.invalid_release or active.released_audio is None:
                        failure = self._fail_active_locked(
                            CueDeliveryFailureReason.QUARANTINE_REJECTED,
                            terminal=True,
                        )
                    else:
                        released = active.released_audio
                        assert active.requested_at is not None
                        release = ReleasedCueAudio(
                            ticket_id=active.ticket_id,
                            authorization=active.authorization,
                            response_id=released.response_id,
                            item_id=released.item_id,
                            content_index=released.content_index,
                            pcm16_mono_24khz=released.pcm16_mono_24khz,
                            queued_at_seconds=active.queued_at,
                            requested_at_seconds=active.requested_at,
                            released_at_seconds=self._now(),
                        )
                        try:
                            # The lock makes mode-transition preemption and the
                            # final handoff linearly ordered.  The callback must
                            # only enqueue into the exclusive speaker arbiter.
                            self._on_audio(release)
                        except Exception as exc:  # scrub before surfacing it
                            callback_error = exc
                            failure = self._fail_active_locked(
                                CueDeliveryFailureReason.OUTPUT_CALLBACK_FAILED,
                                terminal=True,
                            )
                        else:
                            active.finished = True
                            self._active = None
                            self._released_count += 1
                            to_request = self._select_next_locked(self._now())
                    consumed = True
                elif response_id is not None and event.response_id == response_id:
                    if self._current_mode() is not SessionMode.ACTIVE_EXERCISE:
                        cancel_for_mode = (active, response_id)
                        failure = self._fail_active_locked(
                            CueDeliveryFailureReason.MODE_CHANGED,
                            terminal=False,
                        )
                    consumed = True
                elif matching_audio:
                    consumed = True
                else:
                    return False

        if cancel_for_mode is not None:
            self._cancel_wire(cancel_for_mode[1])
        elif failure is not None and cancel_for_failure:
            self._cancel_wire_or_revoke_pending(failure.response_id)
        if failure is not None:
            self._notify_failure(failure)
        if callback_error is not None:
            raise CueDeliveryError("approved cue output callback failed") from callback_error
        if to_request is not None:
            self._request_work(to_request)
        return consumed

    def expire_stale(self) -> int:
        """Drop stale queued work and cancel an over-time active response.

        The application should call this from its existing device-loop tick;
        this component intentionally creates no background thread.
        """

        now = self._now()
        failure: CueDeliveryFailure | None = None
        response_id: str | None = None
        with self._lock:
            before = self._stale_drop_count
            self._drop_stale_pending_locked(now)
            active = self._active
            if (
                active is not None
                and active.requested_at is not None
                and now - active.requested_at > self._config.response_timeout_seconds
            ):
                response_id = active.response_id
                failure = self._fail_active_locked(
                    CueDeliveryFailureReason.RESPONSE_TIMEOUT,
                    terminal=False,
                )
            dropped = self._stale_drop_count - before

        if failure is not None:
            callback_error = self._invoke_preempt_callback()
            cancel_error: Exception | None = None
            try:
                self._cancel_wire_or_revoke_pending(response_id)
            except Exception as exc:
                cancel_error = exc
            self._notify_failure(failure)
            if callback_error is not None or cancel_error is not None:
                cause = callback_error or cancel_error
                raise CueDeliveryError("timed-out cue preemption failed") from cause
        return dropped

    def preempt_model_audio(self) -> None:
        """Scrub all current/pending cue work before cancelling generation."""

        response_id: str | None = None
        had_active = False
        with self._lock:
            if self._closed:
                return
            active = self._active
            if active is not None:
                had_active = True
                response_id = active.response_id
                self._active = None
                self._draining = _DrainingResponse(active.ticket_id, response_id)
                self._cancelled_count += 1
            self._pending.clear()

        callback_error = self._invoke_preempt_callback()
        cancel_error: Exception | None = None
        if had_active:
            try:
                self._cancel_wire_or_revoke_pending(response_id)
            except Exception as exc:
                cancel_error = exc

        if callback_error is not None or cancel_error is not None:
            cause = callback_error or cancel_error
            raise CueDeliveryError("cue preemption failed at a local boundary") from cause

    def close(self) -> None:
        """Permanently preempt this scheduler without closing the shared session."""

        self.preempt_model_audio()
        with self._lock:
            self._closed = True

    @property
    def snapshot(self) -> CueDeliverySnapshot:
        with self._lock:
            return CueDeliverySnapshot(
                active_ticket_id=None if self._active is None else self._active.ticket_id,
                active_cue_id=(None if self._active is None else self._active.authorization.cue_id),
                pending_cue_ids=tuple(
                    work.authorization.cue_id for work in self._ordered_pending_locked()
                ),
                draining_stale_response=self._draining is not None,
                requested_count=self._requested_count,
                released_count=self._released_count,
                coalesced_count=self._coalesced_count,
                superseded_count=self._superseded_count,
                stale_drop_count=self._stale_drop_count,
                capacity_drop_count=self._capacity_drop_count,
                cancelled_count=self._cancelled_count,
                failed_count=self._failed_count,
            )

    def _request_work(self, work: _CueWork) -> None:
        if self._current_mode() is not SessionMode.ACTIVE_EXERCISE:
            with self._lock:
                if self._active is work:
                    failure = self._fail_active_locked(
                        CueDeliveryFailureReason.MODE_CHANGED,
                        terminal=True,
                    )
                    self._request_inflight = False
                else:
                    failure = None
            if failure is not None:
                self._notify_failure(failure)
            raise CueDeliveryError("prompt cues require ACTIVE_EXERCISE mode")

        try:
            self._session.request_approved_prompt_cue(work.authorization)
        except Exception:
            with self._lock:
                self._request_inflight = False
                if self._active is work:
                    # RealtimeSession preserves a pending tombstone because a
                    # failing send may already have reached the service.
                    failure = self._fail_active_locked(
                        CueDeliveryFailureReason.REQUEST_FAILED,
                        terminal=False,
                    )
                else:
                    failure = None
            try:
                self._cancel_wire_or_revoke_pending(work.response_id)
            except Exception:
                pass
            if failure is not None:
                self._notify_failure(failure)
            raise

        stale = False
        to_request: _CueWork | None = None
        with self._lock:
            self._request_inflight = False
            if self._active is work:
                self._requested_count += 1
            elif work.finished:
                # A fast in-memory transport can deliver a complete response
                # before send_event returns.  It is terminal, not stale.
                self._requested_count += 1
                to_request = self._select_next_locked(self._now())
            else:
                stale = True
        if stale:
            # A pause/stop raced the synchronous socket send.  The local state
            # was already scrubbed; close the just-sent response as well.
            self._cancel_wire_or_revoke_pending(work.response_id)
        elif to_request is not None:
            self._request_work(to_request)

    def _handle_draining_locked(
        self,
        result: RealtimeClientResult,
    ) -> tuple[bool, _CueWork | None]:
        drain = self._draining
        assert drain is not None
        event = result.event
        if event.kind is ServerEventKind.RESPONSE_CREATED and drain.response_id is None:
            drain.response_id = event.response_id
            if event.response_id is not None:
                # This out-of-band response finally has a safe wire target.
                # Keep correlation and cancellation ordered before any queued
                # cue can be dispatched on the shared long-lived session.
                self._session.cancel_response(response_id=event.response_id)
            return True, None
        if drain.response_id is None or event.response_id != drain.response_id:
            return False, None
        if event.kind is ServerEventKind.RESPONSE_DONE:
            self._draining = None
            return True, self._select_next_locked(self._now())
        return True, None

    def _fail_active_locked(
        self,
        reason: CueDeliveryFailureReason,
        *,
        terminal: bool,
    ) -> CueDeliveryFailure:
        active = self._active
        assert active is not None
        failure = CueDeliveryFailure(
            ticket_id=active.ticket_id,
            cue_id=active.authorization.cue_id,
            reason=reason,
            response_id=active.response_id,
        )
        self._active = None
        self._pending.clear()
        self._request_inflight = False
        self._failed_count += 1
        active.finished = terminal
        if not terminal:
            self._draining = _DrainingResponse(active.ticket_id, active.response_id)
        return failure

    def _select_next_locked(self, now: float) -> _CueWork | None:
        if self._closed or self._draining is not None or self._request_inflight:
            return None
        self._drop_stale_pending_locked(now)
        if not self._pending:
            return None
        self._pending.sort(key=self._sort_key)
        work = self._pending.pop(0)
        work.requested_at = now
        self._active = work
        self._request_inflight = True
        return work

    def _queue_pending_locked(self, work: _CueWork) -> CueQueueDisposition:
        self._pending.append(work)
        if len(self._pending) <= self._config.max_pending_cues:
            return CueQueueDisposition.QUEUED

        dropped = max(self._pending, key=self._sort_key)
        self._pending.remove(dropped)
        self._capacity_drop_count += 1
        if dropped is work:
            return CueQueueDisposition.DROPPED_CAPACITY
        return CueQueueDisposition.QUEUED

    def _drop_stale_pending_locked(self, now: float) -> None:
        retained: list[_CueWork] = []
        for work in self._pending:
            if now - work.queued_at > self._max_queue_age(work.lane):
                self._stale_drop_count += 1
            else:
                retained.append(work)
        self._pending = retained

    def _ordered_pending_locked(self) -> list[_CueWork]:
        return sorted(self._pending, key=self._sort_key)

    def _lane_for(self, authorization: ApprovedCuePlaybackAuthorization) -> _CueLane:
        if authorization.cue_kind is CueKind.SAFETY:
            return _CueLane.SAFETY
        if authorization.cue_id.value in self._count_cue_ids:
            return _CueLane.COUNT
        if authorization.cue_kind is CueKind.CORRECTION:
            return _CueLane.CORRECTION
        return _CueLane.OTHER

    def _max_queue_age(self, lane: _CueLane) -> float:
        if lane is _CueLane.SAFETY:
            return self._config.safety_max_queue_age_seconds
        if lane is _CueLane.COUNT:
            return self._config.count_max_queue_age_seconds
        if lane is _CueLane.CORRECTION:
            return self._config.correction_max_queue_age_seconds
        return self._config.other_max_queue_age_seconds

    @staticmethod
    def _same_cue(
        first: ApprovedCuePlaybackAuthorization,
        second: ApprovedCuePlaybackAuthorization,
    ) -> bool:
        return first.cue_id is second.cue_id and first.catalog_version == second.catalog_version

    @staticmethod
    def _sort_key(work: _CueWork) -> tuple[int, int]:
        return int(work.lane), work.ticket_id

    @staticmethod
    def _valid_released_audio(audio: ReleasedModelAudio) -> bool:
        return (
            audio.policy is ModelAudioPolicy.PROMPT_CUE_QUARANTINE
            and audio.complete
            and bool(audio.pcm16_mono_24khz)
            and len(audio.pcm16_mono_24khz) % 2 == 0
        )

    def _allocate_ticket_locked(self) -> int:
        ticket_id = self._next_ticket_id
        self._next_ticket_id += 1
        return ticket_id

    def _require_active_exercise(self) -> None:
        if self._current_mode() is not SessionMode.ACTIVE_EXERCISE:
            raise CueDeliveryError("prompt cues require ACTIVE_EXERCISE mode")

    def _current_mode(self) -> SessionMode:
        mode = self._mode_provider.current_mode
        if not isinstance(mode, SessionMode):
            raise TypeError("mode_provider.current_mode must be a SessionMode")
        return mode

    def _now(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise CueDeliveryError("clock must return a finite number")
        return float(value)

    def _cancel_wire(self, response_id: str | None) -> None:
        if response_id is None:
            raise CueDeliveryError("out-of-band response needs an ID before wire cancellation")
        try:
            self._session.cancel_response(response_id=response_id)
        except Exception as exc:
            raise CueDeliveryError("failed to cancel stale cue response") from exc

    def _cancel_wire_or_revoke_pending(self, response_id: str | None) -> None:
        if response_id is None:
            self._session.revoke_pending_response_locally()
            return
        self._cancel_wire(response_id)

    def _invoke_preempt_callback(self) -> Exception | None:
        if self._on_preempt is None:
            return None
        try:
            self._on_preempt()
        except Exception as exc:
            return exc
        return None

    def _notify_failure(self, failure: CueDeliveryFailure) -> None:
        if self._on_failure is not None:
            self._on_failure(failure)


__all__ = [
    "DEFAULT_CUE_DELIVERY_CONFIG",
    "CueDeliveryConfig",
    "CueDeliveryError",
    "CueDeliveryFailure",
    "CueDeliveryFailureReason",
    "CueDeliverySnapshot",
    "CueEnqueueResult",
    "CueQueueDisposition",
    "PromptCueSession",
    "RealtimeCueDelivery",
    "ReleasedCueAudio",
]
