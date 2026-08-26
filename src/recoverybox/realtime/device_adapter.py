"""Composition adapter between the physical controller and Realtime session."""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from recoverybox.core import SessionMode
from recoverybox.device.ports import PCM_S16LE_24K_MONO
from recoverybox.session import SessionModeProvider, session_mode_allows_model_audio

from .client import RealtimeClientResult, RealtimeSession
from .protocol import ServerEventKind
from .safety import ConversationMode, ModelAudioPolicy

if TYPE_CHECKING:
    from recoverybox.device.ports import AudioFormat


AUDIO_APPEND_CHUNK_BYTES = 32_000


@dataclass(frozen=True, slots=True)
class RealtimeTurnPolicy:
    """Realtime wire mode and audio gate selected from one product mode."""

    conversation_mode: ConversationMode
    audio_policy: ModelAudioPolicy

    @property
    def model_audio_allowed(self) -> bool:
        return self.audio_policy is ModelAudioPolicy.CONVERSATIONAL_STREAM


def select_realtime_turn_policy(mode: SessionMode) -> RealtimeTurnPolicy:
    """Fail closed outside the two explicitly conversational product phases.

    ``SessionMode.COMPLETE`` is the product's post-session phase.  All other
    modes request tool-only responses guarded by ``NO_AUDIO``.
    """

    if not isinstance(mode, SessionMode):
        raise TypeError("mode must be a SessionMode")
    if mode is SessionMode.CHECK_IN:
        return RealtimeTurnPolicy(
            ConversationMode.CHECK_IN,
            ModelAudioPolicy.CONVERSATIONAL_STREAM,
        )
    if mode is SessionMode.COMPLETE:
        return RealtimeTurnPolicy(
            ConversationMode.POST_SESSION,
            ModelAudioPolicy.CONVERSATIONAL_STREAM,
        )
    return RealtimeTurnPolicy(
        ConversationMode.ACTIVE_EXERCISE,
        ModelAudioPolicy.NO_AUDIO,
    )


def _policy_allows_audio_in_mode(policy: ModelAudioPolicy, mode: SessionMode) -> bool:
    if policy is ModelAudioPolicy.CONVERSATIONAL_STREAM:
        return session_mode_allows_model_audio(mode)
    return False


@runtime_checkable
class RealtimeEventSink(Protocol):
    """Callbacks implemented by the deterministic device controller."""

    def on_response_started(self, *, turn_id: str, response_id: str) -> None: ...

    def on_response_audio(
        self,
        *,
        turn_id: str,
        response_id: str,
        item_id: str,
        pcm: bytes,
    ) -> None: ...

    def on_response_done(self, *, turn_id: str, response_id: str | None) -> None: ...

    def on_response_error(
        self,
        *,
        turn_id: str,
        response_id: str | None,
        error: Exception,
    ) -> None: ...


class RealtimeConversationAdapter:
    """Implement the device ``ConversationPort`` over ``RealtimeSession``.

    Every turn reads an explicit product mode. CHECK_IN and the post-session
    COMPLETE phase select ``CONVERSATIONAL_STREAM``. Every other ordinary turn
    selects ``NO_AUDIO``. Prompt cues use a separate speaker-arbitration
    composition so ordinary button turns cannot acquire that capability. A
    response that began in a conversational phase is revoked if the product
    mode changes before later bytes are delivered to the sink.

    Reception is pumped separately through :meth:`pump_once`, allowing the app
    to choose a thread/event-loop strategy without putting one inside the
    safety-critical device controller.
    """

    def __init__(
        self,
        *,
        session: RealtimeSession,
        sink: RealtimeEventSink,
        mode_provider: SessionModeProvider,
    ) -> None:
        self._session = session
        self._sink = sink
        self._mode_provider = mode_provider
        self._turn_counter = itertools.count(1)
        self._lock = threading.RLock()
        self._turn_id: str | None = None
        self._response_id: str | None = None
        self._turn_audio_policy: ModelAudioPolicy | None = None
        self._ended = False

    @property
    def active_turn_id(self) -> str | None:
        with self._lock:
            return self._turn_id

    def send_audio_turn(self, pcm: bytes, *, audio_format: AudioFormat) -> str:
        if audio_format != PCM_S16LE_24K_MONO:
            raise ValueError("Realtime requires signed PCM16 mono at 24 kHz")
        if not pcm:
            raise ValueError("audio turn must not be empty")
        if len(pcm) % 2:
            raise ValueError("PCM16 turn must contain complete samples")

        policy = select_realtime_turn_policy(self._current_mode())
        with self._lock:
            if self._ended:
                raise RuntimeError("conversation has ended")
            if self._turn_id is not None:
                raise RuntimeError("a conversation turn is already active")
            turn_id = f"realtime-turn-{next(self._turn_counter)}"
            self._turn_id = turn_id
            self._response_id = None
            self._turn_audio_policy = policy.audio_policy
            try:
                for offset in range(0, len(pcm), AUDIO_APPEND_CHUNK_BYTES):
                    self._session.append_user_audio(pcm[offset : offset + AUDIO_APPEND_CHUNK_BYTES])
                self._session.finish_user_turn_and_request(
                    mode=policy.conversation_mode,
                    policy=policy.audio_policy,
                )
            except Exception:
                self._turn_id = None
                self._turn_audio_policy = None
                try:
                    self._session.clear_input_audio()
                except Exception:
                    pass
                raise
            return turn_id

    def cancel_response(self, turn_id: str, response_id: str | None) -> None:
        with self._lock:
            if turn_id != self._turn_id:
                return
            if (
                response_id is not None
                and self._response_id is not None
                and response_id != self._response_id
            ):
                return
            # Before response.created there is no response ID that can safely
            # identify the queued authorization, so cancel the whole pending
            # gate rather than trusting a caller-supplied ID.
            bound_response_id = self._response_id
            self._session.cancel_response(response_id=bound_response_id)
            self._turn_id = None
            self._response_id = None
            self._turn_audio_policy = None

    def preempt_model_audio(self) -> None:
        """Revoke pending/bound model audio after an edge-mode safety change.

        This closes the local model-audio gate before returning.  Physical
        playback preemption is a separate edge-composition port owned by the
        session coordinator.
        """

        with self._lock:
            if self._ended or self._turn_id is None:
                return
            self._session.cancel_response(response_id=self._response_id)
            self._turn_id = None
            self._response_id = None
            self._turn_audio_policy = None

    def truncate_assistant(self, item_id: str, audio_end_ms: int) -> None:
        # The controller calls this after cancellation and therefore supplies
        # the authoritative assistant item ID explicitly.
        self._session.truncate_assistant(
            item_id=item_id,
            audio_end_ms=audio_end_ms,
        )

    def clear_and_end(self) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
            self._turn_id = None
            self._response_id = None
            self._turn_audio_policy = None
        try:
            self._session.clear_input_audio()
        finally:
            self._session.close()

    def pump_once(self) -> RealtimeClientResult:
        """Compatibility wrapper for apps where this adapter owns reception."""

        result = self._session.receive_once()
        return self.handle_result(result)

    def handle_result(self, result: RealtimeClientResult) -> RealtimeClientResult:
        """Dispatch an already-gated event from the application's sole reader.

        Product composition should prefer this method so prompt cues, session
        control tools, and ordinary button turns never race separate reads on
        the same long-lived Realtime WebSocket.
        """

        if not isinstance(result, RealtimeClientResult):
            raise TypeError("result must be a RealtimeClientResult")
        current_mode = self._current_mode()
        with self._lock:
            turn_id = self._turn_id
            if self._ended or turn_id is None:
                # A concurrent preemption can clear the turn after
                # RealtimeSession has released a streaming delta but before
                # this adapter reacquires its correlation lock. Never expose
                # those now-stale bytes through the return value.
                return replace(result, released_audio=())

            # A mode transition can invalidate an authorization that was safe
            # when response.create was sent. Close it before any returned bytes
            # are handed to the speaker-facing sink.
            audio_policy = self._turn_audio_policy
            if (
                audio_policy is not None
                and not _policy_allows_audio_in_mode(
                    audio_policy,
                    current_mode,
                )
                and audio_policy is not ModelAudioPolicy.NO_AUDIO
            ):
                response_id = self._response_id or result.event.response_id
                self._session.cancel_response(response_id=response_id)
                self._turn_id = None
                self._response_id = None
                self._turn_audio_policy = None
                return replace(result, released_audio=())

            event = result.event
            if event.kind is ServerEventKind.RESPONSE_CREATED:
                response_id = event.response_id
                if response_id is not None and result.response_authorized:
                    self._response_id = response_id
                    if _policy_allows_audio_in_mode(audio_policy, current_mode):
                        self._sink.on_response_started(
                            turn_id=turn_id,
                            response_id=response_id,
                        )

            for audio in result.released_audio:
                if audio.response_id != self._response_id:
                    continue
                if audio_policy is None or not _policy_allows_audio_in_mode(
                    audio_policy,
                    self._current_mode(),
                ):
                    self._session.cancel_response(response_id=self._response_id)
                    self._turn_id = None
                    self._response_id = None
                    self._turn_audio_policy = None
                    return replace(result, released_audio=())
                self._sink.on_response_audio(
                    turn_id=turn_id,
                    response_id=audio.response_id,
                    item_id=audio.item_id,
                    pcm=audio.pcm16_mono_24khz,
                )

            if event.kind is ServerEventKind.RESPONSE_DONE:
                response_id = event.response_id
                if response_id != self._response_id:
                    return result
                if event.response_status == "completed":
                    self._sink.on_response_done(
                        turn_id=turn_id,
                        response_id=response_id,
                    )
                else:
                    self._sink.on_response_error(
                        turn_id=turn_id,
                        response_id=response_id,
                        error=RuntimeError(f"Realtime response ended with {event.response_status}"),
                    )
                self._turn_id = None
                self._response_id = None
                self._turn_audio_policy = None
            elif event.kind is ServerEventKind.ERROR:
                self._sink.on_response_error(
                    turn_id=turn_id,
                    response_id=self._response_id,
                    error=RuntimeError(event.error_message or "Realtime error"),
                )
                self._turn_id = None
                self._response_id = None
                self._turn_audio_policy = None

        return result

    def _current_mode(self) -> SessionMode:
        mode = self._mode_provider.current_mode
        if not isinstance(mode, SessionMode):
            raise TypeError("mode_provider.current_mode must be a SessionMode")
        return mode
