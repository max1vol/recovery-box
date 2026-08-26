"""The only release gate between model audio and the speaker.

There are two intentionally explicit product lanes:

* ``CONVERSATIONAL_STREAM`` provides low-latency model audio during CHECK_IN
  and POST_SESSION.  It is an opt-in latency/safety tradeoff: audio is released
  as validated PCM deltas arrive, before a transcript exists.
* ``TRANSCRIPT_QUARANTINE`` is the strict conversational default.  It buffers
  the complete audio and releases it only after both audio and transcript
  completion and an exact match to text authorized locally.
* ``PROMPT_CUE_QUARANTINE`` applies a stricter whole-response gate to the one
  Guardian-selected cue phrase allowed during an active exercise.  It releases
  exactly one matching audio content only after ``response.done`` reports a
  completed response.

ACTIVE_EXERCISE otherwise uses ``NO_AUDIO``.  A free-form model response can
never enter the prompt-cue lane merely because it resembles an approved cue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from .protocol import ParsedServerEvent, ServerEventKind, parse_server_event

MAX_QUARANTINED_AUDIO_BYTES = 12_000_000


class AudioGateError(ValueError):
    """A response authorization is internally inconsistent or unsafe."""


class ConversationMode(StrEnum):
    IDLE = "IDLE"
    CHECK_IN = "CHECK_IN"
    ACTIVE_EXERCISE = "ACTIVE_EXERCISE"
    POST_SESSION = "POST_SESSION"


class ModelAudioPolicy(StrEnum):
    """How server-generated audio may leave the model-audio boundary."""

    TRANSCRIPT_QUARANTINE = "TRANSCRIPT_QUARANTINE"
    CONVERSATIONAL_STREAM = "CONVERSATIONAL_STREAM"
    PROMPT_CUE_QUARANTINE = "PROMPT_CUE_QUARANTINE"
    NO_AUDIO = "NO_AUDIO"


@dataclass(frozen=True, slots=True)
class ReleasedModelAudio:
    """PCM that the gate has explicitly approved for downstream playback."""

    response_id: str
    item_id: str
    content_index: int
    pcm16_mono_24khz: bytes
    policy: ModelAudioPolicy
    complete: bool


@dataclass(frozen=True, slots=True)
class ResponseAuthorization:
    mode: ConversationMode
    policy: ModelAudioPolicy
    authorized_text: str | None = None


@dataclass(slots=True)
class _Quarantine:
    chunks: list[bytes] = field(default_factory=list)
    byte_count: int = 0
    audio_done: bool = False
    transcript_done: bool = False
    transcript: str | None = None


class ModelAudioGate:
    """Stateful release gate designed for deterministic event replay tests.

    Call :meth:`authorize_next_response` immediately before sending
    ``response.create``.  The authorization binds FIFO to the next unique
    ``response.created`` event.  A request cancelled before its
    ``response.created`` arrives leaves a FIFO tombstone: the corresponding
    late creation is consumed and permanently closed instead of stealing a
    later turn's authorization.

    The Realtime wire events do not echo a caller-selected correlation ID on
    ``response.created``.  FIFO ordering is therefore the only available
    request-to-response binding.  If the service omits a creation for a
    cancelled request, its tombstone conservatively denies the next creation;
    this can lose a response, but it cannot release the wrong response.  A
    reconnect/reset clears tombstones because old events cannot cross the new
    transport boundary.
    """

    def __init__(self) -> None:
        # ``None`` is a cancelled request slot.  It must remain in FIFO order
        # until one response.created consumes it.
        self._pending: deque[ResponseAuthorization | None] = deque()
        self._authorizations: dict[str, ResponseAuthorization] = {}
        self._quarantines: dict[tuple[str, str, int], _Quarantine] = {}
        self._closed_responses: set[str] = set()
        self._closed_content: set[tuple[str, str, int]] = set()
        self._seen_event_ids: set[str] = set()
        self.blocked_audio_bytes = 0

    @property
    def pending_authorizations(self) -> int:
        return sum(authorization is not None for authorization in self._pending)

    @property
    def pending_cancellation_tombstones(self) -> int:
        return sum(authorization is None for authorization in self._pending)

    @property
    def open_authorizations(self) -> int:
        """Count live pending and response-bound authorizations."""

        return self.pending_authorizations + len(self._authorizations)

    @property
    def quarantined_audio_bytes(self) -> int:
        return sum(state.byte_count for state in self._quarantines.values())

    def authorize_next_response(
        self,
        *,
        mode: ConversationMode,
        policy: ModelAudioPolicy = ModelAudioPolicy.TRANSCRIPT_QUARANTINE,
        authorized_text: str | None = None,
    ) -> None:
        """Declare the local audio policy before a model response is requested.

        ``CONVERSATIONAL_STREAM`` is accepted only for CHECK_IN/POST_SESSION.
        ``TRANSCRIPT_QUARANTINE`` requires nonblank locally authorized text.
        ACTIVE_EXERCISE accepts either ``NO_AUDIO`` for tool-only work or
        ``PROMPT_CUE_QUARANTINE`` for one exact phrase already selected by the
        local Guardian.
        """

        authorization = self.validate_next_response(
            mode=mode,
            policy=policy,
            authorized_text=authorized_text,
        )
        self._pending.append(authorization)

    def validate_next_response(
        self,
        *,
        mode: ConversationMode,
        policy: ModelAudioPolicy = ModelAudioPolicy.TRANSCRIPT_QUARANTINE,
        authorized_text: str | None = None,
    ) -> ResponseAuthorization:
        """Validate a proposed authorization without changing gate state."""

        if mode is ConversationMode.ACTIVE_EXERCISE:
            if policy is ModelAudioPolicy.PROMPT_CUE_QUARANTINE:
                if authorized_text is None or not authorized_text.strip():
                    raise AudioGateError("PROMPT_CUE_QUARANTINE needs a nonblank catalog phrase")
            elif policy is ModelAudioPolicy.NO_AUDIO:
                if authorized_text is not None:
                    raise AudioGateError("NO_AUDIO cannot authorize model text")
            else:
                raise AudioGateError(
                    "ACTIVE_EXERCISE responses require NO_AUDIO or PROMPT_CUE_QUARANTINE"
                )
        elif policy is ModelAudioPolicy.PROMPT_CUE_QUARANTINE:
            raise AudioGateError("prompt cues are limited to ACTIVE_EXERCISE")
        elif policy is ModelAudioPolicy.NO_AUDIO:
            if authorized_text is not None:
                raise AudioGateError("NO_AUDIO cannot authorize model text")
        elif policy is ModelAudioPolicy.CONVERSATIONAL_STREAM:
            if mode not in {ConversationMode.CHECK_IN, ConversationMode.POST_SESSION}:
                raise AudioGateError(
                    "CONVERSATIONAL_STREAM is limited to CHECK_IN and POST_SESSION"
                )
            if authorized_text is not None:
                raise AudioGateError(
                    "CONVERSATIONAL_STREAM does not claim transcript authorization"
                )
        elif policy is ModelAudioPolicy.TRANSCRIPT_QUARANTINE:
            if authorized_text is None or not authorized_text.strip():
                raise AudioGateError("TRANSCRIPT_QUARANTINE needs nonblank locally authorized text")
        else:  # Defensive if a non-enum value crosses a Python boundary.
            raise AudioGateError("unsupported model audio policy")

        return ResponseAuthorization(
            mode=mode,
            policy=policy,
            authorized_text=authorized_text,
        )

    def ingest_raw(self, raw: dict[str, object]) -> tuple[ReleasedModelAudio, ...]:
        return self.ingest(parse_server_event(raw))

    def ingest(self, event: ParsedServerEvent) -> tuple[ReleasedModelAudio, ...]:
        """Process one validated server event and return only releasable audio."""

        if event.event_id is not None:
            if event.event_id in self._seen_event_ids:
                return ()
            self._seen_event_ids.add(event.event_id)

        if event.kind is ServerEventKind.RESPONSE_CREATED:
            self._bind_created_response(event)
            return ()

        if event.kind is ServerEventKind.RESPONSE_DONE:
            return self._finish_response(event)

        if event.kind is ServerEventKind.ERROR:
            # An error has no safely attributable audio.  Pending authorizations
            # are retained because the API may report non-response errors.
            return ()

        if event.kind not in {
            ServerEventKind.AUDIO_DELTA,
            ServerEventKind.AUDIO_DONE,
            ServerEventKind.TRANSCRIPT_DONE,
        }:
            return ()

        response_id = _present(event.response_id)
        item_id = _present(event.item_id)
        key = (response_id, item_id, event.content_index)
        authorization = self._authorizations.get(response_id)
        if (
            authorization is None
            or response_id in self._closed_responses
            or key in self._closed_content
        ):
            self._count_blocked_delta(event)
            return ()

        if authorization.policy is ModelAudioPolicy.NO_AUDIO:
            self._count_blocked_delta(event)
            return ()

        if authorization.policy is ModelAudioPolicy.CONVERSATIONAL_STREAM:
            if event.kind is not ServerEventKind.AUDIO_DELTA:
                return ()
            audio = _present_audio(event.audio)
            return (
                ReleasedModelAudio(
                    response_id=response_id,
                    item_id=item_id,
                    content_index=event.content_index,
                    pcm16_mono_24khz=audio,
                    policy=authorization.policy,
                    complete=False,
                ),
            )

        return self._ingest_quarantined(key, authorization, event)

    def discard_response(self, response_id: str) -> None:
        """Fail-close a cancelled/interrupted response and erase buffered audio."""

        if not response_id.strip():
            raise AudioGateError("response id must not be blank")
        for key in tuple(self._quarantines):
            if key[0] == response_id:
                state = self._quarantines.pop(key)
                self.blocked_audio_bytes += state.byte_count
                self._closed_content.add(key)
        self._authorizations.pop(response_id, None)
        self._closed_responses.add(response_id)

    def cancel_pending_authorizations(self) -> int:
        """Replace unbound authorizations with fail-closed FIFO tombstones.

        Returning the number cancelled lets orchestration code distinguish a
        queued response from an idle session without exposing authorization
        content.
        """

        cancelled = 0
        retained: deque[ResponseAuthorization | None] = deque()
        for authorization in self._pending:
            if authorization is None:
                retained.append(None)
            else:
                retained.append(None)
                cancelled += 1
        self._pending = retained
        return cancelled

    def is_response_authorized(self, response_id: str | None) -> bool:
        """Return whether a response is currently bound to a live policy."""

        return (
            response_id is not None
            and response_id in self._authorizations
            and response_id not in self._closed_responses
        )

    def discard_all(self) -> None:
        """Erase every queued/bound model response, for disconnect or reset."""

        bound_response_ids = set(self._authorizations)
        for state in self._quarantines.values():
            self.blocked_audio_bytes += state.byte_count
        self._quarantines.clear()
        self._pending.clear()
        self._authorizations.clear()
        self._closed_responses.update(bound_response_ids)

    def _bind_created_response(self, event: ParsedServerEvent) -> None:
        response_id = _present(event.response_id)
        if response_id in self._authorizations or response_id in self._closed_responses:
            return
        if not self._pending:
            self._closed_responses.add(response_id)
            return
        authorization = self._pending.popleft()
        if authorization is None:
            self._closed_responses.add(response_id)
            return
        self._authorizations[response_id] = authorization

    def _ingest_quarantined(
        self,
        key: tuple[str, str, int],
        authorization: ResponseAuthorization,
        event: ParsedServerEvent,
    ) -> tuple[ReleasedModelAudio, ...]:
        state = self._quarantines.setdefault(key, _Quarantine())
        if event.kind is ServerEventKind.AUDIO_DELTA:
            audio = _present_audio(event.audio)
            if state.audio_done:
                # A fresh event ID cannot smuggle additional, untranscribed PCM
                # into content that the server already declared complete.
                self.blocked_audio_bytes += len(audio)
                self._deny_content(key)
                return ()
            next_count = state.byte_count + len(audio)
            if next_count > MAX_QUARANTINED_AUDIO_BYTES:
                self.blocked_audio_bytes += next_count
                self._quarantines.pop(key, None)
                self._closed_content.add(key)
                return ()
            state.chunks.append(audio)
            state.byte_count = next_count
        elif event.kind is ServerEventKind.AUDIO_DONE:
            state.audio_done = True
        elif event.kind is ServerEventKind.TRANSCRIPT_DONE:
            if state.transcript_done:
                # Conflicting duplicate completion is never allowed to replace
                # the transcript that was first observed.
                if state.transcript != event.transcript:
                    self._deny_content(key)
                return ()
            state.transcript_done = True
            state.transcript = event.transcript

        if not state.audio_done or not state.transcript_done:
            return ()

        if state.transcript != authorization.authorized_text:
            self._deny_content(key)
            return ()

        if authorization.policy is ModelAudioPolicy.PROMPT_CUE_QUARANTINE:
            # A prompt cue is not authorized by content completion alone.  It
            # stays buffered until the terminal response event proves that the
            # *whole* response completed and contained exactly one audio
            # content.  This also prevents a valid first content from escaping
            # before a second, unexpected content arrives.
            return ()

        released = self._release_content(key, authorization)
        return () if released is None else (released,)

    def _release_content(
        self,
        key: tuple[str, str, int],
        authorization: ResponseAuthorization,
    ) -> ReleasedModelAudio | None:
        """Atomically close one quarantine and return its approved PCM."""

        state = self._quarantines.pop(key, None)
        self._closed_content.add(key)
        if state is None:
            return None
        audio = b"".join(state.chunks)
        if not audio:
            return None
        return ReleasedModelAudio(
            response_id=key[0],
            item_id=key[1],
            content_index=key[2],
            pcm16_mono_24khz=audio,
            policy=authorization.policy,
            complete=True,
        )

    def _deny_content(self, key: tuple[str, str, int]) -> None:
        state = self._quarantines.pop(key, None)
        if state is not None:
            self.blocked_audio_bytes += state.byte_count
        self._closed_content.add(key)

    def _finish_response(
        self,
        event: ParsedServerEvent,
    ) -> tuple[ReleasedModelAudio, ...]:
        response_id = _present(event.response_id)
        authorization = self._authorizations.get(response_id)
        released: ReleasedModelAudio | None = None

        if (
            authorization is not None
            and authorization.policy is ModelAudioPolicy.PROMPT_CUE_QUARANTINE
            and event.response_status == "completed"
        ):
            # Count every audio content observed for the response, including a
            # content already denied for a mismatch or size violation.  A
            # matching content may be released only when it is the sole audio
            # content produced by the response.
            content_keys = {
                key for key in (*self._quarantines, *self._closed_content) if key[0] == response_id
            }
            if len(content_keys) == 1:
                key = next(iter(content_keys))
                state = self._quarantines.get(key)
                if state is not None and state.audio_done and state.transcript_done:
                    if state.transcript == authorization.authorized_text:
                        released = self._release_content(key, authorization)

        # response.done is terminal.  Any incomplete, failed, cancelled,
        # mismatched, oversized, or multi-content quarantine is denied so a
        # late/replayed event cannot turn it into audible output.
        for key in tuple(self._quarantines):
            if key[0] == response_id:
                self._deny_content(key)
        self._authorizations.pop(response_id, None)
        self._closed_responses.add(response_id)
        return () if released is None else (released,)

    def _count_blocked_delta(self, event: ParsedServerEvent) -> None:
        if event.kind is ServerEventKind.AUDIO_DELTA and event.audio is not None:
            self.blocked_audio_bytes += len(event.audio)


def normalize_authorized_text(text: str) -> str:
    """Return authorization text unchanged.

    The compatibility name remains public for existing callers, but the audio
    release boundary deliberately performs no Unicode, case, punctuation, or
    whitespace normalization.  Only literal string equality authorizes PCM.
    """

    return text


def _present(value: str | None) -> str:
    if value is None or not value:
        raise AudioGateError("parsed audio event is missing its identifier")
    return value


def _present_audio(value: bytes | None) -> bytes:
    if value is None or not value:
        raise AudioGateError("parsed audio delta is missing PCM data")
    return value
