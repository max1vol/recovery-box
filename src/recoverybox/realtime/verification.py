"""Deterministic latency and audio verification for prompt-delivered cues.

The harness in this module is deliberately outside the speaker path.  It asks
an injected :class:`~recoverybox.realtime.client.RealtimeSession` for one typed
Guardian cue, replays server events through that session's normal audio gate,
and persists only PCM that the gate releases.  A secondary ASR verifier can
check the released WAV, but it never participates in the release decision.
"""

from __future__ import annotations

import json
import math
import ssl
import time
import unicodedata
import urllib.request
import uuid
import wave
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import certifi

from recoverybox.core import DEFAULT_CUE_CATALOG, ApprovedCueCatalog
from recoverybox.session import ApprovedCuePlaybackAuthorization

from .client import RealtimeClientResult
from .protocol import (
    PCM_CHANNELS,
    PCM_SAMPLE_RATE_HZ,
    PCM_SAMPLE_WIDTH_BYTES,
    ParsedServerEvent,
    RealtimeProtocolError,
    ServerEventKind,
)
from .safety import AudioGateError, normalize_authorized_text
from .transport import RealtimeTransport

OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
SUPPORTED_ASR_MODELS = frozenset({"gpt-transcribe", "whisper-1"})
MAX_TRANSCRIPTION_RESPONSE_BYTES = 4_000_000
_SAFE_ASR_ERROR_MESSAGES = frozenset(
    {
        "could not read WAV for transcription",
        "OpenAI transcription request failed",
        "OpenAI transcription response was invalid",
        "OpenAI transcription response was too large",
    }
)


def _open_with_verified_openai_tls(request: urllib.request.Request, *, timeout: float) -> Any:
    """Open an OpenAI HTTPS request with the project's pinned CA bundle."""

    context = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.urlopen(request, timeout=timeout, context=context)


class CueVerificationError(RuntimeError):
    """A cue could not be verified to a terminal Realtime response."""


class ASRVerificationError(RuntimeError):
    """A secondary transcription could not be completed safely."""


@runtime_checkable
class CueRealtimeSession(Protocol):
    """Narrow session surface required by :class:`RealtimeCueVerifier`."""

    def request_approved_prompt_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None: ...

    def handle_event(self, raw: Mapping[str, Any]) -> RealtimeClientResult: ...


@dataclass(frozen=True, slots=True)
class ASRWord:
    """One word timestamp returned by ``whisper-1``."""

    word: str
    start_ms: float
    end_ms: float

    def to_dict(self) -> dict[str, float]:
        """Serialize timings without retaining provider-returned word text."""

        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


@dataclass(frozen=True, slots=True)
class ASRVerificationResult:
    """Independent ASR comparison against the catalog phrase.

    ``transcript_text`` is omitted by default by the production verifier so
    transcript retention remains opt-in.  Word text is present only when the
    caller explicitly enables Whisper word timestamps.
    """

    model: str
    matches_expected: bool
    transcript_text: str | None = None
    words: tuple[ASRWord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return content-free ASR evidence suitable for persisted reports."""

        return {
            "model": self.model,
            "matches_expected": self.matches_expected,
            "word_timestamps_ms": [word.to_dict() for word in self.words],
        }


@runtime_checkable
class ASRVerifier(Protocol):
    """Optional verifier for a WAV that already cleared cue quarantine."""

    def verify(
        self,
        *,
        wav_path: Path,
        expected_text: str,
    ) -> ASRVerificationResult: ...


@dataclass(frozen=True, slots=True)
class CueVerificationRequest:
    """One typed cue authorization and its caller-selected output path."""

    authorization: ApprovedCuePlaybackAuthorization
    wav_path: Path


@dataclass(frozen=True, slots=True)
class CueVerificationReport:
    """JSON-safe timing and content-verification result for one cue."""

    cue_id: str
    catalog_version: str
    response_id: str | None
    response_status: str | None
    response_created_ms: float | None
    first_audio_delta_ms: float | None
    transcript_done_ms: float | None
    quarantine_release_ms: float | None
    response_done_ms: float | None
    received_pcm_bytes: int
    released_pcm_bytes: int
    released_audio_duration_ms: float
    quarantine_released: bool
    realtime_transcript_matches: bool | None
    realtime_transcript_text: str | None
    wav_path: str | None
    asr: ASRVerificationResult | None = None
    asr_verification_ms: float | None = None
    asr_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return content-free evidence accepted by ``json.dumps``.

        Provider-controlled response identifiers, status prose, and retained
        transcripts stay on the in-memory report for an explicitly opted-in
        caller. They never enter the default persisted/stdout representation.
        """

        return {
            "cue_id": self.cue_id,
            "catalog_version": self.catalog_version,
            "response_completed": self.response_status == "completed",
            "latency_ms": {
                "response_created": self.response_created_ms,
                "first_audio_delta": self.first_audio_delta_ms,
                "transcript_done": self.transcript_done_ms,
                "quarantine_release": self.quarantine_release_ms,
                "response_done": self.response_done_ms,
            },
            "released_audio_duration_ms": self.released_audio_duration_ms,
            "quarantine_released": self.quarantine_released,
            "realtime_transcript_matches": self.realtime_transcript_matches,
            "wav_path": self.wav_path,
            "asr": None if self.asr is None else self.asr.to_dict(),
            "asr_verification_ms": self.asr_verification_ms,
            "asr_error": self.asr_error,
        }


class RealtimeCueVerifier:
    """Measure a typed prompt cue by consuming an injected Realtime stream.

    The transport and session are separate dependencies so deterministic tests
    can advance an injected clock as each simulated server event is delivered.
    The session must use the same transport for outgoing cue requests.
    """

    def __init__(
        self,
        *,
        session: CueRealtimeSession,
        transport: RealtimeTransport,
        clock: Callable[[], float] = time.monotonic,
        cue_catalog: ApprovedCueCatalog = DEFAULT_CUE_CATALOG,
        asr_verifier: ASRVerifier | None = None,
        retain_realtime_transcript: bool = False,
        max_events_per_cue: int = 1_000,
    ) -> None:
        if not isinstance(max_events_per_cue, int) or isinstance(max_events_per_cue, bool):
            raise TypeError("max_events_per_cue must be an integer")
        if max_events_per_cue <= 0:
            raise ValueError("max_events_per_cue must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        session_transport = getattr(session, "transport", transport)
        if session_transport is not transport:
            raise ValueError("session and verifier must use the same Realtime transport")
        self._session = session
        self._transport = transport
        self._clock = clock
        self._cue_catalog = cue_catalog
        self._asr_verifier = asr_verifier
        self._retain_realtime_transcript = retain_realtime_transcript
        self._max_events_per_cue = max_events_per_cue

    def verify_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
        *,
        wav_path: str | Path,
    ) -> CueVerificationReport:
        """Request one cue and consume events through its terminal response."""

        if not isinstance(authorization, ApprovedCuePlaybackAuthorization):
            raise TypeError("authorization must be ApprovedCuePlaybackAuthorization")
        try:
            cue = self._cue_catalog[authorization.cue_id.value]
        except KeyError as exc:
            raise CueVerificationError("cue authorization is not in the verifier catalog") from exc
        if cue.kind is not authorization.cue_kind:
            raise CueVerificationError("cue authorization kind does not match verifier catalog")

        request_started = self._read_clock()
        last_timestamp = request_started
        try:
            self._session.request_approved_prompt_cue(authorization)
        except AudioGateError:
            raise CueVerificationError(
                "Realtime cue request failed local safety validation"
            ) from None
        except RealtimeProtocolError:
            raise CueVerificationError(
                "Realtime cue request failed local protocol validation"
            ) from None
        except Exception:
            # WebSocket failures can retain request headers or a server body in
            # their exception objects.  Do not chain them into CLI output.
            raise CueVerificationError("Realtime cue request could not be sent") from None

        response_id: str | None = None
        response_status: str | None = None
        response_created_at: float | None = None
        first_audio_at: float | None = None
        transcript_done_at: float | None = None
        quarantine_release_at: float | None = None
        response_done_at: float | None = None
        realtime_transcript: str | None = None
        realtime_transcript_matches: bool | None = None
        received_pcm_bytes = 0
        released_chunks: list[bytes] = []

        for _ in range(self._max_events_per_cue):
            try:
                raw = self._transport.receive_event()
            except EOFError:
                raise CueVerificationError(
                    "Realtime event stream ended before the cue response completed"
                ) from None
            except RealtimeProtocolError:
                raise CueVerificationError(
                    "Realtime event stream failed local protocol validation"
                ) from None
            except Exception:
                raise CueVerificationError(
                    "Realtime event stream failed before the cue response completed"
                ) from None
            received_at = self._read_clock()
            if received_at < last_timestamp:
                raise CueVerificationError("verification clock moved backwards")
            try:
                result = self._session.handle_event(raw)
            except RealtimeProtocolError:
                raise CueVerificationError(
                    "Realtime server event failed local protocol validation"
                ) from None
            except AudioGateError:
                raise CueVerificationError(
                    "Realtime server event failed the local audio safety gate"
                ) from None
            except Exception:
                # An injected session implementation must not be able to copy
                # a raw event, transport object, or credential into CLI output.
                raise CueVerificationError("Realtime server event processing failed") from None
            handled_at = self._read_clock()
            if handled_at < received_at:
                raise CueVerificationError("verification clock moved backwards")
            last_timestamp = handled_at
            event = result.event

            if event.kind is ServerEventKind.ERROR:
                raise CueVerificationError(_safe_realtime_error_summary(event))

            if event.kind is ServerEventKind.RESPONSE_CREATED:
                if not result.response_authorized:
                    continue
                if response_id is not None and event.response_id != response_id:
                    raise CueVerificationError("more than one response bound to one cue request")
                response_id = event.response_id
                if response_created_at is None:
                    response_created_at = received_at
                continue

            if response_id is None or event.response_id != response_id:
                continue

            if event.kind is ServerEventKind.AUDIO_DELTA:
                if event.audio is not None:
                    received_pcm_bytes += len(event.audio)
                if first_audio_at is None:
                    first_audio_at = received_at
            elif event.kind is ServerEventKind.TRANSCRIPT_DONE:
                if transcript_done_at is None:
                    transcript_done_at = received_at
                    realtime_transcript = event.transcript
                    realtime_transcript_matches = normalize_authorized_text(
                        event.transcript or ""
                    ) == normalize_authorized_text(cue.spoken_text)

            if result.released_audio:
                if quarantine_release_at is None:
                    quarantine_release_at = handled_at
                released_chunks.extend(
                    released.pcm16_mono_24khz for released in result.released_audio
                )

            if event.kind is ServerEventKind.RESPONSE_DONE:
                response_done_at = received_at
                response_status = event.response_status
                break
        else:
            raise CueVerificationError("cue response exceeded the event verification limit")

        released_pcm = b"".join(released_chunks)
        written_path: str | None = None
        asr_result: ASRVerificationResult | None = None
        asr_verification_ms: float | None = None
        asr_error: str | None = None
        output_path = Path(wav_path)
        if released_pcm:
            write_pcm16_wav(output_path, released_pcm)
            written_path = str(output_path)
            if self._asr_verifier is not None:
                asr_started = self._read_clock()
                try:
                    asr_result = self._asr_verifier.verify(
                        wav_path=output_path,
                        expected_text=cue.spoken_text,
                    )
                except ASRVerificationError as exc:
                    message = str(exc)
                    asr_error = (
                        message
                        if message in _SAFE_ASR_ERROR_MESSAGES
                        else "ASR verification failed"
                    )
                except Exception:
                    # An injected ASR implementation may carry an HTTP body,
                    # request object, or credential in its exception.  Reports
                    # retain only this stable failure category.
                    asr_error = "ASR verification failed"
                asr_finished = self._read_clock()
                if asr_finished < asr_started:
                    raise CueVerificationError("verification clock moved backwards during ASR")
                asr_verification_ms = _elapsed_ms(asr_started, asr_finished)

        report = CueVerificationReport(
            cue_id=authorization.cue_id.value,
            catalog_version=authorization.catalog_version,
            response_id=response_id,
            response_status=response_status,
            response_created_ms=_optional_elapsed_ms(request_started, response_created_at),
            first_audio_delta_ms=_optional_elapsed_ms(request_started, first_audio_at),
            transcript_done_ms=_optional_elapsed_ms(request_started, transcript_done_at),
            quarantine_release_ms=_optional_elapsed_ms(
                request_started,
                quarantine_release_at,
            ),
            response_done_ms=_optional_elapsed_ms(request_started, response_done_at),
            received_pcm_bytes=received_pcm_bytes,
            released_pcm_bytes=len(released_pcm),
            released_audio_duration_ms=_pcm_duration_ms(released_pcm),
            quarantine_released=bool(released_pcm),
            realtime_transcript_matches=realtime_transcript_matches,
            realtime_transcript_text=(
                realtime_transcript if self._retain_realtime_transcript else None
            ),
            wav_path=written_path,
            asr=asr_result,
            asr_verification_ms=asr_verification_ms,
            asr_error=asr_error,
        )
        # Keep this invariant close to report construction.  Future fields must
        # remain serializable so simulation reports never need custom encoders.
        json.dumps(report.to_dict(), allow_nan=False)
        return report

    def verify_many(
        self,
        requests: Iterable[CueVerificationRequest],
    ) -> tuple[CueVerificationReport, ...]:
        """Verify cue requests sequentially on one injected session."""

        return tuple(
            self.verify_cue(request.authorization, wav_path=request.wav_path)
            for request in requests
        )

    def _read_clock(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CueVerificationError("verification clock must return a number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise CueVerificationError("verification clock must return a finite number")
        return numeric


class OpenAITranscriptionVerifier:
    """Stdlib multipart client for ``POST /v1/audio/transcriptions``.

    The object has no custom representation, never emits request data, and
    converts HTTP/JSON failures to fixed messages.  Consequently neither the
    API key, the audio body, nor a server response body is copied into logs or
    exceptions by this implementation.
    """

    __slots__ = (
        "_api_key",
        "_endpoint",
        "_http_open",
        "_include_transcript_text",
        "_include_word_timestamps",
        "_language",
        "_model",
        "_timeout_seconds",
    )

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-transcribe",
        endpoint: str = OPENAI_TRANSCRIPTIONS_URL,
        timeout_seconds: float = 30.0,
        language: str | None = "en",
        include_transcript_text: bool = False,
        include_word_timestamps: bool | None = None,
        http_open: Callable[..., Any] = _open_with_verified_openai_tls,
    ) -> None:
        normalized_key = api_key.strip()
        normalized_model = model.strip()
        normalized_endpoint = endpoint.strip()
        if not normalized_key:
            raise ValueError("api_key must not be blank")
        if normalized_model not in SUPPORTED_ASR_MODELS:
            raise ValueError("model must be gpt-transcribe or whisper-1")
        parsed_endpoint = urlparse(normalized_endpoint)
        if (
            parsed_endpoint.scheme != "https"
            or parsed_endpoint.netloc != "api.openai.com"
            or parsed_endpoint.path.rstrip("/") != "/v1/audio/transcriptions"
            or parsed_endpoint.params
            or parsed_endpoint.query
            or parsed_endpoint.fragment
        ):
            raise ValueError("endpoint must be the OpenAI HTTPS /v1/audio/transcriptions URL")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise TypeError("timeout_seconds must be a number")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if language is not None and not language.strip():
            raise ValueError("language must be nonblank when provided")
        if not callable(http_open):
            raise TypeError("http_open must be callable")
        if include_word_timestamps is None:
            include_word_timestamps = False
        if include_word_timestamps and normalized_model != "whisper-1":
            raise ValueError("word timestamps are supported here only with whisper-1")

        self._api_key = normalized_key
        self._model = normalized_model
        self._endpoint = normalized_endpoint
        self._timeout_seconds = float(timeout_seconds)
        self._language = None if language is None else language.strip()
        self._include_transcript_text = include_transcript_text
        self._include_word_timestamps = include_word_timestamps
        self._http_open = http_open

    def verify(
        self,
        *,
        wav_path: Path,
        expected_text: str,
    ) -> ASRVerificationResult:
        expected = normalize_authorized_text(expected_text)
        if not expected:
            raise ValueError("expected_text must not be blank")
        try:
            audio = Path(wav_path).read_bytes()
        except OSError as exc:
            raise ASRVerificationError("could not read WAV for transcription") from exc
        if not audio:
            raise ASRVerificationError("could not read WAV for transcription")

        boundary = f"recoverybox-{uuid.uuid4().hex}"
        fields: list[tuple[str, str]] = [("model", self._model)]
        if self._language is not None:
            fields.append(("language", self._language))
        if self._include_word_timestamps:
            fields.extend(
                (
                    ("response_format", "verbose_json"),
                    ("timestamp_granularities[]", "word"),
                )
            )
        else:
            fields.append(("response_format", "json"))
        body = _multipart_form_data(
            boundary=boundary,
            fields=fields,
            file_bytes=audio,
        )
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
        )
        try:
            with self._http_open(request, timeout=self._timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise ASRVerificationError("OpenAI transcription request failed")
                payload = response.read(MAX_TRANSCRIPTION_RESPONSE_BYTES + 1)
        except ASRVerificationError:
            raise
        except Exception:
            # Deliberately do not include the underlying HTTP exception: it may
            # contain a response body or request details supplied by a custom
            # transport.
            raise ASRVerificationError("OpenAI transcription request failed") from None
        if len(payload) > MAX_TRANSCRIPTION_RESPONSE_BYTES:
            raise ASRVerificationError("OpenAI transcription response was too large")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ASRVerificationError("OpenAI transcription response was invalid") from None
        if not isinstance(decoded, Mapping):
            raise ASRVerificationError("OpenAI transcription response was invalid")
        transcript = decoded.get("text")
        if not isinstance(transcript, str):
            raise ASRVerificationError("OpenAI transcription response was invalid")

        words: tuple[ASRWord, ...] = ()
        if self._include_word_timestamps:
            words = _parse_asr_words(decoded.get("words"))
        return ASRVerificationResult(
            model=self._model,
            # This verifier runs only after the exact Realtime transcript gate
            # has released the audio. File ASR can vary capitalization and
            # punctuation for identical spoken words, so its secondary check
            # intentionally compares lexical content rather than gate syntax.
            matches_expected=_normalize_asr_text(transcript) == _normalize_asr_text(expected),
            transcript_text=transcript if self._include_transcript_text else None,
            words=words,
        )


def write_pcm16_wav(path: str | Path, pcm16_mono_24khz: bytes) -> None:
    """Write complete 24 kHz mono signed-16-bit little-endian PCM as RIFF/WAV."""

    if not pcm16_mono_24khz:
        raise ValueError("PCM must not be empty")
    if len(pcm16_mono_24khz) % PCM_SAMPLE_WIDTH_BYTES:
        raise ValueError("PCM16 must contain complete two-byte samples")
    output_path = Path(path)
    if not output_path.parent.exists():
        raise FileNotFoundError("WAV parent directory does not exist")
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(PCM_CHANNELS)
        output.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        output.setframerate(PCM_SAMPLE_RATE_HZ)
        output.writeframes(pcm16_mono_24khz)


def _multipart_form_data(
    *,
    boundary: str,
    fields: Iterable[tuple[str, str]],
    file_bytes: bytes,
) -> bytes:
    marker = f"--{boundary}\r\n".encode("ascii")
    parts: list[bytes] = []
    for name, value in fields:
        parts.extend(
            (
                marker,
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    parts.extend(
        (
            marker,
            b'Content-Disposition: form-data; name="file"; filename="cue.wav"\r\n',
            b"Content-Type: audio/wav\r\n\r\n",
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        )
    )
    return b"".join(parts)


def _parse_asr_words(value: Any) -> tuple[ASRWord, ...]:
    if not isinstance(value, list):
        raise ASRVerificationError("OpenAI transcription response was invalid")
    parsed: list[ASRWord] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ASRVerificationError("OpenAI transcription response was invalid")
        word = item.get("word")
        start = item.get("start")
        end = item.get("end")
        if (
            not isinstance(word, str)
            or not word
            or isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
        ):
            raise ASRVerificationError("OpenAI transcription response was invalid")
        start_number = float(start)
        end_number = float(end)
        if (
            not math.isfinite(start_number)
            or not math.isfinite(end_number)
            or start_number < 0
            or end_number < start_number
        ):
            raise ASRVerificationError("OpenAI transcription response was invalid")
        parsed.append(
            ASRWord(
                word=word,
                start_ms=start_number * 1_000,
                end_ms=end_number * 1_000,
            )
        )
    return tuple(parsed)


def _normalize_asr_text(text: str) -> str:
    """Normalize formatting-only ASR variation outside the safety gate."""

    folded = unicodedata.normalize("NFKC", text).casefold()
    lexical = "".join(character if character.isalnum() else " " for character in folded)
    return " ".join(lexical.split())


def _pcm_duration_ms(pcm: bytes) -> float:
    bytes_per_second = PCM_SAMPLE_RATE_HZ * PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
    return len(pcm) * 1_000 / bytes_per_second


def _safe_realtime_error_summary(event: ParsedServerEvent) -> str:
    diagnostics: list[str] = []
    if event.error_type is not None:
        diagnostics.append(f"type={event.error_type}")
    if event.error_code is not None:
        diagnostics.append(f"code={event.error_code}")
    suffix = f" ({', '.join(diagnostics)})" if diagnostics else ""
    return f"Realtime rejected the cue request{suffix}"


def _elapsed_ms(start: float, end: float) -> float:
    return round((end - start) * 1_000, 3)


def _optional_elapsed_ms(start: float, end: float | None) -> float | None:
    return None if end is None else _elapsed_ms(start, end)
