from __future__ import annotations

import base64
import json
import ssl
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from recoverybox.core import DEFAULT_CUE_CATALOG, CueId, GuardianReason
from recoverybox.realtime.client import RealtimeSession
from recoverybox.realtime.verification import (
    ASRVerificationError,
    ASRVerificationResult,
    ASRWord,
    CueVerificationError,
    CueVerificationRequest,
    OpenAITranscriptionVerifier,
    RealtimeCueVerifier,
    write_pcm16_wav,
)
from recoverybox.session import (
    DEFAULT_CUE_CATALOG_VERSION,
    ApprovedCuePlaybackAuthorization,
)

PCM_A = b"\x01\x00" * 240
PCM_B = b"\x02\x00" * 240


@dataclass
class _Clock:
    value: float = 10.0

    def __call__(self) -> float:
        return self.value


class _TimelineTransport:
    def __init__(self, clock: _Clock, incoming: list[tuple[float, dict[str, Any]]]) -> None:
        self.clock = clock
        self.incoming = deque(incoming)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def send_event(self, event: dict[str, Any]) -> None:
        self.sent.append(dict(event))

    def receive_event(self) -> dict[str, Any]:
        if not self.incoming:
            raise EOFError("timeline exhausted")
        timestamp, event = self.incoming.popleft()
        self.clock.value = timestamp
        return dict(event)

    def close(self) -> None:
        self.closed = True


class _LeakingTransport:
    def __init__(self, *, failure_point: str) -> None:
        self.failure_point = failure_point
        self.sent: list[dict[str, Any]] = []

    def send_event(self, event: dict[str, Any]) -> None:
        if self.failure_point == "send":
            raise RuntimeError("sk-live-must-not-escape")
        self.sent.append(dict(event))

    def receive_event(self) -> dict[str, Any]:
        raise RuntimeError("sk-live-must-not-escape")

    def close(self) -> None:
        return None


def _authorization(cue_id: CueId = CueId.MOVE_SLOWLY) -> ApprovedCuePlaybackAuthorization:
    cue = DEFAULT_CUE_CATALOG[cue_id.value]
    return ApprovedCuePlaybackAuthorization(
        cue_id=cue_id,
        cue_kind=cue.kind,
        catalog_version=DEFAULT_CUE_CATALOG_VERSION,
        guardian_rule_version="guardian-verification-test-v1",
        reason_codes=(GuardianReason.LEARNED_MODEL_CUE_ACCEPTED,),
    )


def _created(response_id: str, event_id: str) -> dict[str, Any]:
    return {
        "type": "response.created",
        "event_id": event_id,
        "response": {"id": response_id, "status": "in_progress"},
    }


def _audio(
    pcm: bytes,
    *,
    response_id: str,
    item_id: str,
    event_id: str,
) -> dict[str, Any]:
    return {
        "type": "response.output_audio.delta",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": base64.b64encode(pcm).decode("ascii"),
    }


def _audio_done(*, response_id: str, item_id: str, event_id: str) -> dict[str, Any]:
    return {
        "type": "response.output_audio.done",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
    }


def _transcript_done(
    transcript: str,
    *,
    response_id: str,
    item_id: str,
    event_id: str,
) -> dict[str, Any]:
    return {
        "type": "response.output_audio_transcript.done",
        "event_id": event_id,
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "transcript": transcript,
    }


def _done(response_id: str, event_id: str, *, status: str = "completed") -> dict[str, Any]:
    return {
        "type": "response.done",
        "event_id": event_id,
        "response": {"id": response_id, "status": status},
    }


def _successful_timeline(
    *,
    start: float = 10.0,
    response_id: str = "resp-cue-1",
    item_id: str = "item-cue-1",
    event_suffix: str = "1",
) -> list[tuple[float, dict[str, Any]]]:
    phrase = DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value].spoken_text
    return [
        (start + 0.05, _created(response_id, f"evt-created-{event_suffix}")),
        (
            start + 0.08,
            _audio(
                PCM_A,
                response_id=response_id,
                item_id=item_id,
                event_id=f"evt-audio-a-{event_suffix}",
            ),
        ),
        (
            start + 0.10,
            _audio(
                PCM_B,
                response_id=response_id,
                item_id=item_id,
                event_id=f"evt-audio-b-{event_suffix}",
            ),
        ),
        (
            start + 0.12,
            _transcript_done(
                phrase,
                response_id=response_id,
                item_id=item_id,
                event_id=f"evt-transcript-{event_suffix}",
            ),
        ),
        (
            start + 0.13,
            _audio_done(
                response_id=response_id,
                item_id=item_id,
                event_id=f"evt-audio-done-{event_suffix}",
            ),
        ),
        (start + 0.14, _done(response_id, f"evt-done-{event_suffix}")),
    ]


def test_verifier_measures_each_stage_and_writes_released_pcm_wav(tmp_path: Path) -> None:
    clock = _Clock()
    transport = _TimelineTransport(clock, _successful_timeline())
    session = RealtimeSession(transport=transport)
    wav_path = tmp_path / "move-slowly.wav"

    report = RealtimeCueVerifier(
        session=session,
        transport=transport,
        clock=clock,
    ).verify_cue(_authorization(), wav_path=wav_path)

    assert report.response_id == "resp-cue-1"
    assert report.response_status == "completed"
    assert report.response_created_ms == 50.0
    assert report.first_audio_delta_ms == 80.0
    assert report.transcript_done_ms == 120.0
    # Prompt cues remain quarantined through the terminal response event, so
    # release and response completion intentionally share a timestamp.
    assert report.quarantine_release_ms == 140.0
    assert report.response_done_ms == 140.0
    assert report.received_pcm_bytes == len(PCM_A + PCM_B)
    assert report.released_pcm_bytes == len(PCM_A + PCM_B)
    assert report.released_audio_duration_ms == 20.0
    assert report.quarantine_released is True
    assert report.realtime_transcript_matches is True
    assert report.realtime_transcript_text is None
    assert report.wav_path == str(wav_path)
    persisted = json.loads(json.dumps(report.to_dict()))
    assert persisted["latency_ms"] == {
        "response_created": 50.0,
        "first_audio_delta": 80.0,
        "transcript_done": 120.0,
        "quarantine_release": 140.0,
        "response_done": 140.0,
    }
    assert persisted["response_completed"] is True
    assert "response_id" not in persisted
    assert "response_status" not in persisted
    assert "realtime_transcript_text" not in persisted
    assert "received_pcm_bytes" not in persisted
    assert "released_pcm_bytes" not in persisted

    with wave.open(str(wav_path), "rb") as rendered:
        assert rendered.getnchannels() == 1
        assert rendered.getsampwidth() == 2
        assert rendered.getframerate() == 24_000
        assert rendered.getnframes() == 480
        assert rendered.readframes(rendered.getnframes()) == PCM_A + PCM_B

    request = transport.sent[0]
    assert request["type"] == "response.create"
    assert request["response"]["conversation"] == "none"
    assert request["response"]["tools"] == []


def test_verifier_does_not_persist_pcm_when_realtime_transcript_mismatches(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    response_id = "resp-mismatch"
    item_id = "item-mismatch"
    timeline = [
        (10.01, _created(response_id, "evt-created-mismatch")),
        (
            10.02,
            _audio(
                PCM_A,
                response_id=response_id,
                item_id=item_id,
                event_id="evt-audio-mismatch",
            ),
        ),
        (
            10.03,
            _audio_done(
                response_id=response_id,
                item_id=item_id,
                event_id="evt-audio-done-mismatch",
            ),
        ),
        (
            10.04,
            _transcript_done(
                "Move quickly and with control.",
                response_id=response_id,
                item_id=item_id,
                event_id="evt-transcript-mismatch",
            ),
        ),
        (10.05, _done(response_id, "evt-done-mismatch")),
    ]
    transport = _TimelineTransport(clock, timeline)
    wav_path = tmp_path / "must-not-exist.wav"

    report = RealtimeCueVerifier(
        session=RealtimeSession(transport=transport),
        transport=transport,
        clock=clock,
        retain_realtime_transcript=True,
    ).verify_cue(_authorization(), wav_path=wav_path)

    assert report.received_pcm_bytes == len(PCM_A)
    assert report.released_pcm_bytes == 0
    assert report.quarantine_released is False
    assert report.quarantine_release_ms is None
    assert report.realtime_transcript_matches is False
    assert report.realtime_transcript_text == "Move quickly and with control."
    assert report.wav_path is None
    assert not wav_path.exists()


class _FakeASR:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def verify(self, *, wav_path: Path, expected_text: str) -> ASRVerificationResult:
        self.calls.append((wav_path, expected_text))
        return ASRVerificationResult(model="fake-asr", matches_expected=True)


def test_optional_asr_runs_only_after_quarantine_release(tmp_path: Path) -> None:
    clock = _Clock()
    transport = _TimelineTransport(clock, _successful_timeline())
    fake_asr = _FakeASR()
    wav_path = tmp_path / "verified.wav"

    report = RealtimeCueVerifier(
        session=RealtimeSession(transport=transport),
        transport=transport,
        clock=clock,
        asr_verifier=fake_asr,
    ).verify_cue(_authorization(), wav_path=wav_path)

    assert fake_asr.calls == [(wav_path, DEFAULT_CUE_CATALOG[CueId.MOVE_SLOWLY.value].spoken_text)]
    assert report.asr == ASRVerificationResult(model="fake-asr", matches_expected=True)
    assert report.asr_verification_ms == 0.0


def test_verify_many_consumes_simulated_authorizations_sequentially(tmp_path: Path) -> None:
    clock = _Clock()
    timeline = _successful_timeline()
    timeline.extend(
        _successful_timeline(
            start=10.14,
            response_id="resp-cue-2",
            item_id="item-cue-2",
            event_suffix="2",
        )
    )
    transport = _TimelineTransport(clock, timeline)
    verifier = RealtimeCueVerifier(
        session=RealtimeSession(transport=transport),
        transport=transport,
        clock=clock,
    )

    reports = verifier.verify_many(
        (
            CueVerificationRequest(_authorization(), tmp_path / "cue-1.wav"),
            CueVerificationRequest(_authorization(), tmp_path / "cue-2.wav"),
        )
    )

    assert [report.response_id for report in reports] == ["resp-cue-1", "resp-cue-2"]
    assert reports[1].response_created_ms == 50.0
    assert len(transport.sent) == 2


def test_realtime_error_diagnostic_never_echoes_server_message(tmp_path: Path) -> None:
    secret = "sk-live-must-not-escape"
    clock = _Clock()
    transport = _TimelineTransport(
        clock,
        [
            (
                10.01,
                {
                    "type": "error",
                    "event_id": "evt-error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                        "message": f"Incorrect API key provided: {secret}",
                    },
                },
            )
        ],
    )

    with pytest.raises(CueVerificationError) as caught:
        RealtimeCueVerifier(
            session=RealtimeSession(transport=transport),
            transport=transport,
            clock=clock,
        ).verify_cue(_authorization(), wav_path=tmp_path / "must-not-exist.wav")

    assert str(caught.value) == (
        "Realtime rejected the cue request (type=invalid_request_error, code=invalid_api_key)"
    )
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("failure_point", "expected_message"),
    (
        ("send", "Realtime cue request could not be sent"),
        (
            "receive",
            "Realtime event stream failed before the cue response completed",
        ),
    ),
)
def test_transport_exception_text_is_not_exposed(
    tmp_path: Path,
    failure_point: str,
    expected_message: str,
) -> None:
    transport = _LeakingTransport(failure_point=failure_point)

    with pytest.raises(CueVerificationError) as caught:
        RealtimeCueVerifier(
            session=RealtimeSession(transport=transport),
            transport=transport,
        ).verify_cue(_authorization(), wav_path=tmp_path / "must-not-exist.wav")

    assert str(caught.value) == expected_message
    assert "sk-live-must-not-escape" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_injected_session_exception_text_is_not_exposed(tmp_path: Path) -> None:
    clock = _Clock()
    transport = _TimelineTransport(clock, [(10.01, _created("resp-1", "evt-created"))])

    class _LeakingSession(RealtimeSession):
        def handle_event(self, raw: dict[str, Any]):  # type: ignore[no-untyped-def]
            del raw
            raise RuntimeError("sk-live-must-not-escape")

    with pytest.raises(CueVerificationError) as caught:
        RealtimeCueVerifier(
            session=_LeakingSession(transport=transport),
            transport=transport,
            clock=clock,
        ).verify_cue(_authorization(), wav_path=tmp_path / "must-not-exist.wav")

    assert str(caught.value) == "Realtime server event processing failed"
    assert "sk-live-must-not-escape" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_asr_exception_text_is_not_copied_into_report(tmp_path: Path) -> None:
    class _FailingASR:
        def verify(self, *, wav_path: Path, expected_text: str) -> ASRVerificationResult:
            del wav_path, expected_text
            raise ASRVerificationError("sk-live-must-not-escape")

    clock = _Clock()
    transport = _TimelineTransport(clock, _successful_timeline())
    report = RealtimeCueVerifier(
        session=RealtimeSession(transport=transport),
        transport=transport,
        clock=clock,
        asr_verifier=_FailingASR(),
    ).verify_cue(_authorization(), wav_path=tmp_path / "verified.wav")

    assert report.asr is None
    assert report.asr_error == "ASR verification failed"
    assert "sk-live-must-not-escape" not in json.dumps(report.to_dict())


def test_persisted_report_omits_opted_in_transcript_and_provider_identifiers(
    tmp_path: Path,
) -> None:
    sentinel = "provider-controlled-sensitive-text"
    response_id = f"resp-{sentinel}"
    item_id = "item-sensitive"
    clock = _Clock()
    timeline = [
        (10.01, _created(response_id, "evt-created-sensitive")),
        (
            10.02,
            _audio(
                PCM_A,
                response_id=response_id,
                item_id=item_id,
                event_id="evt-audio-sensitive",
            ),
        ),
        (
            10.03,
            _audio_done(
                response_id=response_id,
                item_id=item_id,
                event_id="evt-audio-done-sensitive",
            ),
        ),
        (
            10.04,
            _transcript_done(
                sentinel,
                response_id=response_id,
                item_id=item_id,
                event_id="evt-transcript-sensitive",
            ),
        ),
        (10.05, _done(response_id, "evt-done-sensitive")),
    ]
    transport = _TimelineTransport(clock, timeline)
    report = RealtimeCueVerifier(
        session=RealtimeSession(transport=transport),
        transport=transport,
        clock=clock,
        retain_realtime_transcript=True,
    ).verify_cue(_authorization(), wav_path=tmp_path / "must-not-exist.wav")

    assert report.realtime_transcript_text == sentinel
    assert sentinel not in json.dumps(report.to_dict())


class _HTTPResponse:
    def __init__(self, payload: dict[str, Any], *, status: int = 200) -> None:
        self._payload = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self) -> _HTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]


class _CaptureOpen:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.request: Request | None = None
        self.timeout: float | None = None

    def __call__(self, request: Request, *, timeout: float) -> _HTTPResponse:
        self.request = request
        self.timeout = timeout
        return _HTTPResponse(self.payload)


def test_gpt_transcribe_verifier_posts_json_multipart_without_network(tmp_path: Path) -> None:
    wav_path = tmp_path / "cue.wav"
    write_pcm16_wav(wav_path, PCM_A)
    http_open = _CaptureOpen({"text": "Move slowly and with control."})
    verifier = OpenAITranscriptionVerifier(
        api_key="test-secret-key",
        model="gpt-transcribe",
        include_transcript_text=True,
        http_open=http_open,
    )

    result = verifier.verify(
        wav_path=wav_path,
        expected_text="Move slowly and with control.",
    )

    assert result == ASRVerificationResult(
        model="gpt-transcribe",
        matches_expected=True,
        transcript_text="Move slowly and with control.",
    )
    assert http_open.request is not None
    assert http_open.request.full_url == "https://api.openai.com/v1/audio/transcriptions"
    assert http_open.request.get_method() == "POST"
    assert http_open.request.get_header("Authorization") == "Bearer test-secret-key"
    assert http_open.timeout == 30.0
    body = http_open.request.data
    assert body is not None
    assert b'name="model"\r\n\r\ngpt-transcribe\r\n' in body
    assert b'name="response_format"\r\n\r\njson\r\n' in body
    assert b'filename="cue.wav"' in body
    assert b"RIFF" in body
    assert "test-secret-key" not in repr(verifier)


def test_whisper_verifier_requests_and_parses_word_timestamps(tmp_path: Path) -> None:
    wav_path = tmp_path / "cue.wav"
    write_pcm16_wav(wav_path, PCM_A)
    http_open = _CaptureOpen(
        {
            "text": "Move slowly.",
            "words": [
                {"word": "Move", "start": 0.02, "end": 0.21},
                {"word": "slowly.", "start": 0.22, "end": 0.55},
            ],
        }
    )

    result = OpenAITranscriptionVerifier(
        api_key="test-secret-key",
        model="whisper-1",
        include_word_timestamps=True,
        http_open=http_open,
    ).verify(wav_path=wav_path, expected_text="Move slowly.")

    assert result.matches_expected is True
    assert result.transcript_text is None
    assert result.words == (
        ASRWord(word="Move", start_ms=20.0, end_ms=210.0),
        ASRWord(word="slowly.", start_ms=220.0, end_ms=550.0),
    )
    assert result.to_dict()["word_timestamps_ms"] == [
        {"start_ms": 20.0, "end_ms": 210.0},
        {"start_ms": 220.0, "end_ms": 550.0},
    ]
    assert "Move" not in json.dumps(result.to_dict())
    assert http_open.request is not None
    body = http_open.request.data
    assert body is not None
    assert b'name="response_format"\r\n\r\nverbose_json\r\n' in body
    assert b'name="timestamp_granularities[]"\r\n\r\nword\r\n' in body


def test_transcription_match_ignores_only_asr_case_and_punctuation(
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "cue.wav"
    write_pcm16_wav(wav_path, PCM_A)
    http_open = _CaptureOpen({"text": "BRING your arms back out to a T-shape"})

    matching = OpenAITranscriptionVerifier(
        api_key="test-secret-key",
        model="gpt-transcribe",
        http_open=http_open,
    ).verify(
        wav_path=wav_path,
        expected_text="Bring your arms back out to a T shape.",
    )

    assert matching.matches_expected is True

    different_words = _CaptureOpen({"text": "Bring your arms down to your sides."})
    mismatching = OpenAITranscriptionVerifier(
        api_key="test-secret-key",
        model="gpt-transcribe",
        http_open=different_words,
    ).verify(
        wav_path=wav_path,
        expected_text="Bring your arms back out to a T shape.",
    )

    assert mismatching.matches_expected is False


def test_transcription_default_open_uses_verified_certifi_tls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "cue.wav"
    write_pcm16_wav(wav_path, PCM_A)
    captured: dict[str, Any] = {}

    def fake_urlopen(
        request: Request,
        *,
        timeout: float,
        context: ssl.SSLContext,
    ) -> _HTTPResponse:
        captured.update(request=request, timeout=timeout, context=context)
        return _HTTPResponse({"text": "Move slowly."})

    monkeypatch.setattr(
        "recoverybox.realtime.verification.urllib.request.urlopen",
        fake_urlopen,
    )

    result = OpenAITranscriptionVerifier(
        api_key="test-secret-key",
        model="whisper-1",
    ).verify(wav_path=wav_path, expected_text="Move slowly.")

    assert result.matches_expected is True
    assert captured["timeout"] == 30.0
    context = captured["context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.get_ca_certs()


def test_transcription_http_failure_does_not_echo_transport_secrets(tmp_path: Path) -> None:
    wav_path = tmp_path / "cue.wav"
    write_pcm16_wav(wav_path, PCM_A)

    def failing_open(request: Request, *, timeout: float) -> _HTTPResponse:
        del request, timeout
        raise RuntimeError("test-secret-key and raw-audio-body")

    verifier = OpenAITranscriptionVerifier(
        api_key="test-secret-key",
        http_open=failing_open,
    )
    with pytest.raises(ASRVerificationError) as caught:
        verifier.verify(wav_path=wav_path, expected_text="Approved phrase.")

    assert str(caught.value) == "OpenAI transcription request failed"
    assert "test-secret-key" not in str(caught.value)
    assert "raw-audio-body" not in str(caught.value)


def test_transcription_verifier_rejects_non_openai_endpoint() -> None:
    with pytest.raises(ValueError, match="OpenAI HTTPS"):
        OpenAITranscriptionVerifier(
            api_key="test-secret-key",
            endpoint="https://example.com/v1/audio/transcriptions",
        )


def test_pcm_wav_writer_rejects_partial_samples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="complete two-byte samples"):
        write_pcm16_wav(tmp_path / "bad.wav", b"\x00")
