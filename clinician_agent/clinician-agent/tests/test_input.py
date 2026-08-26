from __future__ import annotations

import json

import pytest

from recoverybox_clinician.input import parse_question, parse_session_summaries_json
from recoverybox_clinician.models import ValidationError


def valid_summary(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "session_id": "sess-test-001",
        "participant_ref": "anon-abc123",
        "session_date": "2026-08-20",
        "exercise_id": "seated-knee-extension",
        "prescribed_reps": 10,
        "completed_reps": 8,
        "duration_seconds": 200,
        "quality_score": 0.78,
        "observation_confidence": 0.91,
        "pain_reported": False,
        "stopped_early": False,
        "safety_flags": [],
        "model_version": "motion-0.1.0",
    }
    value.update(overrides)
    return value


def test_parses_strict_deidentified_summary() -> None:
    sessions = parse_session_summaries_json(json.dumps([valid_summary()]))

    assert len(sessions) == 1
    assert sessions[0].completion_rate == 0.8
    assert sessions[0].participant_ref == "anon-abc123"


@pytest.mark.parametrize("field", ["name", "email", "free_text_note", "raw_video_path"])
def test_rejects_identifying_free_text_and_raw_media_fields(field: str) -> None:
    summary = valid_summary(**{field: "do-not-send"})

    with pytest.raises(ValidationError, match="cannot include"):
        parse_session_summaries_json(json.dumps([summary]))


def test_rejects_unknown_fields_and_duplicate_sessions() -> None:
    with pytest.raises(ValidationError, match="unknown session fields"):
        parse_session_summaries_json(json.dumps([valid_summary(unexpected_metric=1)]))

    with pytest.raises(ValidationError, match="must be unique"):
        parse_session_summaries_json(json.dumps([valid_summary(), valid_summary()]))


def test_rejects_impossible_rep_count_and_unbounded_reason_code() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        parse_session_summaries_json(
            json.dumps([valid_summary(completed_reps=11, prescribed_reps=10)])
        )

    with pytest.raises(ValidationError, match="uppercase reason codes"):
        parse_session_summaries_json(json.dumps([valid_summary(safety_flags=["free text"])]))


def test_question_rejects_obvious_direct_contact_details() -> None:
    assert parse_question("  Which sessions   should I review? ") == (
        "Which sessions should I review?"
    )

    with pytest.raises(ValidationError, match="cannot contain"):
        parse_question("Review the record for clinician@example.test")
