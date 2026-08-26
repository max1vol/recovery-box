from __future__ import annotations

from datetime import date

import pytest

from recoverybox_clinician.models import SessionSummary
from recoverybox_clinician.review import build_review_report, render_markdown


def session(
    session_id: str,
    participant_ref: str = "anon-abc123",
    session_date: date = date(2026, 8, 20),
    prescribed_reps: int = 10,
    completed_reps: int = 10,
    quality_score: float = 0.8,
    observation_confidence: float = 0.9,
    pain_reported: bool = False,
    stopped_early: bool = False,
    safety_flags: tuple[str, ...] = (),
) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        participant_ref=participant_ref,
        session_date=session_date,
        exercise_id="seated-knee-extension",
        prescribed_reps=prescribed_reps,
        completed_reps=completed_reps,
        duration_seconds=200,
        quality_score=quality_score,
        observation_confidence=observation_confidence,
        pain_reported=pain_reported,
        stopped_early=stopped_early,
        safety_flags=safety_flags,
        model_version="motion-0.1.0",
    )


def test_pain_and_safety_flag_rank_before_low_completion() -> None:
    pain = session(
        "sess-test-pain",
        participant_ref="anon-pain01",
        pain_reported=True,
        stopped_early=True,
        safety_flags=("SHARP_PAIN_REPORTED",),
    )
    incomplete = session(
        "sess-test-low1",
        participant_ref="anon-low001",
        completed_reps=2,
        quality_score=0.45,
    )

    report = build_review_report("Which sessions should I review first?", (incomplete, pain))

    assert report.queue[0].session_id == pain.session_id
    assert report.queue[0].attention_level == "high"
    assert report.queue[1].attention_level == "medium"


def test_snapshot_uses_rep_weighted_completion_and_tracks_evidence() -> None:
    sessions = (
        session("sess-test-001", prescribed_reps=10, completed_reps=5),
        session(
            "sess-test-002",
            participant_ref="anon-def456",
            prescribed_reps=20,
            completed_reps=20,
            observation_confidence=0.4,
        ),
    )

    report = build_review_report("Summarize the supplied trends", sessions)

    assert report.cohort.completion_rate == pytest.approx(25 / 30)
    assert report.cohort.low_confidence_session_ids == ("sess-test-002",)
    assert report.evidence_session_ids == ("sess-test-001", "sess-test-002")


def test_participant_trend_compares_first_and_latest_session() -> None:
    first = session(
        "sess-test-001",
        session_date=date(2026, 8, 18),
        completed_reps=5,
        quality_score=0.5,
    )
    latest = session(
        "sess-test-002",
        session_date=date(2026, 8, 22),
        completed_reps=9,
        quality_score=0.75,
    )

    report = build_review_report("Show the trend", (latest, first))
    trend = report.participant_trends[0]

    assert trend.direction == "improving"
    assert trend.completion_delta == pytest.approx(0.4)
    assert trend.quality_delta == pytest.approx(0.25)
    markdown = render_markdown(report)
    assert "[session:sess-test-001]" in markdown
    assert "[session:sess-test-002]" in markdown


@pytest.mark.parametrize(
    "question",
    [
        "Diagnose this participant",
        "Change the exercise plan to 20 reps",
        "Start the device remotely",
        "What medication dose should they take?",
        "Perform emergency triage",
    ],
)
def test_unsupported_clinical_or_control_requests_are_blocked(question: str) -> None:
    report = build_review_report(question, (session("sess-test-001"),))
    markdown = render_markdown(report)

    assert report.scope_status == "unsupported"
    assert "cannot provide" in report.scope_message
    assert "No diagnosis, prescription, triage, or device action was produced." in markdown
    assert "Review queue" not in markdown
