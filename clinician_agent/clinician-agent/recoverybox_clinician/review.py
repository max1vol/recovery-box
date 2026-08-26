"""Deterministic, read-only review queue and trend calculations."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean

from .models import (
    CohortSnapshot,
    ParticipantTrend,
    QueueItem,
    ReviewReport,
    SessionSummary,
)

_UNSUPPORTED_REQUESTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdiagnos(?:e|is|tic|ed)?\b", re.I), "diagnosis"),
    (re.compile(r"\bprescrib(?:e|ing|ed|er|ption)?\b", re.I), "prescribing"),
    (re.compile(r"\b(?:medication|medicine|drug|dosage|dose)\b", re.I), "medication advice"),
    (
        re.compile(r"\b(?:change|set|increase|decrease)\b.{0,30}\b(?:exercise|reps?|plan)\b", re.I),
        "prescription changes",
    ),
    (
        re.compile(r"\b(?:control|start|stop|unlock|restart)\b.{0,30}\bdevice\b", re.I),
        "device control",
    ),
    (re.compile(r"\bemergency\b|\btriage\b", re.I), "emergency triage"),
)

_BOUNDARY_DISCLAIMER_WORDS = frozenset(
    {
        "and",
        "change",
        "control",
        "device",
        "diagnose",
        "diagnosis",
        "do",
        "dose",
        "drug",
        "exercise",
        "increase",
        "medication",
        "medicine",
        "must",
        "never",
        "not",
        "or",
        "plan",
        "prescribe",
        "prescribing",
        "reps",
        "restart",
        "set",
        "start",
        "stop",
        "the",
        "unlock",
    }
)


def _without_boundary_disclaimers(question: str) -> str:
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", question):
        words = re.findall(r"[a-z]+", sentence.lower())
        negated = words[:2] == ["do", "not"] or words[:2] == ["must", "not"]
        negated = negated or words[:1] == ["never"]
        is_boundary_only = negated and set(words).issubset(_BOUNDARY_DISCLAIMER_WORDS)
        if not is_boundary_only:
            kept.append(sentence)
    return " ".join(kept)


def _scope_for(question: str) -> tuple[str, str]:
    scope_text = _without_boundary_disclaimers(question)
    blocked = sorted(
        {label for pattern, label in _UNSUPPORTED_REQUESTS if pattern.search(scope_text)}
    )
    if blocked:
        return (
            "unsupported",
            "This read-only review aid cannot provide "
            + ", ".join(blocked)
            + ". Use the clinical workflow and source record for that decision.",
        )
    return (
        "supported",
        "This output prioritizes review from supplied de-identified measurements; it does "
        "not diagnose, prescribe, control a device, or replace the source record.",
    )


def _queue_item(session: SessionSummary) -> QueueItem:
    score = 0
    evidence: list[str] = []

    if session.pain_reported:
        score += 60
        evidence.append("pain was reported")
    if session.safety_flags:
        score += 40
        evidence.append("safety flags: " + ", ".join(session.safety_flags))
    if session.stopped_early:
        score += 25
        evidence.append("session stopped early")
    if session.completion_rate < 0.5:
        score += 20
        evidence.append(
            f"completed {session.completed_reps}/{session.prescribed_reps} reps "
            f"({session.completion_rate:.0%})"
        )
    elif session.completion_rate < 0.8:
        score += 10
        evidence.append(
            f"completed {session.completed_reps}/{session.prescribed_reps} reps "
            f"({session.completion_rate:.0%})"
        )
    if session.quality_score < 0.5:
        score += 15
        evidence.append(f"quality score {session.quality_score:.2f}")
    elif session.quality_score < 0.7:
        score += 8
        evidence.append(f"quality score {session.quality_score:.2f}")
    if session.observation_confidence < 0.6:
        score += 8
        evidence.append(f"observation confidence {session.observation_confidence:.2f}")

    if score >= 50:
        attention_level = "high"
    elif score >= 20:
        attention_level = "medium"
    else:
        attention_level = "routine"
    if not evidence:
        evidence.append("no configured review trigger was present")

    return QueueItem(
        session_id=session.session_id,
        participant_ref=session.participant_ref,
        attention_level=attention_level,
        score=score,
        evidence=tuple(evidence),
    )


def _trend_direction(completion_delta: float, quality_delta: float) -> str:
    positive = completion_delta >= 0.10 or quality_delta >= 0.05
    negative = completion_delta <= -0.10 or quality_delta <= -0.05
    if positive and negative:
        return "mixed"
    if positive:
        return "improving"
    if negative:
        return "declining"
    return "stable"


def _participant_trends(sessions: Iterable[SessionSummary]) -> tuple[ParticipantTrend, ...]:
    by_participant: dict[str, list[SessionSummary]] = defaultdict(list)
    for session in sessions:
        by_participant[session.participant_ref].append(session)

    trends: list[ParticipantTrend] = []
    for participant_ref, participant_sessions in sorted(by_participant.items()):
        ordered = sorted(
            participant_sessions, key=lambda item: (item.session_date, item.session_id)
        )
        if len(ordered) < 2:
            continue
        first = ordered[0]
        latest = ordered[-1]
        completion_delta = latest.completion_rate - first.completion_rate
        quality_delta = latest.quality_score - first.quality_score
        trends.append(
            ParticipantTrend(
                participant_ref=participant_ref,
                first_session_id=first.session_id,
                latest_session_id=latest.session_id,
                session_count=len(ordered),
                completion_delta=completion_delta,
                quality_delta=quality_delta,
                direction=_trend_direction(completion_delta, quality_delta),
            )
        )
    return tuple(trends)


def _cohort_snapshot(sessions: tuple[SessionSummary, ...]) -> CohortSnapshot:
    if not sessions:
        return CohortSnapshot(
            session_count=0,
            participant_count=0,
            completion_rate=0.0,
            mean_quality_score=0.0,
            mean_observation_confidence=0.0,
            pain_reported_count=0,
            stopped_early_count=0,
            safety_flagged_count=0,
            low_confidence_session_ids=(),
        )

    total_prescribed = sum(session.prescribed_reps for session in sessions)
    return CohortSnapshot(
        session_count=len(sessions),
        participant_count=len({session.participant_ref for session in sessions}),
        completion_rate=sum(session.completed_reps for session in sessions) / total_prescribed,
        mean_quality_score=fmean(session.quality_score for session in sessions),
        mean_observation_confidence=fmean(session.observation_confidence for session in sessions),
        pain_reported_count=sum(session.pain_reported for session in sessions),
        stopped_early_count=sum(session.stopped_early for session in sessions),
        safety_flagged_count=sum(bool(session.safety_flags) for session in sessions),
        low_confidence_session_ids=tuple(
            sorted(
                session.session_id for session in sessions if session.observation_confidence < 0.6
            )
        ),
    )


def build_review_report(question: str, sessions: tuple[SessionSummary, ...]) -> ReviewReport:
    """Build an auditable report without model calls or external side effects."""
    scope_status, scope_message = _scope_for(question)
    queue = tuple(
        sorted(
            (_queue_item(session) for session in sessions),
            key=lambda item: (-item.score, item.session_id),
        )
    )
    evidence_ids = tuple(sorted(session.session_id for session in sessions))
    low_confidence_count = sum(session.observation_confidence < 0.6 for session in sessions)
    limitations = [
        "Only the supplied summary fields were evaluated; raw motion, audio, and clinical records "
        "were not available.",
        "Attention levels order a review queue and are not clinical urgency or emergency triage.",
    ]
    if low_confidence_count:
        limitations.append(
            f"{low_confidence_count} session(s) had observation confidence below 0.60; "
            "treat their movement metrics as limited evidence."
        )
    if not sessions:
        limitations.append(
            "No session summaries were supplied, so no review ranking or trend exists."
        )

    return ReviewReport(
        question=question,
        scope_status=scope_status,
        scope_message=scope_message,
        queue=queue,
        cohort=_cohort_snapshot(sessions),
        participant_trends=_participant_trends(sessions),
        evidence_session_ids=evidence_ids,
        limitations=tuple(limitations),
    )


def render_markdown(report: ReviewReport) -> str:
    """Render the deterministic report with source-session citations."""
    lines = [
        "# Clinician review support",
        "",
        report.scope_message,
    ]
    if report.scope_status == "unsupported":
        lines.extend(
            [
                "",
                "No diagnosis, prescription, triage, or device action was produced.",
            ]
        )
        return "\n".join(lines)

    lines.extend(["", "## Review queue", ""])
    if not report.queue:
        lines.append("No sessions were supplied.")
    else:
        for index, item in enumerate(report.queue, start=1):
            evidence = "; ".join(item.evidence)
            lines.append(
                f"{index}. **{item.attention_level.upper()}** — {item.participant_ref}, "
                f"{item.session_id}: {evidence}. [session:{item.session_id}]"
            )

    cohort = report.cohort
    evidence = ", ".join(f"session:{session_id}" for session_id in report.evidence_session_ids)
    citation = f" [{evidence}]" if evidence else ""
    lines.extend(
        [
            "",
            "## Cohort snapshot",
            "",
            f"- Sessions: {cohort.session_count}; de-identified participants: "
            f"{cohort.participant_count}.{citation}",
            f"- Rep completion: {cohort.completion_rate:.0%}; mean quality: "
            f"{cohort.mean_quality_score:.2f}; mean observation confidence: "
            f"{cohort.mean_observation_confidence:.2f}.{citation}",
            f"- Pain reported: {cohort.pain_reported_count}; stopped early: "
            f"{cohort.stopped_early_count}; safety-flagged: "
            f"{cohort.safety_flagged_count}.{citation}",
            "",
            "## Participant trends",
            "",
        ]
    )
    if not report.participant_trends:
        lines.append("At least two sessions for one participant are required for a trend.")
    else:
        for trend in report.participant_trends:
            lines.append(
                f"- {trend.participant_ref}: **{trend.direction}** across "
                f"{trend.session_count} sessions; completion change "
                f"{trend.completion_delta:+.0%}, quality change {trend.quality_delta:+.2f}. "
                f"[session:{trend.first_session_id}] [session:{trend.latest_session_id}]"
            )

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report.limitations)
    return "\n".join(lines)
