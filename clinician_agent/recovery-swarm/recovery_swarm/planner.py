"""Deterministic planning kernel that cannot rewrite a clinician envelope."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from typing import Any

from .schema import (
    AccessibilityCode,
    CoachingStyle,
    Day,
    EquipmentCode,
    PlanRequest,
    PreferredTime,
    TreatmentEnvelope,
)

RECOVERY_PLAN_SCHEMA = "recoverybox.recovery-plan.v1"


class PlanStatus(StrEnum):
    DRAFT_FOR_CLINICIAN_APPROVAL = "DRAFT_FOR_CLINICIAN_APPROVAL"
    BLOCKED = "BLOCKED"


class BlockerCode(StrEnum):
    INSUFFICIENT_AVAILABLE_DAYS = "INSUFFICIENT_AVAILABLE_DAYS"
    SESSION_EXCEEDS_TIME_LIMIT = "SESSION_EXCEEDS_TIME_LIMIT"
    MISSING_REQUIRED_EQUIPMENT = "MISSING_REQUIRED_EQUIPMENT"
    PRESCRIPTION_BOUNDARY_REVIEW = "PRESCRIPTION_BOUNDARY_REVIEW"
    PREFERENCE_MAPPING_REVIEW = "PREFERENCE_MAPPING_REVIEW"
    FEASIBILITY_REVIEW = "FEASIBILITY_REVIEW"


@dataclass(frozen=True, slots=True)
class ScheduledSession:
    """A weekly placement of the full, unchanged treatment envelope."""

    day: Day
    preferred_time: PreferredTime
    exercise_ids: tuple[str, ...]
    estimated_minutes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day.value,
            "preferred_time": self.preferred_time.value,
            "exercise_ids": list(self.exercise_ids),
            "estimated_minutes": self.estimated_minutes,
        }


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    """A reviewable draft; never an executable or approved prescription."""

    request_id: str
    status: PlanStatus
    diagnosis_code: str
    treatment_envelope: TreatmentEnvelope
    selected_weekly_frequency: int | None
    sessions: tuple[ScheduledSession, ...]
    coaching_style: CoachingStyle
    language: str
    accessibility: tuple[AccessibilityCode, ...]
    blocker_codes: tuple[BlockerCode, ...]
    schema: str = RECOVERY_PLAN_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "status": self.status.value,
            "diagnosis_code": self.diagnosis_code,
            "treatment_envelope": self.treatment_envelope.as_dict(),
            "selected_weekly_frequency": self.selected_weekly_frequency,
            "sessions": [session.as_dict() for session in self.sessions],
            "presentation": {
                "coaching_style": self.coaching_style.value,
                "language": self.language,
                "accessibility": [item.value for item in self.accessibility],
            },
            "blocker_codes": [item.value for item in self.blocker_codes],
            "clinician_approval_required": True,
        }


def _ordered_days(request: PlanRequest, preferred_days: Sequence[Day] | None) -> tuple[Day, ...]:
    available = request.user_preferences.available_days
    if preferred_days is None:
        return available
    candidate = tuple(preferred_days)
    if len(candidate) != len(available) or set(candidate) != set(available):
        return available
    return candidate


def _required_equipment(request: PlanRequest) -> set[EquipmentCode]:
    return {
        code
        for exercise in request.clinician_diagnosis.treatment_envelope.approved_exercises
        for code in exercise.required_equipment
        if code is not EquipmentCode.NONE
    }


def build_deterministic_plan(
    request: PlanRequest,
    *,
    preferred_days: Sequence[Day] | None = None,
    additional_blockers: Iterable[BlockerCode] = (),
) -> RecoveryPlan:
    """Place the immutable prescription inside coded user constraints.

    Agent suggestions can reorder already-available days or add caution. They
    cannot supply exercises, doses, frequency bounds, contraindications, or
    presentation values.
    """

    envelope = request.clinician_diagnosis.treatment_envelope
    preferences = request.user_preferences
    blockers = list(additional_blockers)
    if len(preferences.available_days) < envelope.allowed_weekly_frequency.minimum:
        blockers.append(BlockerCode.INSUFFICIENT_AVAILABLE_DAYS)

    estimated_minutes = ceil(
        sum(item.duration_seconds for item in envelope.approved_exercises) / 60
    )
    if estimated_minutes > preferences.max_minutes:
        blockers.append(BlockerCode.SESSION_EXCEEDS_TIME_LIMIT)

    available_equipment = set(preferences.equipment)
    if not _required_equipment(request).issubset(available_equipment):
        blockers.append(BlockerCode.MISSING_REQUIRED_EQUIPMENT)

    unique_blockers = tuple(dict.fromkeys(blockers))
    if unique_blockers:
        return RecoveryPlan(
            request_id=request.request_id,
            status=PlanStatus.BLOCKED,
            diagnosis_code=request.clinician_diagnosis.diagnosis_code,
            treatment_envelope=envelope,
            selected_weekly_frequency=None,
            sessions=(),
            coaching_style=preferences.coaching_style,
            language=preferences.language,
            accessibility=preferences.accessibility,
            blocker_codes=unique_blockers,
        )

    selected_frequency = min(
        envelope.allowed_weekly_frequency.maximum,
        len(preferences.available_days),
    )
    days = _ordered_days(request, preferred_days)[:selected_frequency]
    exercise_ids = tuple(item.exercise_id for item in envelope.approved_exercises)
    sessions = tuple(
        ScheduledSession(
            day=day,
            preferred_time=preferences.preferred_time,
            exercise_ids=exercise_ids,
            estimated_minutes=estimated_minutes,
        )
        for day in days
    )
    return RecoveryPlan(
        request_id=request.request_id,
        status=PlanStatus.DRAFT_FOR_CLINICIAN_APPROVAL,
        diagnosis_code=request.clinician_diagnosis.diagnosis_code,
        treatment_envelope=envelope,
        selected_weekly_frequency=selected_frequency,
        sessions=sessions,
        coaching_style=preferences.coaching_style,
        language=preferences.language,
        accessibility=preferences.accessibility,
        blocker_codes=(),
    )


def render_plan_markdown(
    plan: RecoveryPlan,
    swarm_result: Any | None = None,
    *,
    organisation_version: str = "recovery-swarm.v1",
    focus: str = "coded schedule and preference fit",
) -> str:
    """Render a concise plan for the Flower chat surface."""

    del swarm_result  # Role internals are private; only their aggregate is shown.
    envelope = plan.treatment_envelope
    lines = [
        "# Recovery Swarm plan",
        "",
        f"**Status:** `{plan.status.value}`",
        "",
        "> This is a synthetic planning demo and a draft for clinician approval. "
        "It does not diagnose, prescribe, change the Guardian, or control hardware.",
        "",
        f"- Organisation: `{organisation_version}`",
        f"- Focus: {focus}",
        f"- Diagnosis code: `{plan.diagnosis_code}`",
        f"- Review date: `{envelope.review_date.isoformat()}`",
        "",
        "## Immutable clinician envelope",
        "",
    ]
    for exercise in envelope.approved_exercises:
        lines.append(
            f"- `{exercise.exercise_id}` — {exercise.sets} sets × {exercise.reps} reps; "
            f"{exercise.duration_seconds} seconds"
        )
    frequency = envelope.allowed_weekly_frequency
    lines.extend(
        [
            "",
            f"Allowed weekly frequency: **{frequency.minimum}–{frequency.maximum}**.",
            "Contraindication codes: "
            + (", ".join(f"`{code}`" for code in envelope.contraindication_codes) or "none"),
            "",
        ]
    )
    if plan.status is PlanStatus.BLOCKED:
        lines.extend(
            [
                "## Blocked for clinician review",
                "",
                *[f"- `{code.value}`" for code in plan.blocker_codes],
            ]
        )
    else:
        lines.extend(["## Personalized weekly placement", ""])
        for session in plan.sessions:
            lines.append(
                f"- **{session.day.value.title()} · {session.preferred_time.value.title()}** "
                f"({session.estimated_minutes} min; full approved envelope)"
            )
        lines.extend(
            [
                "",
                f"Coaching: `{plan.coaching_style.value}` · Language: `{plan.language}`",
                "Accessibility: "
                + (", ".join(f"`{code.value}`" for code in plan.accessibility) or "none"),
            ]
        )
    lines.extend(["", "**Next step:** clinician approves, revises, or rejects this draft."])
    return "\n".join(lines)
