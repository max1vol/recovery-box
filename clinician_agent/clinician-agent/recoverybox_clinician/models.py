"""Strict, de-identified inputs and deterministic review output models."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


class ValidationError(ValueError):
    """Raised when review input is unsafe, ambiguous, or malformed."""


_SESSION_ID_PATTERN = re.compile(r"^sess-[a-z0-9][a-z0-9-]{4,47}$")
_PARTICIPANT_REF_PATTERN = re.compile(r"^anon-[a-z0-9]{6,32}$")
_EXERCISE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,47}$")
_MODEL_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")
_SAFETY_FLAG_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,47}$")

_ALLOWED_FIELDS = frozenset(
    {
        "session_id",
        "participant_ref",
        "session_date",
        "exercise_id",
        "prescribed_reps",
        "completed_reps",
        "duration_seconds",
        "quality_score",
        "observation_confidence",
        "pain_reported",
        "stopped_early",
        "safety_flags",
        "model_version",
    }
)

_FORBIDDEN_FIELD_TOKENS = frozenset(
    {
        "name",
        "email",
        "phone",
        "address",
        "dob",
        "birth",
        "mrn",
        "nhs",
        "transcript",
        "note",
        "audio",
        "video",
        "image",
        "frame",
    }
)


def _ensure_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValidationError(f"{field} must be a boolean")
    return value


def _ensure_int(value: object, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"{field} must be an integer from {minimum} to {maximum}")
    return value


def _ensure_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a number from 0 to 1")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValidationError(f"{field} must be a number from 0 to 1")
    return parsed


def _ensure_string(value: object, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValidationError(f"{field} has an invalid de-identified format")
    return value


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """A bounded summary that cannot carry raw media, transcripts, or free text."""

    session_id: str
    participant_ref: str
    session_date: date
    exercise_id: str
    prescribed_reps: int
    completed_reps: int
    duration_seconds: int
    quality_score: float
    observation_confidence: float
    pain_reported: bool
    stopped_early: bool
    safety_flags: tuple[str, ...]
    model_version: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SessionSummary:
        if not isinstance(value, Mapping):
            raise ValidationError("each session summary must be an object")

        supplied = set(value)
        forbidden = sorted(
            field
            for field in supplied
            if any(token in field.lower().replace("-", "_") for token in _FORBIDDEN_FIELD_TOKENS)
        )
        if forbidden:
            raise ValidationError(
                "session summaries cannot include identifying, free-text, or raw-media fields: "
                + ", ".join(forbidden)
            )

        missing = sorted(_ALLOWED_FIELDS - supplied)
        unknown = sorted(supplied - _ALLOWED_FIELDS)
        if missing:
            raise ValidationError("missing session fields: " + ", ".join(missing))
        if unknown:
            raise ValidationError("unknown session fields: " + ", ".join(unknown))

        try:
            parsed_date = date.fromisoformat(str(value["session_date"]))
        except ValueError as exc:
            raise ValidationError("session_date must use YYYY-MM-DD") from exc

        prescribed_reps = _ensure_int(value["prescribed_reps"], "prescribed_reps", 1, 500)
        completed_reps = _ensure_int(value["completed_reps"], "completed_reps", 0, 500)
        if completed_reps > prescribed_reps:
            raise ValidationError("completed_reps cannot exceed prescribed_reps")

        raw_flags = value["safety_flags"]
        if not isinstance(raw_flags, list) or len(raw_flags) > 16:
            raise ValidationError("safety_flags must be a list with at most 16 items")
        safety_flags: list[str] = []
        for flag in raw_flags:
            if not isinstance(flag, str) or not _SAFETY_FLAG_PATTERN.fullmatch(flag):
                raise ValidationError("safety_flags must contain bounded uppercase reason codes")
            if flag not in safety_flags:
                safety_flags.append(flag)

        return cls(
            session_id=_ensure_string(value["session_id"], "session_id", _SESSION_ID_PATTERN),
            participant_ref=_ensure_string(
                value["participant_ref"], "participant_ref", _PARTICIPANT_REF_PATTERN
            ),
            session_date=parsed_date,
            exercise_id=_ensure_string(value["exercise_id"], "exercise_id", _EXERCISE_ID_PATTERN),
            prescribed_reps=prescribed_reps,
            completed_reps=completed_reps,
            duration_seconds=_ensure_int(value["duration_seconds"], "duration_seconds", 0, 86_400),
            quality_score=_ensure_float(value["quality_score"], "quality_score"),
            observation_confidence=_ensure_float(
                value["observation_confidence"], "observation_confidence"
            ),
            pain_reported=_ensure_bool(value["pain_reported"], "pain_reported"),
            stopped_early=_ensure_bool(value["stopped_early"], "stopped_early"),
            safety_flags=tuple(safety_flags),
            model_version=_ensure_string(
                value["model_version"], "model_version", _MODEL_VERSION_PATTERN
            ),
        )

    @property
    def completion_rate(self) -> float:
        return self.completed_reps / self.prescribed_reps


@dataclass(frozen=True, slots=True)
class QueueItem:
    session_id: str
    participant_ref: str
    attention_level: str
    score: int
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParticipantTrend:
    participant_ref: str
    first_session_id: str
    latest_session_id: str
    session_count: int
    completion_delta: float
    quality_delta: float
    direction: str


@dataclass(frozen=True, slots=True)
class CohortSnapshot:
    session_count: int
    participant_count: int
    completion_rate: float
    mean_quality_score: float
    mean_observation_confidence: float
    pain_reported_count: int
    stopped_early_count: int
    safety_flagged_count: int
    low_confidence_session_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReviewReport:
    question: str
    scope_status: str
    scope_message: str
    queue: tuple[QueueItem, ...]
    cohort: CohortSnapshot
    participant_trends: tuple[ParticipantTrend, ...]
    evidence_session_ids: tuple[str, ...]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
