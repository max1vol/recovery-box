"""Strict, coded input contract for the synthetic Recovery Swarm demo."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

PLAN_REQUEST_SCHEMA = "recoverybox.plan-request.v1"

_SYNTHETIC_CODE = re.compile(r"SYNTH-[A-Z0-9][A-Z0-9._-]{2,63}\Z")
_EXERCISE_ID = re.compile(r"[a-z][a-z0-9-]{2,63}\Z")
_LANGUAGE_CODE = re.compile(r"[a-z]{2}(?:-[A-Z]{2})?\Z")


class PlanRequestValidationError(ValueError):
    """Raised for invalid input without including any submitted value."""


class Day(StrEnum):
    MONDAY = "MONDAY"
    TUESDAY = "TUESDAY"
    WEDNESDAY = "WEDNESDAY"
    THURSDAY = "THURSDAY"
    FRIDAY = "FRIDAY"
    SATURDAY = "SATURDAY"
    SUNDAY = "SUNDAY"


class PreferredTime(StrEnum):
    MORNING = "MORNING"
    MIDDAY = "MIDDAY"
    AFTERNOON = "AFTERNOON"
    EVENING = "EVENING"


class CoachingStyle(StrEnum):
    ENCOURAGING = "ENCOURAGING"
    CONCISE = "CONCISE"
    STEP_BY_STEP = "STEP_BY_STEP"
    QUIET = "QUIET"


class EquipmentCode(StrEnum):
    NONE = "NONE"
    CHAIR = "CHAIR"
    RESISTANCE_BAND = "RESISTANCE_BAND"
    YOGA_MAT = "YOGA_MAT"
    LIGHT_WEIGHTS = "LIGHT_WEIGHTS"
    STEP = "STEP"


class AccessibilityCode(StrEnum):
    LARGE_TEXT = "LARGE_TEXT"
    HIGH_CONTRAST = "HIGH_CONTRAST"
    CAPTIONS = "CAPTIONS"
    REDUCED_AUDIO = "REDUCED_AUDIO"
    SCREEN_READER = "SCREEN_READER"
    SLOW_PACING = "SLOW_PACING"
    SEATED_SETUP = "SEATED_SETUP"


@dataclass(frozen=True, slots=True)
class ExercisePrescription:
    """One clinician-approved exercise and its immutable dose."""

    exercise_id: str
    sets: int
    reps: int
    duration_seconds: int
    required_equipment: tuple[EquipmentCode, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "exercise_id": self.exercise_id,
            "sets": self.sets,
            "reps": self.reps,
            "duration_seconds": self.duration_seconds,
            "required_equipment": [item.value for item in self.required_equipment],
        }


@dataclass(frozen=True, slots=True)
class WeeklyFrequency:
    """Clinician-approved inclusive weekly-frequency interval."""

    minimum: int
    maximum: int

    def as_dict(self) -> dict[str, int]:
        return {"minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True, slots=True)
class TreatmentEnvelope:
    """Immutable prescription boundary supplied by the clinician."""

    approved_exercises: tuple[ExercisePrescription, ...]
    allowed_weekly_frequency: WeeklyFrequency
    contraindication_codes: tuple[str, ...]
    review_date: date

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved_exercises": [item.as_dict() for item in self.approved_exercises],
            "allowed_weekly_frequency": self.allowed_weekly_frequency.as_dict(),
            "contraindication_codes": list(self.contraindication_codes),
            "review_date": self.review_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ClinicianDiagnosis:
    """A coded synthetic diagnosis plus the clinician's treatment envelope."""

    diagnosis_code: str
    treatment_envelope: TreatmentEnvelope

    def as_dict(self) -> dict[str, Any]:
        return {
            "diagnosis_code": self.diagnosis_code,
            "treatment_envelope": self.treatment_envelope.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class UserPreferences:
    """Coded preferences that may shape presentation and scheduling only."""

    available_days: tuple[Day, ...]
    preferred_time: PreferredTime
    max_minutes: int
    equipment: tuple[EquipmentCode, ...]
    coaching_style: CoachingStyle
    language: str
    accessibility: tuple[AccessibilityCode, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "available_days": [item.value for item in self.available_days],
            "preferred_time": self.preferred_time.value,
            "max_minutes": self.max_minutes,
            "equipment": [item.value for item in self.equipment],
            "coaching_style": self.coaching_style.value,
            "language": self.language,
            "accessibility": [item.value for item in self.accessibility],
        }


@dataclass(frozen=True, slots=True)
class PlanRequest:
    """Validated synthetic-only plan request."""

    request_id: str
    clinician_diagnosis: ClinicianDiagnosis
    user_preferences: UserPreferences
    schema: str = PLAN_REQUEST_SCHEMA
    synthetic: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "synthetic": self.synthetic,
            "request_id": self.request_id,
            "clinician_diagnosis": self.clinician_diagnosis.as_dict(),
            "user_preferences": self.user_preferences.as_dict(),
        }


def _object(value: Any, *, path: str, fields: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanRequestValidationError(f"{path} must be an object")
    if set(value) != fields:
        raise PlanRequestValidationError(f"{path} has missing or unsupported fields")
    return value


def _list(value: Any, *, path: str, minimum: int, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise PlanRequestValidationError(f"{path} must be an array")
    if not minimum <= len(value) <= maximum:
        raise PlanRequestValidationError(f"{path} has an invalid item count")
    return value


def _integer(value: Any, *, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanRequestValidationError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise PlanRequestValidationError(f"{path} is outside the allowed range")
    return value


def _string(value: Any, *, path: str) -> str:
    if not isinstance(value, str):
        raise PlanRequestValidationError(f"{path} must be a string")
    return value


def _enum(value: Any, enum_type: type[StrEnum], *, path: str) -> StrEnum:
    raw = _string(value, path=path)
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise PlanRequestValidationError(f"{path} is not an allowed code") from exc


def _unique(values: tuple[Any, ...], *, path: str) -> None:
    if len(set(values)) != len(values):
        raise PlanRequestValidationError(f"{path} contains duplicate codes")


def _synthetic_code(value: Any, *, path: str) -> str:
    code = _string(value, path=path)
    if _SYNTHETIC_CODE.fullmatch(code) is None:
        raise PlanRequestValidationError(f"{path} must be a synthetic code")
    return code


def _parse_exercise(value: Any, *, index: int) -> ExercisePrescription:
    path = f"clinician_diagnosis.treatment_envelope.approved_exercises[{index}]"
    obj = _object(
        value,
        path=path,
        fields=frozenset(
            {"exercise_id", "sets", "reps", "duration_seconds", "required_equipment"}
        ),
    )
    exercise_id = _string(obj["exercise_id"], path=f"{path}.exercise_id")
    if _EXERCISE_ID.fullmatch(exercise_id) is None:
        raise PlanRequestValidationError(f"{path}.exercise_id is not a valid identifier")
    raw_equipment = _list(
        obj["required_equipment"], path=f"{path}.required_equipment", minimum=1, maximum=8
    )
    equipment = tuple(
        _enum(item, EquipmentCode, path=f"{path}.required_equipment[{i}]")
        for i, item in enumerate(raw_equipment)
    )
    _unique(equipment, path=f"{path}.required_equipment")
    if EquipmentCode.NONE in equipment and len(equipment) != 1:
        raise PlanRequestValidationError(f"{path}.required_equipment has incompatible codes")
    return ExercisePrescription(
        exercise_id=exercise_id,
        sets=_integer(obj["sets"], path=f"{path}.sets", minimum=1, maximum=20),
        reps=_integer(obj["reps"], path=f"{path}.reps", minimum=1, maximum=100),
        duration_seconds=_integer(
            obj["duration_seconds"],
            path=f"{path}.duration_seconds",
            minimum=15,
            maximum=3_600,
        ),
        required_equipment=equipment,  # type: ignore[arg-type]
    )


def parse_plan_request(value: Any) -> PlanRequest:
    """Validate and normalize a strict synthetic request.

    Error messages name only schema locations and rules. Submitted values are
    intentionally never copied into an exception.
    """

    root = _object(
        value,
        path="request",
        fields=frozenset(
            {"schema", "synthetic", "request_id", "clinician_diagnosis", "user_preferences"}
        ),
    )
    if root["schema"] != PLAN_REQUEST_SCHEMA:
        raise PlanRequestValidationError("request.schema is unsupported")
    if root["synthetic"] is not True:
        raise PlanRequestValidationError("request.synthetic must be true")
    request_id = _synthetic_code(root["request_id"], path="request.request_id")

    diagnosis_obj = _object(
        root["clinician_diagnosis"],
        path="clinician_diagnosis",
        fields=frozenset({"diagnosis_code", "treatment_envelope"}),
    )
    diagnosis_code = _synthetic_code(
        diagnosis_obj["diagnosis_code"], path="clinician_diagnosis.diagnosis_code"
    )
    envelope_obj = _object(
        diagnosis_obj["treatment_envelope"],
        path="clinician_diagnosis.treatment_envelope",
        fields=frozenset(
            {
                "approved_exercises",
                "allowed_weekly_frequency",
                "contraindication_codes",
                "review_date",
            }
        ),
    )
    exercise_values = _list(
        envelope_obj["approved_exercises"],
        path="clinician_diagnosis.treatment_envelope.approved_exercises",
        minimum=1,
        maximum=16,
    )
    exercises = tuple(
        _parse_exercise(item, index=index) for index, item in enumerate(exercise_values)
    )
    _unique(
        tuple(item.exercise_id for item in exercises),
        path="clinician_diagnosis.treatment_envelope.approved_exercises",
    )

    frequency_obj = _object(
        envelope_obj["allowed_weekly_frequency"],
        path="clinician_diagnosis.treatment_envelope.allowed_weekly_frequency",
        fields=frozenset({"minimum", "maximum"}),
    )
    frequency = WeeklyFrequency(
        minimum=_integer(
            frequency_obj["minimum"],
            path="clinician_diagnosis.treatment_envelope.allowed_weekly_frequency.minimum",
            minimum=1,
            maximum=7,
        ),
        maximum=_integer(
            frequency_obj["maximum"],
            path="clinician_diagnosis.treatment_envelope.allowed_weekly_frequency.maximum",
            minimum=1,
            maximum=7,
        ),
    )
    if frequency.minimum > frequency.maximum:
        raise PlanRequestValidationError(
            "clinician_diagnosis.treatment_envelope.allowed_weekly_frequency is invalid"
        )

    contraindication_values = _list(
        envelope_obj["contraindication_codes"],
        path="clinician_diagnosis.treatment_envelope.contraindication_codes",
        minimum=0,
        maximum=32,
    )
    contraindications = tuple(
        _synthetic_code(
            item,
            path=f"clinician_diagnosis.treatment_envelope.contraindication_codes[{index}]",
        )
        for index, item in enumerate(contraindication_values)
    )
    _unique(
        contraindications,
        path="clinician_diagnosis.treatment_envelope.contraindication_codes",
    )
    review_text = _string(
        envelope_obj["review_date"],
        path="clinician_diagnosis.treatment_envelope.review_date",
    )
    try:
        review_date = date.fromisoformat(review_text)
    except ValueError as exc:
        raise PlanRequestValidationError(
            "clinician_diagnosis.treatment_envelope.review_date must be an ISO date"
        ) from exc

    preferences_obj = _object(
        root["user_preferences"],
        path="user_preferences",
        fields=frozenset(
            {
                "available_days",
                "preferred_time",
                "max_minutes",
                "equipment",
                "coaching_style",
                "language",
                "accessibility",
            }
        ),
    )
    day_values = _list(
        preferences_obj["available_days"],
        path="user_preferences.available_days",
        minimum=1,
        maximum=7,
    )
    days = tuple(
        _enum(item, Day, path=f"user_preferences.available_days[{index}]")
        for index, item in enumerate(day_values)
    )
    _unique(days, path="user_preferences.available_days")
    equipment_values = _list(
        preferences_obj["equipment"],
        path="user_preferences.equipment",
        minimum=1,
        maximum=8,
    )
    equipment = tuple(
        _enum(item, EquipmentCode, path=f"user_preferences.equipment[{index}]")
        for index, item in enumerate(equipment_values)
    )
    _unique(equipment, path="user_preferences.equipment")
    if EquipmentCode.NONE in equipment and len(equipment) != 1:
        raise PlanRequestValidationError("user_preferences.equipment has incompatible codes")
    accessibility_values = _list(
        preferences_obj["accessibility"],
        path="user_preferences.accessibility",
        minimum=0,
        maximum=7,
    )
    accessibility = tuple(
        _enum(item, AccessibilityCode, path=f"user_preferences.accessibility[{index}]")
        for index, item in enumerate(accessibility_values)
    )
    _unique(accessibility, path="user_preferences.accessibility")
    language = _string(preferences_obj["language"], path="user_preferences.language")
    if _LANGUAGE_CODE.fullmatch(language) is None:
        raise PlanRequestValidationError("user_preferences.language is not a language code")

    return PlanRequest(
        request_id=request_id,
        clinician_diagnosis=ClinicianDiagnosis(
            diagnosis_code=diagnosis_code,
            treatment_envelope=TreatmentEnvelope(
                approved_exercises=exercises,
                allowed_weekly_frequency=frequency,
                contraindication_codes=contraindications,
                review_date=review_date,
            ),
        ),
        user_preferences=UserPreferences(
            available_days=days,  # type: ignore[arg-type]
            preferred_time=_enum(
                preferences_obj["preferred_time"],
                PreferredTime,
                path="user_preferences.preferred_time",
            ),  # type: ignore[arg-type]
            max_minutes=_integer(
                preferences_obj["max_minutes"],
                path="user_preferences.max_minutes",
                minimum=5,
                maximum=180,
            ),
            equipment=equipment,  # type: ignore[arg-type]
            coaching_style=_enum(
                preferences_obj["coaching_style"],
                CoachingStyle,
                path="user_preferences.coaching_style",
            ),  # type: ignore[arg-type]
            language=language,
            accessibility=accessibility,  # type: ignore[arg-type]
        ),
    )


DEMO_REQUEST: dict[str, Any] = {
    "schema": PLAN_REQUEST_SCHEMA,
    "synthetic": True,
    "request_id": "SYNTH-REQUEST-001",
    "clinician_diagnosis": {
        "diagnosis_code": "SYNTH-DX-KNEE-01",
        "treatment_envelope": {
            "approved_exercises": [
                {
                    "exercise_id": "seated-knee-extension",
                    "sets": 2,
                    "reps": 8,
                    "duration_seconds": 240,
                    "required_equipment": ["CHAIR"],
                },
                {
                    "exercise_id": "supported-sit-to-stand",
                    "sets": 2,
                    "reps": 6,
                    "duration_seconds": 300,
                    "required_equipment": ["CHAIR"],
                },
            ],
            "allowed_weekly_frequency": {"minimum": 2, "maximum": 3},
            "contraindication_codes": ["SYNTH-CI-ACUTE-PAIN"],
            "review_date": "2026-09-30",
        },
    },
    "user_preferences": {
        "available_days": ["MONDAY", "WEDNESDAY", "FRIDAY"],
        "preferred_time": "MORNING",
        "max_minutes": 20,
        "equipment": ["CHAIR"],
        "coaching_style": "ENCOURAGING",
        "language": "en",
        "accessibility": ["LARGE_TEXT", "CAPTIONS"],
    },
}
