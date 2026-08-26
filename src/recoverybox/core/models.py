"""Typed values shared by local movement analysis and the safety Guardian."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real


class SessionMode(StrEnum):
    """High-level lifecycle of one local rehabilitation session."""

    IDLE = "idle"
    CHECK_IN = "check_in"
    ACTIVE_EXERCISE = "active_exercise"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETE = "complete"


class GuardianAction(StrEnum):
    """Closed set of actions the local safety layer may return."""

    CONTINUE = "continue"
    CUE = "cue"
    PAUSE = "pause"
    STOP = "stop"
    ESCALATE = "escalate"


class GuardianReason(StrEnum):
    """Auditable reason codes attached to each Guardian decision."""

    WITHIN_LIMITS = "within_limits"
    EMERGENCY_REPORTED = "emergency_reported"
    PAIN_REPORTED = "pain_reported"
    WRONG_EXERCISE = "wrong_exercise"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    STALE_OBSERVATION = "stale_observation"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_CAMERA_EVIDENCE = "missing_camera_evidence"
    CAMERA_DISAGREEMENT = "camera_disagreement"
    UNKNOWN_CUE = "unknown_cue"
    CUE_NOT_ALLOWED = "cue_not_allowed"
    LOCAL_CUE_ACCEPTED = "local_cue_accepted"
    LOCAL_CUE_IGNORED_FOR_SAFETY = "local_cue_ignored_for_safety"
    LEARNED_MODEL_INCREASED_CAUTION = "learned_model_increased_caution"
    LEARNED_MODEL_PRESERVED_CAUTION = "learned_model_preserved_caution"
    LEARNED_MODEL_SUGGESTION_IGNORED = "learned_model_suggestion_ignored"
    REALTIME_UNAVAILABLE = "realtime_unavailable"
    CUE_DELIVERY_UNAVAILABLE = "cue_delivery_unavailable"
    INHERITED_RUNTIME_CAUTION = "inherited_runtime_caution"
    RUNTIME_BOUNDARY_FAILURE = "runtime_boundary_failure"
    SAFETY_ENFORCEMENT_FAILURE = "safety_enforcement_failure"


class GuardianRuntimeFault(StrEnum):
    """Closed runtime-fault vocabulary arbitrated by the local Guardian."""

    REALTIME_UNAVAILABLE = "realtime_unavailable"
    CUE_DELIVERY_UNAVAILABLE = "cue_delivery_unavailable"
    INHERITED_CAUTION = "inherited_caution"
    RUNTIME_BOUNDARY_FAILURE = "runtime_boundary_failure"
    SAFETY_ENFORCEMENT_FAILURE = "safety_enforcement_failure"


@dataclass(frozen=True, slots=True)
class MovementObservation:
    """One sanitized, pose-derived observation; it contains no image or audio."""

    exercise_id: str
    timestamp_ms: int
    confidence: float
    camera_disagreement_degrees: float | None
    pose_age_ms: int
    camera_view_count: int = 2
    rep_index: int = 0
    phase: str = "unknown"
    quality_label: str | None = None
    pain_reported: bool = False
    emergency_reported: bool = False
    wrong_exercise: bool = False
    out_of_distribution: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.exercise_id, str):
            raise TypeError("exercise_id must be a string")
        exercise_id = self.exercise_id.strip()
        if not isinstance(self.phase, str):
            raise TypeError("phase must be a string")
        phase = self.phase.strip()
        if not exercise_id:
            raise ValueError("exercise_id must not be empty")
        _validate_non_negative_integer("timestamp_ms", self.timestamp_ms)
        _validate_unit_interval("confidence", self.confidence)
        if self.camera_disagreement_degrees is not None:
            _validate_degrees(
                "camera_disagreement_degrees",
                self.camera_disagreement_degrees,
            )
        _validate_non_negative_integer("pose_age_ms", self.pose_age_ms)
        if (
            isinstance(self.camera_view_count, bool)
            or not isinstance(self.camera_view_count, int)
            or self.camera_view_count < 1
        ):
            raise ValueError("camera_view_count must be a positive integer")
        _validate_non_negative_integer("rep_index", self.rep_index)
        if not phase:
            raise ValueError("phase must not be empty")
        if self.quality_label is not None:
            if not isinstance(self.quality_label, str):
                raise TypeError("quality_label must be a string when provided")
            if not self.quality_label.strip():
                raise ValueError("quality_label must be non-empty when provided")
        for field_name in (
            "pain_reported",
            "emergency_reported",
            "wrong_exercise",
            "out_of_distribution",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        object.__setattr__(self, "exercise_id", exercise_id)
        object.__setattr__(self, "phase", phase)
        if self.quality_label is not None:
            object.__setattr__(self, "quality_label", self.quality_label.strip())


@dataclass(frozen=True, slots=True)
class ExercisePlan:
    """Clinician-configured safety envelope for one prescribed exercise."""

    exercise_id: str
    allowed_cue_ids: frozenset[str]
    target_reps: int = 10
    min_confidence: float = 0.7
    max_camera_disagreement_degrees: float = 12.0
    max_pose_age_ms: int = 500
    required_camera_views: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.exercise_id, str):
            raise TypeError("exercise_id must be a string")
        exercise_id = self.exercise_id.strip()
        if not exercise_id:
            raise ValueError("exercise_id must not be empty")
        if (
            isinstance(self.target_reps, bool)
            or not isinstance(self.target_reps, int)
            or self.target_reps <= 0
        ):
            raise ValueError("target_reps must be positive")
        _validate_unit_interval("min_confidence", self.min_confidence)
        _validate_degrees(
            "max_camera_disagreement_degrees",
            self.max_camera_disagreement_degrees,
        )
        _validate_non_negative_integer("max_pose_age_ms", self.max_pose_age_ms)
        if (
            isinstance(self.required_camera_views, bool)
            or not isinstance(self.required_camera_views, int)
            or self.required_camera_views < 1
        ):
            raise ValueError("required_camera_views must be a positive integer")

        normalized_cues = frozenset(str(cue_id).strip() for cue_id in self.allowed_cue_ids)
        if "" in normalized_cues:
            raise ValueError("allowed_cue_ids must not contain empty identifiers")
        object.__setattr__(self, "exercise_id", exercise_id)
        object.__setattr__(self, "allowed_cue_ids", normalized_cues)


@dataclass(frozen=True, slots=True)
class LearnedSuggestion:
    """Untrusted severity-only proposal from a learned model.

    Learned output may preserve or increase caution, but it cannot nominate a
    cue. Cue selection is reserved for the deterministic local Guardian path.
    """

    action: GuardianAction
    cue_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, GuardianAction):
            raise TypeError("action must be a GuardianAction")
        if self.action is GuardianAction.CUE:
            raise ValueError("learned suggestions cannot select cues")
        if self.cue_id is not None:
            raise ValueError("learned suggestions cannot include cue_id")


@dataclass(frozen=True, slots=True)
class LocalCueRequest:
    """Untrusted local exercise event asking the Guardian to select one cue.

    Pose code can report that a deterministic event occurred, but it cannot
    authorize speech. The Guardian still validates the cue against both the
    reviewed catalog and the active exercise plan before returning ``CUE``.
    """

    cue_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.cue_id, str):
            raise TypeError("cue_id must be a string")
        cue_id = self.cue_id.strip()
        if not cue_id:
            raise ValueError("cue_id must not be empty")
        object.__setattr__(self, "cue_id", cue_id)


@dataclass(frozen=True, slots=True, init=False)
class GuardianDecision:
    """Sealed final local action and its machine-readable audit trail.

    Only :class:`recoverybox.core.guardian.Guardian` issues instances.  The
    public constructor is deliberately closed so application code cannot turn
    an arbitrary enum value into safety authority.
    """

    action: GuardianAction
    reason_codes: tuple[GuardianReason, ...]
    rule_version: str
    sequence: int
    cue_id: str | None = None
    _issuer: object

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("GuardianDecision can only be issued by Guardian")

    @classmethod
    def _issue(
        cls,
        *,
        action: GuardianAction,
        reason_codes: tuple[GuardianReason, ...],
        rule_version: str,
        sequence: int,
        cue_id: str | None = None,
        _issuer: object,
    ) -> GuardianDecision:
        if _issuer is None:
            raise TypeError("GuardianDecision can only be issued by Guardian")
        decision = object.__new__(cls)
        object.__setattr__(decision, "action", action)
        object.__setattr__(decision, "reason_codes", reason_codes)
        object.__setattr__(decision, "rule_version", rule_version)
        object.__setattr__(decision, "sequence", sequence)
        object.__setattr__(decision, "cue_id", cue_id)
        object.__setattr__(decision, "_issuer", _issuer)
        decision.__post_init__()
        return decision

    def __post_init__(self) -> None:
        if not isinstance(self.action, GuardianAction):
            raise TypeError("action must be a GuardianAction")
        if (
            type(self.reason_codes) is not tuple
            or not self.reason_codes
            or not all(isinstance(reason, GuardianReason) for reason in self.reason_codes)
        ):
            raise ValueError("reason_codes must not be empty")
        if not isinstance(self.rule_version, str) or not self.rule_version.strip():
            raise ValueError("rule_version must not be empty")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if self.action is GuardianAction.CUE:
            if self.cue_id is None or not isinstance(self.cue_id, str) or not self.cue_id.strip():
                raise ValueError("a cue decision requires cue_id")
            object.__setattr__(self, "cue_id", self.cue_id.strip())
        elif self.cue_id is not None:
            raise ValueError("cue_id is only valid for a cue decision")


def _issue_guardian_decision(
    *,
    action: GuardianAction,
    reason_codes: tuple[GuardianReason, ...],
    rule_version: str,
    sequence: int,
    cue_id: str | None = None,
    _issuer: object,
) -> GuardianDecision:
    """Issue one sealed verdict for the Guardian implementation only."""

    return GuardianDecision._issue(
        action=action,
        reason_codes=reason_codes,
        rule_version=rule_version,
        sequence=sequence,
        cue_id=cue_id,
        _issuer=_issuer,
    )


def _is_guardian_decision_issued_by(value: object, issuer: object) -> bool:
    """Return whether ``value`` carries one Guardian instance's authority."""

    return isinstance(value, GuardianDecision) and getattr(value, "_issuer", None) is issuer


def _validate_unit_interval(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be finite and between 0 and 1")


def _validate_degrees(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 <= value <= 180.0
    ):
        raise ValueError(f"{name} must be finite and between 0 and 180 degrees")


def _validate_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
