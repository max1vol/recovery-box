"""Deterministic safety arbitration for local exercise coaching."""

from __future__ import annotations

from recoverybox.core.cues import DEFAULT_CUE_CATALOG, ApprovedCueCatalog
from recoverybox.core.models import (
    ExercisePlan,
    GuardianAction,
    GuardianDecision,
    GuardianReason,
    LearnedSuggestion,
    LocalCueRequest,
    MovementObservation,
)

_ACTION_CAUTION = {
    GuardianAction.CONTINUE: 0,
    GuardianAction.CUE: 1,
    GuardianAction.PAUSE: 2,
    GuardianAction.STOP: 3,
    GuardianAction.ESCALATE: 4,
}


class Guardian:
    """Apply fixed safety rules and arbitrate untrusted learned suggestions."""

    RULE_VERSION = "guardian-v1"

    def __init__(
        self,
        cue_catalog: ApprovedCueCatalog = DEFAULT_CUE_CATALOG,
        *,
        rule_version: str = RULE_VERSION,
    ) -> None:
        if not rule_version.strip():
            raise ValueError("rule_version must not be empty")
        self._cue_catalog = cue_catalog
        self._rule_version = rule_version.strip()

    @property
    def cue_catalog(self) -> ApprovedCueCatalog:
        return self._cue_catalog

    @property
    def rule_version(self) -> str:
        return self._rule_version

    def decide(
        self,
        observation: MovementObservation,
        plan: ExercisePlan,
        learned_suggestion: LearnedSuggestion | None = None,
        *,
        local_cue_request: LocalCueRequest | None = None,
    ) -> GuardianDecision:
        """Return a safe action; learned output can never reduce its caution."""

        if learned_suggestion is not None and local_cue_request is not None:
            raise ValueError("provide either a local cue request or a learned suggestion, not both")

        action, reasons = self._deterministic_decision(observation, plan)
        cue_id: str | None = None

        if local_cue_request is not None:
            action, cue_id = self._apply_local_cue_request(
                action,
                local_cue_request,
                plan,
                reasons,
            )
        elif learned_suggestion is not None:
            action, cue_id = self._apply_learned_suggestion(
                action,
                learned_suggestion,
                plan,
                reasons,
            )

        if not reasons:
            reasons.append(GuardianReason.WITHIN_LIMITS)

        return GuardianDecision(
            action=action,
            cue_id=cue_id,
            reason_codes=tuple(reasons),
            rule_version=self._rule_version,
        )

    def _apply_local_cue_request(
        self,
        deterministic_action: GuardianAction,
        request: LocalCueRequest,
        plan: ExercisePlan,
        reasons: list[GuardianReason],
    ) -> tuple[GuardianAction, str | None]:
        """Validate a pose-derived event without transferring cue authority."""

        if _ACTION_CAUTION[deterministic_action] >= _ACTION_CAUTION[GuardianAction.PAUSE]:
            reasons.append(GuardianReason.LOCAL_CUE_IGNORED_FOR_SAFETY)
            return deterministic_action, None
        if not self._cue_catalog.is_approved(request.cue_id):
            reasons.append(GuardianReason.UNKNOWN_CUE)
            return GuardianAction.PAUSE, None
        if request.cue_id not in plan.allowed_cue_ids:
            reasons.append(GuardianReason.CUE_NOT_ALLOWED)
            return GuardianAction.PAUSE, None
        reasons.append(GuardianReason.LOCAL_CUE_ACCEPTED)
        return GuardianAction.CUE, request.cue_id

    def _deterministic_decision(
        self,
        observation: MovementObservation,
        plan: ExercisePlan,
    ) -> tuple[GuardianAction, list[GuardianReason]]:
        reasons: list[GuardianReason] = []

        if observation.emergency_reported:
            reasons.append(GuardianReason.EMERGENCY_REPORTED)
        if observation.pain_reported:
            reasons.append(GuardianReason.PAIN_REPORTED)
        if observation.exercise_id != plan.exercise_id or observation.wrong_exercise:
            reasons.append(GuardianReason.WRONG_EXERCISE)
        if observation.out_of_distribution:
            reasons.append(GuardianReason.OUT_OF_DISTRIBUTION)
        if observation.pose_age_ms > plan.max_pose_age_ms:
            reasons.append(GuardianReason.STALE_OBSERVATION)
        if observation.confidence < plan.min_confidence:
            reasons.append(GuardianReason.LOW_CONFIDENCE)
        if observation.camera_view_count < plan.required_camera_views:
            reasons.append(GuardianReason.MISSING_CAMERA_EVIDENCE)
        elif plan.required_camera_views > 1:
            if observation.camera_disagreement_degrees is None:
                reasons.append(GuardianReason.MISSING_CAMERA_EVIDENCE)
            elif observation.camera_disagreement_degrees > plan.max_camera_disagreement_degrees:
                reasons.append(GuardianReason.CAMERA_DISAGREEMENT)

        if GuardianReason.EMERGENCY_REPORTED in reasons:
            return GuardianAction.ESCALATE, reasons
        if GuardianReason.PAIN_REPORTED in reasons:
            return GuardianAction.STOP, reasons
        if reasons:
            return GuardianAction.PAUSE, reasons
        return GuardianAction.CONTINUE, reasons

    def _apply_learned_suggestion(
        self,
        deterministic_action: GuardianAction,
        suggestion: LearnedSuggestion,
        plan: ExercisePlan,
        reasons: list[GuardianReason],
    ) -> tuple[GuardianAction, str | None]:
        proposed_action = suggestion.action

        if proposed_action is GuardianAction.CUE:
            assert suggestion.cue_id is not None
            if not self._cue_catalog.is_approved(suggestion.cue_id):
                proposed_action = GuardianAction.PAUSE
                reasons.append(GuardianReason.UNKNOWN_CUE)
            elif suggestion.cue_id not in plan.allowed_cue_ids:
                proposed_action = GuardianAction.PAUSE
                reasons.append(GuardianReason.CUE_NOT_ALLOWED)
            elif _ACTION_CAUTION[deterministic_action] < _ACTION_CAUTION[GuardianAction.PAUSE]:
                reasons.append(GuardianReason.LEARNED_MODEL_CUE_ACCEPTED)
                return GuardianAction.CUE, suggestion.cue_id

        deterministic_caution = _ACTION_CAUTION[deterministic_action]
        proposed_caution = _ACTION_CAUTION[proposed_action]

        if proposed_caution > deterministic_caution:
            reasons.append(GuardianReason.LEARNED_MODEL_INCREASED_CAUTION)
            return proposed_action, None
        if proposed_caution == deterministic_caution and deterministic_caution >= 2:
            reasons.append(GuardianReason.LEARNED_MODEL_PRESERVED_CAUTION)
            return deterministic_action, None

        if proposed_action is not GuardianAction.CONTINUE or deterministic_caution > 0:
            reasons.append(GuardianReason.LEARNED_MODEL_SUGGESTION_IGNORED)
        return deterministic_action, None
