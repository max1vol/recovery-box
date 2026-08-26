"""Deterministic safety arbitration for local exercise coaching."""

from __future__ import annotations

import threading

from recoverybox.core.cues import (
    DEFAULT_CUE_CATALOG,
    SQUAT_SCRIPTED_SESSION_CUE_IDS,
    ApprovedCueCatalog,
)
from recoverybox.core.models import (
    ExercisePlan,
    GuardianAction,
    GuardianDecision,
    GuardianReason,
    GuardianRuntimeFault,
    LearnedSuggestion,
    LocalCueRequest,
    MovementObservation,
    _is_guardian_decision_issued_by,
    _issue_guardian_decision,
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
        self.__issuer = object()
        self._issue_lock = threading.Lock()
        self._latest_sequence = 0
        self._consumed_activation_sequences: set[int] = set()

    @property
    def cue_catalog(self) -> ApprovedCueCatalog:
        return self._cue_catalog

    @property
    def rule_version(self) -> str:
        return self._rule_version

    def issued(self, decision: object) -> bool:
        """Return whether this exact Guardian instance issued ``decision``."""

        return _is_guardian_decision_issued_by(decision, self.__issuer)

    @property
    def latest_sequence(self) -> int:
        """Monotonic sequence of the most recently issued local verdict."""

        with self._issue_lock:
            return self._latest_sequence

    def consume_activation(self, decision: object) -> bool:
        """Atomically consume the latest CONTINUE verdict for one activation."""

        with self._issue_lock:
            if not _is_guardian_decision_issued_by(decision, self.__issuer):
                return False
            assert isinstance(decision, GuardianDecision)
            if (
                decision.action is not GuardianAction.CONTINUE
                or decision.sequence != self._latest_sequence
                or decision.sequence in self._consumed_activation_sequences
            ):
                return False
            self._consumed_activation_sequences.add(decision.sequence)
            return True

    def _issue(
        self,
        *,
        action: GuardianAction,
        reason_codes: tuple[GuardianReason, ...],
        cue_id: str | None = None,
    ) -> GuardianDecision:
        with self._issue_lock:
            self._latest_sequence += 1
            return _issue_guardian_decision(
                action=action,
                cue_id=cue_id,
                reason_codes=reason_codes,
                rule_version=self._rule_version,
                sequence=self._latest_sequence,
                _issuer=self.__issuer,
            )

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
                reasons,
            )

        if not reasons:
            reasons.append(GuardianReason.WITHIN_LIMITS)

        return self._issue(
            action=action,
            cue_id=cue_id,
            reason_codes=tuple(reasons),
        )

    def decide_scripted_session_cue(
        self,
        request: LocalCueRequest,
        plan: ExercisePlan,
    ) -> GuardianDecision:
        """Authorize one closed, observation-free session-stage cue.

        This path exists only for the welcome and first-person-detected stages,
        where an exercise observation either does not exist yet or has already
        been validated separately.  It cannot authorize rep, correction,
        safety, or model-selected speech.
        """

        if not isinstance(request, LocalCueRequest):
            raise TypeError("request must be a LocalCueRequest")
        if not isinstance(plan, ExercisePlan):
            raise TypeError("plan must be an ExercisePlan")

        cue_id = request.cue_id
        if not self._cue_catalog.is_approved(cue_id):
            return self._issue(
                action=GuardianAction.PAUSE,
                reason_codes=(GuardianReason.UNKNOWN_CUE,),
            )
        if cue_id not in SQUAT_SCRIPTED_SESSION_CUE_IDS or cue_id not in plan.allowed_cue_ids:
            return self._issue(
                action=GuardianAction.PAUSE,
                reason_codes=(GuardianReason.CUE_NOT_ALLOWED,),
            )
        return self._issue(
            action=GuardianAction.CUE,
            cue_id=cue_id,
            reason_codes=(GuardianReason.LOCAL_CUE_ACCEPTED,),
        )

    def decide_runtime_fault(self, fault: GuardianRuntimeFault) -> GuardianDecision:
        """Fail closed for one typed non-camera runtime fault.

        Runtime code reports the fault category but never selects the action.
        This keeps output/connectivity caution under the same sealed Guardian
        authority without pretending it came from movement evidence.
        """

        if not isinstance(fault, GuardianRuntimeFault):
            raise TypeError("fault must be a GuardianRuntimeFault")
        reasons = {
            GuardianRuntimeFault.REALTIME_UNAVAILABLE: (GuardianReason.REALTIME_UNAVAILABLE,),
            GuardianRuntimeFault.CUE_DELIVERY_UNAVAILABLE: (
                GuardianReason.CUE_DELIVERY_UNAVAILABLE,
            ),
            GuardianRuntimeFault.INHERITED_CAUTION: (GuardianReason.INHERITED_RUNTIME_CAUTION,),
            GuardianRuntimeFault.RUNTIME_BOUNDARY_FAILURE: (
                GuardianReason.RUNTIME_BOUNDARY_FAILURE,
            ),
            GuardianRuntimeFault.SAFETY_ENFORCEMENT_FAILURE: (
                GuardianReason.SAFETY_ENFORCEMENT_FAILURE,
            ),
        }
        return self._issue(
            action=(
                GuardianAction.ESCALATE
                if fault is GuardianRuntimeFault.SAFETY_ENFORCEMENT_FAILURE
                else GuardianAction.PAUSE
            ),
            reason_codes=reasons[fault],
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
        reasons: list[GuardianReason],
    ) -> tuple[GuardianAction, str | None]:
        proposed_action = suggestion.action

        # LearnedSuggestion rejects CUE at construction. Keep this branch as a
        # fail-closed defense against a forged or deserialized instance.
        if proposed_action is GuardianAction.CUE:
            proposed_action = GuardianAction.PAUSE

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
