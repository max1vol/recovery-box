from __future__ import annotations

import math

import pytest

from recoverybox.core import (
    DEFAULT_CUE_CATALOG,
    ApprovedCue,
    ApprovedCueCatalog,
    CueId,
    CueKind,
    ExercisePlan,
    Guardian,
    GuardianAction,
    GuardianDecision,
    GuardianReason,
    LearnedSuggestion,
    LocalCueRequest,
    MovementObservation,
)


def observation(**overrides: object) -> MovementObservation:
    values: dict[str, object] = {
        "exercise_id": "seated-knee-extension",
        "timestamp_ms": 1_000,
        "confidence": 0.92,
        "camera_disagreement_degrees": 3.0,
        "pose_age_ms": 50,
        "rep_index": 3,
        "phase": "extension",
    }
    values.update(overrides)
    return MovementObservation(**values)  # type: ignore[arg-type]


@pytest.fixture
def plan() -> ExercisePlan:
    return ExercisePlan(
        exercise_id="seated-knee-extension",
        allowed_cue_ids=frozenset({CueId.MOVE_SLOWLY, CueId.KNEE_ALIGNMENT}),
        min_confidence=0.7,
        max_camera_disagreement_degrees=12.0,
        max_pose_age_ms=500,
    )


@pytest.fixture
def guardian() -> Guardian:
    return Guardian()


def test_safe_observation_continues(guardian: Guardian, plan: ExercisePlan) -> None:
    decision = guardian.decide(observation(), plan)

    assert decision.action is GuardianAction.CONTINUE
    assert decision.cue_id is None
    assert decision.reason_codes == (GuardianReason.WITHIN_LIMITS,)
    assert decision.rule_version == Guardian.RULE_VERSION


def test_emergency_escalates_and_has_highest_priority(
    guardian: Guardian, plan: ExercisePlan
) -> None:
    decision = guardian.decide(
        observation(
            emergency_reported=True,
            pain_reported=True,
            confidence=0.1,
            pose_age_ms=999,
        ),
        plan,
        LearnedSuggestion(GuardianAction.CONTINUE),
    )

    assert decision.action is GuardianAction.ESCALATE
    assert GuardianReason.EMERGENCY_REPORTED in decision.reason_codes
    assert GuardianReason.PAIN_REPORTED in decision.reason_codes
    assert GuardianReason.LOW_CONFIDENCE in decision.reason_codes
    assert GuardianReason.STALE_OBSERVATION in decision.reason_codes
    assert GuardianReason.LEARNED_MODEL_SUGGESTION_IGNORED in decision.reason_codes


def test_pain_stops_and_model_cannot_reduce_action(guardian: Guardian, plan: ExercisePlan) -> None:
    decision = guardian.decide(
        observation(pain_reported=True),
        plan,
        LearnedSuggestion(GuardianAction.CUE, CueId.MOVE_SLOWLY),
    )

    assert decision.action is GuardianAction.STOP
    assert decision.cue_id is None
    assert decision.reason_codes == (
        GuardianReason.PAIN_REPORTED,
        GuardianReason.LEARNED_MODEL_SUGGESTION_IGNORED,
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    (
        ({"pose_age_ms": 501}, GuardianReason.STALE_OBSERVATION),
        ({"confidence": 0.69}, GuardianReason.LOW_CONFIDENCE),
        ({"camera_view_count": 1}, GuardianReason.MISSING_CAMERA_EVIDENCE),
        (
            {"camera_disagreement_degrees": None},
            GuardianReason.MISSING_CAMERA_EVIDENCE,
        ),
        (
            {"camera_disagreement_degrees": 12.001},
            GuardianReason.CAMERA_DISAGREEMENT,
        ),
        ({"wrong_exercise": True}, GuardianReason.WRONG_EXERCISE),
        (
            {"exercise_id": "different-exercise"},
            GuardianReason.WRONG_EXERCISE,
        ),
        (
            {"out_of_distribution": True},
            GuardianReason.OUT_OF_DISTRIBUTION,
        ),
    ),
)
def test_deterministic_uncertainty_pauses(
    guardian: Guardian,
    plan: ExercisePlan,
    changes: dict[str, object],
    reason: GuardianReason,
) -> None:
    decision = guardian.decide(observation(**changes), plan)

    assert decision.action is GuardianAction.PAUSE
    assert decision.cue_id is None
    assert reason in decision.reason_codes


def test_threshold_boundaries_are_allowed(guardian: Guardian, plan: ExercisePlan) -> None:
    decision = guardian.decide(
        observation(
            confidence=plan.min_confidence,
            camera_disagreement_degrees=plan.max_camera_disagreement_degrees,
            pose_age_ms=plan.max_pose_age_ms,
        ),
        plan,
    )

    assert decision.action is GuardianAction.CONTINUE


def test_single_camera_plan_does_not_claim_or_require_cross_view_agreement(
    guardian: Guardian,
) -> None:
    single_camera_plan = ExercisePlan(
        exercise_id="squat",
        allowed_cue_ids=frozenset({CueId.SQUAT_REP_ONE}),
        required_camera_views=1,
    )
    decision = guardian.decide(
        observation(
            exercise_id="squat",
            camera_view_count=1,
            camera_disagreement_degrees=None,
            phase="standing",
        ),
        single_camera_plan,
        local_cue_request=LocalCueRequest(CueId.SQUAT_REP_ONE),
    )

    assert decision.action is GuardianAction.CUE
    assert decision.cue_id == CueId.SQUAT_REP_ONE


def test_all_uncertainty_reasons_are_retained_for_audit(
    guardian: Guardian, plan: ExercisePlan
) -> None:
    decision = guardian.decide(
        observation(
            wrong_exercise=True,
            out_of_distribution=True,
            pose_age_ms=700,
            confidence=0.2,
            camera_disagreement_degrees=80.0,
        ),
        plan,
    )

    assert decision.action is GuardianAction.PAUSE
    assert decision.reason_codes == (
        GuardianReason.WRONG_EXERCISE,
        GuardianReason.OUT_OF_DISTRIBUTION,
        GuardianReason.STALE_OBSERVATION,
        GuardianReason.LOW_CONFIDENCE,
        GuardianReason.CAMERA_DISAGREEMENT,
    )


def test_approved_and_plan_allowed_cue_is_returned(guardian: Guardian, plan: ExercisePlan) -> None:
    decision = guardian.decide(
        observation(),
        plan,
        LearnedSuggestion(GuardianAction.CUE, CueId.KNEE_ALIGNMENT),
    )

    assert decision.action is GuardianAction.CUE
    assert decision.cue_id == CueId.KNEE_ALIGNMENT
    assert decision.reason_codes == (GuardianReason.LEARNED_MODEL_CUE_ACCEPTED,)
    assert guardian.cue_catalog[decision.cue_id].spoken_text


def test_local_pose_event_must_pass_guardian_before_becoming_a_cue(
    guardian: Guardian, plan: ExercisePlan
) -> None:
    decision = guardian.decide(
        observation(),
        plan,
        local_cue_request=LocalCueRequest(CueId.KNEE_ALIGNMENT),
    )

    assert decision.action is GuardianAction.CUE
    assert decision.cue_id == CueId.KNEE_ALIGNMENT
    assert decision.reason_codes == (GuardianReason.LOCAL_CUE_ACCEPTED,)


def test_local_pose_event_cannot_override_a_safety_pause(
    guardian: Guardian, plan: ExercisePlan
) -> None:
    decision = guardian.decide(
        observation(confidence=0.1),
        plan,
        local_cue_request=LocalCueRequest(CueId.KNEE_ALIGNMENT),
    )

    assert decision.action is GuardianAction.PAUSE
    assert decision.cue_id is None
    assert decision.reason_codes == (
        GuardianReason.LOW_CONFIDENCE,
        GuardianReason.LOCAL_CUE_IGNORED_FOR_SAFETY,
    )


@pytest.mark.parametrize(
    ("cue_id", "reason"),
    (
        ("invented-local-cue", GuardianReason.UNKNOWN_CUE),
        (CueId.HOLD_POSITION, GuardianReason.CUE_NOT_ALLOWED),
    ),
)
def test_invalid_local_pose_cue_fails_closed(
    guardian: Guardian,
    plan: ExercisePlan,
    cue_id: str,
    reason: GuardianReason,
) -> None:
    decision = guardian.decide(
        observation(),
        plan,
        local_cue_request=LocalCueRequest(cue_id),
    )

    assert decision.action is GuardianAction.PAUSE
    assert decision.cue_id is None
    assert decision.reason_codes == (reason,)


def test_local_and_learned_cue_inputs_cannot_be_mixed(
    guardian: Guardian, plan: ExercisePlan
) -> None:
    with pytest.raises(ValueError, match="either a local cue request"):
        guardian.decide(
            observation(),
            plan,
            LearnedSuggestion(GuardianAction.CONTINUE),
            local_cue_request=LocalCueRequest(CueId.KNEE_ALIGNMENT),
        )


def test_unknown_cue_pauses_instead_of_speaking(guardian: Guardian, plan: ExercisePlan) -> None:
    decision = guardian.decide(
        observation(),
        plan,
        LearnedSuggestion(GuardianAction.CUE, "invented-correction"),
    )

    assert decision.action is GuardianAction.PAUSE
    assert decision.cue_id is None
    assert decision.reason_codes == (
        GuardianReason.UNKNOWN_CUE,
        GuardianReason.LEARNED_MODEL_INCREASED_CAUTION,
    )


def test_catalog_cue_not_allowed_by_exercise_plan_pauses(
    guardian: Guardian, plan: ExercisePlan
) -> None:
    decision = guardian.decide(
        observation(),
        plan,
        LearnedSuggestion(GuardianAction.CUE, CueId.HOLD_POSITION),
    )

    assert decision.action is GuardianAction.PAUSE
    assert decision.cue_id is None
    assert decision.reason_codes == (
        GuardianReason.CUE_NOT_ALLOWED,
        GuardianReason.LEARNED_MODEL_INCREASED_CAUTION,
    )


@pytest.mark.parametrize(
    "suggested_action",
    (GuardianAction.PAUSE, GuardianAction.STOP, GuardianAction.ESCALATE),
)
def test_model_can_increase_caution(
    guardian: Guardian,
    plan: ExercisePlan,
    suggested_action: GuardianAction,
) -> None:
    decision = guardian.decide(observation(), plan, LearnedSuggestion(suggested_action))

    assert decision.action is suggested_action
    assert GuardianReason.LEARNED_MODEL_INCREASED_CAUTION in decision.reason_codes


def test_equal_model_action_preserves_deterministic_caution(
    guardian: Guardian, plan: ExercisePlan
) -> None:
    decision = guardian.decide(
        observation(confidence=0.2),
        plan,
        LearnedSuggestion(GuardianAction.PAUSE),
    )

    assert decision.action is GuardianAction.PAUSE
    assert decision.reason_codes == (
        GuardianReason.LOW_CONFIDENCE,
        GuardianReason.LEARNED_MODEL_PRESERVED_CAUTION,
    )


def test_custom_catalog_is_enforced() -> None:
    catalog = ApprovedCueCatalog(
        (ApprovedCue("custom", "Use the custom cue.", CueKind.CORRECTION),)
    )
    guardian = Guardian(catalog, rule_version="site-rules-4")
    plan = ExercisePlan("exercise", frozenset({"custom"}))

    decision = guardian.decide(
        observation(exercise_id="exercise"),
        plan,
        LearnedSuggestion(GuardianAction.CUE, "custom"),
    )

    assert decision.action is GuardianAction.CUE
    assert decision.cue_id == "custom"
    assert decision.rule_version == "site-rules-4"


def test_catalog_rejects_duplicate_identifiers() -> None:
    cue = ApprovedCue("duplicate", "First.", CueKind.STATUS)
    with pytest.raises(ValueError, match="duplicate cue_id"):
        ApprovedCueCatalog((cue, cue))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("confidence", math.nan),
        ("confidence", True),
        ("camera_disagreement_degrees", math.inf),
        ("camera_disagreement_degrees", 180.001),
        ("camera_disagreement_degrees", False),
        ("camera_view_count", 0),
        ("camera_view_count", True),
        ("pose_age_ms", -1),
        ("pose_age_ms", 1.5),
        ("timestamp_ms", -1),
        ("timestamp_ms", True),
        ("rep_index", -1),
    ),
)
def test_observation_rejects_invalid_measurements(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        observation(**{field: value})


def test_plan_normalizes_cue_ids_and_rejects_invalid_limits() -> None:
    plan = ExercisePlan(" exercise ", frozenset({" cue "}))
    assert plan.exercise_id == "exercise"
    assert plan.allowed_cue_ids == frozenset({"cue"})

    with pytest.raises(ValueError, match="target_reps"):
        ExercisePlan("exercise", frozenset(), target_reps=0)
    with pytest.raises(ValueError, match="min_confidence"):
        ExercisePlan("exercise", frozenset(), min_confidence=1.2)
    with pytest.raises(ValueError, match="max_camera_disagreement_degrees"):
        ExercisePlan(
            "exercise",
            frozenset(),
            max_camera_disagreement_degrees=180.001,
        )
    with pytest.raises(ValueError, match="required_camera_views"):
        ExercisePlan("exercise", frozenset(), required_camera_views=0)


def test_suggestion_and_decision_enforce_cue_invariants() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        LocalCueRequest(" ")
    with pytest.raises(ValueError, match="requires cue_id"):
        LearnedSuggestion(GuardianAction.CUE)
    with pytest.raises(ValueError, match="only valid"):
        LearnedSuggestion(GuardianAction.STOP, "cue")
    with pytest.raises(ValueError, match="requires cue_id"):
        GuardianDecision(
            GuardianAction.CUE,
            (GuardianReason.WITHIN_LIMITS,),
            "v1",
        )
    with pytest.raises(TypeError, match="GuardianAction"):
        LearnedSuggestion("cue", CueId.MOVE_SLOWLY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason_codes"):
        GuardianDecision(
            GuardianAction.CONTINUE,
            ("within_limits",),  # type: ignore[arg-type]
            "v1",
        )


def test_default_catalog_contains_unique_nonempty_fixed_phrases() -> None:
    assert set(DEFAULT_CUE_CATALOG) == {cue.value for cue in CueId}
    assert all(cue.spoken_text.strip() for cue in DEFAULT_CUE_CATALOG.values())
