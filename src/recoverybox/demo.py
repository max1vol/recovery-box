"""Deterministic, hardware-free demonstration of the local safety pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from recoverybox.core import (
    CueId,
    ExercisePlan,
    Guardian,
    GuardianAction,
    LearnedSuggestion,
    MovementObservation,
    SessionMode,
)
from recoverybox.federation import PostSessionRepSummary, pose_window_to_post_session_record
from recoverybox.pose import (
    CameraView,
    DualViewPoseFuser,
    DualViewPoseSynchronizer,
    FusedPoseSummary,
    PoseAngles,
    PoseFeatureWindowBuilder,
    PoseViewSummary,
)
from recoverybox.session import (
    ApprovedCuePlaybackAuthorization,
    SessionCoordinator,
    session_mode_allows_model_audio,
)


@dataclass(frozen=True, slots=True)
class DemoEvent:
    step: str
    outcome: str
    evidence: dict[str, Any]


@dataclass(slots=True)
class _DemoCueSpeaker:
    """In-memory approved-cue port used to prove the composition boundary."""

    model_audio_preemptions: int = 0
    played: list[ApprovedCuePlaybackAuthorization] = field(default_factory=list)

    def preempt_model_audio(self) -> None:
        self.model_audio_preemptions += 1

    def play_approved_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None:
        self.played.append(authorization)


def _view(
    view: CameraView,
    timestamp_ms: int,
    angles: tuple[float, float, float],
    confidence: float,
) -> PoseViewSummary:
    return PoseViewSummary(
        view=view,
        monotonic_timestamp_ms=timestamp_ms,
        angles_degrees=PoseAngles.seated_knee_extension(angles),
        confidence=confidence,
    )


def run_safety_demo() -> list[DemoEvent]:
    """Run assessable, camera-loss, and pain paths with no network or hardware."""

    fuser = DualViewPoseFuser()
    guardian = Guardian()
    plan = ExercisePlan(
        exercise_id="seated-knee-extension",
        allowed_cue_ids=frozenset({CueId.MOVE_SLOWLY, CueId.KNEE_ALIGNMENT}),
        target_reps=8,
    )
    feature_builder = PoseFeatureWindowBuilder(maximum_rows=8)
    cue_speaker = _DemoCueSpeaker()
    coordinator = SessionCoordinator(
        cue_playback=cue_speaker,
        initial_mode=SessionMode.ACTIVE_EXERCISE,
    )
    events: list[DemoEvent] = []

    first = fuser.fuse(
        _view(CameraView.PRIMARY, 1_000, (92.0, 121.0, 7.0), 0.94),
        _view(CameraView.SECONDARY, 1_025, (95.0, 118.0, 6.0), 0.91),
        now_monotonic_ms=1_100,
    )
    assert first.fused is not None
    feature_builder.add(first.fused)
    correction = guardian.decide(
        _observation(first.fused, timestamp_ms=1_100),
        plan,
        LearnedSuggestion(GuardianAction.CUE, CueId.MOVE_SLOWLY),
    )
    correction_effect = coordinator.apply_guardian_decision(correction)
    assert correction_effect.cue_authorization is not None
    events.append(
        DemoEvent(
            step="assessable_movement",
            outcome=correction.action.value,
            evidence={
                "confidence": first.fused.confidence,
                "camera_disagreement_degrees": (first.fused.camera_disagreement_degrees),
                "approved_cue_id": correction_effect.cue_authorization.cue_id.value,
                "cue_catalog_version": correction_effect.cue_authorization.catalog_version,
                "local_cue_playback_count": len(cue_speaker.played),
                "session_mode": correction_effect.current_mode.value,
                "guardian_rule_version": correction.rule_version,
            },
        )
    )

    second = fuser.fuse(
        _view(CameraView.PRIMARY, 1_500, (105.0, 114.0, 6.0), 0.93),
        _view(CameraView.SECONDARY, 1_520, (106.0, 112.0, 5.0), 0.90),
        now_monotonic_ms=1_580,
    )
    assert second.fused is not None
    feature_builder.add(second.fused)

    synchronizer = DualViewPoseSynchronizer(fuser)
    synchronizer.arm(now_monotonic_ms=1_900)
    pending = synchronizer.push(
        _view(CameraView.PRIMARY, 1_900, (110.0, 110.0, 5.0), 0.91),
        now_monotonic_ms=1_900,
    )
    assert pending is None
    obscured = synchronizer.expire(now_monotonic_ms=2_020)
    assert obscured is not None
    assert obscured.fused is None
    pause = guardian.decide(
        MovementObservation(
            exercise_id=plan.exercise_id,
            timestamp_ms=2_020,
            confidence=0.0,
            camera_disagreement_degrees=0.0,
            pose_age_ms=120,
            phase="unknown",
        ),
        plan,
    )
    pause_effect = coordinator.apply_guardian_decision(pause)
    events.append(
        DemoEvent(
            step="camera_uncertainty",
            outcome=pause.action.value,
            evidence={
                "fusion_issues": [issue.value for issue in obscured.issues],
                "guardian_reasons": [reason.value for reason in pause.reason_codes],
                "session_mode": pause_effect.current_mode.value,
                "arbitrary_model_audio_allowed": session_mode_allows_model_audio(
                    pause_effect.current_mode
                ),
                "model_audio_preemptions": cue_speaker.model_audio_preemptions,
            },
        )
    )

    stop = guardian.decide(
        MovementObservation(
            exercise_id=plan.exercise_id,
            timestamp_ms=2_100,
            confidence=0.92,
            camera_disagreement_degrees=2.0,
            pose_age_ms=40,
            phase="extension",
            pain_reported=True,
        ),
        plan,
        LearnedSuggestion(GuardianAction.CONTINUE),
    )
    stop_effect = coordinator.apply_guardian_decision(stop)
    events.append(
        DemoEvent(
            step="pain_report",
            outcome=stop.action.value,
            evidence={
                "guardian_reasons": [reason.value for reason in stop.reason_codes],
                "learned_suggestion": GuardianAction.CONTINUE.value,
                "final_guardian_action": stop.action.value,
                "model_was_allowed_to_lower_caution": stop.action is not GuardianAction.STOP,
                "session_mode": stop_effect.current_mode.value,
            },
        )
    )

    snapshot = feature_builder.snapshot()
    flower_record = pose_window_to_post_session_record(
        snapshot,
        PostSessionRepSummary(
            range_progress=0.78,
            rep_duration_s=3.1,
            stability_score=0.82,
            symmetry_score=0.86,
            label=1,
        ),
        joint_index=0,
    )
    raw_media_scan_passed = _passes_raw_media_scan(flower_record)
    events.append(
        DemoEvent(
            step="sanitized_feature_snapshot",
            outcome="ready_for_local_flower_client",
            evidence={
                "source_pose_schema_version": snapshot.schema_version,
                "schema_version": flower_record["schema_version"],
                "exercise_id": flower_record["exercise_id"],
                "label_definition_version": flower_record["label_definition_version"],
                "model_schema_signature": flower_record["model_schema_signature"],
                "source_pose_row_count": len(snapshot.rows),
                "numeric_width": len(flower_record["features"]),
                "contains_raw_media": not raw_media_scan_passed,
                "closed_raw_media_scan_passed": raw_media_scan_passed,
            },
        )
    )
    return events


def _observation(
    fused: FusedPoseSummary,
    *,
    timestamp_ms: int,
) -> MovementObservation:
    return MovementObservation(
        exercise_id="seated-knee-extension",
        timestamp_ms=timestamp_ms,
        confidence=fused.confidence,
        camera_disagreement_degrees=fused.camera_disagreement_degrees,
        pose_age_ms=max(0, timestamp_ms - fused.monotonic_timestamp_ms),
        rep_index=2,
        phase="extension",
    )


def _passes_raw_media_scan(value: object) -> bool:
    """Run the same recursive raw-media guard used by the Flower data loader."""

    from recoverybox.federation.errors import SanitizedDataError
    from recoverybox.federation.schema import reject_raw_media_fields

    try:
        reject_raw_media_fields(value)
    except SanitizedDataError:
        return False
    return True


def demo_as_dicts() -> list[dict[str, Any]]:
    return [asdict(event) for event in run_safety_demo()]
