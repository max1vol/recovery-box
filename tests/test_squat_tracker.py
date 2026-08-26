from __future__ import annotations

from dataclasses import replace
from math import cos, radians, sin

import pytest

from recoverybox.exercise import (
    MEDIAPIPE_POSE_LANDMARK_COUNT,
    MediaPipePoseFrame,
    MediaPipePoseLandmark,
    NormalizedLandmark,
    SquatAssessmentIssue,
    SquatEvent,
    SquatEventType,
    SquatPhase,
    SquatTracker,
    SquatTrackerConfig,
)


def landmark(
    x: float = 0.5,
    y: float = 0.5,
    *,
    visibility: float = 1.0,
    presence: float = 1.0,
) -> NormalizedLandmark:
    return NormalizedLandmark(
        x=x,
        y=y,
        z=0.0,
        visibility=visibility,
        presence=presence,
    )


def pose_frame(
    timestamp_ms: int,
    *,
    left_knee_angle: float = 175.0,
    right_knee_angle: float | None = None,
    arms_in_t: bool = True,
    visibility_overrides: dict[MediaPipePoseLandmark, float] | None = None,
    presence_overrides: dict[MediaPipePoseLandmark, float] | None = None,
    coordinate_overrides: dict[MediaPipePoseLandmark, tuple[float, float]] | None = None,
    image_width: int = 1_000,
    image_height: int = 1_000,
) -> MediaPipePoseFrame:
    values = [landmark() for _ in range(MEDIAPIPE_POSE_LANDMARK_COUNT)]
    right_knee_angle = left_knee_angle if right_knee_angle is None else right_knee_angle

    def placed(x: float, y: float) -> NormalizedLandmark:
        normalized_x = 0.5 + (x - 0.5) * image_height / image_width
        return landmark(normalized_x, y)

    def set_leg(
        hip_name: MediaPipePoseLandmark,
        knee_name: MediaPipePoseLandmark,
        ankle_name: MediaPipePoseLandmark,
        *,
        x: float,
        angle: float,
        direction: float,
    ) -> None:
        knee_y = 0.62
        leg_length = 0.25
        angle_radians = radians(angle)
        values[hip_name] = placed(x, knee_y - leg_length)
        values[knee_name] = placed(x, knee_y)
        values[ankle_name] = placed(
            x + direction * leg_length * sin(angle_radians),
            knee_y - leg_length * cos(angle_radians),
        )

    set_leg(
        MediaPipePoseLandmark.LEFT_HIP,
        MediaPipePoseLandmark.LEFT_KNEE,
        MediaPipePoseLandmark.LEFT_ANKLE,
        x=0.44,
        angle=left_knee_angle,
        direction=-1.0,
    )
    set_leg(
        MediaPipePoseLandmark.RIGHT_HIP,
        MediaPipePoseLandmark.RIGHT_KNEE,
        MediaPipePoseLandmark.RIGHT_ANKLE,
        x=0.56,
        angle=right_knee_angle,
        direction=1.0,
    )

    values[MediaPipePoseLandmark.LEFT_SHOULDER] = placed(0.35, 0.30)
    values[MediaPipePoseLandmark.RIGHT_SHOULDER] = placed(0.65, 0.30)
    if arms_in_t:
        values[MediaPipePoseLandmark.LEFT_ELBOW] = placed(0.20, 0.30)
        values[MediaPipePoseLandmark.LEFT_WRIST] = placed(0.05, 0.30)
        values[MediaPipePoseLandmark.RIGHT_ELBOW] = placed(0.80, 0.30)
        values[MediaPipePoseLandmark.RIGHT_WRIST] = placed(0.95, 0.30)
    else:
        values[MediaPipePoseLandmark.LEFT_ELBOW] = placed(0.20, 0.50)
        values[MediaPipePoseLandmark.LEFT_WRIST] = placed(0.05, 0.70)
        values[MediaPipePoseLandmark.RIGHT_ELBOW] = placed(0.80, 0.50)
        values[MediaPipePoseLandmark.RIGHT_WRIST] = placed(0.95, 0.70)

    for name, value in (visibility_overrides or {}).items():
        values[name] = replace(values[name], visibility=value)
    for name, value in (presence_overrides or {}).items():
        values[name] = replace(values[name], presence=value)
    for name, (x, y) in (coordinate_overrides or {}).items():
        positioned = placed(x, y)
        values[name] = replace(values[name], x=positioned.x, y=positioned.y)
    return MediaPipePoseFrame(
        timestamp_ms=timestamp_ms,
        image_width=image_width,
        image_height=image_height,
        landmarks=tuple(values),
    )


def demo_config(**overrides: object) -> SquatTrackerConfig:
    values: dict[str, object] = {
        "phase_confirmation_frames": 2,
        "arms_correction_confirmation_frames": 2,
        "arms_recovery_confirmation_frames": 2,
    }
    values.update(overrides)
    return SquatTrackerConfig(**values)  # type: ignore[arg-type]


def test_full_standing_down_standing_cycle_emits_one_rep() -> None:
    tracker = SquatTracker(demo_config())

    first_standing = tracker.update(pose_frame(0, left_knee_angle=175.0))
    standing = tracker.update(pose_frame(33, left_knee_angle=175.0))
    transition = tracker.update(pose_frame(66, left_knee_angle=130.0))
    first_down = tracker.update(pose_frame(99, left_knee_angle=90.0))
    down = tracker.update(pose_frame(132, left_knee_angle=90.0))
    first_return = tracker.update(pose_frame(165, left_knee_angle=175.0))
    completed = tracker.update(pose_frame(198, left_knee_angle=175.0))
    still_standing = tracker.update(pose_frame(231, left_knee_angle=175.0))

    assert first_standing.phase is SquatPhase.UNKNOWN
    assert standing.phase is SquatPhase.STANDING
    assert transition.phase is SquatPhase.STANDING
    assert first_down.phase is SquatPhase.STANDING
    assert down.phase is SquatPhase.DOWN
    assert first_return.phase is SquatPhase.DOWN
    assert completed.phase is SquatPhase.STANDING
    assert completed.rep_count == 1
    assert completed.events == (SquatEvent(SquatEventType.REP_COMPLETED, rep_count=1),)
    assert still_standing.events == ()
    assert tracker.rep_count == 1


def test_starting_down_does_not_infer_a_rep() -> None:
    tracker = SquatTracker(demo_config())
    timestamp = 0
    results = []
    for angle in (90.0, 90.0, 175.0, 175.0):
        results.append(tracker.update(pose_frame(timestamp, left_knee_angle=angle)))
        timestamp += 33

    assert all(result.rep_count == 0 for result in results)
    assert results[-1].phase is SquatPhase.STANDING

    for angle in (90.0, 90.0, 175.0, 175.0):
        result = tracker.update(pose_frame(timestamp, left_knee_angle=angle))
        timestamp += 33

    assert result.rep_count == 1
    assert result.events == (SquatEvent(SquatEventType.REP_COMPLETED, 1),)


def test_hysteresis_and_confirmation_reject_threshold_noise() -> None:
    tracker = SquatTracker(demo_config())
    timestamp = 0
    for angle in (175.0, 175.0, 120.0, 95.0, 120.0, 95.0, 120.0, 175.0):
        result = tracker.update(pose_frame(timestamp, left_knee_angle=angle))
        timestamp += 33

    assert result.phase is SquatPhase.STANDING
    assert result.rep_count == 0
    assert not any(event.event_type is SquatEventType.REP_COMPLETED for event in result.events)


def test_low_confidence_frame_invalidates_an_incomplete_cycle() -> None:
    tracker = SquatTracker(demo_config())
    tracker.update(pose_frame(0, left_knee_angle=175.0))
    tracker.update(pose_frame(33, left_knee_angle=175.0))
    tracker.update(pose_frame(66, left_knee_angle=90.0))
    tracker.update(pose_frame(99, left_knee_angle=90.0))
    assert tracker.phase is SquatPhase.DOWN

    withheld = tracker.update(
        pose_frame(
            132,
            left_knee_angle=130.0,
            visibility_overrides={MediaPipePoseLandmark.LEFT_KNEE: 0.2},
            presence_overrides={MediaPipePoseLandmark.LEFT_KNEE: 0.3},
        )
    )
    assert not withheld.assessable
    assert withheld.phase is SquatPhase.UNKNOWN
    assert withheld.issues == (
        SquatAssessmentIssue.LOW_VISIBILITY,
        SquatAssessmentIssue.LOW_PRESENCE,
    )
    assert withheld.events == ()

    tracker.update(pose_frame(165, left_knee_angle=175.0))
    reacquired = tracker.update(pose_frame(198, left_knee_angle=175.0))
    assert reacquired.phase is SquatPhase.STANDING
    assert reacquired.rep_count == 0


@pytest.mark.parametrize(
    "issue",
    (SquatAssessmentIssue.NO_POSE, SquatAssessmentIssue.CAMERA_TIMEOUT),
)
def test_missing_pose_immediately_invalidates_an_incomplete_cycle(
    issue: SquatAssessmentIssue,
) -> None:
    tracker = SquatTracker(demo_config(phase_confirmation_frames=1))
    tracker.update(pose_frame(0, left_knee_angle=175.0))
    tracker.update(pose_frame(33, left_knee_angle=90.0))
    assert tracker.phase is SquatPhase.DOWN

    missing = tracker.update_missing(66, issue=issue)
    assert not missing.assessable
    assert missing.issues == (issue,)
    assert missing.phase is SquatPhase.UNKNOWN

    reacquired = tracker.update(pose_frame(99, left_knee_angle=175.0))
    assert reacquired.phase is SquatPhase.STANDING
    assert reacquired.rep_count == 0


def test_missing_pose_api_rejects_non_missing_issue() -> None:
    tracker = SquatTracker()
    with pytest.raises(ValueError, match="NO_POSE or CAMERA_TIMEOUT"):
        tracker.update_missing(0, issue=SquatAssessmentIssue.LOW_VISIBILITY)


def test_stale_and_non_monotonic_timestamps_fail_closed() -> None:
    tracker = SquatTracker(demo_config(maximum_frame_gap_ms=100))
    tracker.update(pose_frame(100, left_knee_angle=175.0))

    repeated = tracker.update(pose_frame(100, left_knee_angle=175.0))
    assert repeated.issues == (SquatAssessmentIssue.NON_MONOTONIC_TIMESTAMP,)
    assert repeated.phase is SquatPhase.UNKNOWN

    stale = tracker.update(pose_frame(250, left_knee_angle=175.0))
    assert stale.issues == (SquatAssessmentIssue.STALE_FRAME_GAP,)
    assert stale.phase is SquatPhase.UNKNOWN

    accepted = tracker.update(pose_frame(283, left_knee_angle=175.0))
    assert accepted.assessable
    assert accepted.phase is SquatPhase.UNKNOWN


def test_bilateral_knee_disagreement_is_withheld() -> None:
    tracker = SquatTracker(demo_config())
    result = tracker.update(pose_frame(0, left_knee_angle=90.0, right_knee_angle=175.0))

    assert not result.assessable
    assert result.issues == (SquatAssessmentIssue.BILATERAL_KNEE_DISAGREEMENT,)
    assert result.knee_angle_degrees == pytest.approx(132.5)
    assert result.events == ()


def test_both_legs_must_cross_each_phase_threshold() -> None:
    tracker = SquatTracker(demo_config(phase_confirmation_frames=1))

    one_leg_not_standing = tracker.update(
        pose_frame(0, left_knee_angle=150.0, right_knee_angle=170.0)
    )
    assert one_leg_not_standing.assessable
    assert one_leg_not_standing.phase is SquatPhase.UNKNOWN

    standing = tracker.update(pose_frame(33, left_knee_angle=165.0, right_knee_angle=170.0))
    assert standing.phase is SquatPhase.STANDING

    one_leg_not_down = tracker.update(pose_frame(66, left_knee_angle=90.0, right_knee_angle=110.0))
    assert one_leg_not_down.assessable
    assert one_leg_not_down.phase is SquatPhase.STANDING
    assert one_leg_not_down.rep_count == 0


def test_joint_geometry_is_equivalent_across_camera_aspect_ratios() -> None:
    square = SquatTracker(demo_config(phase_confirmation_frames=1)).update(
        pose_frame(
            0,
            left_knee_angle=110.0,
            image_width=1_000,
            image_height=1_000,
        )
    )
    widescreen = SquatTracker(demo_config(phase_confirmation_frames=1)).update(
        pose_frame(
            0,
            left_knee_angle=110.0,
            image_width=1_920,
            image_height=1_080,
        )
    )

    assert square.knee_angle_degrees == pytest.approx(110.0)
    assert widescreen.knee_angle_degrees == pytest.approx(110.0)
    assert square.arms_in_t is True
    assert widescreen.arms_in_t is True


def test_degenerate_leg_geometry_is_withheld() -> None:
    tracker = SquatTracker(demo_config())
    result = tracker.update(
        pose_frame(
            0,
            coordinate_overrides={MediaPipePoseLandmark.LEFT_HIP: (0.44, 0.62)},
        )
    )

    assert not result.assessable
    assert result.issues == (SquatAssessmentIssue.INVALID_LEG_GEOMETRY,)


def test_extreme_extrapolated_landmark_is_withheld() -> None:
    tracker = SquatTracker(demo_config())
    result = tracker.update(
        pose_frame(
            0,
            coordinate_overrides={MediaPipePoseLandmark.LEFT_ANKLE: (50.0, 50.0)},
        )
    )

    assert not result.assessable
    assert result.issues == (SquatAssessmentIssue.OUT_OF_FRAME_LANDMARK,)


def test_arms_not_in_t_event_is_debounced_and_rearmed_after_recovery() -> None:
    tracker = SquatTracker(demo_config())

    first_bad = tracker.update(pose_frame(0, arms_in_t=False))
    correction = tracker.update(pose_frame(33, arms_in_t=False))
    repeated_bad = tracker.update(pose_frame(66, arms_in_t=False))
    first_good = tracker.update(pose_frame(99, arms_in_t=True))
    recovered = tracker.update(pose_frame(132, arms_in_t=True))
    bad_again = tracker.update(pose_frame(165, arms_in_t=False))
    correction_again = tracker.update(pose_frame(198, arms_in_t=False))

    assert first_bad.arms_in_t is False
    assert first_bad.events == ()
    assert correction.events == (SquatEvent(SquatEventType.ARMS_NOT_IN_T),)
    assert repeated_bad.events == ()
    assert first_good.events == ()
    assert recovered.events == ()
    assert bad_again.events == ()
    assert correction_again.events == (SquatEvent(SquatEventType.ARMS_NOT_IN_T),)


def test_low_arm_visibility_withholds_the_whole_t_squat_assessment() -> None:
    tracker = SquatTracker(demo_config())
    result = tracker.update(
        pose_frame(
            0,
            visibility_overrides={MediaPipePoseLandmark.LEFT_WRIST: 0.1},
        )
    )

    assert not result.assessable
    assert result.issues == (SquatAssessmentIssue.LOW_VISIBILITY,)
    assert result.arms_in_t is None


def test_arm_monitor_can_be_disabled_for_an_ordinary_squat() -> None:
    tracker = SquatTracker(
        demo_config(
            monitor_arms_in_t=False,
            phase_confirmation_frames=1,
        )
    )
    result = tracker.update(
        pose_frame(
            0,
            visibility_overrides={
                MediaPipePoseLandmark.LEFT_WRIST: 0.0,
                MediaPipePoseLandmark.RIGHT_WRIST: 0.0,
            },
        )
    )

    assert result.assessable
    assert result.phase is SquatPhase.STANDING
    assert result.arms_in_t is None


def test_reset_starts_a_new_exercise_session() -> None:
    tracker = SquatTracker(demo_config(phase_confirmation_frames=1))
    timestamp = 0
    for angle in (175.0, 90.0, 175.0):
        result = tracker.update(pose_frame(timestamp, left_knee_angle=angle))
        timestamp += 33
    assert result.rep_count == 1

    tracker.reset()

    assert tracker.rep_count == 0
    assert tracker.phase is SquatPhase.UNKNOWN
    restarted = tracker.update(pose_frame(0, left_knee_angle=175.0))
    assert restarted.phase is SquatPhase.STANDING


def test_landmark_accepts_finite_extrapolated_coordinates_but_not_bad_confidence() -> None:
    extrapolated = NormalizedLandmark(-0.2, 1.1, 0.05, 0.8, 0.9)
    assert extrapolated.x == -0.2
    assert extrapolated.y == 1.1

    with pytest.raises(ValueError, match="x must be finite"):
        NormalizedLandmark(float("nan"), 0.5, 0.0, 0.8, 0.9)
    with pytest.raises(ValueError, match="visibility must be between"):
        NormalizedLandmark(0.5, 0.5, 0.0, 1.1, 0.9)


def test_frame_enforces_exact_media_pipe_schema_and_typed_tuple() -> None:
    good = landmark()
    with pytest.raises(ValueError, match="exactly 33"):
        MediaPipePoseFrame(0, 640, 480, tuple(good for _ in range(32)))
    with pytest.raises(TypeError, match="immutable tuple"):
        MediaPipePoseFrame(0, 640, 480, [good] * 33)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="NormalizedLandmark"):
        MediaPipePoseFrame(  # type: ignore[arg-type]
            0,
            640,
            480,
            tuple(object() for _ in range(33)),
        )
    with pytest.raises(ValueError, match="image_width"):
        MediaPipePoseFrame(0, 0, 480, tuple(good for _ in range(33)))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"minimum_visibility": 1.1}, "minimum_visibility"),
        ({"maximum_frame_gap_ms": 0}, "maximum_frame_gap_ms"),
        ({"maximum_coordinate_extrapolation": 1.1}, "coordinate_extrapolation"),
        (
            {
                "down_knee_angle_degrees": 160.0,
                "standing_knee_angle_degrees": 150.0,
            },
            "knee thresholds",
        ),
        ({"phase_confirmation_frames": 0}, "phase_confirmation_frames"),
        ({"arm_horizontal_tolerance_degrees": 90.0}, "arm_horizontal"),
        ({"monitor_arms_in_t": 1}, "monitor_arms_in_t"),
    ),
)
def test_config_rejects_unsafe_thresholds(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        demo_config(**changes)


def test_tracker_rejects_an_untyped_config() -> None:
    with pytest.raises(TypeError, match="SquatTrackerConfig"):
        SquatTracker({})  # type: ignore[arg-type]


def test_semantic_event_payload_invariants() -> None:
    with pytest.raises(ValueError, match="positive rep_count"):
        SquatEvent(SquatEventType.REP_COMPLETED)
    with pytest.raises(ValueError, match="only rep_completed"):
        SquatEvent(SquatEventType.ARMS_NOT_IN_T, rep_count=1)
    with pytest.raises(TypeError, match="SquatEventType"):
        SquatEvent("rep_completed", rep_count=1)  # type: ignore[arg-type]
