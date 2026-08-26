from __future__ import annotations

from dataclasses import asdict, fields

import pytest

from recoverybox.pose import (
    POSE_ANGLE_SCHEMA_ID,
    POSE_FEATURE_SCHEMA_VERSION,
    SEATED_KNEE_EXTENSION_ANGLE_NAMES,
    CameraView,
    DualViewPoseFuser,
    DualViewPoseSynchronizer,
    FusedPoseSummary,
    FusionIssue,
    FusionResult,
    PoseAngleName,
    PoseAngles,
    PoseAngleSchema,
    PoseFeatureWindowBuilder,
    PoseFusionConfig,
    PoseViewSummary,
    SanitizedPoseFeature,
)


def pose_angles(
    knee: float = 90.0,
    hip: float = 120.0,
    trunk: float = 5.0,
) -> PoseAngles:
    return PoseAngles.seated_knee_extension((knee, hip, trunk))


def view(
    camera: CameraView,
    *,
    timestamp: int = 1_000,
    angles: PoseAngles | None = None,
    confidence: float = 0.9,
) -> PoseViewSummary:
    return PoseViewSummary(
        view=camera,
        monotonic_timestamp_ms=timestamp,
        angles_degrees=angles or pose_angles(),
        confidence=confidence,
    )


def test_camera_boundary_has_only_derived_numeric_measurements() -> None:
    summary = view(CameraView.PRIMARY)
    field_names = {field.name for field in fields(summary)}

    assert field_names == {
        "view",
        "monotonic_timestamp_ms",
        "angles_degrees",
        "confidence",
    }
    assert isinstance(summary.angles_degrees, PoseAngles)
    assert all(isinstance(value, float) for value in summary.angles_degrees)
    assert summary.angles_degrees.schema_id == POSE_ANGLE_SCHEMA_ID
    assert summary.angles_degrees.ordered_names == SEATED_KNEE_EXTENSION_ANGLE_NAMES
    assert summary.angles_degrees.named_values() == (
        (PoseAngleName.PRESCRIBED_KNEE_FLEXION, 90.0),
        (PoseAngleName.PRESCRIBED_HIP_FLEXION, 120.0),
        (PoseAngleName.TRUNK_FLEXION, 5.0),
    )
    assert not any(
        forbidden in field_name.lower()
        for field_name in field_names
        for forbidden in ("frame", "image", "pixel", "bytes", "audio", "transcript")
    )
    with pytest.raises(TypeError):
        PoseViewSummary(  # type: ignore[call-arg]
            view=CameraView.PRIMARY,
            monotonic_timestamp_ms=1_000,
            angles_degrees=pose_angles(),
            confidence=0.9,
            frame=b"not allowed",
        )


@pytest.mark.parametrize(
    "raw_angles",
    [
        (90.0, 120.0, 5.0),
        [90.0, 120.0, 5.0],
        b"raw-frame-like-bytes",
        "90,120,5",
        iter((90.0, 120.0, 5.0)),
    ],
)
def test_pose_boundary_rejects_untyped_angle_iterables(raw_angles: object) -> None:
    with pytest.raises(TypeError, match="validated PoseAngles"):
        PoseViewSummary(
            view=CameraView.PRIMARY,
            monotonic_timestamp_ms=1_000,
            angles_degrees=raw_angles,  # type: ignore[arg-type]
            confidence=0.9,
        )


@pytest.mark.parametrize(
    "raw_values",
    [
        [90.0, 120.0, 5.0],
        b"123",
        "123",
        iter((90.0, 120.0, 5.0)),
    ],
)
def test_pose_angles_reject_non_tuple_storage(raw_values: object) -> None:
    with pytest.raises(TypeError, match="immutable tuple"):
        PoseAngles(
            PoseAngleSchema.SEATED_KNEE_EXTENSION_V1,
            raw_values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "values",
    [(), (90.0,), (90.0, 120.0), (90.0, 120.0, 5.0, 1.0)],
)
def test_pose_angle_schema_enforces_exact_width(values: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="exactly 3 ordered angles"):
        PoseAngles.seated_knee_extension(values)


@pytest.mark.parametrize(
    "values",
    [
        (-0.001, 120.0, 5.0),
        (180.001, 120.0, 5.0),
        (float("nan"), 120.0, 5.0),
        (float("inf"), 120.0, 5.0),
    ],
)
def test_pose_angle_schema_enforces_physical_degree_bounds(
    values: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="between 0 and 180"):
        PoseAngles.seated_knee_extension(values)

    assert tuple(PoseAngles.seated_knee_extension((0.0, 180.0, 0.0))) == (
        0.0,
        180.0,
        0.0,
    )


@pytest.mark.parametrize("invalid", [(True, 120.0, 5.0), ("90", 120.0, 5.0)])
def test_pose_angle_schema_rejects_non_real_values(invalid: tuple[object, ...]) -> None:
    with pytest.raises(TypeError, match="real numbers"):
        PoseAngles.seated_knee_extension(invalid)  # type: ignore[arg-type]


def test_valid_views_fuse_with_conservative_confidence() -> None:
    fuser = DualViewPoseFuser()
    result = fuser.fuse(
        view(
            CameraView.PRIMARY,
            timestamp=1_000,
            angles=pose_angles(90.0, 120.0, 5.0),
            confidence=0.8,
        ),
        view(
            CameraView.SECONDARY,
            timestamp=1_025,
            angles=pose_angles(94.0, 116.0, 7.0),
            confidence=1.0,
        ),
        now_monotonic_ms=1_100,
    )

    assert result.assessable
    assert result.issues == ()
    assert result.fused is not None
    assert result.fused.confidence == 0.8
    assert result.fused.camera_disagreement_degrees == 4.0
    assert result.fused.timestamp_skew_ms == 25
    assert tuple(result.fused.angles_degrees) == pytest.approx((92.222222, 117.777778, 6.111111))
    assert result.fused.angles_degrees.schema_id == POSE_ANGLE_SCHEMA_ID


def test_synchronizer_consumes_each_pair_once_and_replaces_an_old_same_view() -> None:
    synchronizer = DualViewPoseSynchronizer()

    assert (
        synchronizer.push(
            view(CameraView.PRIMARY, timestamp=990, angles=pose_angles(80.0)),
            now_monotonic_ms=1_000,
        )
        is None
    )
    assert (
        synchronizer.push(
            view(CameraView.PRIMARY, timestamp=1_000, angles=pose_angles(90.0)),
            now_monotonic_ms=1_000,
        )
        is None
    )
    result = synchronizer.push(
        view(CameraView.SECONDARY, timestamp=1_010, angles=pose_angles(92.0)),
        now_monotonic_ms=1_020,
    )

    assert result is not None and result.fused is not None
    assert tuple(result.fused.angles_degrees) == pytest.approx((91.0, 120.0, 5.0))
    assert synchronizer.pending_view_count == 0
    assert (
        synchronizer.push(
            view(CameraView.SECONDARY, timestamp=1_030, angles=pose_angles(93.0)),
            now_monotonic_ms=1_030,
        )
        is None
    )


@pytest.mark.parametrize(
    ("arriving_view", "expected_issue"),
    [
        (CameraView.PRIMARY, FusionIssue.MISSING_SECONDARY),
        (CameraView.SECONDARY, FusionIssue.MISSING_PRIMARY),
    ],
)
def test_synchronizer_watchdog_closes_a_missing_camera(
    arriving_view: CameraView,
    expected_issue: FusionIssue,
) -> None:
    synchronizer = DualViewPoseSynchronizer(
        DualViewPoseFuser(PoseFusionConfig(maximum_pair_wait_ms=50))
    )
    assert synchronizer.push(view(arriving_view), now_monotonic_ms=1_000) is None
    assert synchronizer.expire(now_monotonic_ms=1_049) is None

    result = synchronizer.expire(now_monotonic_ms=1_050)

    assert result == FusionResult(fused=None, issues=(expected_issue,))
    assert not result.assessable
    assert synchronizer.pending_view_count == 0
    assert synchronizer.expire(now_monotonic_ms=1_099) is None
    assert synchronizer.expire(now_monotonic_ms=1_100) == FusionResult(
        fused=None,
        issues=(FusionIssue.MISSING_PRIMARY, FusionIssue.MISSING_SECONDARY),
    )


def test_same_view_replacements_cannot_hide_a_dead_peer_from_watchdog() -> None:
    synchronizer = DualViewPoseSynchronizer(
        DualViewPoseFuser(PoseFusionConfig(maximum_pair_wait_ms=50))
    )
    synchronizer.push(
        view(CameraView.PRIMARY, timestamp=1_000),
        now_monotonic_ms=1_000,
    )
    synchronizer.push(
        view(CameraView.PRIMARY, timestamp=1_040),
        now_monotonic_ms=1_040,
    )

    result = synchronizer.expire(now_monotonic_ms=1_050)

    assert result is not None
    assert result.issues == (FusionIssue.MISSING_SECONDARY,)
    assert result.fused is None


def test_explicit_arm_detects_complete_camera_loss_at_startup() -> None:
    synchronizer = DualViewPoseSynchronizer(
        DualViewPoseFuser(PoseFusionConfig(maximum_pair_wait_ms=50))
    )
    synchronizer.arm(now_monotonic_ms=1_000)

    assert synchronizer.expire(now_monotonic_ms=1_049) is None
    result = synchronizer.expire(now_monotonic_ms=1_050)

    assert result == FusionResult(
        fused=None,
        issues=(FusionIssue.MISSING_PRIMARY, FusionIssue.MISSING_SECONDARY),
    )
    assert not result.assessable


def test_synchronizer_watchdog_stays_armed_after_pair_and_supports_recovery() -> None:
    synchronizer = DualViewPoseSynchronizer(
        DualViewPoseFuser(PoseFusionConfig(maximum_pair_wait_ms=50))
    )
    synchronizer.arm(now_monotonic_ms=1_000)
    synchronizer.push(view(CameraView.PRIMARY), now_monotonic_ms=1_010)
    paired = synchronizer.push(
        view(CameraView.SECONDARY, timestamp=1_011),
        now_monotonic_ms=1_011,
    )
    assert paired is not None and paired.assessable
    assert synchronizer.expire(now_monotonic_ms=1_060) is None
    assert synchronizer.expire(now_monotonic_ms=1_061) == FusionResult(
        fused=None,
        issues=(FusionIssue.MISSING_PRIMARY, FusionIssue.MISSING_SECONDARY),
    )

    # Both cameras can recover in the newly opened observation window.
    synchronizer.push(
        view(CameraView.PRIMARY, timestamp=1_070),
        now_monotonic_ms=1_070,
    )
    recovered = synchronizer.push(
        view(CameraView.SECONDARY, timestamp=1_071),
        now_monotonic_ms=1_071,
    )
    assert recovered is not None and recovered.assessable


def test_clear_disarms_watchdog_and_allows_explicit_rearm() -> None:
    synchronizer = DualViewPoseSynchronizer(
        DualViewPoseFuser(PoseFusionConfig(maximum_pair_wait_ms=50))
    )
    synchronizer.arm(now_monotonic_ms=1_000)

    with pytest.raises(RuntimeError, match="already armed"):
        synchronizer.arm(now_monotonic_ms=1_010)

    synchronizer.clear()
    assert synchronizer.expire(now_monotonic_ms=2_000) is None
    synchronizer.arm(now_monotonic_ms=2_000)
    assert synchronizer.expire(now_monotonic_ms=2_050) is not None


def test_synchronizer_rejects_a_backwards_watchdog_clock() -> None:
    synchronizer = DualViewPoseSynchronizer()
    synchronizer.push(view(CameraView.PRIMARY), now_monotonic_ms=1_000)

    with pytest.raises(ValueError, match="must not move backwards"):
        synchronizer.expire(now_monotonic_ms=999)


def test_every_synchronization_and_quality_failure_withholds_fused_pose() -> None:
    fuser = DualViewPoseFuser(
        PoseFusionConfig(
            maximum_age_ms=100,
            maximum_timestamp_skew_ms=20,
            minimum_view_confidence=0.8,
            maximum_angle_disagreement_degrees=5.0,
        )
    )
    result = fuser.fuse(
        view(
            CameraView.PRIMARY,
            timestamp=800,
            angles=pose_angles(80.0, 120.0),
            confidence=0.7,
        ),
        view(
            CameraView.SECONDARY,
            timestamp=950,
            angles=pose_angles(100.0, 120.0),
            confidence=0.6,
        ),
        now_monotonic_ms=1_000,
    )

    assert not result.assessable
    assert result.fused is None
    assert set(result.issues) == {
        FusionIssue.STALE_PRIMARY,
        FusionIssue.EXCESSIVE_TIMESTAMP_SKEW,
        FusionIssue.LOW_PRIMARY_CONFIDENCE,
        FusionIssue.LOW_SECONDARY_CONFIDENCE,
        FusionIssue.CAMERA_DISAGREEMENT,
    }


@pytest.mark.parametrize(
    ("primary", "secondary", "now", "expected_issue"),
    [
        (
            view(CameraView.PRIMARY, timestamp=1_001),
            view(CameraView.SECONDARY, timestamp=1_000),
            999,
            FusionIssue.FUTURE_TIMESTAMP,
        ),
        (
            view(CameraView.PRIMARY),
            view(CameraView.PRIMARY),
            1_000,
            FusionIssue.SAME_VIEW,
        ),
    ],
)
def test_invalid_pairs_never_return_a_best_effort_estimate(
    primary: PoseViewSummary,
    secondary: PoseViewSummary,
    now: int,
    expected_issue: FusionIssue,
) -> None:
    result = DualViewPoseFuser().fuse(
        primary,
        secondary,
        now_monotonic_ms=now,
    )

    assert result.fused is None
    assert expected_issue in result.issues


def test_feature_window_is_bounded_and_contains_only_sanitized_derivatives() -> None:
    builder = PoseFeatureWindowBuilder(maximum_rows=2)
    builder.add(
        FusedPoseSummary(
            monotonic_timestamp_ms=1_000,
            angles_degrees=pose_angles(90.0, 120.0, 5.0),
            confidence=0.9,
            camera_disagreement_degrees=2.0,
            timestamp_skew_ms=10,
        )
    )
    second = builder.add(
        FusedPoseSummary(
            monotonic_timestamp_ms=1_500,
            angles_degrees=pose_angles(100.0, 115.0, 7.0),
            confidence=0.85,
            camera_disagreement_degrees=3.0,
            timestamp_skew_ms=12,
        )
    )
    builder.add(
        FusedPoseSummary(
            monotonic_timestamp_ms=2_000,
            angles_degrees=pose_angles(110.0, 110.0, 9.0),
            confidence=0.8,
            camera_disagreement_degrees=4.0,
            timestamp_skew_ms=14,
        )
    )
    snapshot = builder.snapshot()

    assert second.angular_velocity_degrees_per_second == pytest.approx((20.0, -10.0, 4.0))
    assert snapshot.schema_version == POSE_FEATURE_SCHEMA_VERSION
    assert len(snapshot.rows) == 2
    assert snapshot.numeric_rows()[0] == pytest.approx(
        (100.0, 115.0, 7.0, 20.0, -10.0, 4.0, 0.85, 3.0)
    )

    serialized = asdict(snapshot)
    serialized_keys = set(serialized) | {key for row in serialized["rows"] for key in row}
    assert not any(
        forbidden in key.lower()
        for key in serialized_keys
        for forbidden in (
            "timestamp",
            "time_ms",
            "frame",
            "image",
            "pixel",
            "bytes",
            "audio",
            "transcript",
            "patient",
            "session",
            "device",
            "camera_id",
        )
    )
    assert not _contains_bytes(serialized)


def test_feature_velocity_requires_monotonic_time_and_stable_schema() -> None:
    builder = PoseFeatureWindowBuilder()
    first = FusedPoseSummary(1_000, pose_angles(), 0.9, 2.0, 10)
    builder.add(first)

    with pytest.raises(ValueError, match="increase monotonically"):
        builder.add(FusedPoseSummary(1_000, pose_angles(91.0), 0.9, 2.0, 10))


def test_sanitized_feature_requires_typed_angles_and_ordered_velocity() -> None:
    with pytest.raises(TypeError, match="validated PoseAngles"):
        SanitizedPoseFeature(
            angles_degrees=(90.0, 120.0, 5.0),  # type: ignore[arg-type]
            angular_velocity_degrees_per_second=(0.0, 0.0, 0.0),
            confidence=0.9,
            camera_disagreement_degrees=1.0,
        )

    with pytest.raises(TypeError, match="immutable tuple"):
        SanitizedPoseFeature(
            angles_degrees=pose_angles(),
            angular_velocity_degrees_per_second=[0.0, 0.0, 0.0],  # type: ignore[arg-type]
            confidence=0.9,
            camera_disagreement_degrees=1.0,
        )

    with pytest.raises(ValueError, match="same schema width"):
        SanitizedPoseFeature(
            angles_degrees=pose_angles(),
            angular_velocity_degrees_per_second=(0.0, 0.0),
            confidence=0.9,
            camera_disagreement_degrees=1.0,
        )


def _contains_bytes(value: object) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, dict):
        return any(_contains_bytes(key) or _contains_bytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_bytes(item) for item in value)
    return False
