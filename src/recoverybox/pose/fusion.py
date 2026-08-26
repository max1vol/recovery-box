"""Time synchronization and conservative dual-view pose fusion."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .models import (
    CameraView,
    FusedPoseSummary,
    FusionIssue,
    FusionResult,
    PoseAngles,
    PoseViewSummary,
)


@dataclass(frozen=True, slots=True)
class PoseFusionConfig:
    """Safety limits applied before any fused measurement is released."""

    maximum_age_ms: int = 250
    maximum_timestamp_skew_ms: int = 80
    maximum_pair_wait_ms: int = 120
    minimum_view_confidence: float = 0.75
    maximum_angle_disagreement_degrees: float = 12.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_age_ms, bool)
            or not isinstance(self.maximum_age_ms, int)
            or self.maximum_age_ms < 0
        ):
            raise ValueError("maximum_age_ms must be a non-negative integer")
        if (
            isinstance(self.maximum_timestamp_skew_ms, bool)
            or not isinstance(self.maximum_timestamp_skew_ms, int)
            or self.maximum_timestamp_skew_ms < 0
        ):
            raise ValueError("maximum_timestamp_skew_ms must be a non-negative integer")
        if (
            isinstance(self.maximum_pair_wait_ms, bool)
            or not isinstance(self.maximum_pair_wait_ms, int)
            or self.maximum_pair_wait_ms < 0
        ):
            raise ValueError("maximum_pair_wait_ms must be a non-negative integer")
        if (
            not isfinite(self.minimum_view_confidence)
            or not 0.0 <= self.minimum_view_confidence <= 1.0
        ):
            raise ValueError("minimum_view_confidence must be between 0 and 1")
        if (
            not isfinite(self.maximum_angle_disagreement_degrees)
            or self.maximum_angle_disagreement_degrees < 0.0
        ):
            raise ValueError("maximum_angle_disagreement_degrees must be finite and non-negative")


class DualViewPoseFuser:
    """Fuse synchronized views only when every safety precondition passes."""

    def __init__(self, config: PoseFusionConfig | None = None) -> None:
        self.config = config or PoseFusionConfig()

    def fuse(
        self,
        first: PoseViewSummary,
        second: PoseViewSummary,
        *,
        now_monotonic_ms: int,
    ) -> FusionResult:
        _validate_monotonic_timestamp(now_monotonic_ms)

        primary, secondary = self._order_views(first, second)
        issues: list[FusionIssue] = []

        if primary.view == secondary.view:
            issues.append(FusionIssue.SAME_VIEW)

        primary_age = now_monotonic_ms - primary.monotonic_timestamp_ms
        secondary_age = now_monotonic_ms - secondary.monotonic_timestamp_ms
        if primary_age < 0 or secondary_age < 0:
            issues.append(FusionIssue.FUTURE_TIMESTAMP)
        if primary_age > self.config.maximum_age_ms:
            issues.append(FusionIssue.STALE_PRIMARY)
        if secondary_age > self.config.maximum_age_ms:
            issues.append(FusionIssue.STALE_SECONDARY)

        timestamp_skew_ms = abs(primary.monotonic_timestamp_ms - secondary.monotonic_timestamp_ms)
        if timestamp_skew_ms > self.config.maximum_timestamp_skew_ms:
            issues.append(FusionIssue.EXCESSIVE_TIMESTAMP_SKEW)

        if primary.confidence < self.config.minimum_view_confidence:
            issues.append(FusionIssue.LOW_PRIMARY_CONFIDENCE)
        if secondary.confidence < self.config.minimum_view_confidence:
            issues.append(FusionIssue.LOW_SECONDARY_CONFIDENCE)

        angle_schema_matches = primary.angles_degrees.schema == secondary.angles_degrees.schema
        if not angle_schema_matches:
            issues.append(FusionIssue.ANGLE_SCHEMA_MISMATCH)
            disagreement = None
        else:
            disagreement = max(
                abs(primary_angle - secondary_angle)
                for primary_angle, secondary_angle in zip(
                    primary.angles_degrees, secondary.angles_degrees, strict=True
                )
            )
            if disagreement > self.config.maximum_angle_disagreement_degrees:
                issues.append(FusionIssue.CAMERA_DISAGREEMENT)

        if issues:
            # No partial or best-effort estimate escapes when any check fails.
            return FusionResult(fused=None, issues=tuple(issues))

        assert disagreement is not None
        confidence_sum = primary.confidence + secondary.confidence
        if confidence_sum == 0.0:
            # This is reachable only when a caller explicitly configures a zero
            # confidence threshold.  Equal weighting still remains deterministic.
            primary_weight = secondary_weight = 0.5
        else:
            primary_weight = primary.confidence / confidence_sum
            secondary_weight = secondary.confidence / confidence_sum
        fused_angles = tuple(
            primary_weight * primary_angle + secondary_weight * secondary_angle
            for primary_angle, secondary_angle in zip(
                primary.angles_degrees, secondary.angles_degrees, strict=True
            )
        )

        return FusionResult(
            fused=FusedPoseSummary(
                monotonic_timestamp_ms=max(
                    primary.monotonic_timestamp_ms,
                    secondary.monotonic_timestamp_ms,
                ),
                angles_degrees=PoseAngles(
                    schema=primary.angles_degrees.schema,
                    values_degrees=fused_angles,
                ),
                # The less certain view bounds the consensus confidence.
                confidence=min(primary.confidence, secondary.confidence),
                camera_disagreement_degrees=disagreement,
                timestamp_skew_ms=timestamp_skew_ms,
            )
        )

    @staticmethod
    def _order_views(
        first: PoseViewSummary, second: PoseViewSummary
    ) -> tuple[PoseViewSummary, PoseViewSummary]:
        if first.view == CameraView.PRIMARY:
            return first, second
        if second.view == CameraView.PRIMARY:
            return second, first
        # Invalid duplicate SECONDARY observations are ordered consistently so
        # their diagnostics remain deterministic.
        return first, second


class DualViewPoseSynchronizer:
    """Pair the most recent observation from each view, then fuse once.

    Camera workers can publish asynchronously.  A new observation replaces an
    older pending observation from the same view; a completed pair is consumed
    exactly once so downstream code cannot silently reuse a view.
    """

    def __init__(self, fuser: DualViewPoseFuser | None = None) -> None:
        self._fuser = fuser or DualViewPoseFuser()
        self._pending: dict[CameraView, PoseViewSummary] = {}
        self._watchdog_started_ms: int | None = None

    def arm(self, *, now_monotonic_ms: int) -> None:
        """Start complete-camera-loss monitoring for the device session.

        Device startup must arm the synchronizer before waiting for frames.
        Arming an active watchdog is rejected so callers cannot indefinitely
        postpone a missing-camera deadline.
        """

        _validate_monotonic_timestamp(now_monotonic_ms)
        if self._watchdog_started_ms is not None:
            raise RuntimeError("pose synchronizer watchdog is already armed")
        self._watchdog_started_ms = now_monotonic_ms

    def push(
        self,
        summary: PoseViewSummary,
        *,
        now_monotonic_ms: int,
    ) -> FusionResult | None:
        _validate_monotonic_timestamp(now_monotonic_ms)
        if self._watchdog_started_ms is None:
            # Preserve safe standalone use while the device loop uses arm()
            # explicitly so complete startup loss can also be detected.
            self._watchdog_started_ms = now_monotonic_ms
        elif now_monotonic_ms < self._watchdog_started_ms:
            raise ValueError("now_monotonic_ms must not move backwards while awaiting a pair")

        # Replacing a same-view sample does not restart the watchdog.  A healthy
        # camera must not conceal that its peer has stopped publishing.
        self._pending[summary.view] = summary
        if len(self._pending) < 2:
            return None

        primary = self._pending.pop(CameraView.PRIMARY)
        secondary = self._pending.pop(CameraView.SECONDARY)
        # A consumed pair begins the next observation window immediately.  If
        # both cameras die now, expire() will report both as missing.
        self._watchdog_started_ms = now_monotonic_ms
        return self._fuser.fuse(
            primary,
            secondary,
            now_monotonic_ms=now_monotonic_ms,
        )

    def expire(self, *, now_monotonic_ms: int) -> FusionResult | None:
        """Close an unpaired observation after the configured watchdog limit.

        The device loop should poll this method even when no camera messages
        arrive.  Once expired, the orphan is consumed and a closed missing-view
        issue is emitted instead of leaving the movement indefinitely unknown.
        """

        _validate_monotonic_timestamp(now_monotonic_ms)
        if self._watchdog_started_ms is None:
            return None

        if now_monotonic_ms < self._watchdog_started_ms:
            raise ValueError("now_monotonic_ms must not move backwards while awaiting a pair")
        if now_monotonic_ms - self._watchdog_started_ms < self._fuser.config.maximum_pair_wait_ms:
            return None

        if not self._pending:
            issues = (FusionIssue.MISSING_PRIMARY, FusionIssue.MISSING_SECONDARY)
        elif CameraView.PRIMARY in self._pending:
            issues = (FusionIssue.MISSING_SECONDARY,)
        else:
            issues = (FusionIssue.MISSING_PRIMARY,)

        # Consume the failed window and immediately begin another.  This keeps
        # monitoring live while allowing either or both cameras to recover.
        self._pending.clear()
        self._watchdog_started_ms = now_monotonic_ms
        return FusionResult(fused=None, issues=issues)

    @property
    def pending_view_count(self) -> int:
        return len(self._pending)

    def clear(self) -> None:
        self._pending.clear()
        self._watchdog_started_ms = None


def _validate_monotonic_timestamp(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("now_monotonic_ms must be a non-negative integer")
