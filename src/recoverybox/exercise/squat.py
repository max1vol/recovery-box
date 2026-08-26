"""Deterministic single-camera squat analysis.

This module performs geometry and state transitions only.  It does not select
spoken phrases, call a model, use a network, or control audio/hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, degrees, hypot
from numbers import Real

from .models import (
    MediaPipePoseFrame,
    MediaPipePoseLandmark,
    NormalizedLandmark,
    SquatAnalysis,
    SquatAssessmentIssue,
    SquatEvent,
    SquatEventType,
    SquatPhase,
)

_GEOMETRY_EPSILON = 1e-9

_LEG_LANDMARKS = (
    MediaPipePoseLandmark.LEFT_HIP,
    MediaPipePoseLandmark.RIGHT_HIP,
    MediaPipePoseLandmark.LEFT_KNEE,
    MediaPipePoseLandmark.RIGHT_KNEE,
    MediaPipePoseLandmark.LEFT_ANKLE,
    MediaPipePoseLandmark.RIGHT_ANKLE,
)

_ARM_LANDMARKS = (
    MediaPipePoseLandmark.LEFT_SHOULDER,
    MediaPipePoseLandmark.RIGHT_SHOULDER,
    MediaPipePoseLandmark.LEFT_ELBOW,
    MediaPipePoseLandmark.RIGHT_ELBOW,
    MediaPipePoseLandmark.LEFT_WRIST,
    MediaPipePoseLandmark.RIGHT_WRIST,
)


def _finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    converted = float(value)
    if converted != converted or converted in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _unit_interval(value: object, *, field_name: str) -> float:
    converted = _finite_float(value, field_name=field_name)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return converted


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class SquatTrackerConfig:
    """Clinician-reviewable thresholds for the deterministic demo tracker."""

    minimum_visibility: float = 0.65
    minimum_presence: float = 0.65
    maximum_frame_gap_ms: int = 500
    maximum_coordinate_extrapolation: float = 0.25
    standing_knee_angle_degrees: float = 160.0
    down_knee_angle_degrees: float = 100.0
    phase_confirmation_frames: int = 3
    maximum_bilateral_knee_difference_degrees: float = 25.0
    monitor_arms_in_t: bool = True
    arm_horizontal_tolerance_degrees: float = 25.0
    minimum_elbow_extension_degrees: float = 150.0
    arms_correction_confirmation_frames: int = 4
    arms_recovery_confirmation_frames: int = 3

    def __post_init__(self) -> None:
        minimum_visibility = _unit_interval(
            self.minimum_visibility,
            field_name="minimum_visibility",
        )
        minimum_presence = _unit_interval(
            self.minimum_presence,
            field_name="minimum_presence",
        )
        maximum_frame_gap_ms = _positive_int(
            self.maximum_frame_gap_ms,
            field_name="maximum_frame_gap_ms",
        )
        coordinate_extrapolation = _finite_float(
            self.maximum_coordinate_extrapolation,
            field_name="maximum_coordinate_extrapolation",
        )
        if not 0.0 <= coordinate_extrapolation <= 1.0:
            raise ValueError("maximum_coordinate_extrapolation must be between 0 and 1")
        standing = _finite_float(
            self.standing_knee_angle_degrees,
            field_name="standing_knee_angle_degrees",
        )
        down = _finite_float(
            self.down_knee_angle_degrees,
            field_name="down_knee_angle_degrees",
        )
        if not 0.0 <= down < standing <= 180.0:
            raise ValueError("knee thresholds must satisfy 0 <= down < standing <= 180 degrees")
        phase_frames = _positive_int(
            self.phase_confirmation_frames,
            field_name="phase_confirmation_frames",
        )
        maximum_difference = _finite_float(
            self.maximum_bilateral_knee_difference_degrees,
            field_name="maximum_bilateral_knee_difference_degrees",
        )
        if not 0.0 <= maximum_difference <= 180.0:
            raise ValueError("maximum_bilateral_knee_difference_degrees must be between 0 and 180")
        if not isinstance(self.monitor_arms_in_t, bool):
            raise TypeError("monitor_arms_in_t must be a boolean")
        horizontal_tolerance = _finite_float(
            self.arm_horizontal_tolerance_degrees,
            field_name="arm_horizontal_tolerance_degrees",
        )
        if not 0.0 <= horizontal_tolerance < 90.0:
            raise ValueError("arm_horizontal_tolerance_degrees must be in [0, 90)")
        elbow_extension = _finite_float(
            self.minimum_elbow_extension_degrees,
            field_name="minimum_elbow_extension_degrees",
        )
        if not 0.0 <= elbow_extension <= 180.0:
            raise ValueError("minimum_elbow_extension_degrees must be between 0 and 180")
        correction_frames = _positive_int(
            self.arms_correction_confirmation_frames,
            field_name="arms_correction_confirmation_frames",
        )
        recovery_frames = _positive_int(
            self.arms_recovery_confirmation_frames,
            field_name="arms_recovery_confirmation_frames",
        )

        object.__setattr__(self, "minimum_visibility", minimum_visibility)
        object.__setattr__(self, "minimum_presence", minimum_presence)
        object.__setattr__(self, "maximum_frame_gap_ms", maximum_frame_gap_ms)
        object.__setattr__(
            self,
            "maximum_coordinate_extrapolation",
            coordinate_extrapolation,
        )
        object.__setattr__(self, "standing_knee_angle_degrees", standing)
        object.__setattr__(self, "down_knee_angle_degrees", down)
        object.__setattr__(self, "phase_confirmation_frames", phase_frames)
        object.__setattr__(
            self,
            "maximum_bilateral_knee_difference_degrees",
            maximum_difference,
        )
        object.__setattr__(
            self,
            "arm_horizontal_tolerance_degrees",
            horizontal_tolerance,
        )
        object.__setattr__(
            self,
            "minimum_elbow_extension_degrees",
            elbow_extension,
        )
        object.__setattr__(
            self,
            "arms_correction_confirmation_frames",
            correction_frames,
        )
        object.__setattr__(self, "arms_recovery_confirmation_frames", recovery_frames)


def _joint_angle_degrees(
    first: NormalizedLandmark,
    vertex: NormalizedLandmark,
    third: NormalizedLandmark,
    *,
    image_width: int,
    image_height: int,
) -> float | None:
    """Return the 2D image-plane joint angle, or ``None`` if degenerate."""

    first_vector = (
        (first.x - vertex.x) * image_width,
        (first.y - vertex.y) * image_height,
    )
    third_vector = (
        (third.x - vertex.x) * image_width,
        (third.y - vertex.y) * image_height,
    )
    first_norm = hypot(*first_vector)
    third_norm = hypot(*third_vector)
    if first_norm <= _GEOMETRY_EPSILON or third_norm <= _GEOMETRY_EPSILON:
        return None
    cosine = (first_vector[0] * third_vector[0] + first_vector[1] * third_vector[1]) / (
        first_norm * third_norm
    )
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def _angle_from_horizontal_degrees(
    first: NormalizedLandmark,
    second: NormalizedLandmark,
    *,
    image_width: int,
    image_height: int,
) -> float | None:
    delta_x = (second.x - first.x) * image_width
    delta_y = (second.y - first.y) * image_height
    if hypot(delta_x, delta_y) <= _GEOMETRY_EPSILON:
        return None
    return degrees(atan2(abs(delta_y), abs(delta_x)))


class SquatTracker:
    """Count full squats and identify a sustained arms-not-in-T condition.

    A count requires a confirmed standing phase, a separately confirmed down
    phase, and a separately confirmed return to standing.  An unassessable
    frame invalidates that partial cycle, while preserving the count of already
    completed reps.
    """

    def __init__(self, config: SquatTrackerConfig | None = None) -> None:
        if config is not None and not isinstance(config, SquatTrackerConfig):
            raise TypeError("config must be a SquatTrackerConfig")
        self.config = config if config is not None else SquatTrackerConfig()
        self._rep_count = 0
        self._phase = SquatPhase.UNKNOWN
        self._candidate_phase: SquatPhase | None = None
        self._candidate_frames = 0
        self._last_timestamp_ms: int | None = None
        self._arms_violation_frames = 0
        self._arms_recovery_frames = 0
        self._arms_violation_announced = False

    @property
    def rep_count(self) -> int:
        return self._rep_count

    @property
    def phase(self) -> SquatPhase:
        return self._phase

    def reset(self) -> None:
        """Reset the full exercise session, including its completed rep count."""

        self._rep_count = 0
        self._last_timestamp_ms = None
        self._invalidate_partial_cycle()

    def update(self, frame: MediaPipePoseFrame) -> SquatAnalysis:
        """Consume one typed frame and return only derived semantic state."""

        if not isinstance(frame, MediaPipePoseFrame):
            raise TypeError("frame must be a MediaPipePoseFrame")

        timestamp_issues = self._timestamp_issues(frame.timestamp_ms)
        if timestamp_issues:
            self._invalidate_partial_cycle()
            return self._withheld_analysis(
                frame,
                timestamp_issues,
                confidence=0.0,
            )

        required_names = _LEG_LANDMARKS + (_ARM_LANDMARKS if self.config.monitor_arms_in_t else ())
        required = tuple(frame.landmark(name) for name in required_names)
        confidence = min(min(landmark.visibility, landmark.presence) for landmark in required)
        extrapolation = self.config.maximum_coordinate_extrapolation
        if any(
            not -extrapolation <= coordinate <= 1.0 + extrapolation
            for landmark in required
            for coordinate in (landmark.x, landmark.y)
        ):
            self._invalidate_partial_cycle()
            return self._withheld_analysis(
                frame,
                (SquatAssessmentIssue.OUT_OF_FRAME_LANDMARK,),
                confidence=confidence,
            )
        confidence_issues: list[SquatAssessmentIssue] = []
        if any(landmark.visibility < self.config.minimum_visibility for landmark in required):
            confidence_issues.append(SquatAssessmentIssue.LOW_VISIBILITY)
        if any(landmark.presence < self.config.minimum_presence for landmark in required):
            confidence_issues.append(SquatAssessmentIssue.LOW_PRESENCE)
        if confidence_issues:
            self._invalidate_partial_cycle()
            return self._withheld_analysis(
                frame,
                tuple(confidence_issues),
                confidence=confidence,
            )

        left_knee_angle = _joint_angle_degrees(
            frame.landmark(MediaPipePoseLandmark.LEFT_HIP),
            frame.landmark(MediaPipePoseLandmark.LEFT_KNEE),
            frame.landmark(MediaPipePoseLandmark.LEFT_ANKLE),
            image_width=frame.image_width,
            image_height=frame.image_height,
        )
        right_knee_angle = _joint_angle_degrees(
            frame.landmark(MediaPipePoseLandmark.RIGHT_HIP),
            frame.landmark(MediaPipePoseLandmark.RIGHT_KNEE),
            frame.landmark(MediaPipePoseLandmark.RIGHT_ANKLE),
            image_width=frame.image_width,
            image_height=frame.image_height,
        )
        if left_knee_angle is None or right_knee_angle is None:
            self._invalidate_partial_cycle()
            return self._withheld_analysis(
                frame,
                (SquatAssessmentIssue.INVALID_LEG_GEOMETRY,),
                confidence=confidence,
            )
        knee_angle = (left_knee_angle + right_knee_angle) / 2.0
        if (
            abs(left_knee_angle - right_knee_angle)
            > self.config.maximum_bilateral_knee_difference_degrees
        ):
            self._invalidate_partial_cycle()
            return self._withheld_analysis(
                frame,
                (SquatAssessmentIssue.BILATERAL_KNEE_DISAGREEMENT,),
                confidence=confidence,
                knee_angle_degrees=knee_angle,
            )

        arms_in_t: bool | None = None
        if self.config.monitor_arms_in_t:
            arms_in_t = self._arms_in_t(frame)
            if arms_in_t is None:
                self._invalidate_partial_cycle()
                return self._withheld_analysis(
                    frame,
                    (SquatAssessmentIssue.INVALID_ARM_GEOMETRY,),
                    confidence=confidence,
                    knee_angle_degrees=knee_angle,
                )

        events = list(self._advance_rep_state(left_knee_angle, right_knee_angle))
        if arms_in_t is not None:
            events.extend(self._advance_arm_state(arms_in_t))
        return SquatAnalysis(
            timestamp_ms=frame.timestamp_ms,
            assessable=True,
            phase=self._phase,
            rep_count=self._rep_count,
            events=tuple(events),
            issues=(),
            confidence=confidence,
            knee_angle_degrees=knee_angle,
            arms_in_t=arms_in_t,
        )

    def update_missing(
        self,
        timestamp_ms: int,
        *,
        issue: SquatAssessmentIssue = SquatAssessmentIssue.NO_POSE,
    ) -> SquatAnalysis:
        """Fail closed immediately when a camera observation has no usable pose.

        The webcam loop should call this for every no-detection result.  A
        watchdog may call it with ``CAMERA_TIMEOUT`` when capture or inference
        has stopped producing observations.
        """

        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int):
            raise TypeError("timestamp_ms must be an integer")
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if not isinstance(issue, SquatAssessmentIssue) or issue not in (
            SquatAssessmentIssue.NO_POSE,
            SquatAssessmentIssue.CAMERA_TIMEOUT,
        ):
            raise ValueError("missing evidence issue must be NO_POSE or CAMERA_TIMEOUT")
        timestamp_issues = self._timestamp_issues(timestamp_ms)
        self._invalidate_partial_cycle()
        return SquatAnalysis(
            timestamp_ms=timestamp_ms,
            assessable=False,
            phase=SquatPhase.UNKNOWN,
            rep_count=self._rep_count,
            events=(),
            issues=(issue, *timestamp_issues),
            confidence=0.0,
            knee_angle_degrees=None,
            arms_in_t=None,
        )

    def _timestamp_issues(
        self,
        timestamp_ms: int,
    ) -> tuple[SquatAssessmentIssue, ...]:
        previous = self._last_timestamp_ms
        if previous is None:
            self._last_timestamp_ms = timestamp_ms
            return ()
        if timestamp_ms <= previous:
            return (SquatAssessmentIssue.NON_MONOTONIC_TIMESTAMP,)
        self._last_timestamp_ms = timestamp_ms
        if timestamp_ms - previous > self.config.maximum_frame_gap_ms:
            return (SquatAssessmentIssue.STALE_FRAME_GAP,)
        return ()

    def _advance_rep_state(
        self,
        left_knee_angle: float,
        right_knee_angle: float,
    ) -> tuple[SquatEvent, ...]:
        both_standing = (
            min(left_knee_angle, right_knee_angle) >= self.config.standing_knee_angle_degrees
        )
        both_down = max(left_knee_angle, right_knee_angle) <= self.config.down_knee_angle_degrees
        if self._phase is SquatPhase.UNKNOWN:
            if both_standing:
                self._confirm_candidate(SquatPhase.STANDING)
            else:
                self._clear_phase_candidate()
            return ()

        if self._phase is SquatPhase.STANDING:
            if both_down:
                self._confirm_candidate(SquatPhase.DOWN)
            else:
                self._clear_phase_candidate()
            return ()

        if both_standing:
            if self._confirm_candidate(SquatPhase.STANDING):
                self._rep_count += 1
                return (
                    SquatEvent(
                        event_type=SquatEventType.REP_COMPLETED,
                        rep_count=self._rep_count,
                    ),
                )
        else:
            self._clear_phase_candidate()
        return ()

    def _confirm_candidate(self, phase: SquatPhase) -> bool:
        if self._candidate_phase is phase:
            self._candidate_frames += 1
        else:
            self._candidate_phase = phase
            self._candidate_frames = 1
        if self._candidate_frames < self.config.phase_confirmation_frames:
            return False
        self._phase = phase
        self._clear_phase_candidate()
        return True

    def _clear_phase_candidate(self) -> None:
        self._candidate_phase = None
        self._candidate_frames = 0

    def _arms_in_t(self, frame: MediaPipePoseFrame) -> bool | None:
        left_shoulder = frame.landmark(MediaPipePoseLandmark.LEFT_SHOULDER)
        right_shoulder = frame.landmark(MediaPipePoseLandmark.RIGHT_SHOULDER)
        left_elbow = frame.landmark(MediaPipePoseLandmark.LEFT_ELBOW)
        right_elbow = frame.landmark(MediaPipePoseLandmark.RIGHT_ELBOW)
        left_wrist = frame.landmark(MediaPipePoseLandmark.LEFT_WRIST)
        right_wrist = frame.landmark(MediaPipePoseLandmark.RIGHT_WRIST)

        shoulder_width = hypot(
            (left_shoulder.x - right_shoulder.x) * frame.image_width,
            (left_shoulder.y - right_shoulder.y) * frame.image_height,
        )
        left_elbow_angle = _joint_angle_degrees(
            left_shoulder,
            left_elbow,
            left_wrist,
            image_width=frame.image_width,
            image_height=frame.image_height,
        )
        right_elbow_angle = _joint_angle_degrees(
            right_shoulder,
            right_elbow,
            right_wrist,
            image_width=frame.image_width,
            image_height=frame.image_height,
        )
        left_horizontal = _angle_from_horizontal_degrees(
            left_shoulder,
            left_wrist,
            image_width=frame.image_width,
            image_height=frame.image_height,
        )
        right_horizontal = _angle_from_horizontal_degrees(
            right_shoulder,
            right_wrist,
            image_width=frame.image_width,
            image_height=frame.image_height,
        )
        if (
            shoulder_width <= _GEOMETRY_EPSILON
            or left_elbow_angle is None
            or right_elbow_angle is None
            or left_horizontal is None
            or right_horizontal is None
        ):
            return None

        body_center_x = (left_shoulder.x + right_shoulder.x) / 2.0
        left_outward = (left_shoulder.x - body_center_x) * (
            left_wrist.x - body_center_x
        ) > 0.0 and abs(left_wrist.x - body_center_x) > abs(left_shoulder.x - body_center_x)
        right_outward = (right_shoulder.x - body_center_x) * (
            right_wrist.x - body_center_x
        ) > 0.0 and abs(right_wrist.x - body_center_x) > abs(right_shoulder.x - body_center_x)
        return (
            left_horizontal <= self.config.arm_horizontal_tolerance_degrees
            and right_horizontal <= self.config.arm_horizontal_tolerance_degrees
            and left_elbow_angle >= self.config.minimum_elbow_extension_degrees
            and right_elbow_angle >= self.config.minimum_elbow_extension_degrees
            and left_outward
            and right_outward
        )

    def _advance_arm_state(self, arms_in_t: bool) -> tuple[SquatEvent, ...]:
        if arms_in_t:
            self._arms_violation_frames = 0
            if not self._arms_violation_announced:
                self._arms_recovery_frames = 0
                return ()
            self._arms_recovery_frames += 1
            if self._arms_recovery_frames >= self.config.arms_recovery_confirmation_frames:
                self._arms_violation_announced = False
                self._arms_recovery_frames = 0
            return ()

        self._arms_recovery_frames = 0
        self._arms_violation_frames += 1
        if (
            not self._arms_violation_announced
            and self._arms_violation_frames >= self.config.arms_correction_confirmation_frames
        ):
            self._arms_violation_announced = True
            return (SquatEvent(event_type=SquatEventType.ARMS_NOT_IN_T),)
        return ()

    def _invalidate_partial_cycle(self) -> None:
        self._phase = SquatPhase.UNKNOWN
        self._clear_phase_candidate()
        self._arms_violation_frames = 0
        self._arms_recovery_frames = 0
        self._arms_violation_announced = False

    def _withheld_analysis(
        self,
        frame: MediaPipePoseFrame,
        issues: tuple[SquatAssessmentIssue, ...],
        *,
        confidence: float,
        knee_angle_degrees: float | None = None,
    ) -> SquatAnalysis:
        return SquatAnalysis(
            timestamp_ms=frame.timestamp_ms,
            assessable=False,
            phase=SquatPhase.UNKNOWN,
            rep_count=self._rep_count,
            events=(),
            issues=issues,
            confidence=confidence,
            knee_angle_degrees=knee_angle_degrees,
            arms_in_t=None,
        )
