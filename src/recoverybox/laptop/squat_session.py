"""Hardware-free composition for one long-lived laptop squat session.

This module joins the deterministic squat analysis, Guardian, prompt-cue
quarantine, and explicit session-end capability.  It deliberately does not
open a camera, import MediaPipe, initialize audio hardware, or connect a
socket.  A launcher owns those edges and injects an already constructed
Realtime transport plus quick, non-blocking speaker callbacks.

The Realtime connection is configured once and remains open across every
exercise cue.  It closes only after a physical stop or a locally validated
``finish_session`` tool call.  Ordinary Realtime events and completion of the
prescribed rep count do not end the session.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

from recoverybox.core import (
    DEFAULT_CUE_CATALOG,
    CueId,
    ExercisePlan,
    Guardian,
    GuardianAction,
    GuardianDecision,
    LocalCueRequest,
    MovementObservation,
    SessionMode,
)
from recoverybox.exercise import (
    SquatAnalysis,
    SquatEvent,
    SquatEventType,
    SquatPhase,
)
from recoverybox.realtime import (
    SESSION_CONTROL_TOOL_REGISTRY,
    ConversationMode,
    CueDeliveryConfig,
    CueDeliveryFailure,
    ModelAudioPolicy,
    RealtimeClientResult,
    RealtimeCueDelivery,
    RealtimeSession,
    RealtimeTransport,
    ReleasedCueAudio,
    ServerEventKind,
    SessionEndController,
    SessionEndSignal,
)
from recoverybox.session import (
    ApprovedCuePlaybackAuthorization,
    GuardianDecisionEffect,
    SessionCoordinator,
)

SQUAT_EXERCISE_ID = "squat"

SESSION_END_POLICY_INSTRUCTIONS = """\
SESSION END POLICY
Keep this Realtime session open across ordinary user turns and exercise cues.
Call finish_session with an empty object only when the user explicitly asks to
finish or leave, or explicitly says goodbye. Never call finish_session merely
because a response, exercise set, pause, or period of silence has completed.
"""

SQUAT_REP_CUE_IDS: tuple[CueId, ...] = (
    CueId.SQUAT_REP_ONE,
    CueId.SQUAT_REP_TWO,
    CueId.SQUAT_REP_THREE,
    CueId.SQUAT_REP_FOUR,
    CueId.SQUAT_REP_FIVE,
    CueId.SQUAT_REP_SIX,
    CueId.SQUAT_REP_SEVEN,
    CueId.SQUAT_REP_EIGHT,
    CueId.SQUAT_REP_NINE,
    CueId.SQUAT_REP_TEN,
)


def build_single_camera_squat_plan(*, target_reps: int = 10) -> ExercisePlan:
    """Build the reviewed one-camera plan used by the laptop prototype."""

    if isinstance(target_reps, bool) or not isinstance(target_reps, int):
        raise TypeError("target_reps must be an integer")
    if target_reps != len(SQUAT_REP_CUE_IDS):
        raise ValueError("the reviewed laptop squat prompt catalog requires exactly 10 reps")
    return ExercisePlan(
        exercise_id=SQUAT_EXERCISE_ID,
        allowed_cue_ids=frozenset(
            cue_id.value for cue_id in (*SQUAT_REP_CUE_IDS, CueId.ARMS_T_SHAPE)
        ),
        target_reps=target_reps,
        min_confidence=0.70,
        max_pose_age_ms=500,
        required_camera_views=1,
    )


DEFAULT_SINGLE_CAMERA_SQUAT_PLAN = build_single_camera_squat_plan()


def observation_from_squat_analysis(
    analysis: SquatAnalysis,
    *,
    plan: ExercisePlan = DEFAULT_SINGLE_CAMERA_SQUAT_PLAN,
    pose_age_ms: int = 0,
) -> MovementObservation:
    """Sanitize one derived squat result into the Guardian's numeric input.

    A single camera has no cross-camera disagreement measurement, so that
    field remains ``None`` and the explicit view count is one.  A withheld
    squat result is marked out-of-distribution as well as retaining its
    measured confidence; this guarantees that it cannot be inferred safe by
    the Guardian even if a future tracker reports a high confidence alongside
    an assessment issue.
    """

    if not isinstance(analysis, SquatAnalysis):
        raise TypeError("analysis must be a SquatAnalysis")
    if not isinstance(plan, ExercisePlan):
        raise TypeError("plan must be an ExercisePlan")
    if isinstance(pose_age_ms, bool) or not isinstance(pose_age_ms, int):
        raise TypeError("pose_age_ms must be an integer")
    if pose_age_ms < 0:
        raise ValueError("pose_age_ms must be non-negative")

    return MovementObservation(
        exercise_id=plan.exercise_id,
        timestamp_ms=analysis.timestamp_ms,
        confidence=analysis.confidence,
        camera_disagreement_degrees=None,
        pose_age_ms=pose_age_ms,
        camera_view_count=1,
        rep_index=analysis.rep_count,
        phase=analysis.phase.value,
        out_of_distribution=not analysis.assessable,
    )


def local_cue_request_for_squat_event(
    event: SquatEvent,
    *,
    target_reps: int = 10,
) -> LocalCueRequest | None:
    """Map a closed semantic event to a fixed cue ID, never to speech text."""

    if not isinstance(event, SquatEvent):
        raise TypeError("event must be a SquatEvent")
    if isinstance(target_reps, bool) or not isinstance(target_reps, int):
        raise TypeError("target_reps must be an integer")
    if target_reps != len(SQUAT_REP_CUE_IDS):
        raise ValueError("the reviewed laptop squat prompt catalog requires exactly 10 reps")

    if event.event_type is SquatEventType.ARMS_NOT_IN_T:
        return LocalCueRequest(CueId.ARMS_T_SHAPE.value)
    if event.event_type is SquatEventType.REP_COMPLETED:
        assert event.rep_count is not None
        if 1 <= event.rep_count <= target_reps:
            return LocalCueRequest(SQUAT_REP_CUE_IDS[event.rep_count - 1].value)
        return None
    return None


@dataclass(frozen=True, slots=True)
class SquatSessionAnalysisResult:
    """Observable deterministic result for one camera-analysis callback."""

    analysis: SquatAnalysis
    observation: MovementObservation
    decisions: tuple[GuardianDecision, ...]
    effects: tuple[GuardianDecisionEffect, ...]
    cue_delivery_failed: bool
    mode: SessionMode


@dataclass(frozen=True, slots=True)
class SquatSessionDispatchResult:
    """Result of routing exactly one event from the shared Realtime receiver."""

    realtime_result: RealtimeClientResult | None
    cue_event_consumed: bool
    end_signal: SessionEndSignal | None
    failure_kind: str | None = None


@dataclass(frozen=True, slots=True)
class SquatAudioTurnResult:
    """Content-free result of submitting one button-defined user audio turn."""

    submitted: bool
    failure_kind: str | None = None


class _CoordinatorModeProvider:
    """Break the delivery/coordinator construction cycle without mutable mode."""

    def __init__(self) -> None:
        self._coordinator: SessionCoordinator | None = None

    def bind(self, coordinator: SessionCoordinator) -> None:
        if self._coordinator is not None:
            raise RuntimeError("mode provider is already bound")
        self._coordinator = coordinator

    @property
    def current_mode(self) -> SessionMode:
        if self._coordinator is None:
            # Construction is synchronous; no request is allowed before bind.
            return SessionMode.IDLE
        return self._coordinator.current_mode


class _AvailabilityAwarePromptCueSession:
    """Prevent stale cue cleanup from writing after the cloud edge is offline."""

    def __init__(
        self,
        session: RealtimeSession,
        is_available: Callable[[], bool],
    ) -> None:
        self._session = session
        self._is_available = is_available

    def request_approved_prompt_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None:
        if not self._is_available():
            raise RuntimeError("Realtime is unavailable for this session")
        self._session.request_approved_prompt_cue(authorization)

    def cancel_response(self, *, response_id: str | None = None) -> None:
        if self._is_available():
            self._session.cancel_response(response_id=response_id)
            return
        if response_id is None:
            self._session.revoke_pending_response_locally()
        else:
            self._session.audio_gate.discard_response(response_id)
        self._session.current_response_id = None
        self._session.current_assistant_item_id = None

    def revoke_pending_response_locally(self) -> int:
        return self._session.revoke_pending_response_locally()


class LaptopSquatSession:
    """Compose one persistent Realtime connection with local squat safety.

    ``start`` sends one session configuration event but does not activate pose
    safety yet.  A launcher first obtains an assessable standing pose and then
    explicitly calls :meth:`activate_exercise`.  Once active, any missing or
    withheld analysis enters PAUSED and never auto-resumes.

    ``on_cue_audio`` must enqueue the complete, already approved clip into a
    speaker arbiter and return promptly.  It must not play synchronously.
    ``on_audio_preempt`` must synchronously revoke that arbiter's queued and
    current model audio.
    """

    def __init__(
        self,
        *,
        transport: RealtimeTransport,
        on_cue_audio: Callable[[ReleasedCueAudio], None],
        on_audio_preempt: Callable[[], None],
        guardian: Guardian | None = None,
        plan: ExercisePlan = DEFAULT_SINGLE_CAMERA_SQUAT_PLAN,
        cue_delivery_config: CueDeliveryConfig | None = None,
        cue_delivery_enabled: bool = True,
    ) -> None:
        if not isinstance(transport, RealtimeTransport):
            raise TypeError("transport must implement RealtimeTransport")
        if not callable(on_cue_audio):
            raise TypeError("on_cue_audio must be callable")
        if not callable(on_audio_preempt):
            raise TypeError("on_audio_preempt must be callable")
        if guardian is not None and not isinstance(guardian, Guardian):
            raise TypeError("guardian must be a Guardian")
        if not isinstance(plan, ExercisePlan):
            raise TypeError("plan must be an ExercisePlan")
        if plan.required_camera_views != 1:
            raise ValueError("the laptop squat session requires exactly one camera view")
        if plan.exercise_id != SQUAT_EXERCISE_ID:
            raise ValueError("the laptop squat session requires the squat exercise plan")
        if plan.target_reps != len(SQUAT_REP_CUE_IDS):
            raise ValueError("the reviewed laptop squat prompt catalog requires exactly 10 reps")
        if cue_delivery_config is not None and not isinstance(
            cue_delivery_config,
            CueDeliveryConfig,
        ):
            raise TypeError("cue_delivery_config must be a CueDeliveryConfig")
        if not isinstance(cue_delivery_enabled, bool):
            raise TypeError("cue_delivery_enabled must be a boolean")

        self._guardian = guardian or Guardian()
        self._plan = plan
        self._realtime_session = RealtimeSession(
            transport=transport,
            tools=SESSION_CONTROL_TOOL_REGISTRY,
            cue_catalog=DEFAULT_CUE_CATALOG,
        )
        self._state_lock = threading.RLock()
        self._request_lock = threading.RLock()
        self._cue_delivery_enabled = cue_delivery_enabled
        self._started = False
        self._closed = False
        self._realtime_available = cue_delivery_enabled
        self._realtime_failure_kind: str | None = None
        self._last_cue_failure: CueDeliveryFailure | None = None
        self._cue_callback_failed = False
        self._control_turn_inflight = False
        self._control_response_id: str | None = None
        self._deferred_cue_decisions: list[GuardianDecision] = []
        self._last_observed_rep_count = 0

        mode_provider = _CoordinatorModeProvider()
        delivery_options: dict[str, object] = {}
        if cue_delivery_config is not None:
            delivery_options["config"] = cue_delivery_config
        cue_session = _AvailabilityAwarePromptCueSession(
            self._realtime_session,
            self._is_realtime_available,
        )
        self._cue_delivery = RealtimeCueDelivery(
            session=cue_session,
            mode_provider=mode_provider,
            on_audio=on_cue_audio,
            on_preempt=on_audio_preempt,
            on_failure=self._on_cue_delivery_failure,
            count_cue_ids=SQUAT_REP_CUE_IDS,
            **delivery_options,  # type: ignore[arg-type]
        )
        self._coordinator = SessionCoordinator(
            cue_playback=self._cue_delivery,
            initial_mode=SessionMode.IDLE,
        )
        mode_provider.bind(self._coordinator)
        self._end_controller = SessionEndController(self._on_session_end)

    @property
    def realtime_session(self) -> RealtimeSession:
        """The single shared session, exposed for an injected audio-turn edge."""

        return self._realtime_session

    @property
    def coordinator(self) -> SessionCoordinator:
        return self._coordinator

    @property
    def cue_delivery(self) -> RealtimeCueDelivery:
        return self._cue_delivery

    @property
    def end_controller(self) -> SessionEndController:
        return self._end_controller

    @property
    def plan(self) -> ExercisePlan:
        return self._plan

    @property
    def started(self) -> bool:
        with self._state_lock:
            return self._started

    @property
    def ended(self) -> bool:
        return self._end_controller.ended

    @property
    def realtime_available(self) -> bool:
        """Whether this workout may still use its original cloud session."""

        with self._state_lock:
            return self._realtime_available

    @property
    def cue_delivery_enabled(self) -> bool:
        """Whether this composition intentionally uses Realtime voice cues."""

        return self._cue_delivery_enabled

    @property
    def realtime_failure_kind(self) -> str | None:
        """Scrubbed exception/event category for the permanent offline state."""

        with self._state_lock:
            return self._realtime_failure_kind

    @property
    def last_cue_failure(self) -> CueDeliveryFailure | None:
        with self._state_lock:
            return self._last_cue_failure

    def start(self, *, instructions: str, voice: str) -> None:
        """Configure the existing connection exactly once; keep it in IDLE."""

        with self._state_lock:
            if self._closed:
                raise RuntimeError("squat session is closed")
            if self._started:
                raise RuntimeError("squat session is already started")
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("instructions must not be blank")
            if self._cue_delivery_enabled:
                session_instructions = (
                    f"{instructions.strip()}\n\n{SESSION_END_POLICY_INSTRUCTIONS.strip()}"
                )
                self._realtime_session.configure(
                    instructions=session_instructions,
                    voice=voice,
                )
            self._started = True

    def activate_exercise(
        self,
        analysis: SquatAnalysis,
        *,
        pose_age_ms: int = 0,
    ) -> bool:
        """Enter ACTIVE only from an explicitly supplied assessable stand."""

        self._require_running()
        observation = observation_from_squat_analysis(
            analysis,
            plan=self._plan,
            pose_age_ms=pose_age_ms,
        )
        with self._request_lock:
            if self._cue_delivery_enabled and not self._is_realtime_available():
                return False
            if self._coordinator.current_mode is not SessionMode.IDLE:
                raise RuntimeError("exercise activation requires IDLE mode")
            if not analysis.assessable or analysis.phase is not SquatPhase.STANDING:
                return False
            decision = self._guardian.decide(observation, self._plan)
            if decision.action is not GuardianAction.CONTINUE:
                return False
            self._coordinator.transition_to(SessionMode.ACTIVE_EXERCISE)
            with self._state_lock:
                self._last_observed_rep_count = analysis.rep_count
            return True

    def resume_after_assessable_pose(
        self,
        analysis: SquatAnalysis,
        *,
        pose_age_ms: int = 0,
    ) -> bool:
        """Explicitly resume a pause after a fresh assessable standing pose."""

        self._require_running()
        observation = observation_from_squat_analysis(
            analysis,
            plan=self._plan,
            pose_age_ms=pose_age_ms,
        )
        with self._request_lock:
            if self._cue_delivery_enabled and not self._is_realtime_available():
                return False
            if self._coordinator.current_mode is not SessionMode.PAUSED:
                raise RuntimeError("exercise resume requires PAUSED mode")
            if not analysis.assessable or analysis.phase is not SquatPhase.STANDING:
                return False
            decision = self._guardian.decide(observation, self._plan)
            if decision.action is not GuardianAction.CONTINUE:
                return False
            self._coordinator.transition_to(SessionMode.ACTIVE_EXERCISE)
            with self._state_lock:
                self._last_observed_rep_count = analysis.rep_count
            return True

    def process_analysis(
        self,
        analysis: SquatAnalysis,
        *,
        pose_age_ms: int = 0,
    ) -> SquatSessionAnalysisResult:
        """Apply one derived frame without ever ending the persistent session."""

        self._require_running()
        observation = observation_from_squat_analysis(
            analysis,
            plan=self._plan,
            pose_age_ms=pose_age_ms,
        )
        with self._request_lock:
            mode = self._coordinator.current_mode
            if mode is SessionMode.IDLE or mode not in {
                SessionMode.ACTIVE_EXERCISE,
                SessionMode.PAUSED,
            }:
                return SquatSessionAnalysisResult(
                    analysis=analysis,
                    observation=observation,
                    decisions=(),
                    effects=(),
                    cue_delivery_failed=False,
                    mode=mode,
                )

            with self._state_lock:
                events_consistent = self._events_are_consistent_locked(analysis)
            if not events_consistent:
                observation = replace(observation, out_of_distribution=True)

            decisions: list[GuardianDecision] = []
            effects: list[GuardianDecisionEffect] = []
            cue_delivery_failed = False

            baseline = self._guardian.decide(observation, self._plan)
            decisions.append(baseline)
            try:
                effects.append(self._coordinator.apply_guardian_decision(baseline))
            except Exception:
                # The coordinator changes to PAUSED before surfacing a cue or
                # preemption boundary failure. Keep the camera loop alive.
                cue_delivery_failed = True

            if (
                baseline.action is GuardianAction.CONTINUE
                and self._coordinator.current_mode is SessionMode.ACTIVE_EXERCISE
            ):
                for event in analysis.events:
                    cue_request = local_cue_request_for_squat_event(
                        event,
                        target_reps=self._plan.target_reps,
                    )
                    if cue_request is None:
                        continue
                    decision = self._guardian.decide(
                        observation,
                        self._plan,
                        local_cue_request=cue_request,
                    )
                    decisions.append(decision)
                    if not self._cue_delivery_enabled:
                        continue
                    if not self._is_realtime_available():
                        break
                    if self._control_turn_inflight:
                        self._defer_cue_decision_locked(decision)
                    else:
                        try:
                            effects.append(self._coordinator.apply_guardian_decision(decision))
                        except Exception as exc:
                            # A failed request makes this socket permanently
                            # unsafe for later FIFO response correlation.
                            self._mark_realtime_unavailable_locked(type(exc).__name__)
                            cue_delivery_failed = True
                            break

            with self._state_lock:
                cue_delivery_failed = cue_delivery_failed or self._cue_callback_failed
                self._cue_callback_failed = False
            return SquatSessionAnalysisResult(
                analysis=analysis,
                observation=observation,
                decisions=tuple(decisions),
                effects=tuple(effects),
                cue_delivery_failed=cue_delivery_failed,
                mode=self._coordinator.current_mode,
            )

    def pump_once(self) -> SquatSessionDispatchResult:
        """Route one event through the one and only Realtime receiver."""

        self._require_running()
        if not self._cue_delivery_enabled:
            return SquatSessionDispatchResult(
                realtime_result=None,
                cue_event_consumed=False,
                end_signal=None,
                failure_kind="VoiceDisabled",
            )
        if not self._is_realtime_available():
            return SquatSessionDispatchResult(
                realtime_result=None,
                cue_event_consumed=False,
                end_signal=None,
                failure_kind="RealtimeUnavailable",
            )
        try:
            # Socket receive may block, so it never owns the request lock used
            # by local pose/Guardian decisions.
            raw = self._realtime_session.transport.receive_event()
        except Exception as exc:
            failure_kind = type(exc).__name__
            with self._request_lock:
                self._mark_realtime_unavailable_locked(failure_kind)
            return SquatSessionDispatchResult(
                realtime_result=None,
                cue_event_consumed=False,
                end_signal=None,
                failure_kind=failure_kind,
            )

        with self._request_lock:
            if not self._is_realtime_available():
                return SquatSessionDispatchResult(
                    realtime_result=None,
                    cue_event_consumed=False,
                    end_signal=None,
                    failure_kind="RealtimeUnavailable",
                )
            try:
                result = self._realtime_session.handle_event(raw)
                if result.event.kind is ServerEventKind.ERROR:
                    self._mark_realtime_unavailable_locked("RealtimeServerError")
                cue_consumed = self._cue_delivery.handle_result(result)
            except Exception as exc:
                failure_kind = type(exc).__name__
                self._mark_realtime_unavailable_locked(failure_kind)
                return SquatSessionDispatchResult(
                    realtime_result=None,
                    cue_event_consumed=False,
                    end_signal=None,
                    failure_kind=failure_kind,
                )

            # Bind response.created before any later function-call event can
            # be considered. Only the currently authorized manual control
            # response is allowed to request the end capability.
            if result.event.kind is ServerEventKind.RESPONSE_CREATED:
                self._finish_control_response_if_terminal(result)
            control_response_matches = self._control_response_matches_locked(result)
            end_signal: SessionEndSignal | None = None
            if control_response_matches:
                for call in result.validated_tool_calls:
                    signal = self._end_controller.accept_validated_tool_call(call)
                    if end_signal is None and signal is not None:
                        end_signal = signal
            if not self.ended and result.event.kind is not ServerEventKind.RESPONSE_CREATED:
                self._finish_control_response_if_terminal(result)
            return SquatSessionDispatchResult(
                realtime_result=result,
                cue_event_consumed=cue_consumed,
                end_signal=end_signal,
                failure_kind=(
                    None if self._is_realtime_available() else self.realtime_failure_kind
                ),
            )

    def submit_user_audio_turn(self, pcm16_mono_24khz: bytes) -> SquatAudioTurnResult:
        """Submit one manual/button-defined turn for session-control intent.

        The response uses ACTIVE_EXERCISE plus ``NO_AUDIO``.  This lets the
        model request ``finish_session`` after an explicit spoken goodbye but
        gives ordinary conversational speech no path to the speaker.  If a
        cue response is already authorized, the turn is rejected before any
        input bytes are sent; the launcher can retry after that cue completes.

        Network or protocol errors are returned as a scrubbed class name. They
        pause cloud coaching, keep the end controller open, and do not stop
        deterministic local tracking.
        """

        self._require_running()
        if not isinstance(pcm16_mono_24khz, bytes):
            raise TypeError("pcm16_mono_24khz must be bytes")
        if not pcm16_mono_24khz or len(pcm16_mono_24khz) % 2:
            raise ValueError("user audio must contain complete PCM16 samples")
        with self._request_lock:
            if not self._cue_delivery_enabled:
                return SquatAudioTurnResult(
                    submitted=False,
                    failure_kind="VoiceDisabled",
                )
            if not self._is_realtime_available():
                return SquatAudioTurnResult(
                    submitted=False,
                    failure_kind="RealtimeUnavailable",
                )
            if self._control_turn_inflight or self._realtime_session.audio_gate.open_authorizations:
                return SquatAudioTurnResult(
                    submitted=False,
                    failure_kind="RealtimeResponseBusy",
                )

            self._control_turn_inflight = True
            self._control_response_id = None
            try:
                for offset in range(0, len(pcm16_mono_24khz), 32_000):
                    self._realtime_session.append_user_audio(
                        pcm16_mono_24khz[offset : offset + 32_000]
                    )
                self._realtime_session.finish_user_turn_and_request(
                    mode=ConversationMode.ACTIVE_EXERCISE,
                    policy=ModelAudioPolicy.NO_AUDIO,
                )
            except Exception as exc:
                failure_kind = type(exc).__name__
                self._mark_realtime_unavailable_locked(failure_kind)
                return SquatAudioTurnResult(
                    submitted=False,
                    failure_kind=failure_kind,
                )
        return SquatAudioTurnResult(submitted=True)

    def report_speaker_failure(self) -> None:
        """Permanently disable voice after an asynchronous speaker error.

        The launcher filters intentional playback cancellation before calling
        this method.  Once an actual output failure reaches this boundary, the
        original cue lane is not safe to resume for the current workout.
        """

        self._require_running()
        with self._request_lock:
            with self._state_lock:
                self._cue_callback_failed = True
            self._mark_realtime_unavailable_locked("SpeakerPlaybackError")

    def tick(self) -> int:
        """Expire stale cue work without creating a background thread."""

        self._require_running()
        with self._request_lock:
            if not self._is_realtime_available():
                return 0
            try:
                return self._cue_delivery.expire_stale()
            except Exception as exc:
                # Realtime failure must not stop deterministic pose processing.
                self._mark_realtime_unavailable_locked(type(exc).__name__)
                return 0

    def request_physical_stop(self) -> SessionEndSignal | None:
        """End through the same one-shot controller used by the model tool."""

        return self._end_controller.request_physical_stop()

    def close(self) -> None:
        """Treat an explicit launcher close as the physical stop boundary."""

        self.request_physical_stop()

    def _on_cue_delivery_failure(self, failure: CueDeliveryFailure) -> None:
        with self._request_lock:
            with self._state_lock:
                self._last_cue_failure = failure
                self._cue_callback_failed = True
            self._mark_realtime_unavailable_locked("CueDeliveryUnavailable")

    def _pause_after_cue_failure(self) -> None:
        if self._coordinator.current_mode is not SessionMode.ACTIVE_EXERCISE:
            return
        try:
            self._coordinator.transition_to(SessionMode.PAUSED)
        except Exception:
            # The coordinator publishes the fail-closed target in a finally
            # block even if an output boundary fails to preempt cleanly.
            pass

    def _on_session_end(self, _: SessionEndSignal) -> None:
        with self._request_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
            try:
                self._coordinator.transition_to(SessionMode.STOPPED)
            finally:
                try:
                    self._cue_delivery.close()
                finally:
                    self._realtime_session.close()

    def _events_are_consistent_locked(self, analysis: SquatAnalysis) -> bool:
        rep_events = tuple(
            event for event in analysis.events if event.event_type is SquatEventType.REP_COMPLETED
        )
        arm_events = tuple(
            event for event in analysis.events if event.event_type is SquatEventType.ARMS_NOT_IN_T
        )
        if len(rep_events) > 1 or len(arm_events) > 1:
            return False
        if analysis.rep_count < self._last_observed_rep_count:
            return False
        if analysis.rep_count == self._last_observed_rep_count:
            if rep_events:
                return False
        elif analysis.rep_count == self._last_observed_rep_count + 1:
            if len(rep_events) != 1 or rep_events[0].rep_count != analysis.rep_count:
                return False
        else:
            return False
        if arm_events and analysis.arms_in_t is not False:
            return False
        self._last_observed_rep_count = analysis.rep_count
        return True

    def _defer_cue_decision_locked(self, decision: GuardianDecision) -> None:
        cue_id = CueId(decision.cue_id or "")
        if cue_id in SQUAT_REP_CUE_IDS:
            self._deferred_cue_decisions = [
                existing
                for existing in self._deferred_cue_decisions
                if CueId(existing.cue_id or "") not in SQUAT_REP_CUE_IDS
            ]
        elif any(existing.cue_id == decision.cue_id for existing in self._deferred_cue_decisions):
            return
        self._deferred_cue_decisions.append(decision)

    def _control_response_matches_locked(self, result: RealtimeClientResult) -> bool:
        return (
            self._control_turn_inflight
            and self._control_response_id is not None
            and result.event.response_id == self._control_response_id
        )

    def _is_realtime_available(self) -> bool:
        with self._state_lock:
            return self._realtime_available

    def _mark_realtime_unavailable_locked(self, failure_kind: str) -> None:
        """Permanently disable cloud I/O while preserving the local session."""

        with self._state_lock:
            if not self._realtime_available:
                return
            self._realtime_available = False
            self._realtime_failure_kind = failure_kind
        self._control_turn_inflight = False
        self._control_response_id = None
        self._deferred_cue_decisions.clear()
        self._realtime_session.audio_gate.discard_all()
        self._realtime_session.current_response_id = None
        self._realtime_session.current_assistant_item_id = None
        mode = self._coordinator.current_mode
        if mode in {
            SessionMode.IDLE,
            SessionMode.CHECK_IN,
            SessionMode.ACTIVE_EXERCISE,
        }:
            try:
                self._coordinator.transition_to(SessionMode.PAUSED)
            except Exception:
                # The target mode is still published by the coordinator.
                pass

    def _finish_control_response_if_terminal(self, result: RealtimeClientResult) -> None:
        event = result.event
        with self._request_lock:
            if not self._control_turn_inflight:
                return
            if event.kind is ServerEventKind.RESPONSE_CREATED and result.response_authorized:
                self._control_response_id = event.response_id
                return
            if event.kind is ServerEventKind.ERROR or (
                event.kind is ServerEventKind.RESPONSE_DONE and self._control_response_id is None
            ):
                # A response-level error is not safely attributable, and a
                # terminal response before its creation is a protocol anomaly.
                # The same socket cannot safely recover FIFO correlation.
                self._mark_realtime_unavailable_locked("RealtimeProtocolOrderError")
                return
            elif (
                event.kind is not ServerEventKind.RESPONSE_DONE
                or event.response_id != self._control_response_id
            ):
                return
            else:
                self._control_turn_inflight = False
                self._control_response_id = None
                deferred = tuple(self._deferred_cue_decisions)
                self._deferred_cue_decisions.clear()
                for decision in deferred:
                    if self._coordinator.current_mode is not SessionMode.ACTIVE_EXERCISE:
                        break
                    try:
                        self._coordinator.apply_guardian_decision(decision)
                    except Exception as exc:
                        self._mark_realtime_unavailable_locked(type(exc).__name__)
                        break

    def _require_running(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("squat session is closed")
            if not self._started:
                raise RuntimeError("squat session has not been started")


__all__ = [
    "DEFAULT_SINGLE_CAMERA_SQUAT_PLAN",
    "SESSION_END_POLICY_INSTRUCTIONS",
    "SQUAT_EXERCISE_ID",
    "SQUAT_REP_CUE_IDS",
    "LaptopSquatSession",
    "SquatAudioTurnResult",
    "SquatSessionAnalysisResult",
    "SquatSessionDispatchResult",
    "build_single_camera_squat_plan",
    "local_cue_request_for_squat_event",
    "observation_from_squat_analysis",
]
