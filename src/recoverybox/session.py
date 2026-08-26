"""Edge-session composition for safety-gated voice and exercise behavior.

This module connects the domain-level :class:`GuardianDecision` to two narrow
audio boundaries:

* arbitrary conversational model audio can only be preempted here;
* an exercise cue request contains only a catalog-selected :class:`CueId` plus
  provenance.  A Realtime adapter resolves the phrase itself and may speak it
  only through transcript quarantine; callers cannot submit coaching text.

The coordinator owns the product-level :class:`SessionMode`.  Realtime code
reads that mode through :class:`SessionModeProvider` and applies its own wire
policy without gaining authority to change the exercise lifecycle.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from recoverybox.core import (
    ApprovedCueCatalog,
    CueId,
    CueKind,
    Guardian,
    GuardianAction,
    GuardianDecision,
    GuardianReason,
    GuardianRuntimeFault,
    SessionMode,
)
from recoverybox.core.cues import SQUAT_SCRIPTED_SESSION_CUE_IDS
from recoverybox.session_end import SessionEndSignal


class SessionCompositionError(RuntimeError):
    """A local mode, cue, or preemption boundary could not be enforced."""


class CueAuthorizationError(SessionCompositionError):
    """A Guardian cue decision did not resolve to an approved prompt cue."""


DEFAULT_CUE_CATALOG_VERSION = "prompt-cues-v3"


@dataclass(frozen=True, slots=True)
class ApprovedCuePlaybackAuthorization:
    """Capability-like request for one fixed, prompt-delivered cue.

    A Realtime output adapter must revalidate the provenance, resolve the
    phrase from the same immutable catalog, place it in response instructions,
    and release generated audio only after its completed transcript matches.
    Free-form text and PCM are deliberately absent so callers cannot bypass
    that gate.
    """

    cue_id: CueId
    cue_kind: CueKind
    catalog_version: str
    guardian_rule_version: str
    reason_codes: tuple[GuardianReason, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cue_id, CueId):
            raise TypeError("cue_id must be a CueId")
        if not isinstance(self.cue_kind, CueKind):
            raise TypeError("cue_kind must be a CueKind")
        if not self.catalog_version.strip():
            raise ValueError("catalog_version must not be empty")
        if not self.guardian_rule_version.strip():
            raise ValueError("guardian_rule_version must not be empty")
        if not self.reason_codes or not all(
            isinstance(reason, GuardianReason) for reason in self.reason_codes
        ):
            raise ValueError("reason_codes must contain GuardianReason values")


@runtime_checkable
class SessionModeProvider(Protocol):
    """Read-only view of the coordinator-owned product mode."""

    @property
    def current_mode(self) -> SessionMode: ...


@runtime_checkable
class ModelAudioPreemptionPort(Protocol):
    """Close a model-audio gate and cancel any model response in progress."""

    def preempt_model_audio(self) -> None: ...


@runtime_checkable
class ApprovedCuePlaybackPort(Protocol):
    """Exclusive prompt-cue lane accepting catalog-derived capabilities."""

    def preempt_model_audio(self) -> None:
        """Stop queued/audible model output before an exercise-state change."""

    def play_approved_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None:
        """Request the exact authorized phrase through the gated speech path."""


@runtime_checkable
class SessionEndAuthorityPort(Protocol):
    """Verify termination signals from one exact session-end controller."""

    def issued(self, signal: object) -> bool: ...


@dataclass(frozen=True, slots=True)
class GuardianEscalationRecord:
    """Non-authoritative audit record of one applied Guardian escalation."""

    guardian_rule_version: str
    reason_codes: tuple[GuardianReason, ...]
    guardian_sequence: int


@runtime_checkable
class EmergencyEscalationPort(Protocol):
    """Local side effect accepting the sealed Guardian verdict itself."""

    def request_emergency_escalation(
        self,
        decision: GuardianDecision,
    ) -> None: ...


class _BoundGuardianEscalationAudit:
    """Default production sink retaining sealed escalation provenance."""

    def __init__(self, guardian: Guardian) -> None:
        self._guardian = guardian
        self._decisions: list[GuardianDecision] = []
        self._lock = threading.Lock()

    def request_emergency_escalation(self, decision: GuardianDecision) -> None:
        if not self._guardian.issued(decision) or decision.action is not GuardianAction.ESCALATE:
            raise TypeError("escalation requires the bound Guardian's sealed verdict")
        with self._lock:
            self._decisions.append(decision)


@dataclass(frozen=True, slots=True)
class GuardianDecisionEffect:
    """Observable result of applying one Guardian decision at the edge."""

    previous_mode: SessionMode
    current_mode: SessionMode
    action: GuardianAction
    cue_authorization: ApprovedCuePlaybackAuthorization | None = None
    escalation_record: GuardianEscalationRecord | None = None
    model_audio_preempted: bool = False


class SessionCoordinator(SessionModeProvider):
    """Own session mode and apply Guardian decisions to safe output ports.

    ``SessionMode.COMPLETE`` is the post-session conversation phase at the
    Realtime boundary.  Every other non-check-in mode is fail-closed to
    arbitrary model audio.  Active exercise has one narrower exception: an
    exact catalog phrase selected through a Guardian ``CUE`` decision.
    """

    DEFAULT_CATALOG_VERSION = DEFAULT_CUE_CATALOG_VERSION

    def __init__(
        self,
        *,
        guardian: Guardian,
        cue_playback: ApprovedCuePlaybackPort,
        session_end_authority: SessionEndAuthorityPort,
        escalation_port: EmergencyEscalationPort | None = None,
        cue_catalog: ApprovedCueCatalog | None = None,
        catalog_version: str = DEFAULT_CATALOG_VERSION,
    ) -> None:
        if not isinstance(guardian, Guardian):
            raise TypeError("guardian must be a Guardian")
        if not isinstance(session_end_authority, SessionEndAuthorityPort):
            raise TypeError("session_end_authority must verify one end controller")
        if escalation_port is not None and not isinstance(
            escalation_port,
            EmergencyEscalationPort,
        ):
            raise TypeError("escalation_port must implement EmergencyEscalationPort")
        approved_catalog = guardian.cue_catalog if cue_catalog is None else cue_catalog
        if approved_catalog is not guardian.cue_catalog:
            raise ValueError("cue_catalog must be the bound Guardian's catalog")
        if not catalog_version.strip():
            raise ValueError("catalog_version must not be empty")
        self._guardian = guardian
        self._cue_playback = cue_playback
        self._session_end_authority = session_end_authority
        self._escalation_port = escalation_port or _BoundGuardianEscalationAudit(guardian)
        self._cue_catalog = approved_catalog
        self._catalog_version = catalog_version.strip()
        self._mode = SessionMode.IDLE
        self._model_audio_preemptors: list[ModelAudioPreemptionPort] = []
        self._last_guardian_action: GuardianAction | None = None
        self._last_escalation_record: GuardianEscalationRecord | None = None
        self._termination_signal: SessionEndSignal | None = None
        # Coordinator operations must be linearly ordered without making mode
        # reads wait on output ports.  In particular, cue delivery holds its
        # own lock while it performs the final mode check and speaker handoff.
        # A restricted transition may wait for that lock during synchronous
        # preemption, so ``current_mode`` must never acquire this operation
        # lock or a lock cycle would be possible.
        self._operation_lock = threading.RLock()
        self._lock = threading.RLock()

    @property
    def current_mode(self) -> SessionMode:
        with self._lock:
            return self._mode

    @property
    def cue_catalog(self) -> ApprovedCueCatalog:
        return self._cue_catalog

    @property
    def catalog_version(self) -> str:
        return self._catalog_version

    @property
    def last_guardian_action(self) -> GuardianAction | None:
        """Last sealed Guardian action applied, including ESCALATE."""

        with self._lock:
            return self._last_guardian_action

    @property
    def termination_signal(self) -> SessionEndSignal | None:
        """Typed lifecycle end, kept separate from Guardian STOP/ESCALATE."""

        with self._lock:
            return self._termination_signal

    @property
    def last_escalation_record(self) -> GuardianEscalationRecord | None:
        """Last retained ESCALATE audit record, independent of STOPPED mode."""

        with self._lock:
            return self._last_escalation_record

    def register_model_audio_preemptor(
        self,
        preemptor: ModelAudioPreemptionPort,
    ) -> None:
        """Register an additional gate, such as the Realtime adapter.

        The cue-playback port is always preempted.  Extra ports let composition
        close the network/audio gate in the same deterministic operation.
        """

        if not isinstance(preemptor, ModelAudioPreemptionPort):
            raise TypeError("preemptor must implement ModelAudioPreemptionPort")
        with self._operation_lock, self._lock:
            if all(existing is not preemptor for existing in self._model_audio_preemptors):
                self._model_audio_preemptors.append(preemptor)

    def _transition_to(self, mode: SessionMode) -> SessionMode:
        """Set product mode, preempting before publishing a restricted mode.

        The prompt-cue port's synchronous preemption is mutually exclusive
        with its final PCM handoff.  Waiting for that boundary before changing
        ``current_mode`` therefore guarantees that no cue accepted under the
        previous mode remains queued or audible when the restricted mode first
        becomes observable.  A preemption error still publishes the target
        fail-closed mode before the error is surfaced.
        """

        if not isinstance(mode, SessionMode):
            raise TypeError("mode must be a SessionMode")
        if mode is SessionMode.ACTIVE_EXERCISE:
            raise SessionCompositionError(
                "ACTIVE_EXERCISE requires a successful Guardian-authorized activation"
            )
        with self._operation_lock:
            with self._lock:
                previous = self._mode

            if session_mode_allows_model_audio(mode):
                with self._lock:
                    self._mode = mode
                return previous

            try:
                # Do not hold the mode lock here.  Output ports are allowed to
                # read current_mode while synchronously draining their lanes.
                self._preempt_model_audio()
            finally:
                with self._lock:
                    self._mode = mode
            return previous

    def enter_check_in(self) -> SessionMode:
        """Enter the conversational check-in phase from IDLE only."""

        with self._operation_lock:
            if self.current_mode is not SessionMode.IDLE:
                raise SessionCompositionError("check-in requires IDLE mode")
            return self._transition_to(SessionMode.CHECK_IN)

    def activate_after_guardian_continue(
        self,
        decision: GuardianDecision,
    ) -> SessionMode:
        """Enter or resume ACTIVE only with this Guardian's CONTINUE verdict."""

        self._require_guardian_decision(decision)
        if decision.action is not GuardianAction.CONTINUE:
            raise SessionCompositionError("exercise activation requires Guardian CONTINUE")
        with self._operation_lock:
            if not self._guardian.consume_activation(decision):
                raise SessionCompositionError(
                    "exercise activation requires the latest unused Guardian CONTINUE"
                )
            previous = self.current_mode
            if previous not in {
                SessionMode.IDLE,
                SessionMode.CHECK_IN,
                SessionMode.PAUSED,
            }:
                raise SessionCompositionError(
                    "exercise activation requires IDLE, CHECK_IN, or PAUSED mode"
                )
            with self._lock:
                self._last_guardian_action = decision.action
            if previous is SessionMode.CHECK_IN:
                return self._activate_from_check_in_locked()
            return self._activate_from_idle_or_pause_locked()

    def _activate_from_idle_or_pause_locked(self) -> SessionMode:
        """Preempt first and publish ACTIVE only after every boundary succeeds."""

        previous = self.current_mode
        try:
            self._preempt_model_audio()
        except Exception as exc:
            try:
                self._apply_runtime_fault_locked(GuardianRuntimeFault.RUNTIME_BOUNDARY_FAILURE)
            except Exception:
                pass
            raise SessionCompositionError(
                "model audio preemption failed during exercise activation"
            ) from exc
        with self._lock:
            self._mode = SessionMode.ACTIVE_EXERCISE
        return previous

    def _activate_from_check_in_locked(self) -> SessionMode:
        """Enter ACTIVE while preserving only the exact prompt-cue lane.

        A CHECK_IN welcome or detection cue may still be in the quarantined
        lane when the first assessable standing pose arrives.  This transition
        closes every separately registered arbitrary-model audio gate, but does
        not cancel those already Guardian-authorized fixed cues.  Any failure
        publishes PAUSED and scrubs the cue lane fail-closed.
        """

        with self._operation_lock:
            if self.current_mode is not SessionMode.CHECK_IN:
                raise SessionCompositionError("scripted exercise activation requires CHECK_IN mode")
            with self._lock:
                preemptors = tuple(self._model_audio_preemptors)
            try:
                for preemptor in preemptors:
                    preemptor.preempt_model_audio()
            except Exception as exc:
                try:
                    self._apply_runtime_fault_locked(GuardianRuntimeFault.RUNTIME_BOUNDARY_FAILURE)
                except Exception:
                    pass
                raise SessionCompositionError(
                    "arbitrary model audio preemption failed during exercise activation"
                ) from exc
            with self._lock:
                self._mode = SessionMode.ACTIVE_EXERCISE
            return SessionMode.CHECK_IN

    def apply_guardian_decision(
        self,
        decision: GuardianDecision,
    ) -> GuardianDecisionEffect:
        """Apply a deterministic Guardian result during an exercise.

        ``CONTINUE`` preserves ACTIVE_EXERCISE. ``CUE`` resolves to a typed,
        catalog-approved prompt capability. ``PAUSE`` enters PAUSED, while
        ``STOP`` and ``ESCALATE`` enter STOPPED.  The three cautionary actions
        synchronously preempt both the speaker lane and registered model gates.
        """

        self._require_guardian_decision(decision)
        with self._operation_lock:
            return self._apply_guardian_decision(decision)

    def apply_runtime_fault(
        self,
        fault: GuardianRuntimeFault,
    ) -> GuardianDecisionEffect:
        """Ask the bound Guardian to arbitrate one typed runtime fault."""

        if not isinstance(fault, GuardianRuntimeFault):
            raise TypeError("fault must be a GuardianRuntimeFault")
        with self._operation_lock:
            return self._apply_runtime_fault_locked(fault)

    def apply_session_end(self, signal: SessionEndSignal) -> SessionMode:
        """Apply one sealed lifecycle termination without forging Guardian STOP."""

        if not self._session_end_authority.issued(signal):
            raise TypeError("signal must be issued by the bound SessionEndController")
        with self._operation_lock:
            with self._lock:
                previous_signal = self._termination_signal
            if previous_signal is not None:
                if previous_signal is signal:
                    return self.current_mode
                raise SessionCompositionError("session already has a different termination")
            try:
                previous = self._transition_to(SessionMode.STOPPED)
            finally:
                # _transition_to publishes STOPPED in its own finally block.
                # Commit the already-minted one-shot source even when output
                # preemption reports a cleanup failure afterward.
                with self._lock:
                    self._termination_signal = signal
            return previous

    def _apply_runtime_fault_locked(
        self,
        fault: GuardianRuntimeFault,
    ) -> GuardianDecisionEffect:
        decision = self._guardian.decide_runtime_fault(fault)
        return self._apply_guardian_decision(decision, allow_idle=True)

    def _require_guardian_decision(self, decision: object) -> None:
        if not isinstance(decision, GuardianDecision) or not self._guardian.issued(decision):
            raise TypeError("decision must be issued by the bound Guardian")

    def _apply_guardian_decision(
        self,
        decision: GuardianDecision,
        *,
        allow_idle: bool = False,
    ) -> GuardianDecisionEffect:
        """Apply one already-validated decision under the operation lock."""

        previous = self.current_mode
        allowed_modes = {
            SessionMode.CHECK_IN,
            SessionMode.ACTIVE_EXERCISE,
            SessionMode.PAUSED,
        }
        if allow_idle:
            allowed_modes.add(SessionMode.IDLE)
        if previous not in allowed_modes:
            raise SessionCompositionError(
                "Guardian decisions require CHECK_IN, ACTIVE_EXERCISE, or PAUSED mode"
            )

        with self._lock:
            self._last_guardian_action = decision.action

        if decision.action is GuardianAction.CONTINUE:
            # A safe observation never silently resumes a prior pause.
            return GuardianDecisionEffect(
                previous_mode=previous,
                current_mode=previous,
                action=decision.action,
            )

        if decision.action is GuardianAction.CUE:
            if previous is SessionMode.CHECK_IN:
                if decision.cue_id not in SQUAT_SCRIPTED_SESSION_CUE_IDS:
                    self._apply_runtime_fault_locked(GuardianRuntimeFault.RUNTIME_BOUNDARY_FAILURE)
                    raise CueAuthorizationError("only reviewed scripted cues can play in CHECK_IN")
            elif previous is not SessionMode.ACTIVE_EXERCISE:
                raise CueAuthorizationError(
                    "approved cues can play only in CHECK_IN or ACTIVE_EXERCISE"
                )
            try:
                authorization = self._authorize_cue(decision)
            except (CueAuthorizationError, KeyError, TypeError, ValueError) as exc:
                self._apply_runtime_fault_locked(GuardianRuntimeFault.RUNTIME_BOUNDARY_FAILURE)
                raise CueAuthorizationError("Guardian cue is not in the approved catalog") from exc

            try:
                self._cue_playback.play_approved_cue(authorization)
            except Exception:
                # Speaker/cue resolution failures make the movement
                # non-coachable until the local runtime explicitly resumes.
                self._apply_runtime_fault_locked(GuardianRuntimeFault.CUE_DELIVERY_UNAVAILABLE)
                raise
            return GuardianDecisionEffect(
                previous_mode=previous,
                current_mode=previous,
                action=decision.action,
                cue_authorization=authorization,
                model_audio_preempted=False,
            )

        if decision.action is GuardianAction.PAUSE:
            target = SessionMode.PAUSED
        elif decision.action in {GuardianAction.STOP, GuardianAction.ESCALATE}:
            target = SessionMode.STOPPED
        else:  # Defensive if an invalid enum-like value crosses a Python boundary.
            raise SessionCompositionError("unsupported Guardian action")

        escalation: GuardianEscalationRecord | None = None
        transition_error: Exception | None = None
        try:
            self._transition_to(target)
        except Exception as exc:
            transition_error = exc
        escalation_error: Exception | None = None
        if decision.action is GuardianAction.ESCALATE:
            escalation = GuardianEscalationRecord(
                guardian_rule_version=decision.rule_version,
                reason_codes=decision.reason_codes,
                guardian_sequence=decision.sequence,
            )
            with self._lock:
                self._last_escalation_record = escalation
            try:
                self._escalation_port.request_emergency_escalation(decision)
            except Exception as exc:
                escalation_error = exc
        if escalation_error is not None:
            raise SessionCompositionError(
                "emergency escalation side effect failed after safe stop"
            ) from escalation_error
        if transition_error is not None:
            raise transition_error
        return GuardianDecisionEffect(
            previous_mode=previous,
            current_mode=target,
            action=decision.action,
            escalation_record=escalation,
            model_audio_preempted=True,
        )

    def _authorize_cue(
        self,
        decision: GuardianDecision,
    ) -> ApprovedCuePlaybackAuthorization:
        if GuardianReason.LOCAL_CUE_ACCEPTED not in decision.reason_codes:
            raise CueAuthorizationError("cue decision lacks Guardian authorization provenance")
        cue_id = CueId(decision.cue_id or "")
        cue = self._cue_catalog[cue_id.value]
        if not self._cue_catalog.is_approved(cue_id.value):
            raise CueAuthorizationError("cue is not approved")
        return ApprovedCuePlaybackAuthorization(
            cue_id=cue_id,
            cue_kind=cue.kind,
            catalog_version=self._catalog_version,
            guardian_rule_version=decision.rule_version,
            reason_codes=decision.reason_codes,
        )

    def _preempt_model_audio(self) -> None:
        with self._lock:
            preemptors = tuple(self._model_audio_preemptors)

        failures: list[Exception] = []
        try:
            self._cue_playback.preempt_model_audio()
        except Exception as exc:  # keep closing every independent gate
            failures.append(exc)
        for preemptor in preemptors:
            try:
                preemptor.preempt_model_audio()
            except Exception as exc:  # keep closing every independent gate
                failures.append(exc)
        if failures:
            raise SessionCompositionError(
                f"model audio preemption failed at {len(failures)} local boundary(s)"
            ) from failures[0]


def session_mode_allows_model_audio(mode: SessionMode) -> bool:
    """Return whether the product phase permits arbitrary conversational audio.

    An ACTIVE_EXERCISE prompt cue is a separate, exact-text capability and does
    not make this predicate true.
    """

    if not isinstance(mode, SessionMode):
        raise TypeError("mode must be a SessionMode")
    return mode in {SessionMode.CHECK_IN, SessionMode.COMPLETE}
