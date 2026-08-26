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
    DEFAULT_CUE_CATALOG,
    ApprovedCueCatalog,
    CueId,
    CueKind,
    GuardianAction,
    GuardianDecision,
    GuardianReason,
    SessionMode,
)


class SessionCompositionError(RuntimeError):
    """A local mode, cue, or preemption boundary could not be enforced."""


class CueAuthorizationError(SessionCompositionError):
    """A Guardian cue decision did not resolve to an approved prompt cue."""


DEFAULT_CUE_CATALOG_VERSION = "prompt-cues-v2"


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


@dataclass(frozen=True, slots=True)
class GuardianDecisionEffect:
    """Observable result of applying one Guardian decision at the edge."""

    previous_mode: SessionMode
    current_mode: SessionMode
    action: GuardianAction
    cue_authorization: ApprovedCuePlaybackAuthorization | None = None
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
        cue_playback: ApprovedCuePlaybackPort,
        cue_catalog: ApprovedCueCatalog = DEFAULT_CUE_CATALOG,
        catalog_version: str = DEFAULT_CATALOG_VERSION,
        initial_mode: SessionMode = SessionMode.IDLE,
    ) -> None:
        if not isinstance(initial_mode, SessionMode):
            raise TypeError("initial_mode must be a SessionMode")
        if not catalog_version.strip():
            raise ValueError("catalog_version must not be empty")
        self._cue_playback = cue_playback
        self._cue_catalog = cue_catalog
        self._catalog_version = catalog_version.strip()
        self._mode = initial_mode
        self._model_audio_preemptors: list[ModelAudioPreemptionPort] = []
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

    def transition_to(self, mode: SessionMode) -> SessionMode:
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

        if not isinstance(decision, GuardianDecision):
            raise TypeError("decision must be a GuardianDecision")
        with self._operation_lock:
            return self._apply_guardian_decision(decision)

    def _apply_guardian_decision(
        self,
        decision: GuardianDecision,
    ) -> GuardianDecisionEffect:
        """Apply one already-validated decision under the operation lock."""

        previous = self.current_mode
        if previous not in {SessionMode.ACTIVE_EXERCISE, SessionMode.PAUSED}:
            raise SessionCompositionError(
                "Guardian exercise decisions require ACTIVE_EXERCISE or PAUSED mode"
            )

        if decision.action is GuardianAction.CONTINUE:
            # A safe observation never silently resumes a prior pause.
            return GuardianDecisionEffect(
                previous_mode=previous,
                current_mode=previous,
                action=decision.action,
            )

        if decision.action is GuardianAction.CUE:
            if previous is not SessionMode.ACTIVE_EXERCISE:
                raise CueAuthorizationError("approved cues can play only in ACTIVE_EXERCISE")
            try:
                authorization = self._authorize_cue(decision)
            except (KeyError, TypeError, ValueError) as exc:
                self.transition_to(SessionMode.PAUSED)
                raise CueAuthorizationError("Guardian cue is not in the approved catalog") from exc

            try:
                self._cue_playback.play_approved_cue(authorization)
            except Exception:
                # Speaker/cue resolution failures make the movement
                # non-coachable until the local runtime explicitly resumes.
                self.transition_to(SessionMode.PAUSED)
                raise
            return GuardianDecisionEffect(
                previous_mode=previous,
                current_mode=SessionMode.ACTIVE_EXERCISE,
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

        self.transition_to(target)
        return GuardianDecisionEffect(
            previous_mode=previous,
            current_mode=target,
            action=decision.action,
            model_audio_preempted=True,
        )

    def _authorize_cue(
        self,
        decision: GuardianDecision,
    ) -> ApprovedCuePlaybackAuthorization:
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
