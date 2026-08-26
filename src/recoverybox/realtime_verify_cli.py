"""Named live verification launcher for prompt-cue latency and ASR checks."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from recoverybox.core import (
    CueId,
    ExercisePlan,
    Guardian,
    LocalCueRequest,
    MovementObservation,
)
from recoverybox.realtime import RealtimeSession, SessionEndController, WebSocketJsonTransport
from recoverybox.realtime.session_control import SESSION_CONTROL_TOOL_REGISTRY
from recoverybox.realtime.verification import (
    CueVerificationRequest,
    OpenAITranscriptionVerifier,
    RealtimeCueVerifier,
)
from recoverybox.session import ApprovedCuePlaybackAuthorization, SessionCoordinator

_VERIFICATION_CUES = (
    CueId.SQUAT_SET_INTRO,
    CueId.SQUAT_PERSON_DETECTED,
    CueId.SQUAT_REP_ONE,
    CueId.SQUAT_REP_TWO,
    CueId.SQUAT_REP_THREE,
)

_SIMULATED_EVENTS = (
    "script_intro_requested",
    "first_assessable_stand",
    "squat_rep_completed_1",
    "squat_rep_completed_2",
    "squat_rep_completed_3",
)

_SESSION_INSTRUCTIONS = """
You are the voice connection for one RecoveryBox workout session. Keep the
session open across all turns and exercise cue responses. Call finish_session
with an empty object only after the user unambiguously asks to end, leave, or
says goodbye. Never call it for silence, a completed repetition or set, a
pause, a network problem, or wording such as "finish this set". During an
active exercise, ordinary speech is blocked by local code; only isolated
Guardian-authorized prompt cues may be spoken.
"""


class _AuthorizationCollector:
    def __init__(self) -> None:
        self.authorizations: list[ApprovedCuePlaybackAuthorization] = []

    def preempt_model_audio(self) -> None:
        return

    def play_approved_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None:
        self.authorizations.append(authorization)


def build_simulated_squat_authorizations() -> tuple[ApprovedCuePlaybackAuthorization, ...]:
    """Run the five-stage squat script through its real Guardian boundaries."""

    guardian = Guardian()
    plan = ExercisePlan(
        exercise_id="squat",
        allowed_cue_ids=frozenset(cue.value for cue in _VERIFICATION_CUES),
        target_reps=3,
        min_confidence=0.7,
        max_pose_age_ms=500,
        required_camera_views=1,
    )
    collector = _AuthorizationCollector()
    end_controller = SessionEndController()
    coordinator = SessionCoordinator(
        guardian=guardian,
        cue_playback=collector,
        session_end_authority=end_controller,
    )
    coordinator.enter_check_in()

    intro = guardian.decide_scripted_session_cue(
        LocalCueRequest(CueId.SQUAT_SET_INTRO.value),
        plan,
    )
    coordinator.apply_guardian_decision(intro)

    first_stand = MovementObservation(
        exercise_id=plan.exercise_id,
        timestamp_ms=1_000,
        confidence=0.98,
        camera_view_count=1,
        camera_disagreement_degrees=None,
        pose_age_ms=10,
        rep_index=0,
        phase="standing",
    )
    stand_decision = guardian.decide(first_stand, plan)
    coordinator.apply_guardian_decision(stand_decision)
    detected = guardian.decide_scripted_session_cue(
        LocalCueRequest(CueId.SQUAT_PERSON_DETECTED.value),
        plan,
    )
    coordinator.apply_guardian_decision(detected)
    activation_decision = guardian.decide(first_stand, plan)
    coordinator.activate_after_guardian_continue(activation_decision)

    for rep_index, cue_id in enumerate(
        (
            CueId.SQUAT_REP_ONE,
            CueId.SQUAT_REP_TWO,
            CueId.SQUAT_REP_THREE,
        ),
        start=1,
    ):
        observation = MovementObservation(
            exercise_id=plan.exercise_id,
            timestamp_ms=(rep_index + 1) * 1_000,
            confidence=0.98,
            camera_view_count=1,
            camera_disagreement_degrees=None,
            pose_age_ms=10,
            rep_index=rep_index,
            phase="standing",
        )
        decision = guardian.decide(
            observation,
            plan,
            local_cue_request=LocalCueRequest(cue_id.value),
        )
        coordinator.apply_guardian_decision(decision)

    return tuple(collector.authorizations)


def run_live_realtime_verification(
    *,
    output_dir: str | Path,
    voice: str,
    asr_model: str = "whisper-1",
    skip_asr: bool = False,
) -> int:
    """Verify several cues on one real Realtime connection and write evidence."""

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for live Realtime verification")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    transport = WebSocketJsonTransport.connect(api_key=api_key, timeout_seconds=30.0)
    session = RealtimeSession(
        transport=transport,
        tools=SESSION_CONTROL_TOOL_REGISTRY,
    )
    session.configure(instructions=_SESSION_INSTRUCTIONS, voice=voice)
    asr = None
    if not skip_asr:
        asr = OpenAITranscriptionVerifier(
            api_key=api_key,
            model=asr_model,
            include_word_timestamps=asr_model == "whisper-1",
            include_transcript_text=False,
        )
    verifier = RealtimeCueVerifier(
        session=session,
        transport=transport,
        asr_verifier=asr,
        retain_realtime_transcript=False,
    )
    authorizations = build_simulated_squat_authorizations()
    requests = tuple(
        CueVerificationRequest(
            authorization=authorization,
            wav_path=destination / f"{index:02d}-{authorization.cue_id.value}.wav",
        )
        for index, authorization in enumerate(authorizations, start=1)
    )

    try:
        reports = verifier.verify_many(requests)
    finally:
        session.close()

    payload = {
        "connection_lifetime": "one Realtime WebSocket reused for every cue",
        "model": "gpt-realtime-2.1",
        "voice": voice,
        "simulated_events": list(_SIMULATED_EVENTS),
        "reports": [report.to_dict() for report in reports],
    }
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({**payload, "report_path": str(report_path)}, indent=2))
    return 0 if all(report.quarantine_released for report in reports) else 1


def parse_verification_cues(values: Sequence[str]) -> tuple[CueId, ...]:
    """Small testable parser reserved for future selectable live trials."""

    return tuple(CueId(value) for value in values)


__all__ = [
    "build_simulated_squat_authorizations",
    "parse_verification_cues",
    "run_live_realtime_verification",
]
