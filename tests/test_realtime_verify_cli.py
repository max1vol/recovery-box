from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from recoverybox import cli, realtime_verify_cli
from recoverybox.core import DEFAULT_CUE_CATALOG, CueId, GuardianReason
from recoverybox.realtime import MemoryTransport
from recoverybox.realtime_verify_cli import (
    _SIMULATED_EVENTS,
    build_simulated_squat_authorizations,
)


def test_simulated_trials_cross_guardian_as_typed_catalog_authorizations() -> None:
    authorizations = build_simulated_squat_authorizations()

    assert [authorization.cue_id for authorization in authorizations] == [
        CueId.SQUAT_SET_INTRO,
        CueId.SQUAT_PERSON_DETECTED,
        CueId.SQUAT_REP_ONE,
        CueId.SQUAT_REP_TWO,
        CueId.SQUAT_REP_THREE,
    ]
    assert all(
        authorization.reason_codes == (GuardianReason.LOCAL_CUE_ACCEPTED,)
        for authorization in authorizations
    )
    assert all(not hasattr(authorization, "spoken_text") for authorization in authorizations)


def test_simulated_event_names_are_closed_and_content_free() -> None:
    assert _SIMULATED_EVENTS == (
        "script_intro_requested",
        "first_assessable_stand",
        "squat_rep_completed_1",
        "squat_rep_completed_2",
        "squat_rep_completed_3",
    )
    assert all("max" not in event_name for event_name in _SIMULATED_EVENTS)
    assert all("phrase" not in event_name for event_name in _SIMULATED_EVENTS)


def test_live_verifier_reports_all_five_events_without_content(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    transport = MemoryTransport()
    seen_cue_ids: list[str] = []

    class FakeVerifier:
        def __init__(self, **_kwargs: object) -> None:
            return

        def verify_many(self, requests: object) -> tuple[SimpleNamespace, ...]:
            reports: list[SimpleNamespace] = []
            for request in requests:  # type: ignore[union-attr]
                cue_id = request.authorization.cue_id.value
                seen_cue_ids.append(cue_id)
                reports.append(
                    SimpleNamespace(
                        quarantine_released=True,
                        to_dict=lambda cue_id=cue_id: {"cue_id": cue_id},
                    )
                )
            return tuple(reports)

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr(
        realtime_verify_cli.WebSocketJsonTransport,
        "connect",
        lambda **_kwargs: transport,
    )
    monkeypatch.setattr(realtime_verify_cli, "RealtimeCueVerifier", FakeVerifier)

    assert (
        realtime_verify_cli.run_live_realtime_verification(
            output_dir=tmp_path,
            voice="marin",
            skip_asr=True,
        )
        == 0
    )

    report_text = (tmp_path / "report.json").read_text()
    payload = json.loads(report_text)
    assert payload["simulated_events"] == list(_SIMULATED_EVENTS)
    assert seen_cue_ids == [
        cue.value
        for cue in (
            CueId.SQUAT_SET_INTRO,
            CueId.SQUAT_PERSON_DETECTED,
            CueId.SQUAT_REP_ONE,
            CueId.SQUAT_REP_TWO,
            CueId.SQUAT_REP_THREE,
        )
    ]
    assert len(payload["reports"]) == 5
    captured = capsys.readouterr()
    for cue_id in seen_cue_ids:
        spoken_text = DEFAULT_CUE_CATALOG[cue_id].spoken_text
        assert spoken_text not in report_text
        assert spoken_text not in captured.out
    assert captured.err == ""


def test_download_pose_model_command_is_explicit(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "pose.task"
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "install_pose_model",
        lambda path: calls.append(path) or destination,
    )

    assert cli.main(["download-pose-model", "--output", str(destination)]) == 0
    assert calls == [str(destination)]


def test_live_verification_command_forwards_options_without_running_network(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "run_live_realtime_verification",
        lambda **kwargs: calls.append(kwargs) or 7,
    )

    result = cli.main(
        [
            "verify-realtime-cues",
            "--output-dir",
            str(tmp_path),
            "--voice",
            "marin",
            "--asr-model",
            "gpt-transcribe",
            "--skip-asr",
        ]
    )

    assert result == 7
    assert calls == [
        {
            "output_dir": str(tmp_path),
            "voice": "marin",
            "asr_model": "gpt-transcribe",
            "skip_asr": True,
        }
    ]


def test_live_verification_command_never_prints_exception_text(
    monkeypatch,
    capsys,
) -> None:
    secret = "credential-and-provider-detail-must-not-escape"

    def fail_verification(**_kwargs: object) -> int:
        raise RuntimeError(secret)

    monkeypatch.setattr(cli, "run_live_realtime_verification", fail_verification)

    assert cli.main(["verify-realtime-cues"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: live Realtime verification failed (RuntimeError)\n"
    assert secret not in captured.err
