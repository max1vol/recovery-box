from __future__ import annotations

from pathlib import Path

from recoverybox import cli
from recoverybox.core import CueId, GuardianReason
from recoverybox.realtime_verify_cli import build_simulated_squat_authorizations


def test_simulated_trials_cross_guardian_as_typed_catalog_authorizations() -> None:
    authorizations = build_simulated_squat_authorizations()

    assert [authorization.cue_id for authorization in authorizations] == [
        CueId.SQUAT_REP_ONE,
        CueId.SQUAT_REP_TWO,
        CueId.ARMS_T_SHAPE,
    ]
    assert all(
        authorization.reason_codes == (GuardianReason.LOCAL_CUE_ACCEPTED,)
        for authorization in authorizations
    )
    assert all(not hasattr(authorization, "spoken_text") for authorization in authorizations)


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
