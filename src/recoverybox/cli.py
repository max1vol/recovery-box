"""Small command-line surface for local verification and the demo."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from recoverybox.config import ConfigurationError, Settings
from recoverybox.demo import demo_as_dicts
from recoverybox.laptop import DEFAULT_POSE_MODEL_PATH, install_pose_model
from recoverybox.laptop.doctor import collect_laptop_doctor_report
from recoverybox.laptop.squat_launcher import run_squat_demo
from recoverybox.realtime_verify_cli import run_live_realtime_verification
from recoverybox.seed import seed_flower_demo
from recoverybox.voice_cli import run_voice_checkin


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recoverybox",
        description="RecoveryBox edge and federated-learning prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="check local configuration without revealing secrets"
    )
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="fail when physical-device or cloud prerequisites are absent",
    )
    subparsers.add_parser("demo-safety", help="run a deterministic camera/Guardian demonstration")
    subparsers.add_parser(
        "voice-checkin",
        help="run a developer-only ALSA + Realtime check-in (not active-exercise use)",
    )
    seed = subparsers.add_parser(
        "seed-flower-demo", help="write three synthetic sanitized feature stores"
    )
    seed.add_argument("--output-dir", default="data/demo")
    seed.add_argument("--rows-per-client", type=int, default=30)
    pose_model = subparsers.add_parser(
        "download-pose-model",
        help="explicitly download and verify the pinned MediaPipe pose model",
    )
    pose_model.add_argument("--output", default=str(DEFAULT_POSE_MODEL_PATH))
    verify = subparsers.add_parser(
        "verify-realtime-cues",
        help="run five scripted Realtime cue latency + ASR trials on one connection",
    )
    verify.add_argument("--output-dir", default="artifacts/realtime-verification")
    verify.add_argument("--voice", default="marin")
    verify.add_argument(
        "--asr-model",
        choices=("whisper-1", "gpt-transcribe"),
        default="whisper-1",
    )
    verify.add_argument("--skip-asr", action="store_true")
    squat = subparsers.add_parser(
        "squat-demo",
        help="run the named one-camera MediaPipe + Realtime squat integration",
    )
    squat.add_argument("--camera-index", type=int)
    squat.add_argument("--model-path")
    squat.add_argument("--voice")
    squat.add_argument("--no-preview", action="store_true")
    squat.add_argument(
        "--no-voice",
        action="store_true",
        help="camera/pose/Guardian integration only; do not connect Realtime",
    )
    squat.add_argument(
        "--no-mic",
        action="store_true",
        help="disable terminal push-to-talk while keeping Realtime cues",
    )
    squat.add_argument(
        "--max-frames",
        type=int,
        help="stop after a positive number of frames for a bounded integration run",
    )
    squat.add_argument(
        "--pose-peer",
        metavar="HOST:PORT",
        help="publish numeric pose analysis to a RecoveryBox HOST:PORT peer",
    )
    squat.add_argument(
        "--pose-token-file",
        metavar="PATH",
        help="read the remote-pose authentication token from this file",
    )
    return parser


def _doctor(*, strict: bool) -> int:
    try:
        settings = Settings.from_environment()
    except ConfigurationError as exc:
        print(json.dumps({"status": "error", "configuration": str(exc)}, indent=2))
        return 2

    report = collect_laptop_doctor_report(settings)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 1 if strict and not report.ready else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(strict=args.strict)
    if args.command == "demo-safety":
        print(json.dumps(demo_as_dicts(), indent=2))
        return 0
    if args.command == "voice-checkin":
        return run_voice_checkin()
    if args.command == "seed-flower-demo":
        try:
            paths = seed_flower_demo(args.output_dir, rows_per_client=args.rows_per_client)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"written": [str(path) for path in paths]}, indent=2))
        return 0
    if args.command == "download-pose-model":
        try:
            installed = install_pose_model(args.output)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"installed": str(installed)}, indent=2))
        return 0
    if args.command == "verify-realtime-cues":
        try:
            return run_live_realtime_verification(
                output_dir=args.output_dir,
                voice=args.voice,
                asr_model=args.asr_model,
                skip_asr=args.skip_asr,
            )
        except Exception as exc:
            print(
                f"error: live Realtime verification failed ({type(exc).__name__})",
                file=sys.stderr,
            )
            return 2
    if args.command == "squat-demo":
        return run_squat_demo(
            camera_index=args.camera_index,
            model_path=args.model_path,
            voice=args.voice,
            no_preview=args.no_preview,
            no_voice=args.no_voice,
            no_mic=args.no_mic,
            max_frames=args.max_frames,
            pose_peer=args.pose_peer,
            pose_token_file=args.pose_token_file,
        )
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
