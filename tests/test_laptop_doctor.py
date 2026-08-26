from __future__ import annotations

import hashlib
import importlib
import json
import tomllib
from importlib import metadata
from pathlib import Path

from recoverybox.config import Settings
from recoverybox.laptop import doctor


def test_report_contains_laptop_readiness_without_retaining_api_key(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_bytes = b"known pose model fixture"
    model_path = tmp_path / "pose.task"
    model_path.write_bytes(model_bytes)
    expected_sha256 = hashlib.sha256(model_bytes).hexdigest()
    inspect_pose_model = doctor._inspect_pose_model

    monkeypatch.setattr(
        doctor,
        "_inspect_pose_model",
        lambda path: inspect_pose_model(
            path,
            expected_size_bytes=len(model_bytes),
            expected_sha256=expected_sha256,
        ),
    )
    versions = dict(doctor.EXPECTED_LAPTOP_RUNTIME_PINS)
    settings = Settings.from_environment(
        {
            "OPENAI_API_KEY": "must-not-enter-the-report",
            "RECOVERYBOX_CAMERA_INDEX": "3",
            "RECOVERYBOX_POSE_MODEL_PATH": str(model_path),
        }
    )

    report = doctor.collect_laptop_doctor_report(
        settings,
        version_provider=versions.__getitem__,
    )
    serialized = json.dumps(report.as_dict(), sort_keys=True)

    assert report.ready is True
    assert report.camera_index == 3
    assert report.pose_model.status is doctor.PoseModelStatus.VALID
    assert report.pose_model.size_matches is True
    assert report.pose_model.sha256_matches is True
    assert report.openai_api_key_present is True
    assert report.as_dict()["hardware_probed"] is False
    assert "must-not-enter-the-report" not in serialized


def test_missing_model_and_packages_fail_closed(tmp_path: Path) -> None:
    settings = Settings.from_environment(
        {"RECOVERYBOX_POSE_MODEL_PATH": str(tmp_path / "missing.task")}
    )

    def missing(distribution: str) -> str:
        raise metadata.PackageNotFoundError(distribution)

    report = doctor.collect_laptop_doctor_report(settings, version_provider=missing)
    payload = report.as_dict()

    assert report.ready is False
    assert payload["openai_api_key_present"] is False
    assert payload["pose_model"]["status"] == "missing"
    assert payload["pose_model"]["size_matches"] is False
    assert payload["pose_model"]["sha256_matches"] is False
    assert {name: check["status"] for name, check in payload["packages"].items()} == {
        "mediapipe": "missing",
        "opencv-contrib-python": "missing",
    }


def test_pose_model_reports_size_and_checksum_failures_separately(tmp_path: Path) -> None:
    model_path = tmp_path / "pose.task"
    expected_bytes = b"expected"
    expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()

    model_path.write_bytes(b"short")
    size_failure = doctor._inspect_pose_model(
        model_path,
        expected_size_bytes=len(expected_bytes),
        expected_sha256=expected_sha256,
    )
    assert size_failure.status is doctor.PoseModelStatus.SIZE_MISMATCH
    assert size_failure.size_matches is False
    assert size_failure.sha256_matches is False

    model_path.write_bytes(b"expectez")
    checksum_failure = doctor._inspect_pose_model(
        model_path,
        expected_size_bytes=len(expected_bytes),
        expected_sha256=expected_sha256,
    )
    assert checksum_failure.status is doctor.PoseModelStatus.CHECKSUM_MISMATCH
    assert checksum_failure.size_matches is True
    assert checksum_failure.sha256_matches is False


def test_directory_pose_model_path_is_unreadable(tmp_path: Path) -> None:
    check = doctor._inspect_pose_model(tmp_path)

    assert check.status is doctor.PoseModelStatus.UNREADABLE
    assert check.actual_size_bytes is None


def test_package_check_uses_distribution_metadata_and_reports_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = Settings.from_environment(
        {"RECOVERYBOX_POSE_MODEL_PATH": str(tmp_path / "missing.task")}
    )
    queried: list[str] = []

    def version_provider(distribution: str) -> str:
        queried.append(distribution)
        return "1.0.1" if distribution == "mediapipe" else doctor.OPENCV_EXPECTED_VERSION

    def native_import_forbidden(name: str, package: str | None = None):
        raise AssertionError(f"doctor attempted a runtime import: {name} {package}")

    monkeypatch.setattr(importlib, "import_module", native_import_forbidden)
    report = doctor.collect_laptop_doctor_report(
        settings,
        version_provider=version_provider,
    )

    assert queried == ["mediapipe", "opencv-contrib-python"]
    packages = {check.distribution: check for check in report.packages}
    assert packages["mediapipe"].status is doctor.PackagePinStatus.VERSION_MISMATCH
    assert packages["mediapipe"].installed_version == "1.0.1"
    assert packages["opencv-contrib-python"].status is doctor.PackagePinStatus.MATCH


def test_doctor_pin_contract_matches_laptop_optional_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    laptop_dependencies = pyproject["project"]["optional-dependencies"]["laptop"]
    exact_dependencies = {
        dependency.split("==", maxsplit=1)[0]: dependency.split("==", maxsplit=1)[1]
        for dependency in laptop_dependencies
        if "==" in dependency
    }

    assert dict(doctor.EXPECTED_LAPTOP_RUNTIME_PINS) == {
        "mediapipe": exact_dependencies["mediapipe"],
        "opencv-contrib-python": exact_dependencies["opencv-contrib-python"],
    }
    assert "opencv-python" not in exact_dependencies
    assert "opencv-python-headless" not in exact_dependencies
