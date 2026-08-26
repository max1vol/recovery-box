"""Hardware-free readiness checks for the laptop squat demo.

This module deliberately limits itself to environment-backed configuration,
ordinary file reads, and installed-distribution metadata.  In particular it
does not import MediaPipe or OpenCV, open a camera, or initialize audio.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any

from recoverybox.config import Settings
from recoverybox.laptop.pose_model import (
    POSE_LANDMARKER_LITE_V1_SHA256,
    POSE_LANDMARKER_LITE_V1_SIZE_BYTES,
)

MEDIAPIPE_EXPECTED_VERSION = "0.10.35"
OPENCV_EXPECTED_VERSION = "4.14.0.94"
EXPECTED_LAPTOP_RUNTIME_PINS = (
    ("mediapipe", MEDIAPIPE_EXPECTED_VERSION),
    ("opencv-contrib-python", OPENCV_EXPECTED_VERSION),
)


class PoseModelStatus(StrEnum):
    """Summary of local pose-model integrity without installing anything."""

    VALID = "valid"
    MISSING = "missing"
    SIZE_MISMATCH = "size_mismatch"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    UNREADABLE = "unreadable"


class PackagePinStatus(StrEnum):
    """Whether installed distribution metadata matches a required exact pin."""

    MATCH = "match"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    METADATA_ERROR = "metadata_error"


@dataclass(frozen=True, slots=True)
class PoseModelDoctorCheck:
    path: Path
    status: PoseModelStatus
    expected_size_bytes: int
    actual_size_bytes: int | None
    size_matches: bool
    expected_sha256: str
    sha256_matches: bool

    @property
    def valid(self) -> bool:
        return self.status is PoseModelStatus.VALID

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "status": self.status.value,
            "expected_size_bytes": self.expected_size_bytes,
            "actual_size_bytes": self.actual_size_bytes,
            "size_matches": self.size_matches,
            "expected_sha256": self.expected_sha256,
            "sha256_matches": self.sha256_matches,
        }


@dataclass(frozen=True, slots=True)
class PackagePinDoctorCheck:
    distribution: str
    expected_version: str
    installed_version: str | None
    status: PackagePinStatus

    @property
    def matches(self) -> bool:
        return self.status is PackagePinStatus.MATCH

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_version": self.expected_version,
            "installed_version": self.installed_version,
            "status": self.status.value,
            "matches": self.matches,
        }


@dataclass(frozen=True, slots=True)
class LaptopDoctorReport:
    """Serializable laptop readiness evidence gathered without hardware access."""

    camera_index: int
    pose_model: PoseModelDoctorCheck
    packages: tuple[PackagePinDoctorCheck, ...]
    openai_api_key_present: bool

    @property
    def ready(self) -> bool:
        return (
            self.openai_api_key_present
            and self.pose_model.valid
            and all(package.matches for package in self.packages)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_index": self.camera_index,
            "pose_model": self.pose_model.as_dict(),
            "packages": {package.distribution: package.as_dict() for package in self.packages},
            "openai_api_key_present": self.openai_api_key_present,
            "hardware_probed": False,
            "ready": self.ready,
        }


_VersionProvider = Callable[[str], str]


def collect_laptop_doctor_report(
    settings: Settings | None = None,
    *,
    version_provider: _VersionProvider = metadata.version,
) -> LaptopDoctorReport:
    """Collect laptop prerequisites without importing or probing native runtimes.

    ``importlib.metadata.version`` reads installed-distribution metadata; it
    does not import the distributions whose versions are queried.  The
    injectable provider is solely for hardware-free tests.
    """

    configured = Settings.from_environment() if settings is None else settings
    configured.validate()
    return LaptopDoctorReport(
        camera_index=configured.pose.camera_index,
        pose_model=_inspect_pose_model(configured.pose.model_path),
        packages=tuple(
            _inspect_package_pin(distribution, expected, version_provider)
            for distribution, expected in EXPECTED_LAPTOP_RUNTIME_PINS
        ),
        openai_api_key_present=configured.openai_api_key_present,
    )


def _inspect_package_pin(
    distribution: str,
    expected_version: str,
    version_provider: _VersionProvider,
) -> PackagePinDoctorCheck:
    try:
        installed_version = version_provider(distribution)
    except metadata.PackageNotFoundError:
        return PackagePinDoctorCheck(
            distribution=distribution,
            expected_version=expected_version,
            installed_version=None,
            status=PackagePinStatus.MISSING,
        )
    except Exception:
        # Metadata corruption should make the check fail closed, but exception
        # text and local paths do not need to enter a readiness report.
        return PackagePinDoctorCheck(
            distribution=distribution,
            expected_version=expected_version,
            installed_version=None,
            status=PackagePinStatus.METADATA_ERROR,
        )

    status = (
        PackagePinStatus.MATCH
        if installed_version == expected_version
        else PackagePinStatus.VERSION_MISMATCH
    )
    return PackagePinDoctorCheck(
        distribution=distribution,
        expected_version=expected_version,
        installed_version=installed_version,
        status=status,
    )


def _inspect_pose_model(
    path: Path,
    *,
    expected_size_bytes: int = POSE_LANDMARKER_LITE_V1_SIZE_BYTES,
    expected_sha256: str = POSE_LANDMARKER_LITE_V1_SHA256,
) -> PoseModelDoctorCheck:
    model_path = path.expanduser()
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        with model_path.open("rb") as model_file:
            actual_size_bytes = os.fstat(model_file.fileno()).st_size
            while chunk := model_file.read(1024 * 1024):
                digest.update(chunk)
                bytes_read += len(chunk)
    except FileNotFoundError:
        return _pose_model_failure(
            model_path,
            PoseModelStatus.MISSING,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )
    except (IsADirectoryError, OSError):
        return _pose_model_failure(
            model_path,
            PoseModelStatus.UNREADABLE,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )

    if bytes_read != actual_size_bytes:
        return _pose_model_failure(
            model_path,
            PoseModelStatus.UNREADABLE,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
            actual_size_bytes=actual_size_bytes,
        )

    size_matches = actual_size_bytes == expected_size_bytes
    sha256_matches = digest.hexdigest() == expected_sha256
    if not size_matches:
        status = PoseModelStatus.SIZE_MISMATCH
    elif not sha256_matches:
        status = PoseModelStatus.CHECKSUM_MISMATCH
    else:
        status = PoseModelStatus.VALID
    return PoseModelDoctorCheck(
        path=model_path,
        status=status,
        expected_size_bytes=expected_size_bytes,
        actual_size_bytes=actual_size_bytes,
        size_matches=size_matches,
        expected_sha256=expected_sha256,
        sha256_matches=sha256_matches,
    )


def _pose_model_failure(
    path: Path,
    status: PoseModelStatus,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    actual_size_bytes: int | None = None,
) -> PoseModelDoctorCheck:
    return PoseModelDoctorCheck(
        path=path,
        status=status,
        expected_size_bytes=expected_size_bytes,
        actual_size_bytes=actual_size_bytes,
        size_matches=False,
        expected_sha256=expected_sha256,
        sha256_matches=False,
    )


__all__ = [
    "EXPECTED_LAPTOP_RUNTIME_PINS",
    "MEDIAPIPE_EXPECTED_VERSION",
    "OPENCV_EXPECTED_VERSION",
    "LaptopDoctorReport",
    "PackagePinDoctorCheck",
    "PackagePinStatus",
    "PoseModelDoctorCheck",
    "PoseModelStatus",
    "collect_laptop_doctor_report",
]
