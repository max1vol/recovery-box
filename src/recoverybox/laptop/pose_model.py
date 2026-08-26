"""Explicit installation and validation of the laptop pose model asset.

Nothing in this module imports MediaPipe or downloads data at import time.  A
setup command must call :func:`install_pose_model` explicitly.  The camera
runtime should call :func:`validate_pose_model` before it attempts to open a
camera or construct a pose landmarker.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import ssl
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol

import certifi

POSE_LANDMARKER_LITE_V1_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_LANDMARKER_LITE_V1_SIZE_BYTES = 5_777_746
POSE_LANDMARKER_LITE_V1_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
DEFAULT_POSE_MODEL_PATH = Path("models/mediapipe/pose_landmarker_lite-v1.task")


class PoseModelError(RuntimeError):
    """Base error for pose-model installation and validation."""


class PoseModelNotInstalledError(PoseModelError):
    """Raised when the expected local pose-model asset is absent."""


class PoseModelIntegrityError(PoseModelError):
    """Raised when a local or downloaded asset is not the pinned model."""


class PoseModelDownloadError(PoseModelError):
    """Raised when the pinned model could not be downloaded or installed."""


@dataclass(frozen=True, slots=True)
class _PoseModelSpec:
    url: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.url.startswith("https://storage.googleapis.com/"):
            raise ValueError("pose-model URL must use the official HTTPS host")
        if self.size_bytes <= 0:
            raise ValueError("pose-model size must be positive")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("pose-model SHA-256 must be 64 lowercase hex digits")


_POSE_LANDMARKER_LITE_V1 = _PoseModelSpec(
    url=POSE_LANDMARKER_LITE_V1_URL,
    size_bytes=POSE_LANDMARKER_LITE_V1_SIZE_BYTES,
    sha256=POSE_LANDMARKER_LITE_V1_SHA256,
)


class _DownloadResponse(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> _DownloadResponse: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


_UrlOpener = Callable[..., _DownloadResponse]


def _default_url_opener(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> _DownloadResponse:
    tls_context = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.urlopen(request, timeout=timeout, context=tls_context)


def _normalized_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _part_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part")


def _validate_pose_model(path: Path, spec: _PoseModelSpec) -> Path:
    try:
        with path.open("rb") as model_file:
            stat = os.fstat(model_file.fileno())
            if stat.st_size != spec.size_bytes:
                raise PoseModelIntegrityError(
                    f"pose model at {path} has size {stat.st_size}; "
                    f"expected {spec.size_bytes} bytes"
                )

            digest = hashlib.sha256()
            bytes_read = 0
            while chunk := model_file.read(1024 * 1024):
                digest.update(chunk)
                bytes_read += len(chunk)
    except FileNotFoundError as exc:
        raise PoseModelNotInstalledError(
            f"pose model is not installed at {path}; run the laptop setup command"
        ) from exc
    except IsADirectoryError as exc:
        raise PoseModelIntegrityError(
            f"pose model path is a directory, not an asset file: {path}"
        ) from exc
    except OSError as exc:
        raise PoseModelIntegrityError(f"could not read pose model at {path}: {exc}") from exc

    if bytes_read != spec.size_bytes:
        raise PoseModelIntegrityError(
            f"pose model at {path} changed while being read; "
            f"read {bytes_read} of {spec.size_bytes} expected bytes"
        )

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != spec.sha256:
        raise PoseModelIntegrityError(
            f"pose model at {path} failed SHA-256 verification; "
            f"expected {spec.sha256}, got {actual_sha256}"
        )
    return path


def validate_pose_model(
    path: str | Path = DEFAULT_POSE_MODEL_PATH,
) -> Path:
    """Require the exact pinned pose model without making a network request.

    Call this before camera access.  The returned path has passed both exact
    byte-size and SHA-256 checks.
    """

    return _validate_pose_model(_normalized_path(path), _POSE_LANDMARKER_LITE_V1)


def install_pose_model(
    destination: str | Path = DEFAULT_POSE_MODEL_PATH,
    *,
    timeout_seconds: float = 60.0,
    chunk_size: int = 256 * 1024,
    _url_opener: _UrlOpener = _default_url_opener,
) -> Path:
    """Explicitly install the pinned official MediaPipe pose asset.

    A valid existing asset is reused without network access.  Otherwise the
    response is streamed to ``<destination>.part`` in the destination
    directory.  Only an exact size-and-hash match is atomically moved into
    place.  A failed attempt leaves any existing destination untouched and
    removes the partial download.

    ``_url_opener`` exists only to make the network boundary replaceable in
    tests.  Runtime callers should not supply it.
    """

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    destination_path = _normalized_path(destination)
    try:
        return _validate_pose_model(destination_path, _POSE_LANDMARKER_LITE_V1)
    except PoseModelNotInstalledError:
        pass
    except PoseModelIntegrityError:
        # A replacement is kept quarantined until it verifies, so the invalid
        # destination remains available for diagnosis if installation fails.
        pass

    part_path = _part_path(destination_path)
    request = urllib.request.Request(
        _POSE_LANDMARKER_LITE_V1.url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "RecoveryBox/0.1 pose-model-setup",
        },
        method="GET",
    )

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.unlink(missing_ok=True)

        with _url_opener(request, timeout=timeout_seconds) as response:
            with part_path.open("xb") as partial_file:
                downloaded_bytes = 0
                while chunk := response.read(chunk_size):
                    if not isinstance(chunk, bytes):
                        raise PoseModelDownloadError(
                            "pose-model server returned a non-bytes response"
                        )
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > _POSE_LANDMARKER_LITE_V1.size_bytes:
                        raise PoseModelIntegrityError(
                            "downloaded pose model is larger than the pinned asset: "
                            f"expected {_POSE_LANDMARKER_LITE_V1.size_bytes} bytes"
                        )
                    partial_file.write(chunk)
                partial_file.flush()
                os.fsync(partial_file.fileno())

        _validate_pose_model(part_path, _POSE_LANDMARKER_LITE_V1)
        os.replace(part_path, destination_path)
        return _validate_pose_model(destination_path, _POSE_LANDMARKER_LITE_V1)
    except (PoseModelIntegrityError, PoseModelDownloadError):
        raise
    except (OSError, TimeoutError) as exc:
        raise PoseModelDownloadError(
            f"could not install pose model from the pinned official URL: {exc}"
        ) from exc
    finally:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            # Preserve the original installation error.  A later explicit
            # setup call removes this exact quarantined path before download.
            pass


__all__ = [
    "DEFAULT_POSE_MODEL_PATH",
    "POSE_LANDMARKER_LITE_V1_SHA256",
    "POSE_LANDMARKER_LITE_V1_SIZE_BYTES",
    "POSE_LANDMARKER_LITE_V1_URL",
    "PoseModelDownloadError",
    "PoseModelError",
    "PoseModelIntegrityError",
    "PoseModelNotInstalledError",
    "install_pose_model",
    "validate_pose_model",
]
