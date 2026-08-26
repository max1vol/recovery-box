from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from types import TracebackType

import pytest

from recoverybox.laptop import pose_model

PAYLOAD = b"verified tiny model fixture"
TEST_SPEC = pose_model._PoseModelSpec(
    url=pose_model.POSE_LANDMARKER_LITE_V1_URL,
    size_bytes=len(PAYLOAD),
    sha256=hashlib.sha256(PAYLOAD).hexdigest(),
)


class FakeResponse:
    def __init__(self, payload: bytes, *, read_observer=None) -> None:
        self._payload = payload
        self._offset = 0
        self._read_observer = read_observer
        self.read_sizes: list[int] = []
        self.exited = False

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self._read_observer is not None:
            self._read_observer()
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exited = True


class RecordingOpener:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[urllib.request.Request, float]] = []

    def __call__(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeResponse:
        self.calls.append((request, timeout))
        return self.response


@pytest.fixture
def tiny_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pose_model, "_POSE_LANDMARKER_LITE_V1", TEST_SPEC)


def test_pins_official_v1_asset_identity_and_default_ignored_location() -> None:
    assert pose_model.POSE_LANDMARKER_LITE_V1_URL == (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
    assert pose_model.POSE_LANDMARKER_LITE_V1_SIZE_BYTES == 5_777_746
    assert pose_model.POSE_LANDMARKER_LITE_V1_SHA256 == (
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
    )
    assert pose_model.DEFAULT_POSE_MODEL_PATH == Path(
        "models/mediapipe/pose_landmarker_lite-v1.task"
    )


def test_default_url_opener_uses_verified_certifi_trust_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = urllib.request.Request(pose_model.POSE_LANDMARKER_LITE_V1_URL)
    response = FakeResponse(PAYLOAD)
    tls_context = object()
    calls: list[tuple[object, ...]] = []

    def fake_certifi_where() -> str:
        calls.append(("certifi",))
        return "/verified/certifi/cacert.pem"

    def fake_create_default_context(*, cafile: str):
        calls.append(("context", cafile))
        return tls_context

    def fake_urlopen(
        opened_request: urllib.request.Request,
        *,
        timeout: float,
        context: object,
    ) -> FakeResponse:
        calls.append(("urlopen", opened_request, timeout, context))
        return response

    monkeypatch.setattr(pose_model.certifi, "where", fake_certifi_where)
    monkeypatch.setattr(pose_model.ssl, "create_default_context", fake_create_default_context)
    monkeypatch.setattr(pose_model.urllib.request, "urlopen", fake_urlopen)

    assert pose_model._default_url_opener(request, timeout=9.5) is response
    assert calls == [
        ("certifi",),
        ("context", "/verified/certifi/cacert.pem"),
        ("urlopen", request, 9.5, tls_context),
    ]


def test_validation_is_local_only_and_missing_asset_does_not_create_directories(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "not-created" / "model.task"

    with pytest.raises(pose_model.PoseModelNotInstalledError, match="setup command"):
        pose_model.validate_pose_model(missing)

    assert not missing.parent.exists()


def test_validate_accepts_only_the_exact_pinned_bytes(
    tmp_path: Path,
    tiny_spec: None,
) -> None:
    model_path = tmp_path / "model.task"
    model_path.write_bytes(PAYLOAD)

    assert pose_model.validate_pose_model(model_path) == model_path

    model_path.write_bytes(PAYLOAD[:-1])
    with pytest.raises(pose_model.PoseModelIntegrityError, match="has size"):
        pose_model.validate_pose_model(model_path)

    model_path.write_bytes(b"x" * len(PAYLOAD))
    with pytest.raises(pose_model.PoseModelIntegrityError, match="SHA-256"):
        pose_model.validate_pose_model(model_path)


def test_explicit_install_streams_to_same_directory_then_atomically_replaces(
    tmp_path: Path,
    tiny_spec: None,
) -> None:
    destination = tmp_path / "models" / "model.task"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old invalid asset")
    part_path = destination.with_name("model.task.part")

    def observe_read() -> None:
        assert destination.read_bytes() == b"old invalid asset"
        assert part_path.parent == destination.parent

    response = FakeResponse(PAYLOAD, read_observer=observe_read)
    opener = RecordingOpener(response)

    installed = pose_model.install_pose_model(
        destination,
        timeout_seconds=12.5,
        chunk_size=5,
        _url_opener=opener,
    )

    assert installed == destination
    assert destination.read_bytes() == PAYLOAD
    assert not part_path.exists()
    assert len(response.read_sizes) > 2
    assert response.exited
    assert len(opener.calls) == 1
    request, timeout = opener.calls[0]
    assert request.full_url == TEST_SPEC.url
    assert request.get_method() == "GET"
    assert timeout == 12.5


def test_valid_existing_asset_skips_network(
    tmp_path: Path,
    tiny_spec: None,
) -> None:
    destination = tmp_path / "model.task"
    destination.write_bytes(PAYLOAD)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network opener must not run for a valid local asset")

    assert pose_model.install_pose_model(destination, _url_opener=fail_if_called) == destination


def test_invalid_download_preserves_destination_and_cleans_partial_file(
    tmp_path: Path,
    tiny_spec: None,
) -> None:
    destination = tmp_path / "model.task"
    destination.write_bytes(b"keep this diagnostic asset")
    response = FakeResponse(b"wrong download")

    with pytest.raises(pose_model.PoseModelIntegrityError, match="has size"):
        pose_model.install_pose_model(
            destination,
            chunk_size=3,
            _url_opener=RecordingOpener(response),
        )

    assert destination.read_bytes() == b"keep this diagnostic asset"
    assert not destination.with_name("model.task.part").exists()
    assert response.exited


def test_network_failure_preserves_destination_and_removes_stale_partial(
    tmp_path: Path,
    tiny_spec: None,
) -> None:
    destination = tmp_path / "model.task"
    destination.write_bytes(b"existing invalid asset")
    part_path = destination.with_name("model.task.part")
    part_path.write_bytes(b"stale")

    def unavailable(*args, **kwargs):
        raise OSError("offline")

    with pytest.raises(pose_model.PoseModelDownloadError, match="pinned official URL"):
        pose_model.install_pose_model(destination, _url_opener=unavailable)

    assert destination.read_bytes() == b"existing invalid asset"
    assert not part_path.exists()


@pytest.mark.parametrize(
    ("timeout", "chunk_size", "message"),
    [
        (0, 1, "timeout_seconds"),
        (-1, 1, "timeout_seconds"),
        (True, 1, "timeout_seconds"),
        (float("nan"), 1, "timeout_seconds"),
        (float("inf"), 1, "timeout_seconds"),
        (1, 0, "chunk_size"),
        (1, -1, "chunk_size"),
        (1, True, "chunk_size"),
        (1, 1.5, "chunk_size"),
    ],
)
def test_rejects_invalid_download_bounds(
    timeout: float,
    chunk_size: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pose_model.install_pose_model(
            timeout_seconds=timeout,
            chunk_size=chunk_size,  # type: ignore[arg-type]
            _url_opener=lambda *args, **kwargs: FakeResponse(PAYLOAD),
        )
