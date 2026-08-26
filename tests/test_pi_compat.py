from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import types
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from recoverybox.device import pi_pose_ncnn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEPLOY_ROOT = _REPO_ROOT / "deploy"
_STATUS_SCRIPT = _DEPLOY_ROOT / "recoverybox_status.py"
_TREE_DIGEST_SCRIPT = _DEPLOY_ROOT / "recoverybox_tree_digest.py"
_SYSTEMD_ROOT = _DEPLOY_ROOT / "systemd"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def debug_web() -> types.ModuleType:
    name = "_recoverybox_test_status_web"
    module = _load_module(name, _STATUS_SCRIPT)
    yield module
    sys.modules.pop(name, None)


def _status(**updates: object) -> dict[str, object]:
    status: dict[str, object] = {
        "service": "connected",
        "peer": "100.70.100.93",
        "session": "a" * 32,
        "mode": "active_exercise",
        "rep": 3,
        "age": 0,
        "voice": "silent",
        "button": "available",
        "failure": None,
    }
    status.update(updates)
    return status


def _write_status(path: Path, status: object, *, mode: int = 0o600) -> None:
    path.write_text(json.dumps(status), encoding="utf-8")
    path.chmod(mode)


def test_status_reader_returns_closed_sanitized_schema(
    tmp_path: Path,
    debug_web: types.ModuleType,
) -> None:
    path = tmp_path / "status.json"
    expected = _status()
    _write_status(path, expected)

    assert debug_web._read_status(path) == expected


@pytest.mark.parametrize(
    "updates",
    [
        {"service": "patient_name"},
        {"peer": "203.0.113.8"},
        {"session": "not-a-session"},
        {"mode": "free form"},
        {"rep": True},
        {"age": -1},
        {"voice": "speaking"},
        {"button": "pressed by patient_name"},
        {"failure": "provider said a secret"},
        {"failure": "FutureProviderError"},
    ],
)
def test_status_reader_rejects_values_outside_closed_schema(
    tmp_path: Path,
    debug_web: types.ModuleType,
    updates: dict[str, object],
) -> None:
    path = tmp_path / "status.json"
    _write_status(path, _status(**updates))

    with pytest.raises(debug_web.StatusUnavailable):
        debug_web._read_status(path)


def test_status_reader_rejects_missing_invalid_or_unsafe_file(
    tmp_path: Path,
    debug_web: types.ModuleType,
) -> None:
    missing_field = tmp_path / "missing.json"
    value = _status()
    del value["voice"]
    _write_status(missing_field, value)

    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b'{"rep":' + (b"9" * 5000) + b"}")
    invalid.chmod(0o600)

    writable = tmp_path / "writable.json"
    _write_status(writable, _status(), mode=0o620)

    target = tmp_path / "target.json"
    _write_status(target, _status())
    symlink = tmp_path / "status-link.json"
    symlink.symlink_to(target)

    unknown_field = tmp_path / "unknown.json"
    _write_status(unknown_field, {**_status(), "transcript": "must-not-leave"})

    for path in (missing_field, invalid, writable, symlink, unknown_field):
        with pytest.raises(debug_web.StatusUnavailable):
            debug_web._read_status(path)


def test_status_reader_rejects_stale_or_future_file(
    tmp_path: Path,
    debug_web: types.ModuleType,
) -> None:
    path = tmp_path / "status.json"
    _write_status(path, _status())
    timestamp = path.stat().st_mtime

    with pytest.raises(debug_web.StatusUnavailable, match="stale"):
        debug_web._read_status(path, now_seconds=timestamp + 2.1, max_age_seconds=2.0)
    with pytest.raises(debug_web.StatusUnavailable, match="stale"):
        debug_web._read_status(path, now_seconds=timestamp - 1.1, max_age_seconds=2.0)


def test_status_reader_accepts_fixed_fail_closed_gpio_state(
    tmp_path: Path,
    debug_web: types.ModuleType,
) -> None:
    path = tmp_path / "status.json"
    expected = _status(
        service="failed",
        peer=None,
        session=None,
        mode="stopped",
        button="unavailable",
        failure="GPIOInputUnavailable",
    )
    _write_status(path, expected)

    assert debug_web._read_status(path) == expected


def test_status_reader_accepts_clock_order_protocol_failure(
    tmp_path: Path,
    debug_web: types.ModuleType,
) -> None:
    path = tmp_path / "status.json"
    expected = _status(
        service="listening",
        peer=None,
        mode="paused",
        failure="PoseResponseBeforeRequest",
    )
    _write_status(path, expected)

    assert debug_web._read_status(path) == expected


def test_tailnet_bind_validation_has_no_wildcard_fallback(
    debug_web: types.ModuleType,
) -> None:
    assert debug_web._tailscale_bind_host("100.106.237.106") == "100.106.237.106"
    assert debug_web._tailscale_peer("100.70.100.93") == "100.70.100.93"
    for invalid in (
        "",
        "0.0.0.0",
        "127.0.0.1",
        "fd7a:115c:a1e0::1",
        "example.test",
    ):
        with pytest.raises(ValueError):
            debug_web._tailscale_bind_host(invalid)
        with pytest.raises(ValueError):
            debug_web._tailscale_peer(invalid)

    with pytest.raises(ValueError, match="IPv4"):
        debug_web._StatusServer(
            ("127.0.0.1", 0),
            Path("/does/not/matter"),
            allowed_peer="fd7a:115c:a1e0::1",
        )


@pytest.fixture
def status_server(
    tmp_path: Path,
    debug_web: types.ModuleType,
) -> Iterator[tuple[str, types.ModuleType]]:
    path = tmp_path / "status.json"
    _write_status(path, _status())
    server = debug_web._StatusServer(("127.0.0.1", 0), path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", debug_web
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _request_json(url: str, *, method: str = "GET") -> tuple[int, str, dict[str, Any]]:
    request = urllib.request.Request(url, method=method)
    try:
        response = urllib.request.urlopen(request, timeout=2.0)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return (
            response.status,
            response.headers.get_content_type(),
            json.loads(response.read()),
        )


def test_http_surface_serves_only_status_and_sanitized_errors(
    status_server: tuple[str, types.ModuleType],
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_url, _debug_web = status_server

    for path in ("/", "/healthz"):
        status, content_type, payload = _request_json(base_url + path)
        assert status == 200
        assert content_type == "application/json"
        assert payload == _status()

    status, content_type, payload = _request_json(base_url + "/not-a-file")
    assert status == 404
    assert content_type == "application/json"
    assert payload == {"service": "recoverybox", "mode": "not_found"}

    status, content_type, payload = _request_json(base_url + "/", method="POST")
    assert status == 501
    assert content_type == "application/json"
    assert payload == {"service": "recoverybox", "mode": "request_rejected"}
    assert capsys.readouterr() == ("", "")


def test_http_unavailable_response_contains_no_path_or_exception(
    tmp_path: Path,
    debug_web: types.ModuleType,
) -> None:
    server = debug_web._StatusServer(("127.0.0.1", 0), tmp_path / "absent.json")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        status, content_type, payload = _request_json(f"http://{host}:{port}/healthz")
        assert status == 503
        assert content_type == "application/json"
        assert payload == {
            "service": "recoverybox",
            "mode": "unavailable",
            "failure": "status_unavailable",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_http_failed_producer_is_503_with_only_closed_status(
    tmp_path: Path,
    debug_web: types.ModuleType,
) -> None:
    path = tmp_path / "status.json"
    payload = _status(
        service="failed",
        peer=None,
        session=None,
        mode="stopped",
        button="unavailable",
        failure="GPIOInputUnavailable",
    )
    _write_status(path, payload)
    server = debug_web._StatusServer(("127.0.0.1", 0), path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        status, content_type, response = _request_json(f"http://{host}:{port}/healthz")
        assert status == 503
        assert content_type == "application/json"
        assert response == payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


@pytest.mark.parametrize(
    "updates",
    [
        {"mode": "paused", "failure": "PoseTimeout", "age": 500},
        {"mode": "active_exercise", "age": 500},
        {"mode": "stopped", "failure": "PhysicalStop"},
    ],
)
def test_http_fail_closed_session_state_is_503(
    tmp_path: Path,
    debug_web: types.ModuleType,
    updates: dict[str, object],
) -> None:
    path = tmp_path / "status.json"
    payload = _status(**updates)
    _write_status(path, payload)
    server = debug_web._StatusServer(("127.0.0.1", 0), path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        status, content_type, response = _request_json(f"http://{host}:{port}/healthz")
        assert status == 503
        assert content_type == "application/json"
        assert response == payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_active_session_at_499_ms_is_healthy(
    tmp_path: Path,
    debug_web: types.ModuleType,
) -> None:
    path = tmp_path / "status.json"
    payload = _status(age=499)
    _write_status(path, payload)
    server = debug_web._StatusServer(("127.0.0.1", 0), path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        status, _content_type, response = _request_json(f"http://{host}:{port}/healthz")
        assert status == 200
        assert response == payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_http_rejects_every_unconfigured_peer_without_reading_status(
    tmp_path: Path,
    debug_web: types.ModuleType,
) -> None:
    path = tmp_path / "status.json"
    _write_status(path, _status())
    server = debug_web._StatusServer(
        ("127.0.0.1", 0),
        path,
        allowed_peer="127.0.0.2",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with pytest.raises(ConnectionError):
            urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_status_port_rejects_privileged_or_invalid_values(
    debug_web: types.ModuleType,
) -> None:
    assert debug_web._port("45874") == 45874
    for invalid in ("400", "0", "65536", "not-a-port"):
        with pytest.raises(ValueError):
            debug_web._port(invalid)


def test_system_units_run_pi_local_pose_with_camera_and_audio_devices_closed() -> None:
    main = (_SYSTEMD_ROOT / "recoverybox.service").read_text(encoding="utf-8")
    status = (_SYSTEMD_ROOT / "recoverybox-status.service").read_text(encoding="utf-8")

    assert "User=pi" in main
    assert "Group=pi" in main
    assert "SupplementaryGroups=gpio video" in main
    assert "WorkingDirectory=/opt/recoverybox/app" in main
    assert "EnvironmentFile=/etc/recoverybox/recoverybox.env" in main
    assert "Environment=PYTHONPATH=/opt/recoverybox/app/src" in main
    assert "LoadCredential=openai-api-key:/etc/recoverybox/credentials/openai-api-key" in main
    assert "Environment=RECOVERYBOX_OPENAI_CREDENTIAL_FILE=%d/openai-api-key" in main
    assert "pose-token" not in main
    assert "RECOVERYBOX_POSE_TOKEN_FILE" not in main
    assert "ExecStart=/usr/bin/python3 -m recoverybox.device.remote_pose_service" in main
    assert "ExecStopPost=-/usr/bin/rm -f /run/recoverybox/status.json" in main
    assert "RuntimeDirectory=recoverybox" in main
    assert "RuntimeDirectoryPreserve=restart" in main
    assert "LimitCORE=0" in main
    assert "DevicePolicy=closed" in main
    assert "DeviceAllow=/dev/gpiochip0 rw" in main
    assert "DeviceAllow=/dev/video0 rw" in main
    assert "ProtectClock=" not in main
    assert "/dev/snd" not in main
    assert (
        "ReadOnlyPaths=/opt/recoverybox/app /opt/recoverybox/runtime /opt/recoverybox/models"
    ) in main

    assert "User=pi" in status
    assert "WorkingDirectory=/opt/recoverybox/app" in status
    assert "EnvironmentFile=/etc/recoverybox/recoverybox.env" in status
    assert "Environment=PYTHONPATH=/opt/recoverybox/app/src" in status
    assert "ExecStart=/usr/bin/python3 /opt/recoverybox/app/deploy/" in status
    assert "recoverybox_status.py" in status
    assert "DevicePolicy=closed" in status
    assert "DeviceAllow=" not in status
    assert "ProtectClock=" not in status
    assert "/dev/snd" not in status
    assert "AmbientCapabilities=" not in status
    assert "CapabilityBoundingSet=\n" in status
    assert "LoadCredential=" not in status
    assert "RECOVERYBOX_OPENAI_CREDENTIAL_FILE" not in status
    assert "RECOVERYBOX_POSE_TOKEN_FILE" not in status
    assert "InaccessiblePaths=/etc/recoverybox/credentials" in status

    for unit in (main, status):
        assert "IPAddressDeny=any" in unit
        assert "IPAddressAllow=100.64.0.0/10" in unit
        assert "voice-ai-bot" not in unit
        assert "ReadOnlyPaths=/opt/recoverybox/app" in unit
        assert "OPENAI_API_KEY" not in unit


def test_deployment_has_no_legacy_compatibility_tree() -> None:
    assert not (_DEPLOY_ROOT / "pi-compat").exists()


def test_deployment_shell_assets_are_syntactically_valid_and_executable() -> None:
    assets = (
        _REPO_ROOT / "scripts" / "deploy-pi3.sh",
        _REPO_ROOT / "scripts" / "fetch-pi-pose-ncnn-models.sh",
        _REPO_ROOT / "scripts" / "fetch-pi-pose-ncnn-runtime.sh",
        _REPO_ROOT / "scripts" / "verify-pi-pose-ncnn-assets.sh",
        _DEPLOY_ROOT / "remove-legacy-voice-ai-bot.sh",
    )
    for asset in assets:
        assert asset.stat().st_mode & 0o111
        subprocess.run(["bash", "-n", str(asset)], check=True)


def test_ncnn_fetch_and_verifier_helpers_match_python_asset_pins_without_downloading() -> None:
    runtime_fetch = (_REPO_ROOT / "scripts" / "fetch-pi-pose-ncnn-runtime.sh").read_text(
        encoding="utf-8"
    )
    model_fetch = (_REPO_ROOT / "scripts" / "fetch-pi-pose-ncnn-models.sh").read_text(
        encoding="utf-8"
    )
    verifier = (_REPO_ROOT / "scripts" / "verify-pi-pose-ncnn-assets.sh").read_text(
        encoding="utf-8"
    )

    runtime_pins = (
        pi_pose_ncnn.NCNN_ARMV7_WHEEL_FILENAME,
        pi_pose_ncnn.NCNN_ARMV7_WHEEL_URL,
        str(pi_pose_ncnn.NCNN_ARMV7_WHEEL_SIZE_BYTES),
        pi_pose_ncnn.NCNN_ARMV7_WHEEL_SHA256,
        pi_pose_ncnn.NCNN_INIT_FILENAME,
        str(pi_pose_ncnn.NCNN_INIT_SIZE_BYTES),
        pi_pose_ncnn.NCNN_INIT_SHA256,
        pi_pose_ncnn.NCNN_NATIVE_FILENAME,
        str(pi_pose_ncnn.NCNN_NATIVE_SIZE_BYTES),
        pi_pose_ncnn.NCNN_NATIVE_SHA256,
        pi_pose_ncnn.NCNN_LIBGOMP_FILENAME,
        str(pi_pose_ncnn.NCNN_LIBGOMP_SIZE_BYTES),
        pi_pose_ncnn.NCNN_LIBGOMP_SHA256,
    )
    for pin in runtime_pins:
        assert pin in runtime_fetch

    model_pins = (
        pi_pose_ncnn.RTMPOSE_ARCHIVE_URL,
        str(pi_pose_ncnn.RTMPOSE_ARCHIVE_SIZE_BYTES),
        pi_pose_ncnn.RTMPOSE_ARCHIVE_SHA256,
        str(pi_pose_ncnn.RTMPOSE_PARAM_SIZE_BYTES),
        pi_pose_ncnn.RTMPOSE_PARAM_SHA256,
        str(pi_pose_ncnn.RTMPOSE_BIN_SIZE_BYTES),
        pi_pose_ncnn.RTMPOSE_BIN_SHA256,
        pi_pose_ncnn.NANODET_ARCHIVE_URL,
        str(pi_pose_ncnn.NANODET_ARCHIVE_SIZE_BYTES),
        pi_pose_ncnn.NANODET_ARCHIVE_SHA256,
        str(pi_pose_ncnn.NANODET_PARAM_SIZE_BYTES),
        pi_pose_ncnn.NANODET_PARAM_SHA256,
        str(pi_pose_ncnn.NANODET_BIN_SIZE_BYTES),
        pi_pose_ncnn.NANODET_BIN_SHA256,
    )
    for pin in model_pins:
        assert pin in model_fetch

    for pin in (
        pi_pose_ncnn.NCNN_INIT_SHA256,
        pi_pose_ncnn.NCNN_NATIVE_SHA256,
        pi_pose_ncnn.NCNN_LIBGOMP_SHA256,
        pi_pose_ncnn.RTMPOSE_PARAM_SHA256,
        pi_pose_ncnn.RTMPOSE_BIN_SHA256,
        pi_pose_ncnn.NANODET_PARAM_SHA256,
        pi_pose_ncnn.NANODET_BIN_SHA256,
    ):
        assert pin in verifier

    assert pi_pose_ncnn.DEFAULT_NCNN_RUNTIME_PATH == Path("/opt/recoverybox/runtime/ncnn")
    assert pi_pose_ncnn.DEFAULT_NCNN_MODEL_DIRECTORY == Path("/opt/recoverybox/models/ncnn")


def test_deploy_builds_fresh_silent_pi_local_pose_environment() -> None:
    deploy = (_REPO_ROOT / "scripts" / "deploy-pi3.sh").read_text(encoding="utf-8")

    assert '"RECOVERYBOX_AUDIO_ENABLED": "0"' in deploy
    assert '"RECOVERYBOX_PLAYBACK_DEVICE": "none"' in deploy
    assert '"RECOVERYBOX_POSE_SOURCE": "local"' in deploy
    assert '"RECOVERYBOX_CAMERA_DEVICE": "/dev/video0"' in deploy
    assert 'getattr(library, "YUY2ToARGB", None)' in deploy
    assert 'getattr(library, "I420ToARGB", None)' not in deploy
    assert '"RECOVERYBOX_CAMERA_WIDTH": "640"' in deploy
    assert '"RECOVERYBOX_CAMERA_HEIGHT": "480"' in deploy
    assert '"RECOVERYBOX_CAMERA_FPS": "10"' in deploy
    assert '"RECOVERYBOX_POSE_FRAME_TIMEOUT_SECONDS": "0.1"' in deploy
    assert '"RECOVERYBOX_V4L2_BUFFER_COUNT": "8"' in deploy
    assert '"RECOVERYBOX_NCNN_THREADS": "2"' in deploy
    assert '"RECOVERYBOX_LOCAL_POSE_MAX_AGE_SECONDS": "0.5"' in deploy
    assert 'readonly REMOTE_NCNN_RUNTIME="$REMOTE_ROOT/runtime/ncnn"' in deploy
    assert 'readonly REMOTE_NCNN_MODELS="$REMOTE_ROOT/models/ncnn"' in deploy
    assert "scripts/fetch-pi-pose-ncnn-runtime.sh" in deploy
    assert "scripts/fetch-pi-pose-ncnn-models.sh" in deploy
    assert "scripts/verify-pi-pose-ncnn-assets.sh" in deploy
    # The Pi mounts the /run deployment stage noexec. Invoke the already
    # digest-verified shell helper through the trusted system shell.
    assert deploy.count('/bin/bash "$verifier" "$assets" root:root') == 1
    assert deploy.count('/bin/bash "$asset_verifier"') == 2
    assert deploy.count('/bin/bash "$asset_verifier_source"') == 2
    assert '[ -f "$asset_verifier" ] && [ ! -L "$asset_verifier" ]' in deploy
    assert '[ -x "$asset_verifier" ]' not in deploy
    assert "assert_unit_word_set recoverybox.service RestrictAddressFamilies" in deploy
    assert "assert_unit_word_set recoverybox-status.service RestrictAddressFamilies" in deploy
    assert "assert_unit_ip_deny_any recoverybox.service" in deploy
    assert "assert_unit_ip_deny_any recoverybox-status.service" in deploy
    assert "assert_unit_property recoverybox.service IPAddressDeny any" not in deploy
    assert (
        '"$admin_target" /bin/bash \\\n    "$remote_asset_verify_stage" "$REMOTE_ROOT" root:root'
    ) in deploy
    assert '\n"$verifier" "$assets" root:root' not in deploy
    assert '\n"$asset_verifier"' not in deploy
    assert '\n"$asset_verifier_source"' not in deploy
    assert (
        '"$admin_target" \\\n    "$remote_asset_verify_stage" "$REMOTE_ROOT" root:root'
    ) not in deploy
    assert "DevicePolicy=closed" in deploy
    assert 'b"OPENAI_API_KEY" in environment' in deploy
    assert "unset OPENAI_API_KEY" in deploy
    assert "for command in ssh python3 tailscale; do" in deploy
    assert "for command in ssh python3 sha256sum tailscale; do" not in deploy
    assert "hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()" in deploy
    assert "printf '%s' \"$openai_api_key\" | ssh" in deploy
    assert 'readonly REMOTE_OPENAI_CREDENTIAL="$REMOTE_CREDENTIAL_DIR/openai-api-key"' in deploy
    assert "REMOTE_POSE_CREDENTIAL" not in deploy
    assert 'readonly REMOTE_APP="$REMOTE_ROOT/app"' in deploy
    assert 'readonly REMOTE_ROOT="/opt/recoverybox"' in deploy
    assert 'readonly REMOTE_CONFIG="$REMOTE_CONFIG_DIR/recoverybox.env"' in deploy
    assert "install -o root -g root -m 0600" in deploy
    config_reset = 'rm -rf --one-file-system -- "$config_dir"'
    openai_install = 'install -o root -g root -m 0600 "$openai_source" "$openai_target"'
    assert config_reset in deploy
    assert deploy.index(config_reset) < deploy.index(openai_install)
    assert 'find "$config_dir" -mindepth 1 -maxdepth 1 -printf x' in deploy
    assert 'find "$credential_dir" -mindepth 1 -maxdepth 1 -printf x' in deploy
    assert '"$config_dir" = /etc/recoverybox' in deploy
    assert '"$credential_dir" = "$config_dir/credentials"' in deploy
    assert 'grep -Fxq "LoadCredential=openai-api-key:$openai_target"' in deploy
    assert "LoadCredential=pose-token" not in deploy
    assert "ImportCredential|SetCredential|Environment=.*(CREDENTIAL|POSE_TOKEN)" in deploy
    assert "secret in environment_raw" in deploy
    assert '"RECOVERYBOX_POSE_TOKEN_FILE": "/run/recoverybox/unused-local-pose-token"' in deploy
    assert '"PYTHONPATH" in actual' in deploy
    assert 'environment.get(b"PYTHONPATH") != f"{app}/src".encode()' in deploy
    assert "/proc/$pid/root$app" in deploy
    assert "/opt/voice-ai-bot/.env" not in deploy
    assert "deploy/pi-compat" not in deploy
    assert "MOVENET" not in deploy
    assert "fetch-movenet" not in deploy
    assert "/dev/video0 rw" in deploy
    assert "SupplementaryGroups 'gpio video'" in deploy
    assert "recoverybox.service LimitCORE 0" in deploy
    assert '/bin/bash "$asset_verifier_source" "/proc/$pid/root$app_root" root:root' in deploy
    assert "backups/" not in deploy
    assert "releases/" not in deploy
    assert 'readonly ssh_target="${remote_user}@${pi_ip}"' in deploy
    assert 'readonly admin_target="${admin_user}@${pi_ip}"' in deploy
    assert "--host" not in deploy
    assert "--pi-ip" not in deploy
    assert "--mac-ip" not in deploy
    assert 'readonly pi_ip="$DEFAULT_PI_IP"' in deploy
    assert 'readonly mac_ip="$DEFAULT_MAC_IP"' in deploy
    assert "-F /dev/null" in deploy
    assert "ProxyCommand=none" in deploy
    assert "ProxyJump=none" in deploy
    assert "this Mac does not own the pinned Tailscale" in deploy
    assert deploy.index("DeviceAllow --value recoverybox.service") < deploy.index(
        "systemctl restart recoverybox.service recoverybox-status.service"
    )
    assert "cleanup_failed_activation" in deploy
    assert "systemctl disable recoverybox-status.service recoverybox.service" in deploy


def test_deploy_binds_staged_and_activated_app_to_fresh_local_tree_digest() -> None:
    deploy = (_REPO_ROOT / "scripts" / "deploy-pi3.sh").read_text(encoding="utf-8")

    local_digest = 'python3 deploy/recoverybox_tree_digest.py --root "$repo_root"'
    streamed_helper = "<deploy/recoverybox_tree_digest.py >/dev/null"
    staged_proof = "staged RecoveryBox application did not match the trusted local tree digest"
    activated_proof = (
        "activated RecoveryBox application did not match the trusted local tree digest"
    )
    stop = "printf 'Stopping any prior RecoveryBox instance before direct replacement.\\n'"
    delete = "printf 'Permanently deleting the exact legacy assistant targets.\\n'"
    rename = 'mv -- "$stage" "$app"'
    restart = "systemctl restart recoverybox.service recoverybox-status.service"

    assert "deploy/recoverybox_tree_digest.py" in deploy
    assert "deploy/recoverybox_status.py deploy/recoverybox_tree_digest.py" in deploy
    assert deploy.count(local_digest) == 2
    assert "--exclude=__pycache__/ --exclude='*.pyc'" in deploy
    assert deploy.count(streamed_helper) == 2
    assert deploy.count("\nverify_local_tree_helper\n") == 3
    assert '--root "$remote_app_stage" --strict --expect "$expected_app_digest"' in deploy
    assert '--root "$REMOTE_APP" --strict --expect "$expected_app_digest"' in deploy
    assert '[[ $current_app_digest == "$expected_app_digest" ]]' in deploy
    assert staged_proof in deploy
    assert activated_proof in deploy
    assert '"$stage/deploy/recoverybox_tree_digest.py"' not in deploy
    assert '"$app/deploy/recoverybox_tree_digest.py"' not in deploy
    assert '"$admin_target:$remote_unit_stage/recoverybox_tree_digest.py"' in deploy
    assert '"$admin_target:$remote_app_stage/src/"' in deploy
    assert '"$ssh_target:$remote_app_stage/src/"' not in deploy
    assert deploy.count('sha256sum "$tree_helper_source"') == 3
    assert deploy.count('--root "$app" --strict --expect "$expected_app_digest"') == 2
    assert '--root "/proc/$pid/root$app" --strict --expect "$expected_app_digest"' in deploy
    assert 'ss -H -ltn "sport = :$pose_port"' not in deploy
    assert "Pose listener:" not in deploy
    assert "pi_ip=$expected_ip" in deploy
    assert deploy.index(staged_proof) < deploy.index(stop)
    assert deploy.index(local_digest, deploy.index(local_digest) + 1) < deploy.index(stop)
    assert deploy.index(stop) < deploy.index(delete) < deploy.index(rename)
    assert deploy.index('systemctl disable "$unit"') < deploy.index(delete)
    assert deploy.index(rename) < deploy.index(activated_proof) < deploy.index(restart)
    assert deploy.index(activated_proof) < deploy.index(
        'install -o root -g root -m 0600 "$config_source" "$config"'
    )
    assert 'mv -- "$asset_stage/runtime" "$root/runtime"' in deploy
    assert 'mv -- "$asset_stage/models" "$root/models"' in deploy
    assert '/bin/bash "$asset_verifier" "$root" root:root' in deploy
    assert "activated RecoveryBox NCNN runtime/models did not match their exact pins" in deploy
    assert deploy.index(activated_proof) < deploy.index(
        '/bin/bash "$asset_verifier_source" "$app_root" root:root'
    )
    assert deploy.index(activated_proof) < deploy.rindex("activated_app_pending=0")
    assert deploy.index(activated_proof) < deploy.rindex("remote_unit_stage_pending=0")

    assert "activated_app_pending=1" in deploy
    assert 'if [ "$remove_stage" -eq 1 ]' in deploy
    assert 'if [ "$remove_activated" -eq 1 ]' in deploy
    assert 'rm -rf --one-file-system -- "$app_root"' in deploy
    assert 'readonly REMOTE_LEGACY_ROOT="/home/pi/recoverybox"' in deploy
    assert 'rm -rf --one-file-system -- "$legacy_root"' in deploy
    assert deploy.count('reject_target_or_descendant_mounts "$app_root"') == 3
    assert deploy.count('reject_target_or_descendant_mounts "$root"') == 2
    assert deploy.count('reject_target_or_descendant_mounts "$legacy_root"') == 1
    assert deploy.count('reject_target_or_descendant_mounts "$config_dir"') == 3
    assert deploy.count('reject_target_or_descendant_mounts "$credential_dir"') == 1
    assert "findmnt --raw --noheadings --output TARGET" in deploy
    assert '$0 == target || index($0, target "/") == 1' in deploy
    assert "END { exit(unsafe ? 1 : 0) }" in deploy
    assert '"$legacy_root/config/recoverybox.env"' in deploy
    assert "No service-account write descriptor ever existed" in deploy
    assert '"$(stat -c %U:%G "$app_root")" = root:root' in deploy
    assert '"$(stat -c %U:%G "$app")" = root:root' in deploy
    assert "runtime_proven=1" in deploy
    assert 'proof="$app_root/.runtime-proven-v2"' in deploy
    final_canonical_proof = deploy.rindex('--root "$app" --strict --expect "$expected_app_digest"')
    proc_root_proof = deploy.index('--root "/proc/$pid/root$app"')
    proc_asset_proof = deploy.index(
        '/bin/bash "$asset_verifier_source" "/proc/$pid/root$app_root" root:root'
    )
    assert final_canonical_proof < deploy.index(restart) < proc_root_proof < proc_asset_proof
    assert proc_asset_proof < deploy.index("runtime_proven=1")
    assert proc_asset_proof < deploy.rindex("activated_app_pending=0")


def test_tree_digest_helper_is_regular_and_python_311_compatible() -> None:
    assert _TREE_DIGEST_SCRIPT.is_file()
    assert not _TREE_DIGEST_SCRIPT.is_symlink()
    ast.parse(
        _TREE_DIGEST_SCRIPT.read_text(encoding="utf-8"),
        filename=str(_TREE_DIGEST_SCRIPT),
        feature_version=(3, 11),
    )


def test_deploy_runs_bounded_silent_local_pose_acceptance_before_service() -> None:
    deploy = (_REPO_ROOT / "scripts" / "deploy-pi3.sh").read_text(encoding="utf-8")

    acceptance = "recoverybox.device.pi_pose_v4l2 --max-frames 3"
    restart = "systemctl restart recoverybox.service recoverybox-status.service"
    process_proof = '/bin/bash "$asset_verifier_source" "/proc/$pid/root$app_root" root:root'
    endpoint = "Tailnet status endpoint failed Pi-local-pose readiness"

    assert acceptance in deploy
    assert 'pose_check=$(\n    cd "$app"\n    timeout 45 runuser -u pi --' in deploy
    assert ") || pose_check_status=$?" in deploy
    assert 'python3 - "$pose_check" "$pose_check_status"' in deploy
    assert (
        'numeric_fields = ("frames_received", "fresh_frames", "timeouts", "inference_ms_max")'
        in deploy
    )
    assert 'f"Pi pose acceptance failed: {failure}; "' in deploy
    assert "timeout 45 runuser -u pi" in deploy
    assert '"service": "recoverybox-pi-v4l2-ncnn-check/v1"' in deploy
    assert '"capture": "v4l2-mmap-yuyv"' in deploy
    assert '"conversion": "libyuv-yuy2-to-bgra"' in deploy
    assert '"estimator": "ncnn-nanodet-rtmpose"' in deploy
    assert '"frames_received": 3' in deploy
    assert '"fresh_frames"' in deploy
    assert 'not 1 <= fresh_frames <= report["frames_received"]' in deploy
    assert '"timeouts": 0' in deploy
    assert '"raw_frames_persisted": 0' in deploy
    assert '"audio": "disabled"' in deploy
    assert "inference_ms < 500" in deploy
    assert "recoverybox.remote_pose_smoke" not in deploy
    assert "timeout 10 systemctl restart recoverybox.service" not in deploy
    assert 'status.get("service") != "local"' in deploy
    assert '"idle", "check_in", "active_exercise", "paused"' in deploy
    assert 'not 0 <= status["age"] < 500' in deploy
    assert 'status.get("failure") is not None' in deploy
    assert "expected_uid = int(sys.argv[2])" in deploy
    assert "1000" not in deploy
    assert process_proof in deploy
    assert endpoint in deploy

    assert deploy.index(acceptance) < deploy.index(restart)
    assert deploy.index(restart) < deploy.index(process_proof)
    assert deploy.index(process_proof) < deploy.index(endpoint)


def test_legacy_deletion_helper_accepts_no_paths_and_names_exact_targets() -> None:
    cleanup = (_DEPLOY_ROOT / "remove-legacy-voice-ai-bot.sh").read_text(encoding="utf-8")

    for exact_target in (
        "/opt/voice-ai-bot",
        "/var/lib/voice-ai-bot",
        "/etc/systemd/system/voice-ai-bot.service",
        "/etc/systemd/system/voice-ai-bot-debug.service",
    ):
        assert exact_target in cleanup
    assert "${2" not in cleanup
    assert "rm -rf --one-file-system" in cleanup
    assert "voice-ai-bot*" not in cleanup
    assert '"$APP_ROOT" "$STATE_ROOT" "$MAIN_RUNTIME" "$DEBUG_RUNTIME"' in cleanup
    assert "validate_legacy_enablement_links 0" in cleanup
    assert 'systemctl disable "$unit"' not in cleanup


def test_legacy_deletion_helper_rejects_extra_arguments_before_root_work() -> None:
    helper = _DEPLOY_ROOT / "remove-legacy-voice-ai-bot.sh"

    result = subprocess.run(
        [
            str(helper),
            "pi",
            "100.106.237.106",
            "0" * 32,
            "/unexpected/path",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "expected ACCOUNT TAILSCALE_IPV4 MACHINE_ID" in result.stderr


def test_pi_deployment_docs_describe_local_pose_and_remaining_physical_evidence() -> None:
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    documentation = (_REPO_ROOT / "docs" / "pi-mac-pose.md").read_text(encoding="utf-8")
    ncnn_documentation = (_REPO_ROOT / "docs" / "pi-local-pose-ncnn.md").read_text(encoding="utf-8")

    assert "No Mac pose publisher is required" in documentation
    assert "RECOVERYBOX_POSE_SOURCE=local" in documentation
    assert "eight V4L2 mmap buffers" in documentation
    assert "two NCNN threads" in documentation
    assert "only `openai-api-key`" in documentation
    assert "local pose loads no remote-pose token" in documentation
    assert "`assessable=0` is valid" in documentation
    assert "not a human RTMPose result" in " ".join(documentation.split())
    assert "sustained capture-to-numeric age below 500 ms" in documentation
    assert "`/dev/snd` is not allowed" in documentation
    assert "There is no pose listener in local mode" in documentation

    assert "Pi 3 local deployment for Max" in readme
    assert "[Pi 3 local-pose runbook](docs/pi-mac-pose.md)" in readme
    assert ". ./.env" in readme
    assert "scripts/deploy-pi3.sh --apply" in readme
    assert "Production camera and pose inference run" in readme
    assert "entirely on the Pi" in readme
    assert "long-running hardware composition remain" not in readme

    assert "/opt/recoverybox/runtime/ncnn" in ncnn_documentation
    assert "/opt/recoverybox/models/ncnn" in ncnn_documentation
    assert "/home/pi/recoverybox/models/ncnn" not in ncnn_documentation
    assert "eight mmap buffers" in ncnn_documentation
    assert "exact 10.000 FPS timestamps" in ncnn_documentation
    normalized_ncnn_documentation = " ".join(ncnn_documentation.split())
    assert "It is not evidence" in normalized_ncnn_documentation
    assert "that RTMPose produced a human skeleton" in normalized_ncnn_documentation


def test_reviewed_synthetic_fixture_hashes_are_stable() -> None:
    expected = {
        "squat-standing.jpg": "ad0e809c1a181927b10ec946a9b6faf8498015d96988b0867a5bf01b365ffd91",
        "squat-down.jpg": "b8d1885ba8f78256c8dabcff87bb2d55f5ffd073d2ab12420f8da191b127cd95",
    }

    for name, digest in expected.items():
        payload = (_REPO_ROOT / "fixtures" / "synthetic" / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
