#!/usr/bin/env bash
# Install RecoveryBox as fresh, silent system services on the known Pi.

set -euo pipefail

readonly DEFAULT_USER="pi"
readonly DEFAULT_ADMIN_USER="root"
readonly DEFAULT_PI_IP="100.106.237.106"
readonly DEFAULT_MAC_IP="100.70.100.93"
readonly POSE_PORT="45873"
readonly DEBUG_PORT="45874"
readonly WATCHDOG_SECONDS="0.5"
readonly REMOTE_LEGACY_ROOT="/home/pi/recoverybox"
readonly REMOTE_ROOT="/opt/recoverybox"
readonly REMOTE_APP="$REMOTE_ROOT/app"
readonly REMOTE_NCNN_RUNTIME="$REMOTE_ROOT/runtime/ncnn"
readonly REMOTE_NCNN_MODELS="$REMOTE_ROOT/models/ncnn"
readonly REMOTE_LIBYUV="/usr/lib/arm-linux-gnueabihf/libyuv.so.0"
readonly REMOTE_CONFIG_DIR="/etc/recoverybox"
readonly REMOTE_CONFIG="$REMOTE_CONFIG_DIR/recoverybox.env"
readonly REMOTE_CREDENTIAL_DIR="/etc/recoverybox/credentials"
readonly REMOTE_OPENAI_CREDENTIAL="$REMOTE_CREDENTIAL_DIR/openai-api-key"
readonly REMOTE_STATUS="/run/recoverybox/status.json"
readonly MAIN_UNIT_PATH="/etc/systemd/system/recoverybox.service"
readonly STATUS_UNIT_PATH="/etc/systemd/system/recoverybox-status.service"

apply=0
remote_user="$DEFAULT_USER"
admin_user="$DEFAULT_ADMIN_USER"
readonly pi_ip="$DEFAULT_PI_IP"
readonly mac_ip="$DEFAULT_MAC_IP"

usage() {
    cat <<'EOF'
Usage: scripts/deploy-pi3.sh [--apply] [options]

The default is a read-only preflight. --apply permanently deletes the two
legacy voice-ai-bot services, /opt/voice-ai-bot, and /var/lib/voice-ai-bot.
It creates no backup, archive, compatibility shim, or rollback release.

Options:
  --apply             Perform the irreversible direct replacement
  --user USER         Unprivileged service account (default: pi)
  --admin-user USER   SSH account whose uid is 0 (default: root)
  -h, --help          Show this help

The automated --apply path requires direct root SSH and intentionally refuses
to accept or pipe a password.
EOF
}

fail() {
    printf 'deploy-pi3: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --apply)
            apply=1
            shift
            ;;
        --user)
            (($# >= 2)) || fail "--user requires a value"
            remote_user=$2
            shift 2
            ;;
        --admin-user)
            (($# >= 2)) || fail "--admin-user requires a value"
            admin_user=$2
            shift 2
            ;;
        -h | --help)
            usage
            exit
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

openai_api_key=""
if ((apply)); then
    [[ ${OPENAI_API_KEY+x} == x && -n ${OPENAI_API_KEY:-} ]] ||
        fail "OPENAI_API_KEY must already be set for --apply"
    openai_api_key=$OPENAI_API_KEY
fi
# Do not let any child process inherit the raw key. The apply path retains it
# only in this shell until one bounded stdin transfer over Tailscale SSH.
unset OPENAI_API_KEY

for account in "$remote_user" "$admin_user"; do
    case "$account" in
        "" | *[!A-Za-z0-9_-]*) fail "SSH account contains unsupported characters" ;;
    esac
done
[[ $remote_user == pi ]] || fail "the reviewed Pi unit files require --user pi"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cd "$repo_root"

required_files=(
    pyproject.toml
    deploy/recoverybox_pi_power_gate.py
    deploy/recoverybox_status.py
    deploy/recoverybox_tree_digest.py
    deploy/remove-legacy-voice-ai-bot.sh
    deploy/systemd/recoverybox.service
    deploy/systemd/recoverybox-status.service
    scripts/fetch-pi-pose-ncnn-models.sh
    scripts/fetch-pi-pose-ncnn-runtime.sh
    scripts/verify-pi-pose-ncnn-assets.sh
    src/recoverybox/device/pi_pose_ncnn.py
    src/recoverybox/device/pi_pose_v4l2.py
    src/recoverybox/device/remote_pose_service.py
)
for required in "${required_files[@]}"; do
    [[ -f $required && ! -L $required ]] || fail "missing required file: $required"
done
[[ -x deploy/remove-legacy-voice-ai-bot.sh ]] || fail "legacy deletion helper is not executable"
for helper in scripts/fetch-pi-pose-ncnn-models.sh \
    scripts/fetch-pi-pose-ncnn-runtime.sh scripts/verify-pi-pose-ncnn-assets.sh; do
    [[ -x $helper ]] || fail "required NCNN helper is not executable: $helper"
done

for command in ssh python3; do
    command -v "$command" >/dev/null || fail "$command is required"
done
if command -v tailscale >/dev/null 2>&1; then
    tailscale_cli=$(command -v tailscale)
elif [[ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]]; then
    tailscale_cli=/Applications/Tailscale.app/Contents/MacOS/Tailscale
else
    fail "Tailscale CLI is required (install the macOS app or put tailscale on PATH)"
fi
readonly tailscale_cli
if ((apply)); then
    command -v rsync >/dev/null || fail "rsync is required for --apply"
    command -v scp >/dev/null || fail "scp is required for --apply"
    command -v curl >/dev/null || fail "curl is required for --apply"
    command -v sha256sum >/dev/null || fail "sha256sum is required for --apply"
    command -v unzip >/dev/null || fail "unzip is required for --apply"
    printf '%s' "$openai_api_key" | python3 -c '
import sys

value = sys.stdin.buffer.read(4097)
raise SystemExit(not (value and len(value) <= 4096 and all(33 <= byte <= 126 for byte in value)))
' || fail "OPENAI_API_KEY must be one printable non-whitespace token of at most 4096 bytes"
fi

expected_app_digest=$(
    python3 deploy/recoverybox_tree_digest.py --root "$repo_root"
) || fail "could not compute the local RecoveryBox application digest"
[[ $expected_app_digest =~ ^[0-9a-f]{64}$ ]] ||
    fail "local RecoveryBox application digest is invalid"
readonly expected_app_digest
local_file_digest() {
    python3 - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}
tree_helper_digest=$(local_file_digest deploy/recoverybox_tree_digest.py) ||
    fail "could not hash the trusted local tree-digest helper"
[[ $tree_helper_digest =~ ^[0-9a-f]{64}$ ]] || fail "tree-digest helper hash is invalid"
readonly tree_helper_digest
power_gate_digest=$(local_file_digest deploy/recoverybox_pi_power_gate.py) ||
    fail "could not hash the trusted Pi power-gate helper"
[[ $power_gate_digest =~ ^[0-9a-f]{64}$ ]] || fail "Pi power-gate helper hash is invalid"
readonly power_gate_digest
verify_local_tree_helper() {
    local actual
    actual=$(local_file_digest deploy/recoverybox_tree_digest.py) ||
        fail "could not re-hash the trusted local tree-digest helper"
    [[ $actual == "$tree_helper_digest" ]] || fail "trusted local tree-digest helper changed"
}

python3 - "$pi_ip" "$mac_ip" <<'PY'
import ipaddress
import sys

tailnet = ipaddress.ip_network("100.64.0.0/10")
pi_address, mac_address = (ipaddress.ip_address(value) for value in sys.argv[1:])
if pi_address.version != 4 or mac_address.version != 4:
    raise SystemExit("deployment requires Tailscale IPv4 addresses")
if pi_address not in tailnet or mac_address not in tailnet or pi_address == mac_address:
    raise SystemExit("Pi and Mac must have distinct Tailscale IPv4 addresses")
PY

"$tailscale_cli" ip -4 | grep -Fqx "$mac_ip" ||
    fail "this Mac does not own the pinned Tailscale IPv4 address $mac_ip"

# Both privilege levels connect to the verified literal Tailnet endpoint. A
# LAN hostname, SSH alias, ProxyJump destination, or other caller-selected
# host can therefore never become the target of the destructive root phase.
readonly ssh_target="${remote_user}@${pi_ip}"
readonly admin_target="${admin_user}@${pi_ip}"
ssh_options=(
    -F /dev/null
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ProxyCommand=none
    -o ProxyJump=none
)

printf 'Read-only Pi preflight: %s (expected Tailnet IP %s)\n' "$ssh_target" "$pi_ip"
ssh "${ssh_options[@]}" "$ssh_target" bash -s -- \
    "$remote_user" "$pi_ip" "$REMOTE_LEGACY_ROOT" "$REMOTE_STATUS" "$REMOTE_LIBYUV" \
    <<'REMOTE_PI_PREFLIGHT'
set -eu

expected_user=$1
expected_ip=$2
legacy_root=$3
recovery_status=$4
libyuv=$5

die() {
    printf 'Pi preflight failed: %s\n' "$*" >&2
    exit 1
}

[ "$(id -un)" = "$expected_user" ] || die "unexpected SSH account"
[ "$(readlink -f "$HOME")" = "/home/$expected_user" ] || die "unexpected home directory"
command -v tailscale >/dev/null 2>&1 || die "tailscale is unavailable"
tailscale ip -4 | grep -Fqx "$expected_ip" || die "expected Pi Tailnet IP is absent"
for command in python3 ss curl sha256sum timeout; do
    command -v "$command" >/dev/null 2>&1 || die "$command is unavailable"
done
python3 - <<'PY' || die "system Python/runtime ABI is not the pinned Pi 3 target"
import platform
import sys

if sys.version_info[:2] != (3, 13):
    raise SystemExit(1)
if platform.machine() != "armv7l":
    raise SystemExit(1)
libc_name, libc_version = platform.libc_ver()
if libc_name != "glibc" or tuple(int(part) for part in libc_version.split(".")[:2]) < (2, 31):
    raise SystemExit(1)
PY
python3 - <<'PY' || die "system Python requires libgpiod 2.x"
import gpiod

if not getattr(gpiod, "__version__", "").startswith("2."):
    raise SystemExit(1)
PY
[ -c /dev/gpiochip0 ] && [ -r /dev/gpiochip0 ] && [ -w /dev/gpiochip0 ] ||
    die "/dev/gpiochip0 is unavailable to the service account"
[ -c /dev/video0 ] && [ -r /dev/video0 ] && [ -w /dev/video0 ] ||
    die "/dev/video0 is unavailable to the service account"
groups=$(id -nG)
printf '%s\n' "$groups" | tr ' ' '\n' | grep -Fqx gpio || die "account is not in gpio group"
printf '%s\n' "$groups" | tr ' ' '\n' | grep -Fqx video || die "account is not in video group"
[ -f "$libyuv" ] && [ -r "$libyuv" ] ||
    die "$libyuv is unavailable"
python3 - "$libyuv" <<'PY' || die "libyuv runtime could not be loaded"
import ctypes
import sys

library = ctypes.CDLL(sys.argv[1])
if not callable(getattr(library, "YUY2ToARGB", None)):
    raise SystemExit(1)
PY

validate_file_if_present() {
    path=$1
    mode=$2
    if [ -e "$path" ] || [ -L "$path" ]; then
        [ -f "$path" ] && [ ! -L "$path" ] || die "$path is not a regular file"
        [ "$(stat -c %U:%G "$path")" = "${expected_user}:${expected_user}" ] ||
            die "$path has an unexpected owner"
        [ "$(stat -c %a "$path")" = "$mode" ] || die "$path has an unexpected mode"
    fi
}
if [ -e "$legacy_root" ] || [ -L "$legacy_root" ]; then
    [ -d "$legacy_root" ] && [ ! -L "$legacy_root" ] ||
        die "$legacy_root is not a real directory"
    [ "$(readlink -f "$legacy_root")" = "$legacy_root" ] ||
        die "$legacy_root resolves elsewhere"
    owner=$(stat -c %U:%G "$legacy_root")
    mode=$(stat -c %a "$legacy_root")
    if [ "$owner" != "${expected_user}:${expected_user}" ] || [ "$mode" != 700 ]; then
        [ "$owner" = "root:${expected_user}" ] && [ "$mode" = 1770 ] ||
            die "$legacy_root is not the known managed tree"
    fi
    marker="$legacy_root/.recoverybox-managed-v1"
    [ -f "$marker" ] && [ ! -L "$marker" ] ||
        die "legacy RecoveryBox tree lacks its exact marker"
    [ "$(stat -c %h "$marker")" = 1 ] || die "legacy marker is hard-linked"
    [ "$(cat "$marker")" = recoverybox-managed-v1 ] ||
        die "legacy RecoveryBox marker is invalid"
fi
validate_file_if_present "$recovery_status" 600
REMOTE_PI_PREFLIGHT

pi_machine_id=$(ssh "${ssh_options[@]}" "$ssh_target" 'tr -d "\n" </etc/machine-id')
[[ $pi_machine_id =~ ^[0-9a-f]{32}$ ]] || fail "Pi returned an invalid machine id"

printf 'Read-only root preflight: %s\n' "$admin_target"
ssh "${ssh_options[@]}" "$admin_target" bash -s -- \
    "$remote_user" "$pi_ip" "$pi_machine_id" "$REMOTE_ROOT" "$REMOTE_APP" \
    "$REMOTE_NCNN_RUNTIME" "$REMOTE_NCNN_MODELS" \
    "$REMOTE_CONFIG_DIR" "$REMOTE_CONFIG" "$REMOTE_CREDENTIAL_DIR" \
    "$REMOTE_OPENAI_CREDENTIAL" <<'REMOTE_ROOT_PREFLIGHT'
set -eu

expected_user=$1
expected_ip=$2
expected_machine_id=$3
app_root=$4
app=$5
runtime=$6
models=$7
config_dir=$8
config=$9
credential_dir=${10}
openai_credential=${11}
die() {
    printf 'root preflight failed: %s\n' "$*" >&2
    exit 1
}

reject_target_or_descendant_mounts() {
    target=$1
    mount_targets=$(LC_ALL=C findmnt --raw --noheadings --output TARGET) ||
        die "could not enumerate mountpoints before inspecting $target"
    if ! printf '%s\n' "$mount_targets" | awk -v target="$target" '
        $0 == target || index($0, target "/") == 1 { unsafe = 1 }
        END { exit(unsafe ? 1 : 0) }
    '; then
        die "$target or one of its descendants is a mountpoint"
    fi
}

[ "$(id -u)" -eq 0 ] || die "admin SSH account does not have uid 0"
[ "$(tr -d '\n' </etc/machine-id)" = "$expected_machine_id" ] ||
    die "admin SSH reached a different machine"
command -v tailscale >/dev/null 2>&1 || die "tailscale is unavailable"
tailscale ip -4 | grep -Fqx "$expected_ip" ||
    die "admin SSH target does not own the expected Tailnet address"
for command in systemctl systemd-analyze timeout python3 runuser findmnt; do
    command -v "$command" >/dev/null 2>&1 || die "$command is unavailable"
done
systemd_version=$(systemd-analyze --version | sed -n '1s/^systemd \([0-9][0-9]*\).*/\1/p')
[ -n "$systemd_version" ] && [ "$systemd_version" -ge 247 ] ||
    die "systemd 247 or newer is required for file credentials"

validate_regular_if_present() {
    path=$1
    owner=$2
    if [ -e "$path" ] || [ -L "$path" ]; then
        [ -f "$path" ] && [ ! -L "$path" ] || die "$path is not a regular file"
        [ "$(stat -c %U:%G "$path")" = "$owner" ] || die "$path has an unexpected owner"
    fi
}
validate_directory_if_present() {
    path=$1
    if [ -e "$path" ] || [ -L "$path" ]; then
        [ -d "$path" ] && [ ! -L "$path" ] || die "$path is not a real directory"
        [ "$(stat -c %U "$path")" = "$expected_user" ] ||
            die "$path has an unexpected owner"
        [ "$(readlink -f "$path")" = "$path" ] || die "$path resolves elsewhere"
    fi
}
validate_root_directory_if_present() {
    path=$1
    mode=$2
    if [ -e "$path" ] || [ -L "$path" ]; then
        [ -d "$path" ] && [ ! -L "$path" ] || die "$path is not a real directory"
        [ "$(stat -c %U:%G "$path")" = root:root ] || die "$path has an unexpected owner"
        [ "$(stat -c %a "$path")" = "$mode" ] || die "$path has an unexpected mode"
        [ "$(readlink -f "$path")" = "$path" ] || die "$path resolves elsewhere"
    fi
}

validate_root_file_if_present() {
    path=$1
    mode=$2
    if [ -e "$path" ] || [ -L "$path" ]; then
        [ -f "$path" ] && [ ! -L "$path" ] || die "$path is not a regular file"
        [ "$(stat -c %U:%G "$path")" = root:root ] ||
            die "$path has an unexpected owner"
        [ "$(stat -c %a "$path")" = "$mode" ] || die "$path has an unexpected mode"
        [ "$(stat -c %h "$path")" = 1 ] || die "$path is hard-linked"
    fi
}

validate_regular_if_present /etc/systemd/system/voice-ai-bot.service root:root
validate_regular_if_present /etc/systemd/system/voice-ai-bot-debug.service root:root
validate_regular_if_present /etc/systemd/system/recoverybox.service root:root
validate_regular_if_present /etc/systemd/system/recoverybox-status.service root:root
[ "$(dirname "$credential_dir")" = "$config_dir" ] || die "credential path is inconsistent"
reject_target_or_descendant_mounts "$app_root"
reject_target_or_descendant_mounts "/home/$expected_user/recoverybox"
reject_target_or_descendant_mounts "$config_dir"
validate_root_directory_if_present "$app_root" 755
if [ -e "$app_root" ] || [ -L "$app_root" ]; then
    marker="$app_root/.recoverybox-managed-v2"
    validate_root_file_if_present "$marker" 600
    [ -f "$marker" ] || die "existing canonical RecoveryBox root lacks its marker"
    case "$(cat "$marker")" in recoverybox-managed-v2:*) ;; *) die "invalid marker" ;; esac
    proof="$app_root/.runtime-proven-v2"
    if [ -e "$proof" ] || [ -L "$proof" ]; then
        validate_root_file_if_present "$proof" 600
        [ "$(cat "$proof")" = "$(cat "$marker")" ] || die "runtime proof marker is stale"
    fi
    if [ -e "$app" ] || [ -L "$app" ]; then
        [ -d "$app" ] && [ ! -L "$app" ] || die "$app is not a real directory"
        [ "$(stat -c %U:%G "$app")" = root:root ] || die "$app has an unexpected owner"
        [ "$(stat -c %a "$app")" = 755 ] || die "$app has an unexpected mode"
        [ "$(readlink -f "$app")" = "$app" ] || die "$app resolves elsewhere"
    fi
    for asset_directory in "$app_root/runtime" "$runtime" "$app_root/models" "$models"; do
        if [ -e "$asset_directory" ] || [ -L "$asset_directory" ]; then
            [ -d "$asset_directory" ] && [ ! -L "$asset_directory" ] ||
                die "$asset_directory is not a real directory"
            [ "$(stat -c %U:%G "$asset_directory")" = root:root ] ||
                die "$asset_directory has an unexpected owner"
            [ "$(stat -c %a "$asset_directory")" = 755 ] ||
                die "$asset_directory has an unexpected mode"
            [ "$(readlink -f "$asset_directory")" = "$asset_directory" ] ||
                die "$asset_directory resolves elsewhere"
        fi
    done
fi
validate_root_directory_if_present "$config_dir" 700
validate_root_directory_if_present "$credential_dir" 700
validate_root_file_if_present "$config" 600
validate_root_file_if_present "$openai_credential" 600
if [ -e "$openai_credential" ] || [ -L "$openai_credential" ]; then
    python3 - "$openai_credential" <<'PY' || die "existing OpenAI credential is invalid"
import sys
from pathlib import Path

value = Path(sys.argv[1]).read_bytes()
if not value or len(value) > 4096 or any(not 33 <= byte <= 126 for byte in value):
    raise SystemExit(1)
PY
fi
validate_directory_if_present /opt/voice-ai-bot
validate_directory_if_present /var/lib/voice-ai-bot
for dropin in \
    /etc/systemd/system/recoverybox.service.d \
    /etc/systemd/system/recoverybox-status.service.d \
    /run/systemd/system/recoverybox.service.d \
    /run/systemd/system/recoverybox-status.service.d
do
    [ ! -e "$dropin" ] && [ ! -L "$dropin" ] ||
        die "unexpected RecoveryBox unit override exists: $dropin"
done

for spec in \
    'voice-ai-bot.service:/etc/systemd/system/voice-ai-bot.service' \
    'voice-ai-bot-debug.service:/etc/systemd/system/voice-ai-bot-debug.service' \
    'recoverybox.service:/etc/systemd/system/recoverybox.service' \
    'recoverybox-status.service:/etc/systemd/system/recoverybox-status.service'
do
    unit=${spec%%:*}
    expected=${spec#*:}
    load_state=$(systemctl show --property=LoadState --value "$unit" 2>/dev/null) ||
        die "could not inspect $unit load state"
    fragment=$(systemctl show --property=FragmentPath --value "$unit" 2>/dev/null) ||
        die "could not inspect $unit fragment"
    if [ "$load_state" != not-found ] && [ -n "$load_state" ]; then
        [ "$fragment" = "$expected" ] || die "$unit resolves outside its exact expected fragment"
    fi
    case "$unit" in
        recoverybox.service | recoverybox-status.service)
            [ -z "$(systemctl show --property=DropInPaths --value "$unit")" ] ||
                die "$unit has an unexpected override"
            ;;
    esac
done
REMOTE_ROOT_PREFLIGHT

# Run the reviewed helper directly from stdin so dry-run preflight can reject
# active undervoltage/throttling before any deployment stage exists.
ssh "${ssh_options[@]}" "$admin_target" /usr/bin/python3 - \
    <deploy/recoverybox_pi_power_gate.py

printf 'Preflight passed: Pi 3 ABI, camera, GPIO, fresh paths, root authority, and Tailnet identity.\n'
if ((!apply)); then
    cat <<EOF
Dry run only; nothing was changed.

With --apply this script will:
  1. stage only newly written RecoveryBox source plus the exact pinned ARMv7
     NCNN runtime and four reviewed model files under $REMOTE_ROOT;
  2. provision the already-set OPENAI_API_KEY as a root-only systemd file
     credential and write a new local-pose environment;
  3. permanently delete the exact legacy units, /opt/voice-ai-bot, and
     /var/lib/voice-ai-bot without copying or renaming any of them;
  4. install recoverybox.service and recoverybox-status.service as root-owned
     units that run as pi; and
  5. start Pi-local camera/pose with audio disabled, /dev/video0 and GPIO
     allowed, and /dev/snd blocked by DevicePolicy=closed.

There is no automatic or retained rollback. Re-run with --apply for the direct replacement.
EOF
    exit
fi

readonly deployment_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly deployment_marker="recoverybox-managed-v2:$deployment_id"
readonly remote_unit_stage="/run/recoverybox-deploy-$deployment_id"
readonly remote_app_stage="$remote_unit_stage/app"
readonly remote_asset_stage="$remote_unit_stage/assets"
readonly remote_asset_verify_stage="$remote_unit_stage/verify-pi-pose-ncnn-assets.sh"
readonly remote_config_stage="$remote_unit_stage/recoverybox.env"
readonly remote_openai_stage="$remote_unit_stage/openai-api-key"
local_asset_stage=""

remote_unit_stage_pending=0
activated_app_pending=0

cleanup_deployment_stages() {
    set +e
    if [[ -n $local_asset_stage ]]; then
        case "$local_asset_stage" in
            /private/tmp/recoverybox-pi3-assets.*)
                [[ -d $local_asset_stage && ! -L $local_asset_stage ]] &&
                    rm -rf -- "$local_asset_stage"
                ;;
        esac
    fi
    if ((remote_unit_stage_pending || activated_app_pending)); then
        ssh "${ssh_options[@]}" "$admin_target" bash -s -- \
            "$pi_ip" "$pi_machine_id" "$remote_unit_stage" "$REMOTE_ROOT" \
            "$deployment_marker" "$remote_unit_stage_pending" \
            "$activated_app_pending" <<'REMOTE_CLEAN_DEPLOYMENT' \
            >/dev/null 2>&1
set -eu
expected_ip=$1
expected_machine_id=$2
stage=$3
app_root=$4
expected_marker=$5
remove_stage=$6
remove_activated=$7
reject_target_or_descendant_mounts() {
    target=$1
    mount_targets=$(LC_ALL=C findmnt --raw --noheadings --output TARGET) || {
        printf 'could not enumerate mountpoints before inspecting %s\n' "$target" >&2
        exit 1
    }
    if ! printf '%s\n' "$mount_targets" | awk -v target="$target" '
        $0 == target || index($0, target "/") == 1 { unsafe = 1 }
        END { exit(unsafe ? 1 : 0) }
    '; then
        printf 'refusing to remove mounted target tree: %s\n' "$target" >&2
        exit 1
    fi
}
[ "$(id -u)" -eq 0 ] || exit 1
[ "$(tr -d '\n' </etc/machine-id)" = "$expected_machine_id" ] || exit 1
tailscale ip -4 | grep -Fqx "$expected_ip" || exit 1
if [ "$remove_stage" -eq 1 ] && { [ -e "$stage" ] || [ -L "$stage" ]; }; then
    case "$stage" in /run/recoverybox-deploy-*) ;; *) exit 1 ;; esac
    [ -d "$stage" ] && [ ! -L "$stage" ] || exit 1
    [ "$(stat -c %U:%G "$stage")" = root:root ] || exit 1
    rm -rf --one-file-system -- "$stage"
fi
if [ "$remove_activated" -eq 1 ] && { [ -e "$app_root" ] || [ -L "$app_root" ]; }; then
    [ "$app_root" = /opt/recoverybox ] || exit 1
    [ -d "$app_root" ] && [ ! -L "$app_root" ] || exit 1
    [ "$(readlink -f "$app_root")" = "$app_root" ] || exit 1
    marker="$app_root/.recoverybox-managed-v2"
    [ -f "$marker" ] && [ ! -L "$marker" ] || exit 1
    [ "$(cat "$marker")" = "$expected_marker" ] || exit 1
    [ "$(stat -c %U:%G "$app_root")" = root:root ] || exit 1
    [ "$(stat -c %a "$app_root")" = 755 ] || exit 1
    proof="$app_root/.runtime-proven-v2"
    if [ ! -f "$proof" ] || [ -L "$proof" ] ||
        [ "$(cat "$proof" 2>/dev/null)" != "$expected_marker" ]; then
        reject_target_or_descendant_mounts "$app_root"
        rm -rf --one-file-system -- "$app_root"
    fi
fi
REMOTE_CLEAN_DEPLOYMENT
    fi
}
trap cleanup_deployment_stages EXIT

printf 'Fetching and verifying the exact pinned Pi 3 NCNN runtime and models.\n'
local_asset_stage=$(mktemp -d /private/tmp/recoverybox-pi3-assets.XXXXXX) ||
    fail "could not create the local NCNN asset stage"
scripts/fetch-pi-pose-ncnn-runtime.sh "$local_asset_stage/runtime/ncnn" ||
    fail "could not fetch the pinned ARMv7 NCNN runtime"
scripts/fetch-pi-pose-ncnn-models.sh "$local_asset_stage/models/ncnn" ||
    fail "could not fetch the pinned NCNN pose models"

printf 'Preparing one root-owned deployment stage (no legacy data is copied).\n'
remote_unit_stage_pending=1
ssh "${ssh_options[@]}" "$admin_target" bash -s -- \
    "$pi_ip" "$pi_machine_id" "$remote_unit_stage" "$remote_app_stage" \
    "$remote_asset_stage" \
    <<'REMOTE_PREPARE_STAGE'
set -eu
expected_ip=$1
expected_machine_id=$2
stage=$3
app_stage=$4
asset_stage=$5
[ "$(id -u)" -eq 0 ] || exit 1
[ "$(tr -d '\n' </etc/machine-id)" = "$expected_machine_id" ] || exit 1
tailscale ip -4 | grep -Fqx "$expected_ip" || exit 1
stage_prefix=/run/recoverybox-deploy-
stage_suffix=${stage#"$stage_prefix"}
[ "$stage_suffix" != "$stage" ] && [ -n "$stage_suffix" ] &&
    [ "${stage_suffix#*/}" = "$stage_suffix" ] || exit 1
[ "$app_stage" = "$stage/app" ] || exit 1
[ "$asset_stage" = "$stage/assets" ] || exit 1
[ ! -e "$stage" ] && [ ! -L "$stage" ] || exit 1
mkdir -m 700 "$stage"
mkdir -m 755 "$app_stage" "$app_stage/src" "$app_stage/deploy"
mkdir -m 755 "$asset_stage" "$asset_stage/runtime" "$asset_stage/models"
REMOTE_PREPARE_STAGE

rsync_options=(-rpt --delete --exclude=__pycache__/ --exclude='*.pyc')
rsync_ssh='ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o ProxyCommand=none -o ProxyJump=none'
rsync "${rsync_options[@]}" -e "$rsync_ssh" src/ "$admin_target:$remote_app_stage/src/"
rsync -pt -e "$rsync_ssh" \
    deploy/recoverybox_pi_power_gate.py deploy/recoverybox_status.py \
    deploy/recoverybox_tree_digest.py \
    "$admin_target:$remote_app_stage/deploy/"
rsync "${rsync_options[@]}" -e "$rsync_ssh" \
    "$local_asset_stage/runtime/" "$admin_target:$remote_asset_stage/runtime/"
rsync "${rsync_options[@]}" -e "$rsync_ssh" \
    "$local_asset_stage/models/" "$admin_target:$remote_asset_stage/models/"
scp -q "${ssh_options[@]}" scripts/verify-pi-pose-ncnn-assets.sh \
    "$admin_target:$remote_asset_verify_stage"

printf 'Verifying the staged RecoveryBox application.\n'
verify_local_tree_helper
if ! ssh "${ssh_options[@]}" "$admin_target" /usr/bin/python3 - \
    --root "$remote_app_stage" --strict --expect "$expected_app_digest" \
    <deploy/recoverybox_tree_digest.py >/dev/null; then
    fail "staged RecoveryBox application did not match the trusted local tree digest"
fi
ssh "${ssh_options[@]}" "$admin_target" bash -s -- \
    "$remote_app_stage" <<'REMOTE_VERIFY_STAGE'
set -eu
stage=$1

if find "$stage" ! -user root -print -quit | grep -q .; then
    printf 'staged application has a non-root owner\n' >&2
    exit 1
fi
chmod -R u=rwX,go=rX "$stage"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$stage/src" /usr/bin/python3 - "$stage" <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

stage = Path(sys.argv[1]).resolve()
import recoverybox
import recoverybox.device.pi_pose_ncnn
import recoverybox.device.pi_pose_v4l2
import recoverybox.device.remote_pose_service

for module, root in (
    (recoverybox, stage / "src/recoverybox"),
    (recoverybox.device.pi_pose_ncnn, stage / "src/recoverybox/device"),
    (recoverybox.device.pi_pose_v4l2, stage / "src/recoverybox/device"),
    (recoverybox.device.remote_pose_service, stage / "src/recoverybox/device"),
):
    if os.path.commonpath((Path(module.__file__).resolve(), root)) != str(root):
        raise SystemExit(1)

for filename, module_name in (
    ("recoverybox_pi_power_gate.py", "_recoverybox_staged_power_gate"),
    ("recoverybox_status.py", "_recoverybox_staged_status"),
):
    path = stage / "deploy" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(1)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    if not callable(module.main):
        raise SystemExit(1)
PY
REMOTE_VERIFY_STAGE

ssh "${ssh_options[@]}" "$admin_target" bash -s -- \
    "$remote_asset_stage" "$remote_asset_verify_stage" <<'REMOTE_VERIFY_ASSET_STAGE'
set -eu
assets=$1
verifier=$2
[ -d "$assets" ] && [ ! -L "$assets" ] || exit 1
[ -f "$verifier" ] && [ ! -L "$verifier" ] || exit 1
[ "$(stat -c %U:%G "$assets")" = root:root ] || exit 1
[ "$(stat -c %U:%G "$verifier")" = root:root ] || exit 1
chmod 0755 "$verifier"
find "$assets" -type d -exec chmod 0755 {} +
find "$assets" -type f -exec chmod 0644 {} +
/bin/bash "$verifier" "$assets" root:root
REMOTE_VERIFY_ASSET_STAGE

printf 'Writing a new RecoveryBox-only environment with audio disabled.\n'
ssh "${ssh_options[@]}" "$admin_target" python3 - \
    "$remote_config_stage" "$pi_ip" "$mac_ip" "$POSE_PORT" \
    "$DEBUG_PORT" "$WATCHDOG_SECONDS" "$REMOTE_STATUS" "$REMOTE_NCNN_RUNTIME" \
    "$REMOTE_NCNN_MODELS" "$REMOTE_LIBYUV" <<'PY'
import os
import sys
from pathlib import Path

(
    config_raw,
    pi_ip,
    mac_ip,
    pose_port,
    debug_port,
    watchdog,
    status,
    runtime,
    models,
    libyuv,
) = sys.argv[1:]
config = Path(config_raw)
temporary = config.with_name(f".{config.name}.{os.getpid()}.tmp")
values = {
    "RECOVERYBOX_POSE_SOURCE": "local",
    "RECOVERYBOX_POSE_BIND_HOST": pi_ip,
    "RECOVERYBOX_POSE_PORT": pose_port,
    "RECOVERYBOX_POSE_ALLOWED_PEER": mac_ip,
    "RECOVERYBOX_POSE_TOKEN_FILE": "/run/recoverybox/unused-local-pose-token",
    "RECOVERYBOX_POSE_WATCHDOG_SECONDS": watchdog,
    "RECOVERYBOX_CAMERA_DEVICE": "/dev/video0",
    "RECOVERYBOX_LIBYUV_LIBRARY_PATH": libyuv,
    "RECOVERYBOX_CAMERA_WIDTH": "640",
    "RECOVERYBOX_CAMERA_HEIGHT": "480",
    "RECOVERYBOX_CAMERA_FPS": "10",
    "RECOVERYBOX_POSE_FRAME_TIMEOUT_SECONDS": "0.1",
    "RECOVERYBOX_LOCAL_POSE_WORKER_TIMEOUT_SECONDS": "0.5",
    "RECOVERYBOX_LOCAL_POSE_START_TIMEOUT_SECONDS": "25",
    "RECOVERYBOX_V4L2_BUFFER_COUNT": "8",
    "RECOVERYBOX_NCNN_RUNTIME_PATH": runtime,
    "RECOVERYBOX_RTMPOSE_PARAM_PATH": f"{models}/rtmpose-t.param",
    "RECOVERYBOX_RTMPOSE_BIN_PATH": f"{models}/rtmpose-t.bin",
    "RECOVERYBOX_NANODET_PARAM_PATH": f"{models}/nanodet-m-int8.param",
    "RECOVERYBOX_NANODET_BIN_PATH": f"{models}/nanodet-m-int8.bin",
    "RECOVERYBOX_NCNN_THREADS": "2",
    "RECOVERYBOX_PERSON_SCORE_THRESHOLD": "0.55",
    "RECOVERYBOX_PERSON_NMS_IOU": "0.3",
    "RECOVERYBOX_MIN_PERSON_AREA_FRACTION": "0.05",
    "RECOVERYBOX_LOCAL_POSE_MAX_AGE_SECONDS": "0.5",
    "RECOVERYBOX_STATUS_PATH": status,
    "RECOVERYBOX_STATUS_MAX_AGE_SECONDS": "2.0",
    "RECOVERYBOX_STATUS_ALLOWED_PEER": mac_ip,
    "RECOVERYBOX_DEBUG_PORT": debug_port,
    "RECOVERYBOX_AUDIO_ENABLED": "0",
    "RECOVERYBOX_PLAYBACK_DEVICE": "none",
    "RECOVERYBOX_BUTTON_GPIO": "23",
    "RECOVERYBOX_TRANSCRIPT_RETENTION": "false",
}
if any("\n" in key or "\n" in value for key, value in values.items()):
    raise SystemExit(1)
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, config)
finally:
    temporary.unlink(missing_ok=True)
config.chmod(0o600)
PY

main_digest=$(local_file_digest deploy/systemd/recoverybox.service)
status_digest=$(local_file_digest deploy/systemd/recoverybox-status.service)
asset_verify_digest=$(local_file_digest scripts/verify-pi-pose-ncnn-assets.sh)

# The OpenAI secret crosses the pinned Tailnet root SSH connection on stdin
# only. It is never placed in argv, the shared environment, or command output.
printf '%s' "$openai_api_key" | ssh "${ssh_options[@]}" "$admin_target" \
    "set -eu; umask 077; target=$remote_openai_stage; temporary=\${target}.tmp; trap 'rm -f -- \"\$temporary\"' EXIT HUP INT TERM; cat >\"\$temporary\"; python3 -c 'import sys; from pathlib import Path; value = Path(sys.argv[1]).read_bytes(); raise SystemExit(not (value and len(value) <= 4096 and all(33 <= byte <= 126 for byte in value)))' \"\$temporary\"; chmod 600 \"\$temporary\"; mv -- \"\$temporary\" \"\$target\"; trap - EXIT HUP INT TERM"
openai_api_key=""

scp -q "${ssh_options[@]}" deploy/systemd/recoverybox.service \
    "$admin_target:$remote_unit_stage/recoverybox.service"
scp -q "${ssh_options[@]}" deploy/systemd/recoverybox-status.service \
    "$admin_target:$remote_unit_stage/recoverybox-status.service"
scp -q "${ssh_options[@]}" deploy/recoverybox_tree_digest.py \
    "$admin_target:$remote_unit_stage/recoverybox_tree_digest.py"
config_digest=$(ssh "${ssh_options[@]}" "$admin_target" \
    "sha256sum $remote_config_stage" | awk '{print $1}')
[[ $config_digest =~ ^[0-9a-f]{64}$ ]] || fail "staged environment hash is invalid"
ssh "${ssh_options[@]}" "$admin_target" bash -s -- \
    "$pi_ip" "$pi_machine_id" "$remote_unit_stage" "$main_digest" \
    "$status_digest" "$config_digest" "$tree_helper_digest" "$asset_verify_digest" \
    "$mac_ip" "$POSE_PORT" "$DEBUG_PORT" "$WATCHDOG_SECONDS" "$REMOTE_STATUS" \
    "$REMOTE_NCNN_RUNTIME" "$REMOTE_NCNN_MODELS" "$REMOTE_LIBYUV" \
    <<'REMOTE_VERIFY_UNIT_STAGE'
set -eu
expected_ip=$1
expected_machine_id=$2
stage=$3
main_digest=$4
status_digest=$5
config_digest=$6
tree_helper_digest=$7
asset_verify_digest=$8
allowed_peer=$9
pose_port=${10}
debug_port=${11}
watchdog=${12}
status_path=${13}
runtime=${14}
models=${15}
libyuv=${16}
[ "$(id -u)" -eq 0 ] || exit 1
[ "$(tr -d '\n' </etc/machine-id)" = "$expected_machine_id" ] || exit 1
tailscale ip -4 | grep -Fqx "$expected_ip" || exit 1
[ -d "$stage" ] && [ ! -L "$stage" ] || exit 1
[ "$(stat -c %U:%G "$stage")" = root:root ] || exit 1
main="$stage/recoverybox.service"
status="$stage/recoverybox-status.service"
config="$stage/recoverybox.env"
openai_credential="$stage/openai-api-key"
tree_helper="$stage/recoverybox_tree_digest.py"
asset_verifier="$stage/verify-pi-pose-ncnn-assets.sh"
assets="$stage/assets"
for path in "$main" "$status" "$config" "$openai_credential" \
    "$tree_helper" "$asset_verifier"; do
    [ -f "$path" ] && [ ! -L "$path" ] || exit 1
    [ "$(stat -c %U:%G "$path")" = root:root ] || exit 1
    [ "$(stat -c %h "$path")" = 1 ] || exit 1
done
for path in "$config" "$openai_credential"; do
    [ "$(stat -c %a "$path")" = 600 ] || exit 1
done
python3 - "$openai_credential" "$config" "$expected_ip" \
    "$allowed_peer" "$pose_port" "$debug_port" "$watchdog" "$status_path" \
    "$runtime" "$models" "$libyuv" <<'PY'
import sys
from pathlib import Path

openai_path, config_path = (Path(raw) for raw in sys.argv[1:3])
value = openai_path.read_bytes()
if not value or len(value) > 4096 or any(not 33 <= byte <= 126 for byte in value):
    raise SystemExit(1)
runtime, models, libyuv = sys.argv[9:12]
expected = {
    "RECOVERYBOX_POSE_SOURCE": "local",
    "RECOVERYBOX_POSE_BIND_HOST": sys.argv[3],
    "RECOVERYBOX_POSE_PORT": sys.argv[5],
    "RECOVERYBOX_POSE_ALLOWED_PEER": sys.argv[4],
    "RECOVERYBOX_POSE_TOKEN_FILE": "/run/recoverybox/unused-local-pose-token",
    "RECOVERYBOX_POSE_WATCHDOG_SECONDS": sys.argv[7],
    "RECOVERYBOX_CAMERA_DEVICE": "/dev/video0",
    "RECOVERYBOX_LIBYUV_LIBRARY_PATH": libyuv,
    "RECOVERYBOX_CAMERA_WIDTH": "640",
    "RECOVERYBOX_CAMERA_HEIGHT": "480",
    "RECOVERYBOX_CAMERA_FPS": "10",
    "RECOVERYBOX_POSE_FRAME_TIMEOUT_SECONDS": "0.1",
    "RECOVERYBOX_LOCAL_POSE_WORKER_TIMEOUT_SECONDS": "0.5",
    "RECOVERYBOX_LOCAL_POSE_START_TIMEOUT_SECONDS": "25",
    "RECOVERYBOX_V4L2_BUFFER_COUNT": "8",
    "RECOVERYBOX_NCNN_RUNTIME_PATH": runtime,
    "RECOVERYBOX_RTMPOSE_PARAM_PATH": f"{models}/rtmpose-t.param",
    "RECOVERYBOX_RTMPOSE_BIN_PATH": f"{models}/rtmpose-t.bin",
    "RECOVERYBOX_NANODET_PARAM_PATH": f"{models}/nanodet-m-int8.param",
    "RECOVERYBOX_NANODET_BIN_PATH": f"{models}/nanodet-m-int8.bin",
    "RECOVERYBOX_NCNN_THREADS": "2",
    "RECOVERYBOX_PERSON_SCORE_THRESHOLD": "0.55",
    "RECOVERYBOX_PERSON_NMS_IOU": "0.3",
    "RECOVERYBOX_MIN_PERSON_AREA_FRACTION": "0.05",
    "RECOVERYBOX_LOCAL_POSE_MAX_AGE_SECONDS": "0.5",
    "RECOVERYBOX_STATUS_PATH": sys.argv[8],
    "RECOVERYBOX_STATUS_MAX_AGE_SECONDS": "2.0",
    "RECOVERYBOX_STATUS_ALLOWED_PEER": sys.argv[4],
    "RECOVERYBOX_DEBUG_PORT": sys.argv[6],
    "RECOVERYBOX_AUDIO_ENABLED": "0",
    "RECOVERYBOX_PLAYBACK_DEVICE": "none",
    "RECOVERYBOX_BUTTON_GPIO": "23",
    "RECOVERYBOX_TRANSCRIPT_RETENTION": "false",
}
actual = {}
for line in config_path.read_text(encoding="utf-8").splitlines():
    if not line or "=" not in line:
        raise SystemExit(1)
    key, item = line.split("=", 1)
    if key in actual:
        raise SystemExit(1)
    actual[key] = item
if actual != expected or "PYTHONPATH" in actual:
    raise SystemExit(1)
PY
[ "$(sha256sum "$main" | awk '{print $1}')" = "$main_digest" ] || exit 1
[ "$(sha256sum "$status" | awk '{print $1}')" = "$status_digest" ] || exit 1
[ "$(sha256sum "$config" | awk '{print $1}')" = "$config_digest" ] || exit 1
[ "$(sha256sum "$tree_helper" | awk '{print $1}')" = "$tree_helper_digest" ] || exit 1
[ "$(sha256sum "$asset_verifier" | awk '{print $1}')" = "$asset_verify_digest" ] || exit 1
/bin/bash "$asset_verifier" "$assets" root:root
systemd-analyze verify "$main" "$status"
REMOTE_VERIFY_UNIT_STAGE

verify_local_tree_helper
current_app_digest=$(
    python3 deploy/recoverybox_tree_digest.py --root "$repo_root"
) || fail "could not recompute the local RecoveryBox application digest"
[[ $current_app_digest == "$expected_app_digest" ]] ||
    fail "local RecoveryBox application tree changed while deployment was staged"

printf 'Stopping any prior RecoveryBox instance before direct replacement.\n'
ssh "${ssh_options[@]}" "$admin_target" bash -s -- \
    "$pi_ip" "$pi_machine_id" "$MAIN_UNIT_PATH" "$STATUS_UNIT_PATH" \
    "$remote_app_stage/deploy/recoverybox_pi_power_gate.py" "$power_gate_digest" \
    <<'REMOTE_STOP_RECOVERYBOX'
set -eu
expected_ip=$1
expected_machine_id=$2
main_fragment=$3
status_fragment=$4
power_gate=$5
power_gate_digest=$6
[ "$(id -u)" -eq 0 ] || exit 1
[ "$(tr -d '\n' </etc/machine-id)" = "$expected_machine_id" ] || exit 1
tailscale ip -4 | grep -Fqx "$expected_ip" || exit 1
[ -f "$power_gate" ] && [ ! -L "$power_gate" ] || exit 1
[ "$(stat -c %U:%G "$power_gate")" = root:root ] || exit 1
[ "$(stat -c %a "$power_gate")" = 644 ] || exit 1
[ "$(stat -c %h "$power_gate")" = 1 ] || exit 1
[ "$(sha256sum "$power_gate" | awk '{print $1}')" = "$power_gate_digest" ] || exit 1
/usr/bin/python3 "$power_gate"
for spec in \
    "recoverybox-status.service:$status_fragment" \
    "recoverybox.service:$main_fragment"
do
    unit=${spec%%:*}
    expected=${spec#*:}
    load_state=$(systemctl show --property=LoadState --value "$unit")
    if [ "$load_state" = not-found ]; then
        continue
    fi
    [ "$(systemctl show --property=FragmentPath --value "$unit")" = "$expected" ] || exit 1
    [ -z "$(systemctl show --property=DropInPaths --value "$unit")" ] || exit 1
    timeout 10 systemctl stop "$unit"
    [ "$(systemctl show --property=MainPID --value "$unit")" = 0 ] || exit 1
    systemctl disable "$unit" >/dev/null
done
REMOTE_STOP_RECOVERYBOX

printf 'Permanently deleting the exact legacy assistant targets.\n'
ssh "${ssh_options[@]}" "$admin_target" bash -s -- \
    "$remote_user" "$pi_ip" "$pi_machine_id" \
    "$remote_app_stage/deploy/recoverybox_pi_power_gate.py" "$power_gate_digest" \
    <deploy/remove-legacy-voice-ai-bot.sh

printf 'Activating the fresh app without retaining a prior copy.\n'
activated_app_pending=1
ssh "${ssh_options[@]}" "$admin_target" bash -s -- \
    "$remote_user" "$pi_ip" "$pi_machine_id" "$REMOTE_ROOT" "$REMOTE_APP" \
    "$remote_app_stage" "$remote_asset_stage" "$remote_asset_verify_stage" \
    "$deployment_marker" "$REMOTE_LEGACY_ROOT" \
    <<'REMOTE_ACTIVATE_APP'
set -eu
expected_user=$1
expected_ip=$2
expected_machine_id=$3
root=$4
app=$5
stage=$6
asset_stage=$7
asset_verifier=$8
expected_marker=$9
legacy_root=${10}
new_root=0
activation_fail() {
    printf 'RecoveryBox app activation rejected: %s\n' "$1" >&2
    exit 1
}
reject_target_or_descendant_mounts() {
    target=$1
    mount_targets=$(LC_ALL=C findmnt --raw --noheadings --output TARGET) || {
        printf 'could not enumerate mountpoints before inspecting %s\n' "$target" >&2
        exit 1
    }
    if ! printf '%s\n' "$mount_targets" | awk -v target="$target" '
        $0 == target || index($0, target "/") == 1 { unsafe = 1 }
        END { exit(unsafe ? 1 : 0) }
    '; then
        printf 'refusing to remove mounted target tree: %s\n' "$target" >&2
        exit 1
    fi
}
cleanup_failed_activation() {
    result=$?
    trap - EXIT
    if [ "$result" -ne 0 ] && [ "$new_root" -eq 1 ] &&
        [ "$root" = /opt/recoverybox ] && [ -d "$root" ] && [ ! -L "$root" ] &&
        [ "$(readlink -f "$root")" = "$root" ] &&
        [ "$(stat -c %U:%G "$root")" = root:root ] &&
        [ "$(stat -c %a "$root")" = 755 ] &&
        [ "$(cat "$root/.recoverybox-managed-v2" 2>/dev/null)" = "$expected_marker" ]; then
        reject_target_or_descendant_mounts "$root"
        rm -rf --one-file-system -- "$root"
    fi
    exit "$result"
}
trap cleanup_failed_activation EXIT
[ "$(id -u)" -eq 0 ] || activation_fail "root authority was lost"
[ "$(tr -d '\n' </etc/machine-id)" = "$expected_machine_id" ] ||
    activation_fail "Pi identity changed"
tailscale ip -4 | grep -Fqx "$expected_ip" ||
    activation_fail "Pi Tailnet identity changed"
[ "$root" = /opt/recoverybox ] && [ "$app" = "$root/app" ] ||
    activation_fail "canonical app paths changed"
case "$stage" in
    /run/recoverybox-deploy-*/app) ;;
    *) activation_fail "application stage path is outside the deployment root" ;;
esac
case "$asset_stage" in
    /run/recoverybox-deploy-*/assets) ;;
    *) activation_fail "asset stage path is outside the deployment root" ;;
esac
[ "${stage%/app}" = "${asset_stage%/assets}" ] ||
    activation_fail "application and asset stages do not share one root"
[ -d "$stage" ] && [ ! -L "$stage" ] ||
    activation_fail "application stage is not a real directory"
[ -d "$asset_stage" ] && [ ! -L "$asset_stage" ] ||
    activation_fail "asset stage is not a real directory"
[ -f "$asset_verifier" ] && [ ! -L "$asset_verifier" ] ||
    activation_fail "asset verifier is not a trusted regular file"
[ "$(stat -c %U:%G "$asset_verifier")" = root:root ] ||
    activation_fail "asset verifier is not root-owned"
[ "$(stat -c %a "$asset_verifier")" = 755 ] ||
    activation_fail "asset verifier has an unexpected mode"
[ "$(stat -c %h "$asset_verifier")" = 1 ] ||
    activation_fail "asset verifier is hard-linked"
[ "$(stat -c %U:%G "$stage")" = root:root ] ||
    activation_fail "application stage is not root-owned"
[ "$(stat -c %U:%G "$asset_stage")" = root:root ] ||
    activation_fail "asset stage is not root-owned"
if [ -e "$root" ] || [ -L "$root" ]; then
    [ -d "$root" ] && [ ! -L "$root" ] ||
        activation_fail "existing managed root is not a real directory"
    [ "$(readlink -f "$root")" = "$root" ] ||
        activation_fail "existing managed root is not canonical"
    [ "$(stat -c %U:%G "$root")" = root:root ] ||
        activation_fail "existing managed root is not root-owned"
    [ "$(stat -c %a "$root")" = 755 ] ||
        activation_fail "existing managed root has an unexpected mode"
    marker="$root/.recoverybox-managed-v2"
    [ -f "$marker" ] && [ ! -L "$marker" ] ||
        activation_fail "existing managed root lacks its marker"
    case "$(cat "$marker")" in
        recoverybox-managed-v2:*) ;;
        *) activation_fail "existing managed root has an invalid marker" ;;
    esac
    reject_target_or_descendant_mounts "$root"
    rm -rf --one-file-system -- "$root"
fi
mkdir -m 755 "$root"
new_root=1
printf '%s\n' "$expected_marker" >"$root/.recoverybox-managed-v2"
chmod 600 "$root/.recoverybox-managed-v2"
mv -- "$stage" "$app"
mv -- "$asset_stage/runtime" "$root/runtime"
mv -- "$asset_stage/models" "$root/models"
rmdir -- "$asset_stage"
[ "$(stat -c %U:%G "$app")" = root:root ] ||
    activation_fail "activated application is not root-owned"
chmod -R u=rwX,go=rX "$app"
[ "$(stat -c %a "$app")" = 755 ] ||
    activation_fail "activated application root has an unexpected mode"
if find "$app" ! -user root -print -quit | grep -q .; then
    activation_fail "activated application contains a non-root-owned inode"
fi
if find "$app" \( -type d -o -type f \) -perm /0022 -print -quit | grep -q .; then
    activation_fail "activated application contains a writable inode"
fi
/bin/bash "$asset_verifier" "$root" root:root

# This is the exact pre-migration RecoveryBox tree. It is destroyed after the
# new root-owned inodes are canonical; nothing is copied from it or retained.
if [ -e "$legacy_root" ] || [ -L "$legacy_root" ]; then
    [ "$legacy_root" = "/home/$expected_user/recoverybox" ] ||
        activation_fail "legacy application path changed"
    [ -d "$legacy_root" ] && [ ! -L "$legacy_root" ] ||
        activation_fail "legacy application is not a real directory"
    [ "$(readlink -f "$legacy_root")" = "$legacy_root" ] ||
        activation_fail "legacy application path is not canonical"
    owner=$(stat -c %U:%G "$legacy_root")
    mode=$(stat -c %a "$legacy_root")
    if [ "$owner" != "${expected_user}:${expected_user}" ] || [ "$mode" != 700 ]; then
        [ "$owner" = "root:${expected_user}" ] && [ "$mode" = 1770 ] ||
            activation_fail "legacy application owner or mode changed"
    fi
    marker="$legacy_root/.recoverybox-managed-v1"
    [ -f "$marker" ] && [ ! -L "$marker" ] ||
        activation_fail "legacy application lacks its marker"
    [ "$(cat "$marker")" = recoverybox-managed-v1 ] ||
        activation_fail "legacy application marker changed"
    reject_target_or_descendant_mounts "$legacy_root"
    rm -rf --one-file-system -- "$legacy_root"
fi
[ ! -e "$legacy_root/app" ] && [ ! -L "$legacy_root/app" ] ||
    activation_fail "legacy application survived deletion"
[ ! -e "$legacy_root/config/recoverybox.env" ] &&
    [ ! -L "$legacy_root/config/recoverybox.env" ] ||
    activation_fail "legacy configuration survived deletion"
trap - EXIT
REMOTE_ACTIVATE_APP

verify_local_tree_helper
if ! ssh "${ssh_options[@]}" "$admin_target" /usr/bin/python3 - \
    --root "$REMOTE_APP" --strict --expect "$expected_app_digest" \
    <deploy/recoverybox_tree_digest.py >/dev/null; then
    fail "activated RecoveryBox application did not match the trusted local tree digest"
fi
if ! ssh "${ssh_options[@]}" "$admin_target" /bin/bash \
    "$remote_asset_verify_stage" "$REMOTE_ROOT" root:root; then
    fail "activated RecoveryBox NCNN runtime/models did not match their exact pins"
fi

printf 'Installing and starting the root-owned, device-confined RecoveryBox units.\n'
activation_result=$(ssh "${ssh_options[@]}" "$admin_target" bash -s -- \
    "$pi_ip" "$pi_machine_id" "$remote_unit_stage" "$main_digest" \
    "$status_digest" "$config_digest" "$MAIN_UNIT_PATH" "$STATUS_UNIT_PATH" \
    "$POSE_PORT" "$DEBUG_PORT" "$REMOTE_ROOT" "$REMOTE_APP" \
    "$REMOTE_CONFIG_DIR" "$REMOTE_CONFIG" "$REMOTE_STATUS" "$mac_ip" \
    "$REMOTE_CREDENTIAL_DIR" "$REMOTE_OPENAI_CREDENTIAL" "$expected_app_digest" \
    "$tree_helper_digest" "$asset_verify_digest" "$deployment_marker" \
    "$REMOTE_LEGACY_ROOT" "$REMOTE_NCNN_RUNTIME" "$REMOTE_NCNN_MODELS" \
    "$REMOTE_LIBYUV" "$power_gate_digest" <<'REMOTE_INSTALL_UNITS'
set -eu
expected_ip=$1
expected_machine_id=$2
stage=$3
main_digest=$4
status_digest=$5
config_digest=$6
main_target=$7
status_target=$8
pi_ip=$expected_ip
pose_port=$9
debug_port=${10}
app_root=${11}
app=${12}
config_dir=${13}
config=${14}
status_path=${15}
allowed_peer=${16}
credential_dir=${17}
openai_target=${18}
expected_app_digest=${19}
tree_helper_digest=${20}
asset_verify_digest=${21}
expected_marker=${22}
legacy_root=${23}
runtime=${24}
models=${25}
libyuv=${26}
power_gate_digest=${27}
power_gate="$app/deploy/recoverybox_pi_power_gate.py"
activation_started=0
runtime_proven=0
config_tree_created=0

reject_target_or_descendant_mounts() {
    target=$1
    mount_targets=$(LC_ALL=C findmnt --raw --noheadings --output TARGET) || {
        printf 'could not enumerate mountpoints before inspecting %s\n' "$target" >&2
        exit 1
    }
    if ! printf '%s\n' "$mount_targets" | awk -v target="$target" '
        $0 == target || index($0, target "/") == 1 { unsafe = 1 }
        END { exit(unsafe ? 1 : 0) }
    '; then
        printf 'refusing to remove mounted target tree: %s\n' "$target" >&2
        exit 1
    fi
}

cleanup_failed_activation() {
    result=$?
    trap - EXIT
    set +e
    if [ "$result" -ne 0 ]; then
        systemctl stop recoverybox-status.service recoverybox.service >/dev/null 2>&1 || true
        systemctl disable recoverybox-status.service recoverybox.service >/dev/null 2>&1 || true
        if [ "$activation_started" -eq 1 ]; then
            for spec in "$main_target:$main_digest" "$status_target:$status_digest"; do
                target=${spec%%:*}
                digest=${spec#*:}
                if [ -f "$target" ] && [ ! -L "$target" ] &&
                    [ "$(stat -c %U:%G "$target")" = root:root ] &&
                    [ "$(sha256sum "$target" | awk '{print $1}')" = "$digest" ]; then
                    rm -f -- "$target"
                fi
            done
            if [ "$config_tree_created" -eq 1 ] &&
                [ "$config_dir" = /etc/recoverybox ] && [ -d "$config_dir" ] &&
                [ ! -L "$config_dir" ] &&
                [ "$(readlink -f "$config_dir")" = "$config_dir" ] &&
                [ "$(stat -c %U:%G "$config_dir")" = root:root ] &&
                [ "$(stat -c %a "$config_dir")" = 700 ]; then
                reject_target_or_descendant_mounts "$config_dir"
                rm -rf --one-file-system -- "$config_dir"
            fi
            systemctl daemon-reload >/dev/null 2>&1 || true
        fi
        if [ "$runtime_proven" -eq 0 ] && [ "$app_root" = /opt/recoverybox ] &&
            [ -d "$app_root" ] && [ ! -L "$app_root" ] &&
            [ "$(readlink -f "$app_root")" = "$app_root" ] &&
            [ "$(stat -c %U:%G "$app_root")" = root:root ] &&
            [ "$(stat -c %a "$app_root")" = 755 ] &&
            [ "$(cat "$app_root/.recoverybox-managed-v2" 2>/dev/null)" = \
                "$expected_marker" ]; then
            reject_target_or_descendant_mounts "$app_root"
            rm -rf --one-file-system -- "$app_root"
        fi
    fi
    exit "$result"
}
trap cleanup_failed_activation EXIT

[ "$(id -u)" -eq 0 ] || { printf 'root authority lost\n' >&2; exit 1; }
[ "$(tr -d '\n' </etc/machine-id)" = "$expected_machine_id" ] || exit 1
tailscale ip -4 | grep -Fqx "$expected_ip" || exit 1
for path in "$stage" "$app"; do
    [ -d "$path" ] && [ ! -L "$path" ] || exit 1
done
stage_prefix=/run/recoverybox-deploy-
stage_suffix=${stage#"$stage_prefix"}
[ "$stage_suffix" != "$stage" ] && [ -n "$stage_suffix" ] &&
    [ "${stage_suffix#*/}" = "$stage_suffix" ] || exit 1
[ "$app_root" = /opt/recoverybox ] && [ "$app" = "$app_root/app" ] || exit 1
[ "$config_dir" = /etc/recoverybox ] && [ "$config" = "$config_dir/recoverybox.env" ] || exit 1
[ "$credential_dir" = "$config_dir/credentials" ] || exit 1
[ "$openai_target" = "$credential_dir/openai-api-key" ] || exit 1
[ "$runtime" = "$app_root/runtime/ncnn" ] || exit 1
[ "$models" = "$app_root/models/ncnn" ] || exit 1
[ "$libyuv" = /usr/lib/arm-linux-gnueabihf/libyuv.so.0 ] || exit 1
[ -f "$power_gate" ] && [ ! -L "$power_gate" ] || exit 1
[ "$(stat -c %U:%G "$power_gate")" = root:root ] || exit 1
[ "$(stat -c %a "$power_gate")" = 644 ] || exit 1
[ "$(stat -c %h "$power_gate")" = 1 ] || exit 1
[ "$(sha256sum "$power_gate" | awk '{print $1}')" = "$power_gate_digest" ] || exit 1
[ "$(stat -c %U:%G "$stage")" = root:root ] || exit 1
[ "$(stat -c %a "$stage")" = 700 ] || exit 1
[ "$(stat -c %U:%G "$app_root")" = root:root ] || exit 1
[ "$(stat -c %a "$app_root")" = 755 ] || exit 1
[ "$(cat "$app_root/.recoverybox-managed-v2")" = "$expected_marker" ] || exit 1
[ "$(stat -c %U:%G "$app")" = root:root ] || exit 1
[ "$(stat -c %a "$app")" = 755 ] || exit 1
[ ! -e "$legacy_root" ] && [ ! -L "$legacy_root" ] || exit 1
if find "$app" ! -user root -print -quit | grep -q .; then exit 1; fi
if find "$app" \( -type d -o -type f \) -perm /0022 -print -quit | grep -q .; then
    exit 1
fi

main_source="$stage/recoverybox.service"
status_source="$stage/recoverybox-status.service"
config_source="$stage/recoverybox.env"
openai_source="$stage/openai-api-key"
tree_helper_source="$stage/recoverybox_tree_digest.py"
asset_verifier_source="$stage/verify-pi-pose-ncnn-assets.sh"
[ "$(sha256sum "$main_source" | awk '{print $1}')" = "$main_digest" ] || exit 1
[ "$(sha256sum "$status_source" | awk '{print $1}')" = "$status_digest" ] || exit 1
[ "$(sha256sum "$config_source" | awk '{print $1}')" = "$config_digest" ] || exit 1
[ "$(sha256sum "$tree_helper_source" | awk '{print $1}')" = "$tree_helper_digest" ] || exit 1
[ "$(sha256sum "$asset_verifier_source" | awk '{print $1}')" = \
    "$asset_verify_digest" ] || exit 1
systemd-analyze verify "$main_source" "$status_source"
for path in "$config_source" "$openai_source"; do
    [ -f "$path" ] && [ ! -L "$path" ] || exit 1
    [ "$(stat -c %U:%G "$path")" = root:root ] || exit 1
    [ "$(stat -c %a "$path")" = 600 ] || exit 1
    [ "$(stat -c %h "$path")" = 1 ] || exit 1
done

# The canonical tree consists of fresh root-owned inodes created directly in
# the root-only stage. No service-account write descriptor ever existed.
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 "$tree_helper_source" \
    --root "$app" --strict --expect "$expected_app_digest" >/dev/null
/bin/bash "$asset_verifier_source" "$app_root" root:root

for target in "$main_target" "$status_target"; do
    if [ -e "$target" ] || [ -L "$target" ]; then
        [ -f "$target" ] && [ ! -L "$target" ] || exit 1
        [ "$(stat -c %U:%G "$target")" = root:root ] || exit 1
    fi
done
activation_started=1
# Local pose has no pose credential. Recreate the entire configuration tree so
# no obsolete environment, token, or other unreviewed secret survives.
[ "$config_dir" = /etc/recoverybox ] || exit 1
[ "$credential_dir" = "$config_dir/credentials" ] || exit 1
if [ -e "$config_dir" ] || [ -L "$config_dir" ]; then
    [ -d "$config_dir" ] && [ ! -L "$config_dir" ] || exit 1
    [ "$(readlink -f "$config_dir")" = "$config_dir" ] || exit 1
    [ "$(stat -c %U:%G "$config_dir")" = root:root ] || exit 1
    [ "$(stat -c %a "$config_dir")" = 700 ] || exit 1
    reject_target_or_descendant_mounts "$config_dir"
    reject_target_or_descendant_mounts "$credential_dir"
    rm -rf --one-file-system -- "$config_dir"
fi
install -d -o root -g root -m 0700 "$config_dir"
config_tree_created=1
install -d -o root -g root -m 0700 "$credential_dir"
install -o root -g root -m 0600 "$config_source" "$config"
install -o root -g root -m 0600 "$openai_source" "$openai_target"
install -o root -g root -m 0644 "$main_source" "$main_target"
install -o root -g root -m 0644 "$status_source" "$status_target"
[ "$(sha256sum "$main_target" | awk '{print $1}')" = "$main_digest" ] || exit 1
[ "$(sha256sum "$status_target" | awk '{print $1}')" = "$status_digest" ] || exit 1
[ "$(sha256sum "$config" | awk '{print $1}')" = "$config_digest" ] || exit 1
for path in "$config" "$openai_target"; do
    [ -f "$path" ] && [ ! -L "$path" ] || exit 1
    [ "$(stat -c %U:%G "$path")" = root:root ] || exit 1
    [ "$(stat -c %a "$path")" = 600 ] || exit 1
    [ "$(stat -c %h "$path")" = 1 ] || exit 1
done
for directory in "$config_dir" "$credential_dir"; do
    [ -d "$directory" ] && [ ! -L "$directory" ] || exit 1
    [ "$(readlink -f "$directory")" = "$directory" ] || exit 1
    [ "$(stat -c %U:%G "$directory")" = root:root ] || exit 1
    [ "$(stat -c %a "$directory")" = 700 ] || exit 1
done
[ "$(find "$config_dir" -mindepth 1 -maxdepth 1 -printf x | wc -c)" -eq 2 ] ||
    exit 1
[ "$(find "$credential_dir" -mindepth 1 -maxdepth 1 -printf x | wc -c)" -eq 1 ] ||
    exit 1
[ -f "$openai_target" ] && [ ! -L "$openai_target" ] || exit 1
rm -f -- "$main_source" "$status_source" "$config_source" \
    "$openai_source"

systemctl daemon-reload

assert_unit_property() {
    unit=$1
    property=$2
    expected=$3
    actual=$(systemctl show --property="$property" --value "$unit")
    [ "$actual" = "$expected" ] || {
        printf '%s has unexpected effective %s\n' "$unit" "$property" >&2
        exit 1
    }
}

assert_unit_word_set() {
    unit=$1
    property=$2
    expected=$3
    actual=$(systemctl show --property="$property" --value "$unit")
    normalized_actual=$(
        printf '%s\n' "$actual" | tr ' ' '\n' | sed '/^$/d' |
            LC_ALL=C sort | tr '\n' ' '
    )
    normalized_expected=$(
        printf '%s\n' "$expected" | tr ' ' '\n' | sed '/^$/d' |
            LC_ALL=C sort | tr '\n' ' '
    )
    [ "$normalized_actual" = "$normalized_expected" ] || {
        printf '%s has unexpected effective %s\n' "$unit" "$property" >&2
        exit 1
    }
}

assert_unit_ip_deny_any() {
    unit=$1
    actual=$(systemctl show --property=IPAddressDeny --value "$unit")
    if [ "$actual" = any ]; then
        return
    fi
    normalized_actual=$(
        printf '%s\n' "$actual" | tr ' ' '\n' | sed '/^$/d' |
            LC_ALL=C sort | tr '\n' ' '
    )
    [ "$normalized_actual" = '0.0.0.0/0 ::/0 ' ] || {
        printf '%s has unexpected effective IPAddressDeny\n' "$unit" >&2
        exit 1
    }
}

assert_unit_property recoverybox.service FragmentPath "$main_target"
assert_unit_property recoverybox-status.service FragmentPath "$status_target"
assert_unit_property recoverybox.service User pi
assert_unit_property recoverybox-status.service User pi
assert_unit_property recoverybox.service Group pi
assert_unit_property recoverybox-status.service Group pi
assert_unit_property recoverybox.service SupplementaryGroups 'gpio video'
assert_unit_property recoverybox-status.service SupplementaryGroups ''
assert_unit_property recoverybox.service WorkingDirectory /opt/recoverybox/app
assert_unit_property recoverybox-status.service WorkingDirectory /opt/recoverybox/app
assert_unit_property recoverybox.service Restart on-failure
assert_unit_property recoverybox-status.service Restart on-failure
assert_unit_property recoverybox.service RuntimeDirectoryPreserve restart
assert_unit_property recoverybox.service LimitCORE 0
assert_unit_property recoverybox.service NoNewPrivileges yes
assert_unit_property recoverybox-status.service NoNewPrivileges yes
assert_unit_property recoverybox.service DevicePolicy closed
assert_unit_property recoverybox-status.service DevicePolicy closed
assert_unit_property recoverybox.service CapabilityBoundingSet ''
assert_unit_property recoverybox.service AmbientCapabilities ''
assert_unit_property recoverybox-status.service CapabilityBoundingSet ''
assert_unit_property recoverybox-status.service AmbientCapabilities ''
assert_unit_property recoverybox.service ProtectSystem strict
assert_unit_property recoverybox-status.service ProtectSystem strict
assert_unit_property recoverybox.service ProtectHome yes
assert_unit_property recoverybox-status.service ProtectHome yes
assert_unit_property recoverybox.service ReadOnlyPaths \
    '/opt/recoverybox/app /opt/recoverybox/runtime /opt/recoverybox/models'
assert_unit_property recoverybox-status.service ReadOnlyPaths \
    '/opt/recoverybox/app /run/recoverybox'
assert_unit_property recoverybox-status.service InaccessiblePaths /etc/recoverybox/credentials
assert_unit_word_set recoverybox.service RestrictAddressFamilies 'AF_UNIX AF_INET AF_INET6'
assert_unit_word_set recoverybox-status.service RestrictAddressFamilies 'AF_UNIX AF_INET AF_INET6'
assert_unit_ip_deny_any recoverybox.service
assert_unit_ip_deny_any recoverybox-status.service
assert_unit_word_set recoverybox.service IPAddressAllow 100.64.0.0/10
assert_unit_word_set recoverybox-status.service IPAddressAllow 100.64.0.0/10
[ "$(systemctl show --property=EnvironmentFiles --value recoverybox.service)" = \
    '/etc/recoverybox/recoverybox.env (ignore_errors=no)' ] || exit 1
[ "$(systemctl show --property=EnvironmentFiles --value recoverybox-status.service)" = \
    '/etc/recoverybox/recoverybox.env (ignore_errors=no)' ] || exit 1
systemctl show --property=ExecStart --value recoverybox.service |
    grep -Fq 'argv[]=/usr/bin/python3 -m recoverybox.device.remote_pose_service ;' || exit 1
systemctl show --property=ExecStart --value recoverybox-status.service |
    grep -Fq \
        'argv[]=/usr/bin/python3 /opt/recoverybox/app/deploy/recoverybox_status.py ;' ||
    exit 1
[ -z "$(systemctl show --property=DropInPaths --value recoverybox.service)" ] || exit 1
[ -z "$(systemctl show --property=DropInPaths --value recoverybox-status.service)" ] || exit 1
# systemd exposes LoadCredential as an unprintable structured D-Bus property on
# this Pi. The installed fragment hashes and absent drop-ins already establish
# the effective source, so verify the exact reviewed directives in that fragment.
grep -Fxq "LoadCredential=openai-api-key:$openai_target" "$main_target" || exit 1
grep -Fxq 'Environment=RECOVERYBOX_OPENAI_CREDENTIAL_FILE=%d/openai-api-key' \
    "$main_target" || exit 1
grep -Eq 'pose-token|POSE_TOKEN' "$main_target" && exit 1
grep -Fxq 'Environment=PYTHONPATH=/opt/recoverybox/app/src' "$main_target" || exit 1
grep -Fxq 'Environment=PYTHONPATH=/opt/recoverybox/app/src' "$status_target" || exit 1
if grep -Eq \
    '^(LoadCredential|ImportCredential|SetCredential|Environment=.*(CREDENTIAL|POSE_TOKEN))' \
    "$status_target"; then
    exit 1
fi

device_allow=$(systemctl show --property=DeviceAllow --value recoverybox.service)
normalized_device_allow=$(printf '%s\n' "$device_allow" | sed '/^$/d' | sort)
[ "$normalized_device_allow" = '/dev/gpiochip0 rw
/dev/video0 rw' ] || exit 1
printf '%s\n' "$device_allow" | grep -Fq /dev/snd && exit 1
[ -z "$(systemctl show --property=DeviceAllow --value recoverybox-status.service)" ] || exit 1

# No service executes until all effective identity, environment, capability,
# network, drop-in, and device policy checks above have passed.
[ "$(sha256sum "$tree_helper_source" | awk '{print $1}')" = "$tree_helper_digest" ] || exit 1
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 "$tree_helper_source" \
    --root "$app" --strict --expect "$expected_app_digest" >/dev/null

# Exercise the exact activated camera/libyuv/NCNN path once, bounded and silent,
# before handing long-lived ownership of /dev/video0 to the service.
/usr/bin/python3 "$power_gate"
set -a
. "$config"
set +a
export PYTHONPATH="$app/src"
export PYTHONDONTWRITEBYTECODE=1
pose_check_status=0
pose_check=$(
    cd "$app"
    timeout 45 runuser -u pi -- \
        /usr/bin/python3 -m recoverybox.device.pi_pose_v4l2 --max-frames 3
) || pose_check_status=$?
python3 - "$pose_check" "$pose_check_status" <<'PY'
import json
import math
import re
import sys

try:
    report = json.loads(sys.argv[1])
    pose_check_status = int(sys.argv[2])
except (IndexError, ValueError, json.JSONDecodeError):
    raise SystemExit(1) from None
if pose_check_status != 0:
    failure = report.get("failure")
    if (
        report.get("service") != "recoverybox-pi-v4l2-ncnn-check/v2"
        or report.get("raw_frames_persisted") != 0
        or report.get("audio") != "disabled"
        or not isinstance(failure, str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", failure) is None
    ):
        raise SystemExit(1)
    numeric_fields = (
        "frames",
        "frames_received",
        "fresh_frames",
        "assessable",
        "timeouts",
        "capture_misses",
        "worker_timeouts",
        "parent_stale_count",
        "detector_ms_max",
        "pose_ms_max",
        "inference_ms_max",
        "evidence_age_ms_max",
    )
    numeric_summary = []
    for name in numeric_fields:
        value = report.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise SystemExit(1)
        numeric_summary.append(f"{name}={value}")
    print(
        f"Pi pose acceptance failed: {failure}; " + " ".join(numeric_summary),
        file=sys.stderr,
    )
    raise SystemExit(1)
expected = {
    "service": "recoverybox-pi-v4l2-ncnn-check/v2",
    "capture": "v4l2-mmap-yuyv",
    "conversion": "libyuv-yuy2-to-bgra",
    "estimator": "ncnn-nanodet-rtmpose",
    "frames": 3,
    "frames_received": 3,
    "fresh_frames": 3,
    "timeouts": 0,
    "capture_misses": 0,
    "worker_timeouts": 0,
    "parent_stale_count": 0,
    "raw_frames_persisted": 0,
    "audio": "disabled",
}
if any(report.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
if frozenset(report) != {
    *expected,
    "assessable",
    "detector_ms_max",
    "pose_ms_max",
    "inference_ms_max",
    "evidence_age_ms_max",
}:
    raise SystemExit(1)
if isinstance(report.get("assessable"), bool) or not isinstance(report.get("assessable"), int):
    raise SystemExit(1)
if not 0 <= report["assessable"] <= 3:
    raise SystemExit(1)

def bounded_duration(name, *, nullable=False):
    value = report.get(name)
    if value is None and nullable:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value < 500
    ):
        raise SystemExit(1)
    return float(value)

detector_ms = bounded_duration("detector_ms_max")
pose_ms = bounded_duration("pose_ms_max", nullable=True)
inference_ms = bounded_duration("inference_ms_max")
evidence_age_ms = bounded_duration("evidence_age_ms_max")
if report["assessable"] > 0 and pose_ms is None:
    raise SystemExit(1)
if (
    detector_ms > inference_ms
    or (pose_ms is not None and pose_ms > inference_ms)
    or inference_ms > evidence_age_ms
):
    raise SystemExit(1)
PY
/usr/bin/python3 "$power_gate"
systemctl enable recoverybox.service recoverybox-status.service >/dev/null
systemctl restart recoverybox.service recoverybox-status.service

ready=0
attempt=0
while [ "$attempt" -lt 90 ]; do
    attempt=$((attempt + 1))
    main_pid=$(systemctl show --property=MainPID --value recoverybox.service)
    status_pid=$(systemctl show --property=MainPID --value recoverybox-status.service)
    if [ "$main_pid" -gt 1 ] 2>/dev/null && [ "$status_pid" -gt 1 ] 2>/dev/null &&
        [ "$(systemctl show --property=ActiveState --value recoverybox.service)" = active ] &&
        [ "$(systemctl show --property=SubState --value recoverybox.service)" = running ] &&
        [ "$(systemctl show --property=ActiveState --value recoverybox-status.service)" = active ] &&
        [ "$(systemctl show --property=SubState --value recoverybox-status.service)" = running ] &&
        ss -H -ltn "sport = :$debug_port" 2>/dev/null |
            awk -v endpoint="${pi_ip}:${debug_port}" '$4 == endpoint { found=1 } END { exit !found }' &&
        python3 - "$status_path" "$(id -u pi)" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

try:
    path = Path(sys.argv[1])
    expected_uid = int(sys.argv[2])
    metadata = path.stat()
    if metadata.st_uid != expected_uid or metadata.st_mode & 0o022 or metadata.st_nlink != 1:
        raise SystemExit(1)
    if not 0 <= time.time() - metadata.st_mtime <= 2.0:
        raise SystemExit(1)
    status = json.loads(path.read_bytes())
    if frozenset(status) != {
        "service", "peer", "session", "mode", "rep", "age", "voice", "failure", "button"
    }:
        raise SystemExit(1)
    if status.get("service") != "local":
        raise SystemExit(1)
    if status.get("session") != "0" * 32:
        raise SystemExit(1)
    if status.get("mode") not in {"idle", "check_in", "active_exercise", "paused"}:
        raise SystemExit(1)
    if isinstance(status.get("age"), bool) or not isinstance(status.get("age"), int):
        raise SystemExit(1)
    if not 0 <= status["age"] < 500:
        raise SystemExit(1)
    if status.get("failure") is not None:
        raise SystemExit(1)
    if status.get("voice") != "silent" or status.get("button") != "available":
        raise SystemExit(1)
except (OSError, ValueError, TypeError):
    raise SystemExit(1) from None
PY
    then
        ready=1
        break
    fi
    sleep 1
done
[ "$ready" -eq 1 ] || { printf 'RecoveryBox services failed readiness\n' >&2; exit 1; }

python3 - "$main_pid" "$status_pid" "$app" "$app_root" "$allowed_peer" \
    "$openai_target" "$runtime" "$models" "$libyuv" <<'PY'
import sys
from pathlib import Path

main_pid, status_pid, app, app_root, allowed_peer, openai_path, runtime, models, libyuv = (
    sys.argv[1:]
)
secrets = (Path(openai_path).read_bytes().rstrip(b"\n"),)
expected = {
    main_pid: [b"/usr/bin/python3", b"-m", b"recoverybox.device.remote_pose_service"],
    status_pid: [b"/usr/bin/python3", f"{app}/deploy/recoverybox_status.py".encode()],
}
for raw_pid, expected_argv in expected.items():
    process = Path("/proc") / raw_pid
    argv = (process / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    if argv != expected_argv:
        raise SystemExit(1)
    environment_raw = (process / "environ").read_bytes()
    environment = dict(
        item.split(b"=", 1)
        for item in environment_raw.split(b"\0")
        if b"=" in item
    )
    if environment.get(b"RECOVERYBOX_AUDIO_ENABLED") != b"0":
        raise SystemExit(1)
    if environment.get(b"RECOVERYBOX_POSE_SOURCE") != b"local":
        raise SystemExit(1)
    if environment.get(b"RECOVERYBOX_STATUS_ALLOWED_PEER") != allowed_peer.encode():
        raise SystemExit(1)
    if environment.get(b"PYTHONPATH") != f"{app}/src".encode():
        raise SystemExit(1)
    if environment.get(b"PYTHONDONTWRITEBYTECODE") != b"1":
        raise SystemExit(1)
    required_local_pose = {
        b"RECOVERYBOX_CAMERA_DEVICE": b"/dev/video0",
        b"RECOVERYBOX_LIBYUV_LIBRARY_PATH": libyuv.encode(),
        b"RECOVERYBOX_CAMERA_WIDTH": b"640",
        b"RECOVERYBOX_CAMERA_HEIGHT": b"480",
        b"RECOVERYBOX_CAMERA_FPS": b"10",
        b"RECOVERYBOX_POSE_FRAME_TIMEOUT_SECONDS": b"0.1",
        b"RECOVERYBOX_LOCAL_POSE_WORKER_TIMEOUT_SECONDS": b"0.5",
        b"RECOVERYBOX_LOCAL_POSE_START_TIMEOUT_SECONDS": b"25",
        b"RECOVERYBOX_V4L2_BUFFER_COUNT": b"8",
        b"RECOVERYBOX_NCNN_RUNTIME_PATH": runtime.encode(),
        b"RECOVERYBOX_RTMPOSE_PARAM_PATH": f"{models}/rtmpose-t.param".encode(),
        b"RECOVERYBOX_RTMPOSE_BIN_PATH": f"{models}/rtmpose-t.bin".encode(),
        b"RECOVERYBOX_NANODET_PARAM_PATH": f"{models}/nanodet-m-int8.param".encode(),
        b"RECOVERYBOX_NANODET_BIN_PATH": f"{models}/nanodet-m-int8.bin".encode(),
        b"RECOVERYBOX_NCNN_THREADS": b"2",
        b"RECOVERYBOX_PERSON_SCORE_THRESHOLD": b"0.55",
        b"RECOVERYBOX_PERSON_NMS_IOU": b"0.3",
        b"RECOVERYBOX_MIN_PERSON_AREA_FRACTION": b"0.05",
        b"RECOVERYBOX_LOCAL_POSE_MAX_AGE_SECONDS": b"0.5",
    }
    if any(environment.get(key) != value for key, value in required_local_pose.items()):
        raise SystemExit(1)
    if environment.get(b"RECOVERYBOX_POSE_TOKEN_FILE") != b"/run/recoverybox/unused-local-pose-token":
        raise SystemExit(1)
    if b"OPENAI_API_KEY" in environment:
        raise SystemExit(1)
    command_line = (process / "cmdline").read_bytes()
    if any(secret and (secret in environment_raw or secret in command_line) for secret in secrets):
        raise SystemExit(1)
    openai_file = environment.get(b"RECOVERYBOX_OPENAI_CREDENTIAL_FILE")
    pose_file = environment.get(b"RECOVERYBOX_POSE_TOKEN_FILE")
    if raw_pid == main_pid:
        if openai_file != b"/run/credentials/recoverybox.service/openai-api-key":
            raise SystemExit(1)
        if environment.get(b"CREDENTIALS_DIRECTORY") != b"/run/credentials/recoverybox.service":
            raise SystemExit(1)
    elif (
        openai_file is not None
        or b"CREDENTIALS_DIRECTORY" in environment
    ):
        raise SystemExit(1)
PY

# Prove the exact source tree as each already-running service resolves it in
# its own mount namespace. This is the executed-tree integrity boundary.
[ "$(sha256sum "$tree_helper_source" | awk '{print $1}')" = "$tree_helper_digest" ] || exit 1
for pid in "$main_pid" "$status_pid"; do
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 "$tree_helper_source" \
        --root "/proc/$pid/root$app" --strict --expect "$expected_app_digest" >/dev/null
    /bin/bash "$asset_verifier_source" "/proc/$pid/root$app_root" root:root
done

for path in \
    /home/pi/recoverybox \
    /opt/voice-ai-bot /var/lib/voice-ai-bot \
    /etc/systemd/system/voice-ai-bot.service \
    /etc/systemd/system/voice-ai-bot-debug.service \
    /etc/systemd/system/multi-user.target.wants/voice-ai-bot.service \
    /etc/systemd/system/multi-user.target.wants/voice-ai-bot-debug.service
do
    [ ! -e "$path" ] && [ ! -L "$path" ] || exit 1
done

proof="$app_root/.runtime-proven-v2"
temporary="$app_root/.runtime-proven-v2.tmp"
umask 077
printf '%s\n' "$expected_marker" >"$temporary"
chmod 600 "$temporary"
mv -f -- "$temporary" "$proof"
[ -f "$proof" ] && [ ! -L "$proof" ] || exit 1
[ "$(stat -c %U:%G "$proof")" = root:root ] || exit 1
[ "$(stat -c %a "$proof")" = 600 ] || exit 1
[ "$(cat "$proof")" = "$expected_marker" ] || exit 1
runtime_proven=1
rm -f -- "$tree_helper_source" "$asset_verifier_source"
rmdir -- "$stage"
trap - EXIT
printf '%s %s\n' "$main_pid" "$status_pid"
REMOTE_INSTALL_UNITS
)
remote_unit_stage_pending=0
activated_app_pending=0

read -r main_pid status_pid <<<"$activation_result"
python3 - "$pi_ip" "$DEBUG_PORT" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(
        f"http://{sys.argv[1]}:{sys.argv[2]}/healthz", timeout=2.0
    ) as response:
        payload = response.read(65_537)
        if response.status != 200 or len(payload) > 65_536:
            raise SystemExit(1)
        status = json.loads(payload)
        if status.get("service") != "local":
            raise SystemExit(1)
        if status.get("voice") != "silent" or status.get("button") != "available":
            raise SystemExit(1)
except Exception:
    raise SystemExit("Tailnet status endpoint failed Pi-local-pose readiness") from None
PY
printf 'RecoveryBox direct replacement verified: main PID %s, status PID %s.\n' \
    "$main_pid" "$status_pid"
printf 'Pi-local pose: /dev/video0; status: http://%s:%s/healthz\n' \
    "$pi_ip" "$DEBUG_PORT"
printf 'Audio is disabled and /dev/snd is outside both services device policy.\n'
printf 'NCNN ARMv7 runtime and all four model files match their exact pins.\n'
printf 'Legacy code, state, services, and stored legacy images are absent.\n'
printf 'No legacy backup, archive, compatibility layer, or rollback was created.\n'
