#!/usr/bin/env bash
# Permanently remove only the validated legacy assistant targets.

set -euo pipefail

die() {
    printf 'legacy deletion refused: %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 3 ]] || die "expected ACCOUNT TAILSCALE_IPV4 MACHINE_ID"
readonly EXPECTED_ACCOUNT=$1
readonly EXPECTED_TAILSCALE_IP=$2
readonly EXPECTED_MACHINE_ID=$3
readonly MAIN_UNIT="voice-ai-bot.service"
readonly DEBUG_UNIT="voice-ai-bot-debug.service"
readonly MAIN_FRAGMENT="/etc/systemd/system/voice-ai-bot.service"
readonly DEBUG_FRAGMENT="/etc/systemd/system/voice-ai-bot-debug.service"
readonly MAIN_WANT="/etc/systemd/system/multi-user.target.wants/voice-ai-bot.service"
readonly DEBUG_WANT="/etc/systemd/system/multi-user.target.wants/voice-ai-bot-debug.service"
readonly MAIN_DROPIN_ETC="/etc/systemd/system/voice-ai-bot.service.d"
readonly DEBUG_DROPIN_ETC="/etc/systemd/system/voice-ai-bot-debug.service.d"
readonly MAIN_DROPIN_RUN="/run/systemd/system/voice-ai-bot.service.d"
readonly DEBUG_DROPIN_RUN="/run/systemd/system/voice-ai-bot-debug.service.d"
readonly APP_ROOT="/opt/voice-ai-bot"
readonly STATE_ROOT="/var/lib/voice-ai-bot"
readonly MAIN_RUNTIME="/run/voice-ai-bot"
readonly DEBUG_RUNTIME="/run/voice-ai-bot-debug"

[[ $(id -u) -eq 0 ]] || die "run this exact helper as root (for example with sudo)"
case "$EXPECTED_ACCOUNT" in
    "" | *[!A-Za-z0-9_-]*) die "invalid expected account" ;;
esac
case "$EXPECTED_TAILSCALE_IP" in
    "" | *[!0-9.]*) die "invalid expected Tailscale IPv4 address" ;;
esac
[[ $EXPECTED_MACHINE_ID =~ ^[0-9a-f]{32}$ ]] || die "invalid expected machine id"
id "$EXPECTED_ACCOUNT" >/dev/null 2>&1 || die "expected account does not exist"
[[ $(tr -d '\n' </etc/machine-id) == "$EXPECTED_MACHINE_ID" ]] ||
    die "root SSH reached a different machine"
command -v tailscale >/dev/null 2>&1 || die "tailscale is unavailable"
tailscale ip -4 | grep -Fqx "$EXPECTED_TAILSCALE_IP" ||
    die "root SSH target does not own the expected Tailnet address"
command -v systemctl >/dev/null 2>&1 || die "systemctl is unavailable"
command -v timeout >/dev/null 2>&1 || die "timeout is unavailable"
systemctl show --property=LoadState --value systemd-journald.service >/dev/null ||
    die "systemd manager is unavailable"

python3 - "$EXPECTED_TAILSCALE_IP" <<'PY' || die "expected address is not Tailscale IPv4"
import ipaddress
import sys

address = ipaddress.ip_address(sys.argv[1])
if address.version != 4 or address not in ipaddress.ip_network("100.64.0.0/10"):
    raise SystemExit(1)
PY

validate_fragment() {
    local unit=$1
    local expected=$2
    local load_state fragment
    load_state=$(systemctl show --property=LoadState --value "$unit") ||
        die "could not inspect $unit"
    fragment=$(systemctl show --property=FragmentPath --value "$unit") ||
        die "could not inspect $unit fragment"
    if [[ $load_state != not-found ]]; then
        [[ $fragment == "$expected" ]] ||
            die "$unit resolves outside its exact expected fragment"
    fi
    if [[ -e $expected || -L $expected ]]; then
        [[ -f $expected && ! -L $expected ]] || die "$expected is not a regular file"
        [[ $(stat -c %U:%G "$expected") == root:root ]] ||
            die "$expected has an unexpected owner"
        [[ $(stat -c %a "$expected") == 644 ]] ||
            die "$expected has an unexpected mode"
    fi
}

validate_legacy_directory() {
    local path=$1
    if [[ -e $path || -L $path ]]; then
        [[ -d $path && ! -L $path ]] || die "$path is not a real directory"
        [[ $(stat -c %U "$path") == "$EXPECTED_ACCOUNT" ]] ||
            die "$path has an unexpected owner"
        [[ $(readlink -f "$path") == "$path" ]] || die "$path resolves elsewhere"
    fi
}

validate_root_directory() {
    local path=$1
    if [[ -e $path || -L $path ]]; then
        [[ -d $path && ! -L $path ]] || die "$path is not a real directory"
        [[ $(stat -c %U:%G "$path") == root:root ]] ||
            die "$path has an unexpected owner"
        [[ $(readlink -f "$path") == "$path" ]] || die "$path resolves elsewhere"
    fi
}

validate_want() {
    local path=$1
    local expected=$2
    if [[ -e $path || -L $path ]]; then
        [[ -L $path ]] || die "$path is not a symlink"
        [[ $(readlink -f "$path") == "$expected" ]] ||
            die "$path points outside the exact legacy fragment"
    fi
}

validate_legacy_enablement_links() {
    local allow_reviewed=$1
    python3 - "$allow_reviewed" "$MAIN_FRAGMENT" "$DEBUG_FRAGMENT" \
        "$MAIN_WANT" "$DEBUG_WANT" <<'PY'
import os
import sys
from pathlib import Path

allow_reviewed = sys.argv[1] == "1"
targets = {str(Path(raw)) for raw in sys.argv[2:4]}
reviewed = {str(Path(raw)) for raw in sys.argv[4:6]}
roots = (
    Path("/etc/systemd/system"),
    Path("/run/systemd/system"),
    Path("/run/systemd/generator"),
    Path("/run/systemd/generator.early"),
    Path("/run/systemd/generator.late"),
    Path("/run/systemd/transient"),
    Path("/usr/lib/systemd/system"),
    Path("/lib/systemd/system"),
)

def fail(error: OSError) -> None:
    raise error

for root in roots:
    if not root.exists():
        continue
    for directory, names, _files in os.walk(root, followlinks=False, onerror=fail):
        for name in names + _files:
            path = Path(directory, name)
            if not path.is_symlink():
                continue
            resolved = str(path.resolve(strict=False))
            if resolved not in targets:
                continue
            if allow_reviewed and str(path) in reviewed:
                continue
            raise SystemExit(f"unexpected legacy unit link remains: {path}")
PY
}

reject_nested_mounts() {
    python3 - "$@" <<'PY'
import os
import sys
from pathlib import Path

roots = tuple(os.fsencode(str(Path(raw))) for raw in sys.argv[1:])
for line in Path("/proc/self/mountinfo").read_bytes().splitlines():
    fields = line.split()
    if len(fields) < 5:
        raise SystemExit(1)
    mountpoint = fields[4].replace(b"\\040", b" ").replace(b"\\011", b"\t")
    for root in roots:
        if mountpoint == root or mountpoint.startswith(root + b"/"):
            raise SystemExit(f"mount exists inside legacy deletion target: {os.fsdecode(root)}")
PY
}

validate_fragment "$MAIN_UNIT" "$MAIN_FRAGMENT"
validate_fragment "$DEBUG_UNIT" "$DEBUG_FRAGMENT"
validate_want "$MAIN_WANT" "$MAIN_FRAGMENT"
validate_want "$DEBUG_WANT" "$DEBUG_FRAGMENT"
validate_legacy_directory "$APP_ROOT"
validate_legacy_directory "$STATE_ROOT"
validate_legacy_directory "$MAIN_RUNTIME"
validate_legacy_directory "$DEBUG_RUNTIME"
for path in "$MAIN_DROPIN_ETC" "$DEBUG_DROPIN_ETC" "$MAIN_DROPIN_RUN" "$DEBUG_DROPIN_RUN"; do
    validate_root_directory "$path"
done
reject_nested_mounts \
    "$MAIN_DROPIN_ETC" "$DEBUG_DROPIN_ETC" "$MAIN_DROPIN_RUN" "$DEBUG_DROPIN_RUN" \
    "$APP_ROOT" "$STATE_ROOT" "$MAIN_RUNTIME" "$DEBUG_RUNTIME"

validate_legacy_enablement_links 1

for unit in "$MAIN_UNIT" "$DEBUG_UNIT"; do
    dropins=$(systemctl show --property=DropInPaths --value "$unit") ||
        die "could not inspect $unit drop-ins"
    while IFS= read -r dropin; do
        [[ -z $dropin ]] && continue
        case "$dropin" in
            /etc/systemd/system/voice-ai-bot.service.d/* | \
                /etc/systemd/system/voice-ai-bot-debug.service.d/* | \
                /run/systemd/system/voice-ai-bot.service.d/* | \
                /run/systemd/system/voice-ai-bot-debug.service.d/*) ;;
            *) die "$unit has an unexpected drop-in: $dropin" ;;
        esac
    done < <(printf '%s\n' "$dropins" | tr ' ' '\n')

    load_state=$(systemctl show --property=LoadState --value "$unit") ||
        die "could not re-inspect $unit"
    if [[ $load_state == not-found ]]; then
        continue
    fi
    for property in ExecStop ExecStopPost PropagatesStopTo OnFailure; do
        [[ -z $(systemctl show --property="$property" --value "$unit") ]] ||
            die "$unit has an unsafe effective $property directive"
    done
    pid=$(systemctl show --property=MainPID --value "$unit") ||
        die "could not inspect $unit process"
    [[ $pid =~ ^[0-9]+$ ]] || die "$unit returned an invalid process id"
    control_group=$(systemctl show --property=ControlGroup --value "$unit") ||
        die "could not inspect $unit cgroup"
    cgroup_procs=""
    if [[ -n $control_group ]]; then
        cgroup_procs="/sys/fs/cgroup$control_group/cgroup.procs"
    fi
    if ((pid > 1)) || [[ -n $cgroup_procs && -s $cgroup_procs ]]; then
        systemctl kill --kill-whom=all --signal=KILL "$unit" ||
            die "could not kill $unit"
    fi
    timeout 10 systemctl stop "$unit" || die "could not stop $unit"
    [[ $(systemctl show --property=MainPID --value "$unit") == 0 ]] ||
        die "$unit still owns a process after KILL"
    systemctl is-active --quiet "$unit" && die "$unit is still active"
    if [[ -n $control_group && -f /sys/fs/cgroup$control_group/cgroup.procs ]]; then
        [[ ! -s /sys/fs/cgroup$control_group/cgroup.procs ]] ||
            die "$unit cgroup still contains processes"
    fi
done

python3 - "$APP_ROOT" "$STATE_ROOT" "$MAIN_RUNTIME" "$DEBUG_RUNTIME" <<'PY'
import os
import sys
from pathlib import Path

legacy_roots = tuple(os.fsencode(raw) for raw in sys.argv[1:])

def inside(raw: bytes) -> bool:
    value = raw.removesuffix(b" (deleted)")
    return any(value == root or value.startswith(root + b"/") for root in legacy_roots)

for process in Path("/proc").glob("[0-9]*"):
    if process.name == str(os.getpid()):
        continue
    try:
        status = (process / "status").read_text(errors="replace")
        if "\nKthread:\t1\n" in f"\n{status}":
            continue
        argv = (process / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        environment = (process / "environ").read_bytes().split(b"\0")
        cwd = os.fsencode(os.readlink(process / "cwd"))
        executable = os.fsencode(os.readlink(process / "exe"))
        maps = (process / "maps").read_bytes().splitlines()
        descriptors = []
        for descriptor in (process / "fd").iterdir():
            try:
                descriptors.append(os.fsencode(os.readlink(descriptor)))
            except FileNotFoundError:
                continue
    except (FileNotFoundError, ProcessLookupError):
        if process.exists():
            raise SystemExit(f"could not completely inspect live process {process.name}")
        continue
    except PermissionError:
        raise SystemExit(f"permission denied inspecting live process {process.name}") from None
    references = argv + environment + descriptors + maps + [cwd, executable]
    if any(inside(value) or any(root in value for root in legacy_roots) for value in references):
        raise SystemExit("a process still references a legacy deletion target")
PY

# These are the complete fixed legacy targets. They were all validated above;
# no caller-supplied or glob-expanded path reaches rm.
rm -f -- "$MAIN_WANT" "$DEBUG_WANT" "$MAIN_FRAGMENT" "$DEBUG_FRAGMENT"
rm -rf --one-file-system -- \
    "$MAIN_DROPIN_ETC" "$DEBUG_DROPIN_ETC" "$MAIN_DROPIN_RUN" "$DEBUG_DROPIN_RUN" \
    "$APP_ROOT" "$STATE_ROOT" "$MAIN_RUNTIME" "$DEBUG_RUNTIME"
systemctl daemon-reload || die "systemd daemon-reload failed"
systemctl reset-failed "$MAIN_UNIT" "$DEBUG_UNIT" >/dev/null 2>&1 || true

# Re-scan after unlinking so a racing process cannot retain an open deleted
# file, cwd, mapping, environment value, or runtime/state reference.
python3 - "$APP_ROOT" "$STATE_ROOT" "$MAIN_RUNTIME" "$DEBUG_RUNTIME" <<'PY'
import os
import sys
from pathlib import Path

roots = tuple(os.fsencode(raw) for raw in sys.argv[1:])
for process in Path("/proc").glob("[0-9]*"):
    if process.name == str(os.getpid()):
        continue
    try:
        status = (process / "status").read_text(errors="replace")
        if "\nKthread:\t1\n" in f"\n{status}":
            continue
        values = [
            (process / "cmdline").read_bytes(),
            (process / "environ").read_bytes(),
            (process / "maps").read_bytes(),
            os.fsencode(os.readlink(process / "cwd")),
            os.fsencode(os.readlink(process / "exe")),
        ]
        for descriptor in (process / "fd").iterdir():
            try:
                values.append(os.fsencode(os.readlink(descriptor)))
            except FileNotFoundError:
                continue
    except (FileNotFoundError, ProcessLookupError):
        if process.exists():
            raise SystemExit(f"could not completely inspect live process {process.name}")
        continue
    except PermissionError:
        raise SystemExit(f"permission denied inspecting live process {process.name}") from None
    if any(root in value for root in roots for value in values):
        raise SystemExit("a process retained a deleted legacy reference")
PY

for path in \
    "$MAIN_WANT" "$DEBUG_WANT" "$MAIN_FRAGMENT" "$DEBUG_FRAGMENT" \
    "$MAIN_DROPIN_ETC" "$DEBUG_DROPIN_ETC" "$MAIN_DROPIN_RUN" "$DEBUG_DROPIN_RUN" \
    "$APP_ROOT" "$STATE_ROOT" "$MAIN_RUNTIME" "$DEBUG_RUNTIME"
do
    [[ ! -e $path && ! -L $path ]] || die "legacy target remains: $path"
done
main_load_state=$(systemctl show --property=LoadState --value "$MAIN_UNIT") ||
    die "could not verify $MAIN_UNIT removal"
debug_load_state=$(systemctl show --property=LoadState --value "$DEBUG_UNIT") ||
    die "could not verify $DEBUG_UNIT removal"
[[ $main_load_state == not-found ]] || die "$MAIN_UNIT remains loaded"
[[ $debug_load_state == not-found ]] || die "$DEBUG_UNIT remains loaded"

validate_legacy_enablement_links 0

printf 'legacy assistant permanently deleted; no backup or rollback was created\n'
