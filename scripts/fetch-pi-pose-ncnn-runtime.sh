#!/usr/bin/env bash
set -euo pipefail

# Exact CPython 3.13 / ARMv7 NCNN runtime used by the Raspberry Pi 3 lane.
# The wheel is downloaded only by an explicit deployment/integration command;
# ordinary tests merely inspect this helper and never contact the network.
WHEEL_FILENAME='ncnn-1.0.20260526-cp313-cp313-manylinux_2_31_armv7l.whl'
WHEEL_URL='https://files.pythonhosted.org/packages/2f/2f/1a8f4c5d83213ac459865fbea4af7051ab63c5934fd2813cccfaf9bf6409/ncnn-1.0.20260526-cp313-cp313-manylinux_2_31_armv7l.whl'
WHEEL_SIZE='2976956'
WHEEL_SHA256='28c1e2a574b8f9bcbcc8c95de94d7814fb25b1a106f12dae3b7a8a4344d2db4b'

NCNN_INIT='ncnn/__init__.py'
NCNN_INIT_SIZE='118'
NCNN_INIT_SHA256='ef90c76a49b37e74b0cd89f1da9502e764ee6b24a8da44860f4af894dc5838fe'
NCNN_NATIVE='ncnn/ncnn.cpython-313-arm-linux-gnueabihf.so'
NCNN_NATIVE_SIZE='6611269'
NCNN_NATIVE_SHA256='b64fdd46904e1a3379fd71f4179b2a06c3109eb4a6588414a5e0fdc22c7811c9'
NCNN_LIBGOMP='ncnn.libs/libgomp-39027b09.so.1.0.0'
NCNN_LIBGOMP_SIZE='184421'
NCNN_LIBGOMP_SHA256='d94a8c0b47d2371b67b9417ff21cff03204f162f6db9c8000d78b11ce389caf9'

usage() {
  printf 'usage: %s NEW_ABSOLUTE_RUNTIME_DIRECTORY\n' "$0" >&2
  exit 2
}

if [[ $# -ne 1 || "$1" != /* || "$1" == / ]]; then
  usage
fi

destination=${1%/}
if [[ -e "$destination" || -L "$destination" ]]; then
  printf 'runtime destination must not already exist: %s\n' "$destination" >&2
  exit 1
fi

parent=${destination%/*}
[[ -n "$parent" && "$parent" != "$destination" ]] || usage
mkdir -p -- "$parent"
temporary_directory=$(mktemp -d "$parent/.ncnn-runtime.XXXXXX")
cleanup() {
  if [[ -n "${temporary_directory:-}" && -d "$temporary_directory" ]]; then
    rm -rf -- "$temporary_directory"
  fi
}
trap cleanup EXIT HUP INT TERM

verify_file() {
  local candidate=$1
  local expected_size=$2
  local expected_digest=$3
  [[ -f "$candidate" && ! -L "$candidate" ]] || return 1
  [[ $(wc -c <"$candidate" | tr -d ' ') == "$expected_size" ]] || return 1
  [[ $(sha256sum "$candidate" | awk '{print $1}') == "$expected_digest" ]]
}

wheel="$temporary_directory/$WHEEL_FILENAME"
curl \
  --fail \
  --location \
  --silent \
  --show-error \
  --proto '=https' \
  --tlsv1.2 \
  --output "$wheel" \
  "$WHEEL_URL"
if ! verify_file "$wheel" "$WHEEL_SIZE" "$WHEEL_SHA256"; then
  printf 'NCNN ARMv7 wheel failed pinned size/digest verification\n' >&2
  exit 1
fi

payload="$temporary_directory/payload"
mkdir -m 0755 "$payload" "$payload/ncnn" "$payload/ncnn.libs"
unzip -p "$wheel" "$NCNN_INIT" >"$payload/$NCNN_INIT"
unzip -p "$wheel" "$NCNN_NATIVE" >"$payload/$NCNN_NATIVE"
unzip -p "$wheel" "$NCNN_LIBGOMP" >"$payload/$NCNN_LIBGOMP"

if \
  ! verify_file "$payload/$NCNN_INIT" "$NCNN_INIT_SIZE" "$NCNN_INIT_SHA256" || \
  ! verify_file "$payload/$NCNN_NATIVE" "$NCNN_NATIVE_SIZE" "$NCNN_NATIVE_SHA256" || \
  ! verify_file "$payload/$NCNN_LIBGOMP" "$NCNN_LIBGOMP_SIZE" "$NCNN_LIBGOMP_SHA256"
then
  printf 'extracted NCNN ARMv7 runtime failed pinned size/digest verification\n' >&2
  exit 1
fi

chmod 0644 "$payload/$NCNN_INIT" "$payload/$NCNN_NATIVE" "$payload/$NCNN_LIBGOMP"
mv -- "$payload" "$destination"
rm -f -- "$wheel"
rmdir -- "$temporary_directory"
temporary_directory=''
trap - EXIT HUP INT TERM
