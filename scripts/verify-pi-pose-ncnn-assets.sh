#!/usr/bin/env bash
set -euo pipefail

# Verify the closed RecoveryBox NCNN asset tree below one absolute root.  An
# optional OWNER (for example root:root) also binds every inode's ownership.
if [[ $# -lt 1 || $# -gt 2 || "$1" != /* || "$1" == / ]]; then
  printf 'usage: %s ABSOLUTE_RECOVERYBOX_ROOT [OWNER]\n' "$0" >&2
  exit 2
fi

root=${1%/}
expected_owner=${2:-}
runtime="$root/runtime/ncnn"
models="$root/models/ncnn"

die() {
  printf 'NCNN asset verification failed: %s\n' "$*" >&2
  exit 1
}

verify_directory() {
  local path=$1
  [[ -d "$path" && ! -L "$path" ]] || die "$path is not a real directory"
  [[ $(stat -c %a "$path") == 755 ]] || die "$path has an unexpected mode"
  if [[ -n "$expected_owner" ]]; then
    [[ $(stat -c %U:%G "$path") == "$expected_owner" ]] ||
      die "$path has an unexpected owner"
  fi
}

verify_file() {
  local path=$1
  local expected_size=$2
  local expected_digest=$3
  [[ -f "$path" && ! -L "$path" ]] || die "$path is not a regular file"
  [[ $(stat -c %h "$path") == 1 ]] || die "$path is hard-linked"
  [[ $(stat -c %a "$path") == 644 ]] || die "$path has an unexpected mode"
  [[ $(wc -c <"$path" | tr -d ' ') == "$expected_size" ]] ||
    die "$path has an unexpected size"
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected_digest" ]] ||
    die "$path has an unexpected digest"
  if [[ -n "$expected_owner" ]]; then
    [[ $(stat -c %U:%G "$path") == "$expected_owner" ]] ||
      die "$path has an unexpected owner"
  fi
}

for directory in \
  "$root/runtime" "$runtime" "$runtime/ncnn" "$runtime/ncnn.libs" \
  "$root/models" "$models"
do
  verify_directory "$directory"
done

verify_file \
  "$runtime/ncnn/__init__.py" \
  118 \
  ef90c76a49b37e74b0cd89f1da9502e764ee6b24a8da44860f4af894dc5838fe
verify_file \
  "$runtime/ncnn/ncnn.cpython-313-arm-linux-gnueabihf.so" \
  6611269 \
  b64fdd46904e1a3379fd71f4179b2a06c3109eb4a6588414a5e0fdc22c7811c9
verify_file \
  "$runtime/ncnn.libs/libgomp-39027b09.so.1.0.0" \
  184421 \
  d94a8c0b47d2371b67b9417ff21cff03204f162f6db9c8000d78b11ce389caf9
verify_file \
  "$models/rtmpose-t.param" \
  16845 \
  0348745dcfded9842b546a4e904c410e6c9640a09b1713b1dba055c25b646595
verify_file \
  "$models/rtmpose-t.bin" \
  13332744 \
  6fd3738741cfe14c82e40762434cdbe29c484b8a5a0b2bdefbc83d8bacb94c7c
verify_file \
  "$models/nanodet-m-int8.param" \
  17127 \
  ecce99dba4f9bd9298eb753905917a1eedf84e94755a3915868b155755097f04
verify_file \
  "$models/nanodet-m-int8.bin" \
  1004492 \
  719679139b0762d01663508a0893b026fdec09955f881d789b3b6bbc1ca900e1

# The two subtrees are closed: a valid pinned file may not hide alongside an
# unreviewed library, model, device, FIFO, socket, symlink, or extra directory.
runtime_entries=$(find "$runtime" -mindepth 1 -print | wc -l | tr -d ' ')
model_entries=$(find "$models" -mindepth 1 -print | wc -l | tr -d ' ')
[[ "$runtime_entries" == 5 ]] || die "runtime tree contains unexpected entries"
[[ "$model_entries" == 4 ]] || die "model tree contains unexpected entries"
if find "$runtime" "$models" \! \( -type d -o -type f \) -print -quit | grep -q .; then
  die "asset tree contains a special entry"
fi
