#!/usr/bin/env bash
set -euo pipefail

# RTMPose-t comes from OpenMMLab MMDeploy/MMPose (Apache-2.0 projects).
# NanoDet-m comes from the NanoDet v0.4.0 release (Apache-2.0 project).
RTMPOSE_URL='https://mmdeploy-oss.openmmlab.com/model/mmpose/rtmpose-t-ncnn-155ab7.zip'
RTMPOSE_ARCHIVE_SIZE='24967723'
RTMPOSE_ARCHIVE_SHA256='1c0481b760419a2140bf814d396335711ae1ffba91566395d1a529b4a177cd5d'
RTMPOSE_PARAM_SIZE='16845'
RTMPOSE_PARAM_SHA256='0348745dcfded9842b546a4e904c410e6c9640a09b1713b1dba055c25b646595'
RTMPOSE_BIN_SIZE='13332744'
RTMPOSE_BIN_SHA256='6fd3738741cfe14c82e40762434cdbe29c484b8a5a0b2bdefbc83d8bacb94c7c'

NANODET_URL='https://github.com/RangiLyu/nanodet/releases/download/v0.4.0/ncnn-nanodet-m-int8.zip'
NANODET_ARCHIVE_SIZE='910094'
NANODET_ARCHIVE_SHA256='2ac8f6ea9b5bb1cd52f809229c4492d7402b419daa11b068a13478b0fd33bc32'
NANODET_PARAM_SIZE='17127'
NANODET_PARAM_SHA256='ecce99dba4f9bd9298eb753905917a1eedf84e94755a3915868b155755097f04'
NANODET_BIN_SIZE='1004492'
NANODET_BIN_SHA256='719679139b0762d01663508a0893b026fdec09955f881d789b3b6bbc1ca900e1'

usage() {
  printf 'usage: %s ABSOLUTE_MODEL_DIRECTORY\n' "$0" >&2
  exit 2
}

if [[ $# -ne 1 || "$1" != /* ]]; then
  usage
fi

destination=${1%/}
if [[ -z "$destination" || "$destination" == / ]]; then
  usage
fi
mkdir -p -- "$destination"

verify_file() {
  local candidate=$1
  local expected_size=$2
  local expected_digest=$3
  [[ -f "$candidate" && ! -L "$candidate" ]] || return 1
  [[ $(wc -c <"$candidate" | tr -d ' ') == "$expected_size" ]] || return 1
  [[ $(sha256sum "$candidate" | awk '{print $1}') == "$expected_digest" ]]
}

if \
  verify_file "$destination/rtmpose-t.param" "$RTMPOSE_PARAM_SIZE" "$RTMPOSE_PARAM_SHA256" && \
  verify_file "$destination/rtmpose-t.bin" "$RTMPOSE_BIN_SIZE" "$RTMPOSE_BIN_SHA256" && \
  verify_file "$destination/nanodet-m-int8.param" "$NANODET_PARAM_SIZE" "$NANODET_PARAM_SHA256" && \
  verify_file "$destination/nanodet-m-int8.bin" "$NANODET_BIN_SIZE" "$NANODET_BIN_SHA256"
then
  chmod 0644 "$destination/rtmpose-t.param" "$destination/rtmpose-t.bin" \
    "$destination/nanodet-m-int8.param" "$destination/nanodet-m-int8.bin"
  exit 0
fi

temporary_directory=$(mktemp -d "$destination/.pose-models.XXXXXX")
cleanup() {
  if [[ -n "${temporary_directory:-}" && -d "$temporary_directory" ]]; then
    rm -rf -- "$temporary_directory"
  fi
}
trap cleanup EXIT HUP INT TERM

download_archive() {
  local url=$1
  local target=$2
  curl \
    --fail \
    --location \
    --silent \
    --show-error \
    --proto '=https' \
    --tlsv1.2 \
    --output "$target" \
    "$url"
}

rtmpose_archive="$temporary_directory/rtmpose.zip"
nanodet_archive="$temporary_directory/nanodet.zip"
download_archive "$RTMPOSE_URL" "$rtmpose_archive"
download_archive "$NANODET_URL" "$nanodet_archive"

if ! verify_file "$rtmpose_archive" "$RTMPOSE_ARCHIVE_SIZE" "$RTMPOSE_ARCHIVE_SHA256"; then
  printf 'RTMPose archive failed pinned size/digest verification\n' >&2
  exit 1
fi
if ! verify_file "$nanodet_archive" "$NANODET_ARCHIVE_SIZE" "$NANODET_ARCHIVE_SHA256"; then
  printf 'NanoDet archive failed pinned size/digest verification\n' >&2
  exit 1
fi

unzip -p "$rtmpose_archive" end2end.param >"$temporary_directory/rtmpose-t.param"
unzip -p "$rtmpose_archive" end2end.bin >"$temporary_directory/rtmpose-t.bin"
unzip -p "$nanodet_archive" nanodet_m-int8.param >"$temporary_directory/nanodet-m-int8.param"
unzip -p "$nanodet_archive" nanodet_m-int8.bin >"$temporary_directory/nanodet-m-int8.bin"

if \
  ! verify_file "$temporary_directory/rtmpose-t.param" "$RTMPOSE_PARAM_SIZE" "$RTMPOSE_PARAM_SHA256" || \
  ! verify_file "$temporary_directory/rtmpose-t.bin" "$RTMPOSE_BIN_SIZE" "$RTMPOSE_BIN_SHA256" || \
  ! verify_file "$temporary_directory/nanodet-m-int8.param" "$NANODET_PARAM_SIZE" "$NANODET_PARAM_SHA256" || \
  ! verify_file "$temporary_directory/nanodet-m-int8.bin" "$NANODET_BIN_SIZE" "$NANODET_BIN_SHA256"
then
  printf 'extracted NCNN model failed pinned size/digest verification\n' >&2
  exit 1
fi

chmod 0644 "$temporary_directory/rtmpose-t.param" "$temporary_directory/rtmpose-t.bin" \
  "$temporary_directory/nanodet-m-int8.param" "$temporary_directory/nanodet-m-int8.bin"
mv -f -- "$temporary_directory/rtmpose-t.param" "$destination/rtmpose-t.param"
mv -f -- "$temporary_directory/rtmpose-t.bin" "$destination/rtmpose-t.bin"
mv -f -- "$temporary_directory/nanodet-m-int8.param" "$destination/nanodet-m-int8.param"
mv -f -- "$temporary_directory/nanodet-m-int8.bin" "$destination/nanodet-m-int8.bin"
