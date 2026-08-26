#!/usr/bin/env bash
set -euo pipefail

MODEL_URL='https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite'
MODEL_SHA256='cd7cc22fa946e5d146a7b98d496853e1923e22828d3972d579973f27f91bb105'
MODEL_SIZE='2894840'

usage() {
  printf 'usage: %s ABSOLUTE_MODEL_PATH\n' "$0" >&2
  exit 2
}

if [[ $# -ne 1 || "$1" != /* ]]; then
  usage
fi

destination=$1
parent_directory=$(dirname "$destination")
mkdir -p "$parent_directory"

verify_model() {
  local candidate=$1
  [[ -f "$candidate" && ! -L "$candidate" ]] || return 1
  [[ $(wc -c <"$candidate" | tr -d ' ') == "$MODEL_SIZE" ]] || return 1
  [[ $(sha256sum "$candidate" | awk '{print $1}') == "$MODEL_SHA256" ]]
}

if verify_model "$destination"; then
  chmod 0644 "$destination"
  exit 0
fi

temporary_file=$(mktemp "${destination}.tmp.XXXXXX")
cleanup() {
  if [[ -n "${temporary_file:-}" && -f "$temporary_file" ]]; then
    rm -f -- "$temporary_file"
  fi
}
trap cleanup EXIT HUP INT TERM

curl \
  --fail \
  --location \
  --silent \
  --show-error \
  --proto '=https' \
  --tlsv1.2 \
  --output "$temporary_file" \
  "$MODEL_URL"

if ! verify_model "$temporary_file"; then
  printf 'downloaded MoveNet asset failed pinned size/digest verification\n' >&2
  exit 1
fi

chmod 0644 "$temporary_file"
mv -f -- "$temporary_file" "$destination"
temporary_file=''
