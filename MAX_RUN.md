# Max: run RecoveryBox

These commands assume macOS, Python 3.12, `uv`, and a checkout of this
repository. RecoveryBox is a prototype, not a medical device; stop if a
movement hurts or feels unsafe.

## Fastest local squat test

Install the locked laptop runtime and verified pose model once:

```bash
uv sync --extra laptop --extra dev
uv run recoverybox download-pose-model
```

Put `OPENAI_API_KEY=...` in `.env`, load it into the launching shell, then run:

```bash
chmod 600 .env
set -a
. ./.env
set +a
uv run recoverybox squat-demo --no-mic
```

The preview should show the Mac camera, pose bones, squat phase/count, camera
FPS, and pose-model FPS. The fixed spoken sequence is:

1. “Hi Max. Let's start with a set of three squats.”
2. Once a standing person is detected: “I can see you. Now do the squats.”
3. Rep one: “One.”
4. Rep two: “Slower.”
5. Rep three: “Three. Excellent. Now bring your arms out into a T shape.”

Press `q` or Escape in the preview to stop. If macOS asks, grant Camera access
to the terminal or app that launched the command. Add `--no-voice --no-mic`
for a silent tracking-only test.

## Two-process Mac topology test

This reproduces the temporary Prime/pose-client split locally. The Prime side
loads the API key directly from `.env`; the pose client never receives it.

Create one temporary authentication token:

```bash
umask 077
openssl rand -hex 32 > /tmp/recoverybox-pose-token.hex
/Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4
```

Use the printed Mac Tailscale IPv4 in both terminals. In terminal 1:

```bash
uv run recoverybox-local-prime \
  --tailscale-ip MAC_TAILSCALE_IP \
  --token-file /tmp/recoverybox-pose-token.hex \
  --env-file .env
```

In terminal 2:

```bash
uv run recoverybox-pose-client \
  --peer MAC_TAILSCALE_IP:45873 \
  --token-file /tmp/recoverybox-pose-token.hex \
  --authorize
```

The second command owns the Mac camera and opens the debugging preview with
bones, count, phase, frame FPS, pose-model FPS, and request FPS. Stop both
commands with Control-C, then delete the temporary token.

## Pi 3 deployment

The deployed Pi owns `/dev/video0` and runs V4L2, libyuv, NanoDet, RTMPose, the
deterministic Guardian, and BCM23 stop handling locally. It does not send raw
camera frames to the Mac. Tonight's deployment profile is deliberately silent:
it does not open `/dev/snd` or speak in London.

With the Pi powered off, wire one normally-open momentary stop switch directly
between physical pin 16 (`BCM23`) and physical pin 14 (`GND`). Do not connect
the switch to an external voltage; RecoveryBox requests the GPIO's pull-up and
treats a closed switch as the active-low stop signal.

Read the exact irreversible-deployment and physical-acceptance procedure before
running it:

- [Pi 3 deployment runbook](docs/pi-mac-pose.md)
- [Pi V4L2/NCNN backend](docs/pi-local-pose-ncnn.md)

Run the read-only preflight first:

```bash
ssh root@100.106.237.106 'vcgencmd get_throttled'
scripts/deploy-pi3.sh
```

The low hexadecimal nibble is the current power state and must be zero. A
result such as `0xd0000` contains historical flags only and passes with a
warning; `0xd0005` has active flags `0x5` and fails. After a fresh reboot,
prefer `0x0`. If the low nibble is non-zero, replace or reseat the Pi power
supply/cable before continuing; active undervoltage/throttling can make pose
evidence miss the Guardian's fixed 500 ms deadline and can drop SSH entirely.
Do not increase that safety deadline. The deployment script independently
requires three clear samples before replacement and again around live pose
acceptance.

Only after power and the read-only preflight both pass, load `.env` into the
current shell and apply the direct replacement:

```bash
chmod 600 .env
set -a
. ./.env
set +a
scripts/deploy-pi3.sh --apply
curl --fail http://100.106.237.106:45874/healthz
unset OPENAI_API_KEY
```

`--apply` permanently removes the named old assistant deployment and installs
the new root-owned tree at `/opt/recoverybox`; it creates no backup or rollback.
The API key is installed as a root-only systemd file credential and remains
unused while Pi audio is disabled.
