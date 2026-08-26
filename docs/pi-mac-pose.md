# Pi 3 local pose deployment

The production RecoveryBox path no longer sends camera frames or pose requests
between the Mac and Pi. The Pi owns `/dev/video0`, converts YUYV to BGRA with
libyuv, and runs the pinned NanoDet + RTMPose NCNN pipeline in one child capture
process. Only immutable numeric `SquatAnalysis` observations cross the private
child-to-Guardian IPC boundary. Frames are never written, logged, returned to
the Guardian parent, exposed by status, or sent over Tailscale.

The Mac-camera pose preview remains a test-only workflow for a person standing
at the laptop. It is not the deployed camera path. No Mac pose publisher is required
once Pi-local pose is installed.

| Role | Tailnet endpoint |
| --- | --- |
| Pi RecoveryBox | `100.106.237.106` |
| Pi sanitized status | `http://100.106.237.106:45874/healthz` |
| Allowed operator Mac | `100.70.100.93` |

## Direct Pi installation

Run the read-only preflight from the Mac:

```bash
scripts/deploy-pi3.sh
```

The preflight changes nothing. It uses literal pinned Tailscale addresses with
SSH configuration, proxy commands, and jump hosts disabled. It verifies that
the `pi` and root connections reach the same machine and that the target is the
reviewed ARMv7 Pi 3 runtime: CPython 3.13, glibc 2.31 or newer, libgpiod 2.x,
the libyuv `YUY2ToARGB` ABI, readable/writable `/dev/video0` and
`/dev/gpiochip0`, and `pi` membership in `video` and `gpio`. It also reads
`vcgencmd get_throttled` three times, 250 ms apart. The current low-nibble flags
must remain clear; sticky historical flags are reported but do not by
themselves authorize or reject movement evidence. Unknown or malformed flags
fail closed.

Apply the direct replacement only after preflight succeeds:

```bash
chmod 600 .env
set -a
. ./.env
set +a
scripts/deploy-pi3.sh --apply
unset OPENAI_API_KEY
```

`--apply` is intentionally irreversible. It:

1. downloads and verifies the exact CPython 3.13 ARMv7 NCNN runtime and the
   four pinned NanoDet/RTMPose files before transferring them;
2. stages fresh root-owned code and assets, verifies their hashes, and installs
   them at `/opt/recoverybox/{app,runtime,models}`;
3. writes a closed `RECOVERYBOX_POSE_SOURCE=local` environment at
   `/etc/recoverybox/recoverybox.env` with 640x480 YUYV, requested 10 FPS,
   eight V4L2 mmap buffers, two NCNN threads, and the 500 ms hard freshness
   limit;
4. recreates `/etc/recoverybox` as a closed tree containing only the exact
   environment and a credentials directory containing only `openai-api-key`;
   local pose loads no remote-pose token;
5. permanently removes the exact old voice assistant units, code, state, and
   prior `/home/pi/recoverybox` tree without a backup or compatibility shim;
6. runs a bounded, silent three-frame V4L2/libyuv/NCNN acceptance command as
   `pi`; and
7. installs and starts the root-owned RecoveryBox units as `pi`.

The same three-sample power gate runs from the digest-verified staged helper
immediately before the first service stop, immediately before pose acceptance,
at both legacy-assistant termination/removal boundaries, and again after
acceptance before either service starts. Thus a later active undervoltage or
thermal-throttle condition cannot be hidden by an earlier preflight.
`0xd0000` is historical-only; `0xd0005` is an active `0x5` failure. Never
increase the pose deadline to work around a power failure.

Every runtime/model file is checked by exact size and SHA-256 in the transfer
stage, after activation, and through each running service's `/proc/<pid>/root`
view. The running process environment is also checked for the exact local
camera, buffer, FPS, timeout, runtime, and model values. A successful cutover
requires a fresh numeric pose observation less than 500 ms old.

The closed `recoverybox-pi-v4l2-ncnn-check/v2` acceptance report requires three
acquired fresh frames, zero aggregate timeouts, zero capture misses, zero
worker timeouts, zero parent-stale rejections, `raw_frames_persisted=0`, and
`audio=disabled`. Detector, total-inference, and end-to-end evidence-age maxima
must each be finite and below 500 ms; pose time may be null only when no pose
ran, such as an empty room. `assessable=0` is valid in an empty or poorly
framed room: it proves camera acquisition and the detector path, not a human
RTMPose result. A reviewed synthetic-person fixture through the exact RTMPose
runtime, a live fully framed person, and a sustained thermal run remain separate
acceptance evidence.

## Device and credential boundaries

Tonight's deployed service is silent. `RECOVERYBOX_AUDIO_ENABLED=0` prevents
speaker construction, and systemd `DevicePolicy=closed` permits exactly
`/dev/video0` and `/dev/gpiochip0`; `/dev/snd` is not allowed. The main service
also sets `LimitCORE=0`, preventing native camera/model failures from persisting
raw frame memory in a core dump. It receives only the root-owned OpenAI file
credential, which is not read while audio is disabled. The status service
receives no credential and cannot access the credential directory.

There is no pose listener in local mode. Tailscale carries only SSH deployment
traffic and the sanitized status endpoint. The status service has no hardware
allow-list and exposes only the closed fields `service`, `peer`, `session`,
`mode`, `rep`, `age`, `voice`, `button`, and `failure`:

```bash
curl --fail http://100.106.237.106:45874/healthz
```

HTTP 200 requires a fresh, failure-free status. The Guardian remains the sole
authority for pause, stop, escalation, and fixed cue selection. Missing,
timed-out, stale, low-confidence, multiple-person, or clipped-person evidence
is non-assessable and cannot advance the exercise.

## Physical acceptance still required

With the Pi powered off, connect a normally-open momentary switch between
physical pin 16 (`BCM23`) and physical pin 14 (`GND`). No external voltage is
used: the service requests an active-low input with pull-up bias. Power the Pi
back on only after checking the wiring. A disconnected normally-open switch is
not electrically distinguishable from a released switch, so the operator must
also test an actual press.

The automated suite does not establish physical rehabilitation accuracy. Before
calling the Pi deployment fully accepted, record all of the following:

- one fully framed person producing an assessable NanoDet + RTMPose result;
- the reviewed synthetic-person fixture producing numeric RTMPose keypoints
  through the exact activated ARMv7 runtime;
- sustained capture-to-numeric age below 500 ms without thermal throttling;
- three correctly counted squats under representative framing and lighting;
- a deliberate BCM23 button press stopping the live session immediately; and
- later, when London audio is explicitly re-enabled, one reviewed cue with no
  unapproved speech.

Do not weaken the no-person or stale-evidence gates to make these checks pass.
