# RecoveryBox

RecoveryBox is a clean-room hackathon prototype for private, personalized home
rehabilitation. Its runnable laptop slice derives squat events locally from one
webcam, a deterministic Guardian decides which reviewed cue ID may be spoken,
and `gpt-realtime-2.1` voices that fixed phrase through a quarantined audio
path. Flower remains a separate post-session training system and never owns the
live camera or voice loop.

The interactive slice is intentionally narrow: one macOS laptop, one RGB
camera, one configured ten-repetition squat with arms held in a T shape, and
one persistent Realtime session. The root Flower FAB remains a separate
three-client `seated-knee-extension` training demonstration, with a second
read-only clinician-review AgentApp.

> This is research and hackathon software. It has not been clinically
> validated, certified as a medical device, or designed for emergency use.

## Quickstart: run the laptop demo

Use Python 3.12 and [`uv`](https://docs.astral.sh/uv/). This is a checked-out,
`src`-layout project with a committed `uv.lock`, so `uv run` executes the local
RecoveryBox code with its locked project environment; no `.venv/bin/...`
command or manual activation is needed. `uvx` is intended primarily for
running a published, one-off tool in an isolated environment, so it is not the
default for developing and running this checkout.

Install the exact laptop dependencies and the checksum-verified pose model:

```bash
uv sync --extra laptop --extra dev
uv run recoverybox download-pose-model
```

Run the camera-only demo without Realtime voice or microphone input:

```bash
uv run recoverybox squat-demo --no-voice --no-mic
```

For the full demo, inject a valid `OPENAI_API_KEY` into the process environment
with your shell or secret manager, then run:

```bash
uv run recoverybox squat-demo
```

RecoveryBox does not load `.env` files. The preview shows the live camera,
MediaPipe skeleton, and squat count/status overlay. Press `q` or Escape in the
preview to stop. On macOS, the launching terminal or app must have Camera
permission; webcam visual acceptance is currently blocked by that permission
in this environment.

## What is implemented

- An explicit, checksum-verified MediaPipe Pose Landmarker install and a
  webcam adapter with no import-time model download.
- A deterministic single-camera, 2D squat tracker with standing-down-standing
  repetition counting, sustained arms-in-T feedback, and fail-closed missing
  or low-confidence pose handling.
- A strict, versioned three-angle pose contract, conservative dual-camera
  fusion, and a watchdog for the target Pi composition.
- A numeric-only, identity-free pose window and a validated post-session
  adapter to the Flower record schema.
- A deterministic Guardian with a reviewable fixed-phrase cue prompt catalog
  and monotonic safety arbitration.
- A manual 24 kHz PCM Realtime protocol and one-session laptop composition
  that reuses one WebSocket for user turns and every cue.
- Separate audible-output policies for check-in and active exercise, including
  terminal-response transcript quarantine for prompt-generated cues.
- macOS microphone and speaker adapters with bounded capture, serialized
  playback, synchronous preemption, and no exercise TTS fallback.
- A race-tested device state machine with injectable button, LED, recorder,
  playback, and conversation ports, plus explicit connectivity recovery.
- A three-client Flower 1.34 SecAgg+ application over a small NumPy classifier,
  with a signed one-exercise/one-label schema contract.
- An isolated clinician AgentApp project with deterministic, evidence-backed
  review logic.
- Unit and replay tests for the principal safety and privacy boundaries.

See [the laptop squat runbook](docs/laptop-squat-demo.md),
[architecture](docs/architecture.md), [safety and privacy case](docs/safety-and-privacy.md),
and [six-minute demo](docs/hackathon-demo.md).

## Hardware-free checks

For the ordinary hardware-free suite:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run recoverybox doctor
uv run recoverybox demo-safety
```

`doctor` checks the laptop camera index, pose-model size/hash, exact MediaPipe
and OpenCV distribution versions, and API-key presence without importing the
native runtimes or probing camera/audio hardware. It never prints the key;
`hardware_probed` remains false. The safety demo is deterministic and needs no
hardware, cloud account, or network.

## Laptop demo behavior and safety

The vision runtime is pinned to `mediapipe==0.10.35` and
`opencv-contrib-python==4.14.0.94`; the laptop extra also pins
`sounddevice==0.5.6`. Do not install a second OpenCV wheel alongside the
contrib wheel. The pose-model command verifies the official 5,777,746-byte
Pose Landmarker Lite file and installs it at
`models/mediapipe/pose_landmarker_lite-v1.task`; package import never downloads
it implicitly.

In the preview, `q`/Escape stops; in the terminal, Enter toggles push-to-talk,
`r` plus Enter requests resume from a fresh assessable standing pose, and `q`
plus Enter stops. `--no-voice --no-mic` runs the local camera/tracker
intentionally without a socket, microphone turns, or spoken cues; it is not a
speech fallback or voice acceptance test.

The demo uses one Realtime WebSocket and configures one session once. Exercise
cues and manual audio turns reuse it; a cue uses an isolated response on that
same socket rather than opening another connection. Completing the tenth rep,
pausing, silence, or an ordinary response does not end the session. Only a
physical stop or a locally validated, empty `finish_session` tool call is a
normal end capability. OpenAI documents a 60-minute maximum Realtime-session
duration, so this one-connection hackathon workout must remain below that cap.

During active exercise, ordinary model speech is blocked. Cue PCM is released
only after the isolated response completes and its completed transcript
exactly matches the Guardian-selected fixed phrase. There is no TTS,
free-form-speech, or prerecorded-audio fallback: provider or network failure
pauses cloud coaching while local pose analysis, Guardian decisions, and stop
behavior remain available.

The laptop path is deliberately single-camera and 2D. Projected knee and arm
angles are sensitive to viewpoint and occlusion, there is no depth or
cross-camera agreement measurement, and this tracker is not a clinical motion
assessment. See the [runbook](docs/laptop-squat-demo.md) for exact setup,
limitations, and acceptance evidence.

### Realtime cue verification status

The live verification launcher simulates rep one, rep two, and arms leaving the
T shape through the real Guardian, then requests all three cues sequentially
on one Realtime connection:

```bash
uv run recoverybox verify-realtime-cues \
  --output-dir artifacts/realtime-verification \
  --voice marin \
  --asr-model whisper-1
```

Only gate-approved PCM can be written as WAV, alongside a content-free
`report.json`; provider IDs, transcripts, and ASR word text are not persisted.
The live `gpt-realtime-2.1` run with the `marin` voice succeeded: all three
Guardian-selected cues passed the completed-transcript exact-phrase gate, and
all three released cues passed the Whisper ASR check. Safe-release latencies
were 502 ms, 547 ms, and 740 ms. The report remains a local, ignored artifact
under `artifacts/`; it is not committed. This verifies the Realtime cue path
from simulated deterministic exercise events, not the separate webcam visual
or camera-to-speaker path.

For the separate Pi check-in developer lane, expose `OPENAI_API_KEY` to its
service environment and keep the model pinned to `gpt-realtime-2.1`.
`.env.example` is a deployment template;
the library deliberately does not read secret files by itself. The production
transport accepts the key only while opening the WebSocket and does not retain
it. Audio is signed 16-bit little-endian mono PCM at 24 kHz with manual button
turns. A developer can exercise the real check-in lane with:

```bash
uv run recoverybox voice-checkin
```

Press Enter to start/stop a turn and `q` to exit. This command uses ALSA and the
real Realtime WebSocket, but it is intentionally check-in-only: it does not
compose cameras, GPIO, or the active-exercise loop. The automated suite does
not make a live API call.

## Run Flower locally

The root project is one Flower FAB containing only its ServerApp and ClientApp.
It runs only after a session from a closed sanitized feature snapshot; it is
never part of the camera/Guardian/Realtime loop. Configure a local Flower
simulation with three SuperNodes, then run the FAB:

```bash
uv run flwr federation simulation-config --num-supernodes 3
uv run flwr run . local --stream
```

The ClientApp reads a sanitized local JSONL feature snapshot selected through
its Flower node configuration. No image, audio, transcript, or direct patient
identifier belongs in that file. Every row is also bound to the exact exercise,
label definition, normalization ranges, model family, and parameter shapes.
The included demo dataset helper and exact deployment commands are documented
in `docs/flower.md`.

The current FAB is schema-bound to `seated-knee-extension`, so it must reject
the laptop demo's `squat` records. A separate versioned squat adapter, labels,
and model signature are required before squat summaries can enter Flower.

The clinician assistant is intentionally a second Flower project:

```bash
cd clinician_agent/clinician-agent
uv sync --extra dev
uv run pytest
```

Flower Agent is experimental, so the deterministic local review engine remains
usable without a running Agent runtime.

## Physical-device contract

- Hold the blue button to record; release it to commit the turn.
- Press during playback to stop local audio, cancel the response, truncate the
  unheard portion, and begin a new turn.
- Recording and playback cannot overlap.
- During check-in/post-session conversation, model audio may stream under the
  explicit conversational policy. The accepted 502/547/740 ms values are
  cue-verification safe-release measurements, not conversational or
  camera-to-speaker latency.
- During an active exercise, ordinary model speech is blocked. Fixed cue
  phrases live in a versioned, clinician-reviewed prompt catalog, and only the
  local Guardian may select a cue ID.
- For that selected ID, `gpt-realtime-2.1` receives an isolated prompt that
  requires exactly the catalog phrase. Its PCM remains quarantined until the
  response is complete and the completed transcript exactly matches the fixed
  phrase; a missing or mismatched transcript discards the audio.
- Low-confidence, stale, missing, or disagreeing camera evidence pauses
  assessment and preempts model audio instead of producing a correction. In
  this prototype, pause/stop actions are silent; audible exercise feedback is a
  separate explicit Guardian `CUE` decision carrying a reviewed prompt cue ID.
- Cue speech needs a working Realtime connection. If it is unavailable, the
  active exercise is silent while local pose processing, Guardian decisions,
  and fail-safe pause/stop behavior continue offline.
- No speech-generation API, operating-system TTS, generic Realtime response,
  or prerecorded cue is used as an exercise fallback.

Hardware-specific GPIO and ALSA device names are configuration, not application
logic. Defaults in `.env.example` are placeholders to verify on the actual Pi.

## Project layout

```text
src/recoverybox/
  core/             deterministic Guardian and fixed cue prompt catalog
  exercise/         deterministic single-camera squat events and tracking
  vision/           explicit MediaPipe model boundary and webcam adapter
  laptop/           macOS audio, microphone, and long-lived squat session
  pose/             numeric camera boundary, fusion, sanitized features
  realtime/         gpt-realtime-2.1 protocol and audible-output gate
  device/           button/audio state machine and hardware adapters
  federation/       Flower ServerApp, ClientApp, model, and local data
clinician_agent/    separate read-only Flower AgentApp FAB
docs/               architecture, safety case, demo, and operations
tests/              unit, privacy-boundary, and event-replay tests
```

## Current boundaries

The prototype does not prescribe exercises, diagnose injury, monitor
emergencies, change treatment, upload raw media, or promote a federated model
straight into the device runtime. Model signing, a controlled validation gate,
fleet identity, TLS deployment, regulatory work, and clinical validation remain
production work.

The laptop squat is not equivalent to the target Pi/two-camera product. Its
single RGB view and image-plane angles cannot establish depth, anatomical 3D
angles, cross-view agreement, load, balance, pain, or general movement safety.
Concrete Pi GPIO wiring and its long-running hardware composition remain a
separate integration and acceptance slice.

The simulated-event Guardian/Realtime verification harness has completed a
live `gpt-realtime-2.1`/`marin` run: all three exact-phrase gates and all three
Whisper ASR checks passed, with safe release at 502/547/740 ms. This establishes
the isolated live cue gate, not end-to-end camera-to-speaker acceptance. Webcam
visual acceptance remains blocked by macOS camera permission in this
environment.

## Upstream documentation

- [OpenAI gpt-realtime-2.1](https://developers.openai.com/api/docs/models/gpt-realtime-2.1)
- [OpenAI Realtime WebSocket guide](https://developers.openai.com/api/docs/guides/realtime-websocket)
- [OpenAI Realtime conversations and 60-minute session limit](https://developers.openai.com/api/docs/guides/realtime-conversations)
- [Flower quickstart](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
- [Flower secure aggregation](https://flower.ai/docs/examples/flower-secure-aggregation.html)
- [Flower AgentApp runtime](https://flower.ai/docs/agent/explanations/agentapp-runtime.html)
