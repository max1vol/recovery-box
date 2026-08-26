# Laptop squat demo

The runnable hackathon slice is a ten-repetition squat session on a macOS
laptop. One webcam is processed locally, deterministic code decides whether a
frame is assessable and whether a repetition completed, the Guardian selects
only reviewed cue IDs, and `gpt-realtime-2.1` voices the corresponding fixed
phrases. This is a narrow interaction demo, not a clinical assessment.

## Install exactly the laptop stack

Use Python 3.12 and install the laptop and development extras from the project
lockfile:

```bash
uv sync --extra laptop --extra dev
```

After installing the pose model below, `uv run recoverybox doctor --strict`
checks the configured camera index, model size/hash, the exact MediaPipe and
OpenCV distribution versions, and API-key presence. It deliberately does not
import the native runtimes or probe camera/audio hardware, so a passing doctor
report is prerequisite evidence rather than physical acceptance.

The native vision pair is intentionally exact rather than a floating range:

- `mediapipe==0.10.35`
- `opencv-contrib-python==4.14.0.94`

The same laptop extra pins `sounddevice==0.5.6` for microphone capture. Do not
add a second OpenCV wheel such as `opencv-python`: both wheels export `cv2` and
can make the imported native runtime depend on installation order.

Installing the Python packages does not download a pose model. Install the
pinned official MediaPipe Pose Landmarker Lite model explicitly:

```bash
uv run recoverybox download-pose-model
```

A successful default install reports:

```json
{
  "installed": "models/mediapipe/pose_landmarker_lite-v1.task"
}
```

The command installs
`models/mediapipe/pose_landmarker_lite-v1.task`, then requires both its exact
5,777,746-byte size and SHA-256
`59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`.
The camera runtime validates that same file before use; importing the package
never performs an implicit download.

For this documentation pass, the hardware-free doctor reported both package
pins as `match`, the pose model as `valid`, and `hardware_probed: false`. It
checks API-key presence only and is not provider acceptance; the separate live
cue verification evidence is documented below.

## Run the named demo

Expose a valid `OPENAI_API_KEY` in the process environment, then run:

```bash
uv run recoverybox squat-demo
```

The preview and terminal controls are deliberately small:

- `q` or Escape in the preview, or `q` plus Enter in the terminal, is the
  physical-stop capability;
- Enter toggles push-to-talk capture, then sends the completed turn; and
- `r` plus Enter requests an explicit resume, which succeeds only from a fresh,
  assessable standing pose.

`--no-voice` is an explicit camera/pose-only run. It opens no Realtime
connection, disables microphone turns, and produces no exercise cues. It is
useful for checking local tracking but is intentional silence, not a TTS
fallback and not voice acceptance evidence.

Other bounded launcher overrides are `--camera-index`, `--model-path`,
`--voice`, `--no-preview`, `--no-mic`, and the test-oriented `--max-frames`.
Run `uv run recoverybox squat-demo --help` for their exact CLI surface.

The command composes the webcam, MediaPipe adapter, deterministic 2D squat
tracker, Guardian, microphone, macOS speaker arbiter, and one Realtime
WebSocket. It uses 24 kHz mono signed 16-bit PCM. Camera frames and MediaPipe
landmarks remain process-local; downstream safety receives only derived squat
analysis and numeric state.

The demo configures one Realtime session once and reuses that socket for
manual audio turns and every exercise cue. Each cue response is isolated from
the conversation on that same connection; it does not create a new WebSocket.
Client events use one bounded, ordered sender worker, so a slow socket write
cannot block the camera/Guardian loop or a physical stop. Queue exhaustion or
an asynchronous send failure permanently disables cloud coaching for that
workout.
Completing ten repetitions, pausing, silence, or an ordinary model response
does not end the session. The two normal end capabilities are:

1. a local physical-stop action; or
2. a `finish_session` tool call with an empty object that has passed the local
   tool registry after an unambiguous request to leave or finish.

Raw model text, a transcript, a completed set, or wording such as “finish this
set” is not itself an end capability. An unexpected process or transport
failure can still force cleanup. Official OpenAI documentation limits a
Realtime session to 60 minutes, so this one-connection hackathon workout must
finish within that cap. A longer production session would need an explicit,
tested connection-rotation design that preserves local safety state.

## Exercise speech boundary

Ordinary model speech is blocked while the exercise is active. For a cue, the
local Guardian first authorizes a typed cue ID; local code resolves its fixed
catalog phrase and sends an isolated exact-response prompt. All returned PCM
stays in memory quarantine until the response reaches terminal `completed`
status and its completed output transcript exactly matches the catalog phrase.
A mismatch, cancellation, timeout, provider error, stale authorization, or
mode change discards the PCM.

There is deliberately no fallback to OpenAI speech generation, operating-system
TTS, free-form Realtime speech, or prerecorded cue audio. If the Realtime
connection or credential is unavailable, cue delivery fails closed and the
coaching mode pauses; camera analysis, the Guardian, and local stop behavior
remain available.

At shutdown the command prints a content-free JSON summary containing frame
counts, assessable-frame count, rep count, end source, final mode, whether voice
connected, installed runtime versions, and scrubbed failure class/reason fields.
It does not print frames, landmarks, captured audio, transcripts, prompts, or a
credential.

## Live cue verification

This separate command drives three simulated semantic events—rep one, rep two,
and arms out of the T position—through the real Guardian, then requests their
cues sequentially over one live Realtime socket:

```bash
uv run recoverybox verify-realtime-cues \
  --output-dir artifacts/realtime-verification \
  --voice marin \
  --asr-model whisper-1
```

Only PCM released by the exact-transcript quarantine gate can become a WAV.
On success the command writes those gate-released verification WAVs plus
`artifacts/realtime-verification/report.json`, including response-stage timing
and the optional post-gate ASR check. The persisted/stdout report keeps match
booleans and word timing offsets, but omits provider response IDs, transcripts,
and ASR word text. This verifies cue delivery; the three
semantic events are simulated and do not prove webcam pose accuracy.

Current sanitized result:

```text
exit code: 0
model/voice: gpt-realtime-2.1 / marin
exact Realtime transcript gate: 3/3 passed
timestamped whisper-1 ASR: 3/3 passed
safe playback release after request: 502 ms / 547 ms / 740 ms
gate-released verification WAV files: 3
report.json: produced locally under artifacts/realtime-verification/
```

The report and WAVs are local ignored verification artifacts and are not part
of the repository. No provider text, transcript, ASR word text, or credential
fragment is retained in the report. This accepts the simulated-event Realtime
cue/gate/ASR lane; it does not establish webcam pose accuracy, actual speaker
onset, or end-to-end camera-to-speaker latency.

## Single-camera and 2D limits

The laptop tracker uses one RGB webcam and image-plane `x`/`y` geometry. It
does not use a second view, depth sensing, MediaPipe world coordinates, force
measurement, or a clinical motion-capture reference. Consequently:

- there is no cross-camera agreement measurement;
- perspective, camera placement, self-occlusion, loose clothing, lighting, and
  landmarks leaving the frame can change or withhold the result;
- the measured knee and arm angles are 2D projections, not anatomical 3D joint
  angles; and
- the standing-down-standing state machine counts this configured squat only;
  it does not diagnose injury, detect every unsafe movement, or validate load,
  balance, or pain.

Low visibility, low presence, stale or missing frames, invalid geometry, and
large left/right projected-knee disagreement are non-assessable and cannot
advance a repetition. A single camera is an explicit hackathon limitation, not
evidence equivalent to the target dual-view Pi architecture.

## Flower remains post-session

No Flower ClientApp, ServerApp, or training round owns the webcam, Realtime
socket, microphone, speaker, squat tracker, or Guardian loop. Flower may run
only after the workout from a closed snapshot of sanitized numeric features.

The current root Flower FAB is signed for the separate
`seated-knee-extension` feature and label contract. It must reject laptop
`squat` records rather than silently treating them as compatible. Federating
squat data requires a new versioned post-session adapter, label definition,
model signature, and validation gate; that work is not part of this demo.

## OpenAI references

- [GPT-Realtime-2.1 model](https://developers.openai.com/api/docs/models/gpt-realtime-2.1)
- [Realtime conversations and the 60-minute session limit](https://developers.openai.com/api/docs/guides/realtime-conversations)
- [Realtime WebSocket guide](https://developers.openai.com/api/docs/guides/realtime-websocket)
