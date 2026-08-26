# Six-minute laptop hackathon demo

This is the target storyboard for the implemented macOS squat composition. It
keeps the webcam/pose/Guardian loop local, uses one long-lived
`gpt-realtime-2.1` socket for the workout, and runs Flower only after the
session. It is a research prototype, not a clinical assessment or emergency
system.

The Realtime cue path has passed a live credential acceptance run: all three
simulated cues cleared the exact transcript gate and timestamped Whisper ASR.
That evidence covers the provider/gate/ASR lane only. Do not present it as
webcam pose accuracy, actual speaker onset, or camera-to-speaker success.

## Setup

- One macOS laptop with one RGB webcam, microphone, and speaker.
- Python 3.12 with `mediapipe==0.10.35`,
  `opencv-contrib-python==4.14.0.94`, and `sounddevice==0.5.6` from the laptop
  extra.
- The explicitly installed, checksum-verified MediaPipe Pose Landmarker Lite
  model at `models/mediapipe/pose_landmarker_lite-v1.task`.
- One ten-repetition squat plan with arms held in a T shape and reviewed fixed
  cue phrases.
- A valid `OPENAI_API_KEY` for the target live voice portion.
- A separate three-client Flower simulation using its existing
  `seated-knee-extension` demo data; do not present it as squat training.

Install and start with the named commands:

```bash
uv sync --extra laptop --extra dev
uv run recoverybox download-pose-model
uv run recoverybox squat-demo
```

Use `q`/Escape as physical stop, Enter to toggle push-to-talk, and `r` plus
Enter to request an explicit resume from a fresh assessable standing pose. An
explicit `--no-voice` run can demonstrate local camera/tracker behavior, but it
has no cue or microphone path and is not voice acceptance evidence.

## Storyboard

1. **State the boundary (30 seconds).** Show the medical-prototype disclaimer.
   Explain that one webcam provides 2D projected landmarks, not depth,
   anatomical 3D angles, force, pain, or cross-camera agreement.
2. **Start one workout session (45 seconds).** Launch `recoverybox squat-demo`.
   Show that it opens one Realtime WebSocket, configures one session, and keeps
   that socket for all user turns and cues. Mention OpenAI's 60-minute maximum
   Realtime-session duration and keep the demo well below it.
3. **Local squat loop (90 seconds).** Begin from an assessable standing pose,
   then perform full standing-down-standing cycles with arms held in a T shape.
   Show only derived phase, confidence, projected knee angle, repetition count,
   and closed event/issue IDs—not uploaded frames or raw landmarks.
4. **Prompt cue gate (45 seconds).** For a repetition cue, show the local path:
   Guardian-selected cue ID, fixed catalog phrase, isolated response on the
   same socket, quarantined PCM, terminal `completed` response, and exact
   completed-transcript match before speaker release. Ordinary exercise speech
   remains blocked.
5. **Fail closed (45 seconds).** Occlude required landmarks or interrupt the
   network. The result becomes non-assessable or the cue fails, coaching
   pauses, and queued model audio is preempted. There is no TTS, generic model
   speech, or prerecorded fallback. Local pose processing, Guardian state, and
   physical stop remain available.
6. **Prove session-end semantics (75 seconds).** Complete the tenth rep and show
   that the shared session stays open. A pause, silence, ordinary response, or
   “finish this set” also does not end it. End only through an explicit physical
   stop or an unambiguous user request that becomes a locally validated,
   argument-free `finish_session` tool call. Raw text or a transcript is never
   itself an end capability.
7. **Flower after the session (30 seconds).** Close the workout first. Explain
   that Flower can read only a closed, sanitized numeric snapshot in a later
   discrete job. The current FAB is signed for `seated-knee-extension` and must
   reject laptop `squat` records until a new versioned adapter, labels, and
   model signature exist.

## Simulated-event live verification

Before claiming live cue delivery, run:

```bash
uv run recoverybox verify-realtime-cues \
  --output-dir artifacts/realtime-verification \
  --voice marin \
  --asr-model whisper-1
```

This command does not use the webcam. It simulates rep one, rep two, and arms
leaving the T shape, routes all three through the real local Guardian, and
requests the resulting cues sequentially over one live Realtime connection.
Only audio that clears the completed-response transcript quarantine can be
written as WAV; a successful run also writes `report.json` with measured stage
timings and optional post-gate Whisper verification.

Current sanitized output:

```text
exit code: 0
model/voice: gpt-realtime-2.1 / marin
exact Realtime transcript gate: 3/3 passed
timestamped whisper-1 ASR: 3/3 passed
safe playback release after request: 502 ms / 547 ms / 740 ms
gate-released verification WAV files: 3
report.json: produced locally under artifacts/realtime-verification/
```

The WAVs and report are ignored local verification artifacts, not committed
demo assets. The measured release timings are live provider/gate evidence, but
the semantic events are simulated and must not be relabeled as webcam or
camera-to-speaker latency.

## Evidence to keep on screen

- Dependency pins and pose-model install/verification result.
- One Realtime connection/session identifier and elapsed connection age.
- Local mode: idle, active exercise, paused, or stopped.
- Pose assessability, confidence, projected knee angle, phase, rep count, and
  issue IDs without uploading frames.
- Guardian decision, rule version, cue ID, fixed phrase, response status,
  exact-transcript gate result, and released PCM byte count.
- Session-end source: `physical_stop` or validated `finish_session` only.
- A post-session boundary marker before any Flower command.
- Flower exercise/schema identity and aggregate-only round metrics.
