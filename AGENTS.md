# RecoveryBox engineering rules

This repository is a clean-room implementation. Add only newly written code,
tests, fixtures, and documentation with a clear purpose in this product.

## Safety boundary

- The deterministic Guardian is the sole authority for pause, stop, and
  escalation decisions.
- A learned model or language model may preserve or increase caution. It may
  never weaken a Guardian decision, alter a prescription, or control hardware
  directly.
- Fixed exercise-cue phrases live in a versioned, clinician-reviewed prompt
  catalog. Only the local Guardian may select a cue ID, after which
  `gpt-realtime-2.1` receives an isolated prompt requiring exactly that phrase.
- While an exercise is active, ordinary model speech is blocked. Cue PCM stays
  quarantined until the response is complete and its completed transcript
  exactly matches the Guardian-selected catalog phrase; otherwise discard it.
- Missing, stale, low-confidence, or disagreeing camera evidence is
  non-assessable. Never infer a safe movement from insufficient evidence.

## Privacy boundary

- Camera frames are process-local and short-lived. Only numeric pose summaries
  may cross the capture boundary.
- Flower clients read sanitized feature records only. Raw images, audio,
  transcripts, names, and patient identifiers must never enter Flower records.
- Transcript retention is disabled by default. Do not log secrets or patient
  content.

## Runtime boundary

- The device session loop owns cameras, the button, audio, and immediate
  safety. Cue speech requires the Realtime connection, but local pose
  processing, Guardian decisions, and fail-safe pause/stop behavior must keep
  working when cloud services are unavailable.
- Flower runs discrete, restartable training/evaluation jobs after sessions.
  A Flower ClientApp must never own an always-on camera or voice session.
- The clinician AgentApp is a separate FAB and is read-only for this prototype.

## macOS native-runtime safety

RecoveryBox's ordinary test suite must remain hardware-free. The MediaPipe
adapter tests use fake runtime modules; they must not open a real camera or
construct a real `PoseLandmarker`. On this Mac, `mediapipe==1.0.1` aborts
during `PoseLandmarker` creation in `DrishtiMetalHelper` both inside the Codex
sandbox and in an elevated retry. That `SIGABRT` produces a “Python quit
unexpectedly” dialog instead of a catchable Python exception. The validated
laptop pair is the exact `[project.optional-dependencies].laptop` pin:
`mediapipe==0.10.35` with `opencv-contrib-python==4.14.0.94`.

- In the standard sandbox, never call `WebcamPoseSource.open()`, import/probe
  the real `mediapipe` runtime, construct `PoseLandmarker`, open a camera with
  OpenCV, show a native preview window, or run a live pose/camera demo.
- Never install, probe, or retry MediaPipe 1.0.1 on this Mac. Preserve the
  pinned laptop dependency pair, and do not install a second OpenCV wheel that
  also exports `cv2`.
- Do not add a live-hardware test to the default `pytest` suite. Keep MediaPipe,
  OpenCV, camera, microphone, speaker, GPIO, and network behavior behind fakes
  or explicit integration launchers.
- `uv run pytest`, Ruff, compile-only checks, and the deterministic demo are
  safe in the sandbox only while they use the existing fake/runtime-free
  boundaries. If a test unexpectedly begins initializing native vision or
  audio hardware, interrupt it rather than rerunning it.
- Never respond to a native abort by trying another Python executable, venv,
  dependency import, delegate, or fallback command. Preserve the first crash
  evidence and inspect the macOS report read-only.
- When real webcam/MediaPipe verification is explicitly required, run one
  named, repository-owned integration launcher with the pinned laptop
  dependency pair. Use `sandbox_permissions: require_escalated` so it has host
  Metal and camera access. Request approval for the exact command only; do not
  request a broad persistent prefix for `python`, `python3`, `uv`, or a shell.
- If the elevated integration run still exits by signal or reports a Metal,
  camera, or native-runtime failure, stop and report that first result. Do not
  retry alternate interpreters or delegates.

Add or update tests whenever a boundary above changes.
