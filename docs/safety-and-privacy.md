# Prototype safety and privacy case

RecoveryBox is hackathon software. It has not been clinically validated,
certified as a medical device, or evaluated for emergency use. The demo must
state that clearly.

## Safety invariants

These are executable product requirements:

- During an active exercise, ordinary model speech is blocked. The only PCM
  that may be released is a completed `gpt-realtime-2.1` cue response whose
  transcript exactly matches the Guardian-selected fixed phrase in the
  versioned, clinician-reviewed prompt catalog.
- A button stop, pain report, `STOP`, or `ESCALATE` decision preempts queued
  audio immediately.
- A learned-model or language-model proposal cannot reduce Guardian caution.
- Missing, stale, low-confidence, or disagreeing camera evidence never produces
  a corrective movement cue.
- Camera disagreement is measured in degrees at both fusion and Guardian
  boundaries; the plan threshold names that unit explicitly.
- Recording and playback never overlap.
- Missing, partial, canceled, or mismatched cue transcripts discard all
  quarantined PCM.
- Realtime loss makes cue speech unavailable but cannot interrupt local pose
  processing or weaken Guardian pause/stop behavior.
- There is no exercise fallback to a speech-generation endpoint,
  operating-system TTS, generic Realtime speech, or prerecorded cue.
- Completing the prescribed repetitions does not end the Realtime session.
  Normal end requires a physical stop or an argument-free `finish_session`
  call that has passed the local one-tool registry.
- An incompatible feature/model schema is rejected before use.
- A client dataset above the configured SecAgg+ maximum weight is rejected
  before local training.

The language model can converse, classify a user utterance into an allowlisted
dialogue act, or request a tool. It cannot change an exercise plan, diagnose an
injury, give an unapproved exercise instruction, suppress an escalation, or
write directly to GPIO/audio/model storage.

## Laptop squat boundary

The runnable laptop demo is a single-camera, image-plane tracker. It uses the
`x` and `y` projection of MediaPipe landmarks to recognize three configured
standing-down-standing cycles. The third fixed cue then instructs the person to
bring their arms into a T shape; the script does not require the arms to remain
in that shape throughout the squats. It does not measure depth, anatomical 3D
joint angles, force, load, balance, pain, or cross-camera agreement. It is not
equivalent to the target dual-view Pi architecture or to clinical motion
capture.

Perspective, camera placement, self-occlusion, clothing, lighting, and
out-of-frame landmarks can affect projected angles. Low visibility, low
presence, stale or missing frames, invalid geometry, and large bilateral
projected-knee disagreement are non-assessable. They cannot advance a rep and
cannot be converted into an inference of safe movement.

The laptop runtime requires an explicit, integrity-checked install of the
pinned MediaPipe Pose Landmarker Lite file before camera access. Python package
import never downloads that model. The exact native pair is
`mediapipe==0.10.35` with
`opencv-contrib-python==4.14.0.94`; installing another OpenCV wheel beside it is
outside the tested dependency contract.

The workout configures and reuses one Realtime socket/session. A cue is an
isolated response on that same socket, while normal active-exercise speech
remains blocked. The semantic end boundary is physical stop or locally
validated `finish_session` only. OpenAI documents a 60-minute maximum Realtime
session, so this unrotated hackathon composition is limited to a shorter
workout. Completing the third rep and issuing the T-shape instruction leaves
that session open. Connection rotation has not been implemented or accepted.

## Privacy invariants

- Pose extraction occurs on the device. Raw frames are short-lived and cannot
  be represented by the message sent to the fusion process.
- The sanitized feature schema has no image, audio, transcript, name, address,
  account, or patient-identifier field.
- Flower receives model arrays and numeric metrics derived from local feature
  windows. Secure aggregation protects individual updates within the protocol;
  it does not make a three-client demo anonymous or immune to poisoning.
- The active laptop workout sends nothing to Flower. Any future squat
  federation input must be produced after session end by a new allowlisted
  numeric adapter. The current `seated-knee-extension` FAB must reject `squat`
  records as schema-incompatible.
- Transcript retention is disabled in the hackathon profile. The completed cue
  transcript is held only long enough to compare it with the fixed prompt
  phrase, then discarded. Realtime service processing is still a cloud data
  flow and must be disclosed.
- Logs contain event types, reason codes, durations, and anonymous device/run
  IDs only. Secrets and user utterances must never be logged.

## Threats deliberately left for later

- Production device identity, mutual TLS, key rotation, and fleet revocation.
- Production integrity protection for executable configuration and verified
  boot.
- Poisoning, sybil, and inference attacks across a real federation.
- Clinical study design, accessibility validation, incident response, and
  medical-device regulatory work.
- A production retention policy and clinician access controls.

## Demo release checklist

- Run the complete unit suite.
- Run the three-client Flower round and preserve aggregate-only evidence.
- Replay camera loss, disagreement, low-confidence, pain, and button-stop cases.
- Inspect an emitted feature record and prove that it contains numeric derived
  features only.
- Disconnect the network during exercise and prove local pause/stop still work.
- Confirm cue speech becomes unavailable after that disconnect while local
  pose processing and Guardian decisions continue.
- Confirm the active-exercise speaker gate discards quarantined PCM for a
  missing, partial, or mismatched completed transcript and releases it only for
  the exact Guardian-selected prompt phrase.
- Confirm a canceled response arriving late cannot open the following turn's
  speaker authorization.
- Confirm three completed squats and the final T-shape cue do not close the
  shared Realtime session, while physical stop and one locally validated empty
  `finish_session` call each do.
- Confirm the live workout stays below the documented 60-minute session cap.
- Do not show a Flower round as part of the live camera loop; snapshot only
  sanitized numeric data after session end and preserve the current exercise
  schema rejection.
- Show the medical-prototype disclaimer before the demo begins.

## Current Realtime acceptance evidence

The named verifier simulates all five closed stages—introduction, first
assessable standing detection, and completed reps one through three—through the
real Guardian, then attempts five prompt cues on one live Realtime connection:

```bash
uv run recoverybox verify-realtime-cues \
  --output-dir artifacts/realtime-verification \
  --voice marin \
  --asr-model whisper-1
```

On 2026-08-25 the current five-cue catalog passed 5/5 live completed-transcript
gates on one `gpt-realtime-2.1` connection. The run used `--skip-asr`, so no
independent ASR claim is made. The local ignored report omits provider response
IDs, transcripts, prompt text, ASR word text, and credentials; it retains only
content-free event names, match/timing evidence, and paths to audio that already
passed the exact-transcript gate. This accepts the simulated Realtime cue/gate
lane, not webcam pose accuracy, actual speaker onset, or complete
camera-to-speaker latency.
