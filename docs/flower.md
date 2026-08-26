# Flower development and deployment

The root Flower App is the federated-training FAB. It contains a ServerApp and
a ClientApp and targets Flower 1.34.0. The always-on device loop is a separate
process. The clinician AgentApp is a second FAB under
`clinician_agent/clinician-agent`.

The implemented laptop squat workout follows the same hard boundary: no
Flower process owns or receives from its webcam, MediaPipe landmarker,
microphone, speaker, Realtime socket, Guardian, or repetition loop. Flower may
start only after the workout has ended, from an immutable snapshot of
allowlisted numeric features.

The current FAB is not a squat FAB. Its exact signed contract is
`exercise_id = "seated-knee-extension"`; the loader must reject `squat` rather
than coerce it into that model. Federating laptop squat summaries requires a
new post-session adapter, feature schema, label definition, model signature,
validation set, and promotion gate.

## Local three-client SecAgg+ run

Install dependencies and generate deterministic synthetic data:

```bash
uv sync --extra dev
uv run recoverybox seed-flower-demo --output-dir data/demo --rows-per-client 24
```

Configure the local simulation federation and submit the root FAB:

```bash
uv run flwr federation simulation-config --num-supernodes 3
uv run flwr run . local --stream
```

A successful run reports three sampled clients, three results, zero failures,
and `Secure aggregation completed` in each of the three rounds.

The simulation ClientApps obtain Flower's numeric `partition-id` and resolve
`data/demo/client-{partition_id}.jsonl`. For a real SuperNode, configure the
exact device-local path instead.

## Feature stages

There are two deliberately different schemas:

1. `recoverybox.pose.v1` is the short-lived numeric window emitted by the edge
   pose pipeline. It contains joint angles, velocity, confidence, and camera
   disagreement with no absolute time or identity.
2. `rehab-quality-features/v1` is the model-ready post-session record consumed
   by Flower. The explicit pose adapter combines the window with bounded,
   derived repetition duration, stability, symmetry, range progress, and a
   review label.

The adapter validates every scalar and never creates values that were not
supplied or derived. The Flower loader applies a closed allowlist recursively;
raw-media and transcript keys are rejected even when nested.

These two implemented schemas describe the seated-knee-extension Flower demo,
not the laptop squat tracker. During a squat session, derived pose state remains
local and continuous. A future adapter may snapshot only sanitized numeric
post-session summaries after the session-end boundary; raw frames, MediaPipe
landmarks, audio, transcripts, fixed cue phrases, Realtime timing payloads, and
session-control tool calls remain forbidden.

Every model-ready JSONL row also carries:

- `exercise_id = "seated-knee-extension"`;
- `label_definition_version = "seated-knee-extension-rep-quality/v1"`;
- `model_schema_signature`, a digest over the exercise, feature normalization,
  label meanings, model family, and parameter shapes.

Label `0` means `rep-needs-coaching-review`; label `1` means
`rep-meets-prescribed-form-criteria`. These are quality-review labels, not
safety decisions. The same identity fields are checked in every Flower round,
and each client rejects a training weight above the configured SecAgg+ bound.

## Trusted-network hackathon deployment

These commands use unencrypted transport and are appropriate only on an
isolated, trusted demo network.

Run them after the interactive session has closed. Starting SuperNode or
ClientApp work during `recoverybox squat-demo` would violate the runtime
boundary even on a laptop where all processes happen to share one machine.

Start the coordinator:

```bash
flower-superlink --insecure
```

Start one SuperNode on each of three devices, changing only the local data path:

```bash
flower-supernode \
  --insecure \
  --superlink COORDINATOR_IP:9092 \
  --node-config 'sanitized-feature-path="/var/lib/recoverybox/features.jsonl"'
```

Configure a `hackathon` SuperLink connection for `COORDINATOR_IP:9093`, then
submit from the coordinator machine:

```bash
flwr run . hackathon --stream
```

For anything outside an isolated demo, follow Flower's deployment guide for
TLS and SuperNode authentication. SecAgg+ protects the update aggregate; it
does not replace encrypted transport, device authentication, input validation,
poisoning defenses, cohort-size analysis, or model governance.

## SecAgg+ compatibility choice

Flower's current official SecAgg+ example uses the legacy strategy bridge:

- `NumPyClient` plus `secaggplus_mod` on each ClientApp;
- `LegacyContext`, `DefaultWorkflow`, and `SecAggPlusWorkflow` on the ServerApp.

The application follows that documented lane rather than mixing it with the
new Message API. Model/data logic is independently testable and can migrate
when Flower publishes a stable Message-API SecAgg+ path.

## Clinician AgentApp

Build and test the second FAB independently:

```bash
cd clinician_agent/clinician-agent
uv sync --extra dev
uv run pytest
uv run flwr build
uv run recoverybox-clinician-review \
  --sessions examples/session_summaries.json
```

The local command runs deterministic review logic without credentials. The
AgentApp emits that same validated Markdown directly and has no language-model
presentation layer. Flower Agent is experimental, so upgrades must re-run its
build and contract tests.

## References

- [Flower deployment runtime](https://flower.ai/docs/framework/1.34/en/how-to-run-flower-with-deployment-engine.html)
- [Flower secure aggregation example](https://flower.ai/docs/examples/flower-secure-aggregation.html)
- [Flower AgentApp runtime](https://flower.ai/docs/agent/explanations/agentapp-runtime.html)
