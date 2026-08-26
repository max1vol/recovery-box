# RecoveryBox clinician review AgentApp

This is a separate Flower App Bundle (FAB) for asynchronous clinician review. It accepts
a question and a strict JSON array of de-identified session summaries, then produces:

- a deterministic review queue with a source session ID for every reason;
- aggregate completion, movement-quality, and observation-confidence measures;
- first-to-latest trends when a de-identified participant has at least two sessions; and
- explicit evidence limits.

The app is read-only. It has no device connector, write tool, prescription tool, raw-media
input, transcript field, or free-text session note. It does not diagnose, prescribe, perform
emergency triage, or control a RecoveryBox. Its attention levels order a review queue; they
are not clinical urgency.

The validated deterministic Markdown is the only clinician-facing output. The AgentApp does
not call a language model or connector, so an unvalidated paraphrase cannot replace or appear
beside the report. The exact Markdown printed by the run is also returned by the task helper
and persisted in Flower `Context.state` under `clinician-review`.

## Input contract

The AgentApp reads these flattened Flower run configuration values:

- `agent.input`: a bounded clinician question without direct contact details;
- `agent.session-summaries-json`: a JSON array encoded as a string.

Each summary has an exact schema. Unknown fields are rejected. `participant_ref` must be a
non-reversible upstream token shaped like `anon-a13c82`; never put an MRN or other source
identifier in it. Raw images, video, audio, transcripts, names, contact details, and notes are
outside the contract. See `examples/session_summaries.json` for the complete shape.

`agent.model` and `agent.use-model` are deliberately unsupported and rejected if supplied.
Model output cannot be checked before a Flower Agent response is presented, so this prototype
does not use a model as a presentation layer.

## Credential-free local review

From this directory:

```bash
uv sync --extra dev
uv run recoverybox-clinician-review \
  --sessions examples/session_summaries.json
uv run pytest
```

Add `--json` to receive the deterministic structured report. This path does not call a model
or require Flower credentials.

## Build and run as a Flower AgentApp

Validate the AgentApp-only FAB:

```bash
uv run flwr build
```

After authenticating to a Flower federation that has Agent access, submit the included demo
configuration:

```bash
uv run flwr run . supergrid \
  --run-config examples/demo-run-config.toml
```

No model provider or model credential is required. Submit this directory to the Flower
connection; the deterministic report is the sole emitted result.

## Runtime boundary and caveat

Flower currently packages either one `agentapp`, or a `serverapp` plus `clientapp`, in a FAB.
This directory deliberately declares only `agentapp`; the federated training FAB lives
separately. The wiring follows the current synchronous entry point:

```python
from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None: ...
```

Flower Agent is experimental, so its APIs and runtime behavior may change between Flower
releases. This project pins the supported range to Flower 1.34.x and keeps the review engine
independent from the runtime import. Revalidate with `uv run flwr build` when upgrading.

Official references:

- [Write your first AgentApp](https://flower.ai/docs/agent/tutorials/write-your-first-agentapp.html)
- [Understand the AgentApp runtime](https://flower.ai/docs/agent/explanations/agentapp-runtime.html)
- [Run an AgentApp on SuperGrid](https://flower.ai/docs/agent/how-to-guides/run-on-supergrid.html)
