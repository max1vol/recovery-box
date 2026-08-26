# Recovery Swarm

Recovery Swarm is a Flower AgentApp that turns two bounded inputs into a draft,
personalized recovery plan for clinician review:

1. a structured clinician diagnosis plus an immutable treatment envelope; and
2. structured user preferences such as availability, session length, equipment,
   coaching style, language, and accessibility needs.

The swarm uses specialized planning, constraint-checking, and review prompts to
organize those inputs. Preferences can shape the schedule and presentation only
inside the supplied treatment envelope. The final result is a draft, not an active
care plan.

## Clinical and device boundary

Recovery Swarm never invents or changes a prescription. It does not diagnose,
activate a plan, decide whether exercise is safe to continue, select exercise cues,
or control the RecoveryBox Guardian or any hardware. The deterministic Guardian
remains the sole authority for real-time pause, stop, escalation, and cue selection.
A clinician must review and approve any draft before it is used.

The plan request uses an exact, allowlisted schema. Unknown fields and free-text
clinical or personal content are rejected. The treatment envelope is treated as
immutable throughout planning; a user preference cannot add an exercise, increase
sets or repetitions, exceed the allowed weekly frequency, remove a contraindication,
or move the review date.

## Public demo privacy rule

The bundled `/demo` request is entirely synthetic. **Only synthetic data may be used
on public Flower infrastructure. Real clinical data, patient identifiers, names,
contact details, notes, transcripts, images, audio, or video are prohibited pending a
documented review of Flower retention, deletion, consent, access-control, and privacy
behavior.**

The clean sample in
[`examples/demo-plan-request.json`](examples/demo-plan-request.json) contains no
patient identity or free text. `synthetic: true` is mandatory for the public demo.

## Fast Flower chat demo

Select **Recovery Swarm** in Flower Agent, then use these exact chat inputs:

1. Enter `/demo` to load the bundled synthetic diagnosis, treatment envelope, and
   preferences. Inspect the draft and its constraint checks.
2. Enter `1` to choose the first Decision Gate route, `2` for the second route, or `3`
   for the third. Enter one number only. Each choice reorganizes the bounded swarm;
   it does not activate or change treatment.
3. Enter `/reset` to discard Recovery Swarm's app-managed demo state and return to the
   initial gate. Then enter `/demo` to begin again.

To demonstrate every route, use this sequence exactly:

```text
/demo
1
/reset
/demo
2
/reset
/demo
3
/reset
```

`examples/demo-run-config.toml` starts a run at `/demo`. The default model is
`openai/gpt-5.5`; model access is provided by the Flower runtime, not by secrets in
this repository.

## Install, inspect, and build

From this directory:

```bash
uv sync --extra dev
uv run recovery-swarm --help
uv run flwr build
```

The FAB is format version 1, targets Flower 1.35.0, declares one AgentApp component,
and includes the Apache `LICENSE`. The package deliberately has no ServerApp,
ClientApp, camera, audio, Guardian, or device component.

After authenticating to a Flower federation with Agent access, run only the synthetic
demo configuration:

```bash
uv run flwr run . supergrid \
  --federation @ACCOUNT/FEDERATION \
  --run-config examples/demo-run-config.toml
```

## Public publication

Flower Hub publication uploads the app's source, and that published source is public.
Review the complete source tree and verify that it contains no secrets, credentials,
private fixtures, or clinical data before publishing.

The manifest declares `publisher = "max1vol"`. Flower requires the publisher value to
match the username of the Flower account used to publish. The GitHub account
[`max1vol`](https://github.com/max1vol) does not by itself prove ownership of the same
Flower namespace. Verify the authenticated Flower username first; if it differs, stop
and update the manifest deliberately rather than publishing under an assumed identity.

Once that identity and the public source have been reviewed, publication is performed
from this directory with Flower's authenticated CLI. This repository does not publish
automatically, and these setup files do not perform a publication.

## Official Flower documentation

- [Write your first AgentApp](https://flower.ai/docs/agent/tutorials/write-your-first-agentapp.html)
- [Understand the AgentApp runtime](https://flower.ai/docs/agent/explanations/agentapp-runtime.html)
- [Run an AgentApp on SuperGrid](https://flower.ai/docs/agent/how-to-guides/run-on-supergrid.html)
- [Flower command-line reference](https://flower.ai/docs/framework/ref-api-cli.html)

Flower Agent APIs are experimental. Revalidate the app boundary and public-data policy
before changing the pinned Flower 1.35.x range.
