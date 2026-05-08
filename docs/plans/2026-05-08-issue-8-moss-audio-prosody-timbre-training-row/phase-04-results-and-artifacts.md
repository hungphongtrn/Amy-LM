# Phase 4: Results And Artifacts

## Phase Goal

Document whether the issue #8 pilot supports, weakens, or does not move the Amy LM prosody/timbre hypothesis, and optionally preserve valuable artifacts.

## Current Status

Stub only. Detail this phase after Phase 3 produces real metrics and checkpoints.

## Expected Scope

- Add a result record with commands, data path, split protocol, baseline metrics, Amy metrics, and lambda trajectories.
- Interpret the run as hypothesis-supporting, neutral, or null; a null result is acceptable.
- Optionally evaluate the trained Amy checkpoint once with learned lambdas and once with `lambda_p=lambda_t=0` as a diagnostic.
- Push checkpoints/weights to HF Hub only if credentials are available and the run artifact is worth preserving.

## Completion Criteria

- Results are documented in a durable project location.
- Issue #8 acceptance criteria are checked against concrete evidence.
- HF Hub artifact status is explicit: pushed, skipped due credentials, or skipped because the artifact was not worth preserving.
