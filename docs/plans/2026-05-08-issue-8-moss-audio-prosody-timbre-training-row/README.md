# MOSS-Audio Prosody/Timbre Training Row Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Start with the current phase; future phase stubs are intentionally incomplete and will be detailed after earlier work lands.

## Quick Status

- **Source issue:** [#8](https://github.com/hungphongtrn/Amy-LM/issues/8)
- **Current Phase:** Phase 1 - MUStARD Dataset Foundation
- **Next Up:** Phase 2 - MOSS-Audio Internal Residual Extension, after Phase 1 validates the data contract
- **Overall Progress:** 0/4 phases complete

## Start Here

New implementer? Read in this order:

1. [strategy.md](./strategy.md) - Understand the big picture and constraints.
2. [phase-01-mustard-dataset.md](./phase-01-mustard-dataset.md) - Implement only the current phase.
3. [decisions.md](./decisions.md) - Review rationale for planning choices.

Do not read future phase stubs as implementation specs. They will be rewritten after Phase 1 reveals the concrete dataset shape and local FACodec checkpoint behavior.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 - MUStARD Dataset Foundation | Not Started | Local MUStARD-to-HF Dataset/parquet path with raw audio, binary labels, FACodec Prosody Stream, and Timbre Vector | [phase-01-mustard-dataset.md](./phase-01-mustard-dataset.md) |
| 2 - MOSS-Audio Internal Residual Extension | Pending | `AmyForProsodyClassification` computes raw-audio Semantic Stream, fuses Prosody/Timbre, and exposes binary logits | [phase-02-model-integration.md](./phase-02-model-integration.md) |
| 3 - Baseline And Amy Training | Pending | Fair frozen-backbone baseline and Amy LM row train/evaluate under the same split protocol | [phase-03-training-baseline.md](./phase-03-training-baseline.md) |
| 4 - Results And Artifacts | Pending | Metrics, lambdas, interpretation, and optional HF Hub artifacts are documented | [phase-04-results-and-artifacts.md](./phase-04-results-and-artifacts.md) |

## Key Decisions

See [decisions.md](./decisions.md) for rationale on major choices.
