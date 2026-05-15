# Amy LM Pilot Training (Issue #8) — Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 4 — Training & Evaluation (complete)
- **Next Up:** None — all 4 phases complete. Ready for GPU training run.
- **Overall Progress:** 4/4 phases complete

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [Current phase document](./phase-04-training-evaluation.md) — Only the phase you're implementing (30 min)
3. [decisions.md](./decisions.md) — Context on choices made (optional, 5 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 — MOSS-Audio Backbone | ✅ Complete | MOSS-Audio loads, sub-modules extracted, semantic stream verified (11 tests) | [phase-01](./phase-01-moss-audio-backbone.md) |
| 2 — Amy Model Assembly | ✅ Complete | `AmyForProsodyClassification` forward pass working end-to-end (17 tests) | [phase-02](./phase-02-amy-model-assembly.md) |
| 3 — Data Pipeline | ✅ Complete | MUStARD FACodec preprocessed, training Dataset/DataLoader working (147 tests) | [phase-03](./phase-03-data-pipeline.md) |
| 4 — Training & Evaluation | ✅ Complete | Baseline + Amy model trained, metrics reported. 7 commits: BaselineClassifier, AmyTrainer, train_amy CLI | [phase-04](./phase-04-training-evaluation.md) |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.

## Source Issue
[Issue #8: MOSS-Audio Baseline + Prosody/Timbre Training Row](https://github.com/hungphongtrn/Amy-LM/issues/8)
