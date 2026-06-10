# Issue #36: Fix DPO Gradient Starvation — Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 3 — Training + Gate (detailed, ready for execution after Phase 2)
- **Overall Progress:** 3/3 phases code-complete, 3/3 plans detailed (0/3 phases executed)

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [phase-01-generation-pipeline.md](./phase-01-generation-pipeline.md) — Adversarial pair generation (15 min)
3. [phase-02-encoding-hooks.md](./phase-02-encoding-hooks.md) — FACodec encoding + λ hooks (10 min)
4. [phase-03-training-gate.md](./phase-03-training-gate.md) — Run B training + decision gate (10 min)

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 — Generation | Code Complete | 1,000+ adversarial pairs (JSONL) | [phase-01-generation-pipeline.md](./phase-01-generation-pipeline.md) |
| 2 — Encoding + Hooks | Code Complete | Dataset on HF Hub + λ gradient logging | [phase-02-encoding-hooks.md](./phase-02-encoding-hooks.md) |
| 3 — Training + Gate | Detailed | Run B complete, λ movement evaluated | [phase-03-training-gate.md](./phase-03-training-gate.md) |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.

## Related
- Issue: [#36](https://github.com/hungphongtrn/Amy-LM/issues/36)
- Parent: [#27](https://github.com/hungphongtrn/Amy-LM/issues/27) — Run A (baseline DPO)
- Contingency: [#37](https://github.com/hungphongtrn/Amy-LM/issues/37) — Intermediate prosody losses
- CONTEXT.md — Updated with Adversarial Preference Pair, Inverse Emotion, NVTTS-FACodec Adversarial Dataset
