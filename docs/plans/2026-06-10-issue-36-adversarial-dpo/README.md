# Issue #36: Fix DPO Gradient Starvation — Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 1 — Adversarial Pair Generation Pipeline (code complete, pending run)
- **Next Up:** Phase 2 — Dataset Encoding + Lambda Hooks (pending Phase 1 execution on training machine)
- **Overall Progress:** 1/3 phases code-complete (0/3 phases executed)

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [phase-01-generation-pipeline.md](./phase-01-generation-pipeline.md) — Current detailed phase (15 min)
3. [decisions.md](./decisions.md) — Context on choices made (optional, 5 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 — Generation | Code Complete | 1,000+ adversarial pairs (JSONL) | [phase-01-generation-pipeline.md](./phase-01-generation-pipeline.md) |
| 2 — Encoding + Hooks | Pending | Dataset on HF Hub + λ gradient logging | Stub only |
| 3 — Training + Gate | Pending | Run B complete, λ movement evaluated | Stub only |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.

## Related
- Issue: [#36](https://github.com/hungphongtrn/Amy-LM/issues/36)
- Parent: [#27](https://github.com/hungphongtrn/Amy-LM/issues/27) — Run A (baseline DPO)
- Contingency: [#37](https://github.com/hungphongtrn/Amy-LM/issues/37) — Intermediate prosody losses
- CONTEXT.md — Updated with Adversarial Preference Pair, Inverse Emotion, NVTTS-FACodec Adversarial Dataset
