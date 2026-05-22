# DPO Collator + Trainer Implementation Plan

> Issue: [#24](https://github.com/hungphongtrn/Amy-LM/issues/24) — DPO Collator + Training Script
> For agentic workers: Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Overall Progress:** 3/4 phases complete (Phase 4 requires GPU)
- **Test Suite:** 12/12 passing (CPU-only)

## Start Here
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [phase-01-dependencies-and-collator.md](./phase-01-dependencies-and-collator.md) — Current phase (15 min)
3. [decisions.md](./decisions.md) — Context on key choices (optional, 5 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 — Collator | ✅ Complete | DPOCollator + 9 tests | [phase-01-*.md](./phase-01-dependencies-and-collator.md) |
| 2 — Trainer | ✅ Complete | AmyDPOTrainer + 3 tests | [phase-02-*.md](./phase-02-amy-dpo-trainer.md) |
| 3 — Script | ✅ Complete | train_amy_dpo.py | [phase-03-*.md](./phase-03-training-script.md) |
| 4 — Integration | ⏭️ Skipped | GPU-only | [phase-04-*.md](./phase-04-integration-test.md) |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on:
- No `_compute_loss` override — audio kwargs passthrough via collator
- `precompute_ref_log_probs=True` avoids lambda-zeroing complexity
- Tokenization in collator (on-the-fly), not dataset pre-processing
- QLoRA on all Qwen3 linear layers + full-precision FACodec modules
- File organization under `src/training/`
