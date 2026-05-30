# LoRA on MOSS-Audio backbone for Amy classifier — Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 2 — Trainer PEFT adaptation
- **Next Up:** Phase 3 — CLI integration (pending Phase 2 completion)
- **Overall Progress:** 1/4 phases complete

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [phase-01-lora-wrapping-utility.md](./phase-01-lora-wrapping-utility.md) — Only the phase you're implementing (20 min)
3. [decisions.md](./decisions.md) — Context on choices made (optional, 5 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 — LoRA wrapping utility + init tests | ✅ Complete | `wrap_classifier_with_lora()` creates trainable LoRA model; static init test passes (adapter existence, frozen backbone, trainable FACodec+classifier). Gradient flow test written (GPU blocked). | [phase-01](./phase-01-lora-wrapping-utility.md) |
| 2 — Trainer PEFT adaptation | 🔲 In Progress | save/load roundtrip passes; λ hooks work through PEFT wrapper | [phase-02](./phase-02-trainer-peft-adaptation.md) |
| 3 — CLI integration | 🔲 Pending | `--use-lora` flag produces trainable LoRA-wrapped model from CLI | Stub only |
| 4 — Experiment script | 🔲 Pending | `scripts/train_33_lora_amy.py` launches, trains to convergence | Stub only |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.

## Source Issue
[Issue #33: LoRA on MOSS-Audio backbone for Amy classifier](https://github.com/hungphongtrn/Amy-LM/issues/33)
