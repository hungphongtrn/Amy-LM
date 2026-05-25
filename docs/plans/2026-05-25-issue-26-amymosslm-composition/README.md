# AmyMossLM Composition Refactor — Implementation Plan

> **GitHub Issue:** [#26](https://github.com/hungphongtrn/Amy-LM/issues/26)
> **ADR:** [`docs/adr/0001-amymosslm-composition-over-inheritance.md`](../../adr/0001-amymosslm-composition-over-inheritance.md)
> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 1 — Vendor and Scaffold
- **Next Up:** Phase 2 — Rewrite AmyMossLM (pending Phase 1 completion)
- **Overall Progress:** 0/3 phases complete

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [phase-01-vendor-and-scaffold.md](./phase-01-vendor-and-scaffold.md) — Current phase (15 min)
3. [decisions.md](./decisions.md) — Context on choices made (optional, 5 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 - Vendor and Scaffold | Pending | MossAudio source vendored, old code deleted, exports updated | [phase-01-vendor-and-scaffold.md](./phase-01-vendor-and-scaffold.md) |
| 2 - Rewrite AmyMossLM | Pending | Full composition-based AmyMossLM with bootstrap | [phase-02-rewrite-amymosslm.md](./phase-02-rewrite-amymosslm.md) |
| 3 - Training scripts and tests | Pending | DPO training updated, all tests pass | [phase-03-training-scripts-and-tests.md](./phase-03-training-scripts-and-tests.md) |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.
