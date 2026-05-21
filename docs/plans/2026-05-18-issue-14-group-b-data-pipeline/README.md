# Group B — Data Pipeline Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.
> **Parent Issue:** [#14 — PRD: Prosody & Timbre Disambiguation Training Dataset](https://github.com/hungphongtrn/Amy-LM/issues/14)

## Quick Status
- **Current Phase:** Phase 1 — Enrichment + Processor
- **Next Up:** Phase 2 — DeepSeek Pair Generation
- **Overall Progress:** 0/3 phases complete
- **Prerequisite Issues:** #17 (NV Tag Emoji Mapping), #18 (Speaker Lookup Cache), #19 (AmyLM Model + Config)

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [phase-01-enrichment-and-encoding.md](./phase-01-enrichment-and-encoding.md) — Current phase tasks (15 min)
3. [decisions.md](./decisions.md) — Context on choices made (optional, 5 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Issues | Document |
|-------|--------|---------|--------|----------|
| 1 — Enrichment + Processor | Not Started | Enriched NVTTS + PreferenceDatasetProcessor | #20, #21 | [phase-01](./phase-01-enrichment-and-encoding.md) |
| 2 — DeepSeek Generation | Pending | JSONL with chosen/rejected pairs | #22 | [phase-02](./phase-02-pair-generation.md) |
| 3 — Embedding + Filter | Pending | Intermediate parquet ready for FACodec | #23 | [phase-03](./phase-03-embedding-and-filter.md) |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.
