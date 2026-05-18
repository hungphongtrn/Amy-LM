# Group B — Data Pipeline Implementation Plan (v2)

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.
> **Parent Issue:** [#14 — PRD: Prosody & Timbre Disambiguation Training Dataset](https://github.com/hungphongtrn/Amy-LM/issues/14)
>
> **v2 note:** Re-written 2026-05-19 after Group A completion. Old disposable plan at `docs/plans/2026-05-18-issue-14-group-b-data-pipeline/` (kept for reference only; APIs outdated).

## Quick Status
| Field | Value |
|---|---|
| **Current Phase** | Phase 1 — Enrichment + Processor |
| **Next Up** | Phase 2 — DeepSeek Pair Generation |
| **Overall Progress** | 0/3 phases complete |
| **Prerequisite Issues** | **#17, #18, #19 — COMPLETE** (commit `84e15e7`) |

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [decisions.md](./decisions.md) — Context on choices made (5 min)
3. [phase-01-enrichment-and-encoding.md](./phase-01-enrichment-and-encoding.md) — Current phase tasks (15 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Issues | Document |
|-------|--------|---------|--------|----------|
| 1 — Enrichment + Processor | Not Started | Enriched NVTTS + PreferenceDatasetProcessor | #20, #21 | [phase-01](./phase-01-enrichment-and-encoding.md) |
| 2 — DeepSeek Generation | Stubbed | JSONL with chosen/rejected pairs | #22 | [phase-02](./phase-02-pair-generation.md) |
| 3 — Embedding + Filter | Stubbed | Intermediate parquet ready for FACodec | #23 | [phase-03](./phase-03-embedding-and-filter.md) |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.

## Group A API Reference (what was built)
| Module | Key API | Purpose |
|---|---|---|
| `src/data/nv_tag_mapping` | `emojis_to_tags(text: str) -> str` | Emoji→[Tag] conversion |
| `src/data/speaker_cache` | `SpeakerCache` class, `resolve_age_context(speaker)` | Speaker context lookup |
| `src/models/amy_lm` | `AmyLMConfig` (extends `MossAudioConfig`) | Model config schema |
