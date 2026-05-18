# Group A Implementation Plan — Issue #14 Dataset Foundations

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 1 - NV Tag Emoji Mapping
- **Next Up:** Phase 2 - Speaker Context Lookup Cache (independent, can run parallel)
- **Overall Progress:** 0/3 phases complete

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) - Understand the big picture (5 min)
2. [Current phase document] - Only the phase you're implementing (10 min)
3. [decisions.md](./decisions.md) - Context on choices made (optional, 3 min)

**Do NOT read future phases.** They're stubbed and may change based on earlier learnings.

## Phase Overview

| Phase | Issue | Status | Outcome | Document |
|-------|-------|--------|---------|----------|
| 1 - NV Tags | #17 | 🔲 Not Started | Emoji → `[Tag]` mapping + transform | [phase-01-nv-tags.md](./phase-01-nv-tags.md) |
| 2 - Speaker Cache | #18 | 🔲 Not Started | Cascading-fallback speaker JSON | [phase-02-speaker-cache.md](./phase-02-speaker-cache.md) |
| 3 - AmyLM Model | #19 | 🔲 Not Started | AmyLM HF model inheriting MossAudioModel | [phase-03-amylm-model.md](./phase-03-amylm-model.md) |

## Source Issues
- Parent: [#14](https://github.com/hungphongtrn/Amy-LM/issues/14) — PRD: Prosody & Timbre Disambiguation Training Dataset
- Phase 1: [#17](https://github.com/hungphongtrn/Amy-LM/issues/17) — NV Tag Emoji Mapping
- Phase 2: [#18](https://github.com/hungphongtrn/Amy-LM/issues/18) — Speaker Context Lookup Cache
- Phase 3: [#19](https://github.com/hungphongtrn/Amy-LM/issues/19) — AmyLM Model + Config

## Test Commands
```bash
uv run python -m pytest tests/ -x -q           # fast tests, skip GPU-heavy
uv run python -m pytest tests/models/ -x -q     # model-specific tests
```

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.
