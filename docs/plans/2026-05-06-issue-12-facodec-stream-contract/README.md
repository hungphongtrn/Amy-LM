# Issue #12: FACodec Stream Contract Spike

> **For agentic workers:** Use subagent-driven-development or executing-plans. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Done
- **Next Up:** Issue #6 / #7 / #8 implementation
- **Overall Progress:** 3/3 phases complete
- **Parent Issue:** [Issue #12](https://github.com/hungphongtrn/Amy-LM/issues/12) — Spike: Correct FACodec stream semantics before MOSS-Audio Residual Extension

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [phase-03-handoff.md](./phase-03-handoff.md) — Only the phase you're implementing (10 min)
3. [decisions.md](./decisions.md) — Context on choices made (optional, 5 min)

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 - Contract Verification | X Complete | Verified FACodec tensor contract against Amphion evidence + local checkpoint run | [phase-01-contract-verification.md](./phase-01-contract-verification.md) |
| 2 - Audit & Corrected Spec | X Complete | #6 preprocessing and #7 module specs published with field migration tables, interface specs, and test audits | [phase-02-audit-and-spec.md](./phase-02-audit-and-spec.md) |
| 3 - Handoff | X Complete | Blocker recommendation published, follow-up issues defined, CONTEXT.md verified | [phase-03-handoff.md](./phase-03-handoff.md) |

## Phase 2 Learnings
- 20+ test assertions in `test_facodec_encoder.py` assume 3-tuple return — all must be updated to 4-tuple
- 4 distinct areas in `test_dataset_processor.py` need schema-related corrections
- `TimbreEmbedding` tests (lines 87-140) must be replaced with `TimbreProjection` + new `AcousticEmbedding`/`ContentEmbedding` test classes
- `ResidualFusion` tests need expansion from single lambda to per-stream lambda gates
- Multi-codebook embedding modules (Acoustic, Content) share the same pattern — consider extracting a shared base class

## Phase 2 Learnings
- 20+ test assertions in `test_facodec_encoder.py` assume 3-tuple return — all must be updated to 4-tuple
- 4 distinct areas in `test_dataset_processor.py` need schema-related corrections
- `TimbreEmbedding` tests (lines 87-140) must be replaced with `TimbreProjection` + new `AcousticEmbedding`/`ContentEmbedding` test classes
- `ResidualFusion` tests need expansion from single lambda to per-stream lambda gates
- Multi-codebook embedding modules (Acoustic, Content) share the same pattern — consider extracting a shared base class

## Phase 3 Learnings
- All 5 acceptance criteria met and checked off in the spike report
- CONTEXT.md verified complete — all 9 canonical terms present and correct
- Issue #8 approved to proceed with conditions (#6 and #7 must land first)
- Follow-up implementation checklists published for #6 (9 items), #7 (8 items), #8 (7 items)
- Spike report with verified contract: `docs/spikes/issue-12-facodec-stream-contract.md`
- Executable inspector for future runs: `scripts/inspect_facodec_contract.py`
- Old single-file plan can be archived: `docs/plans/2026-05-06-issue-12-facodec-stream-contract.md`

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.
