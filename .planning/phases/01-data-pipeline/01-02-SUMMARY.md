---
phase: 01-data-pipeline
plan: "02"
subsystem: data-pipeline
tags: [neutralization, prosody, tts, jsonl, cli]

# Dependency graph
requires:
  - 01-01-Parse-Data (raw_samples.jsonl canonical dataset)
provides:
  - neutralize.py (lexical neutralization module)
  - prosody_instructions.py (prosody instruction generator)
  - enrich_samples.py (CLI for batch enrichment)
  - enriched_samples.jsonl (1998 enriched samples)
affects:
  - Phase 2: Speech Synthesis (uses enriched_samples.jsonl)
  - Phase 3: Benchmark evaluation (uses prosody instructions)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - stdlib urllib.request for OpenAI API calls (no external dependencies)
    - Emotion keyword matching for prosody style classification
    - CLI with enrich_samples() + main() pattern for orchestration
    - Validation pass over output with --validate flag

key-files:
  created:
    - src/proactive_sat/data_pipeline/neutralize.py - Lexical neutralizer with rule_based + openai modes
    - src/proactive_sat/data_pipeline/prosody_instructions.py - Prosody instruction generator
    - src/proactive_sat/data_pipeline/enrich_samples.py - CLI entrypoint for batch enrichment
    - .data/proactive_sat/enriched_samples.jsonl - 1998 enriched samples (1 skipped - empty source_text)
  modified: []

key-decisions:
  - Used stdlib urllib.request for OpenAI API (no httpx/requests dependency)
  - Emotion keyword matching with frozenset for O(1) lookups
  - Skip samples with empty source_text (can't neutralize nothing)
  - Control/trigger texts equal neutral_text (prosody-only injection)

patterns-established:
  - "Pattern: Dual-mode neutralizer (rule_based default, openai optional)"
  - "Pattern: Emotion-to-prosody mapping with keyword classification"
  - "Pattern: Control/Trigger instruction pair generation for prosodic injection"

# Metrics
duration: 5min
completed: 2026-01-31
---

# Phase 1 Plan 2: Lexical Neutralizer and Prosody Instructions Summary

**Dual-mode lexical neutralizer + prosody instruction generator that enriches 1998 samples with neutral_text and control/trigger speaker instructions**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-31T04:32:00Z
- **Completed:** 2026-01-31T04:37:00Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments

- Implemented `neutralize.py` with rule_based (default) and openai modes
- Implemented `prosody_instructions.py` with emotion-to-style mapping (sarcastic/frustrated/distressed)
- Implemented `enrich_samples.py` CLI with validation support
- Generated `enriched_samples.jsonl` with 1998 samples (1 skipped - empty source_text)
- Each sample includes: neutral_text, prosody_style, control/trigger instructions, control_text, trigger_text

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement lexical neutralizer with safe default behavior** - `5be0192` (feat)
2. **Task 2: Generate prosody instructions + enrich samples JSONL** - `8c5e004` (feat)

**Plan metadata:** `docs(01-02): complete 01-02-Neutralizer plan`

## Files Created/Modified

- `src/proactive_sat/data_pipeline/neutralize.py` - Dual-mode neutralizer (rule_based + openai)
- `src/proactive_sat/data_pipeline/prosody_instructions.py` - Emotion-to-prosody mapping
- `src/proactive_sat/data_pipeline/enrich_samples.py` - Batch enrichment CLI
- `.data/proactive_sat/enriched_samples.jsonl` - 1998 enriched samples (output)

## Decisions Made

- Used stdlib urllib.request for OpenAI API calls (no third-party dependencies)
- Emotion keyword matching using frozenset for efficient O(1) lookups
- Skip samples with empty source_text (can't neutralize nothing)
- Control_text and trigger_text exactly equal neutral_text (prosody-only modification)

## Deviations from Plan

**1. [Rule 3 - Blocking] Handle samples with empty source_text**

- **Found during:** Task 2 validation
- **Issue:** 1 sample in raw_samples.jsonl had empty source_text field
- **Fix:** Skip samples with empty source_text during enrichment (can't neutralize nothing)
- **Files modified:** `src/proactive_sat/data_pipeline/enrich_samples.py`
- **Commit:** `8c5e004`

## Issues Encountered

**None**

## User Setup Required

**OpenAI mode only:** To use `mode='openai'` for neutralization, set:
- `OPENAI_API_KEY` environment variable
- Optional: `PROACTIVE_SAT_OPENAI_MODEL` (defaults to `gpt-5-mini`)

## Next Phase Readiness

- **Ready:** `enriched_samples.jsonl` (1998 samples) for Phase 2 Speech Synthesis
- **Ready:** `neutralize.py` for on-demand neutralization if needed
- **Ready:** `prosody_instructions.py` for TTS instruction generation
- **Note:** Each enriched sample contains neutral_text + control/trigger instruction pairs for prosodic manipulation experiments

---

*Phase: 01-data-pipeline*
*Plan: 01-02*
*Completed: 2026-01-31*
