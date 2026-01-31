---
phase: 01-data-pipeline
plan: "03"
subsystem: data-pipeline
tags: [huggingface, dataset, cli, orchestration, stratified-sampling]

# Dependency graph
requires:
  - 01-01-Parse-Data (raw_samples.jsonl canonical dataset)
  - 01-02-Neutralizer (enriched_samples.jsonl with prosody instructions)
provides:
  - build_hf_dataset.py (HF dataset builder with stratified sampling)
  - run_pipeline.py (One-command Phase 1 orchestrator)
  - .data/proactive_sat/hf_dataset (200-sample DatasetDict)
affects:
  - Phase 2: Speech Synthesis (consumes HF dataset)
  - Phase 3: Benchmark evaluation (uses prosody control/trigger fields)

# Tech tracking
tech-stack:
  added:
    - datasets (HuggingFace datasets library)
  patterns:
    - DatasetDict with "train" split via from_list and save_to_disk
    - Stratified sampling for balanced prosody_style distribution
    - Deterministic selection via random seed for reproducibility
    - CLI orchestration pattern with function imports (not subprocess)

key-files:
  created:
    - src/proactive_sat/data_pipeline/build_hf_dataset.py - HF dataset builder CLI
    - src/proactive_sat/data_pipeline/run_pipeline.py - Phase 1 orchestrator CLI
    - .data/proactive_sat/hf_dataset - 200-sample saved DatasetDict
  modified: []

key-decisions:
  - Used datasets.Dataset.from_list + DatasetDict.save_to_disk for persistence
  - Stratified sampling default: prosody_style for balanced distribution
  - Deterministic sampling: seed=42 produces identical sample_ids across runs
  - Function-level imports for orchestration (no subprocess overhead)

patterns-established:
  - "Pattern: HF dataset builder with stratified sampling CLI"
  - "Pattern: One-command pipeline orchestrator via function imports"
  - "Pattern: Deterministic dataset selection with seed-based reproducibility"

# Metrics
duration: 3min
completed: 2026-01-31
---

# Phase 1 Plan 3: HuggingFace Dataset Builder Summary

**200-sample HuggingFace DatasetDict with Control/Trigger prosody metadata, plus one-command Phase 1 pipeline orchestrator**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-01-31T04:39:00Z
- **Completed:** 2026-01-31T04:41:00Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments

- Implemented `build_hf_dataset.py` CLI with stratified sampling support
- Implemented `run_pipeline.py` orchestrator for end-to-end Phase 1 pipeline
- Generated `.data/proactive_sat/hf_dataset` with exactly 200 samples
- Dataset includes all required fields: neutral_text, prosody_style, control/trigger instructions
- Stratified sampling produces balanced prosody_style distribution (sarcastic, frustrated, distressed)
- Single command produces complete dataset from source files

## Task Commits

Each task was committed atomically:

1. **Task 1: Build 200-sample HuggingFace dataset from enriched JSONL** - `d5b0a07` (feat)
2. **Task 2: Add one-command runner for Phase 1 pipeline** - `a9b8081` (feat)

**Plan metadata:** `docs(01-03): complete 01-03-HF-Dataset plan`

## Files Created/Modified

- `src/proactive_sat/data_pipeline/build_hf_dataset.py` - HF dataset builder CLI with stratified sampling
- `src/proactive_sat/data_pipeline/run_pipeline.py` - Phase 1 orchestrator (parse → enrich → build)
- `.data/proactive_sat/hf_dataset` - 200-sample DatasetDict saved to disk

## Dataset Schema

The HF dataset includes these columns:

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Unique sample identifier |
| `dialog_id` | string | Dialogue identifier |
| `source_text` | string | Original emotional text |
| `neutral_text` | string | Lexically neutralized text |
| `prosody_style` | string | Emotion category (sarcastic/frustrated/distressed) |
| `control_text` | string | Same as neutral_text |
| `control_speaker_instruction` | string | Control instruction for TTS |
| `trigger_text` | string | Same as neutral_text |
| `trigger_speaker_instruction` | string | Trigger instruction for prosodic manipulation |
| `intent` | string | Original annotation (preserved) |
| `emotion` | string | Original annotation (preserved) |
| `speech_act` | string | Original annotation (preserved) |
| `implicature_text` | string | Original annotation (preserved) |
| `confidence` | string | Original annotation (preserved) |

## Decisions Made

- Used `datasets.Dataset.from_list()` + `DatasetDict.save_to_disk()` for persistence
- Stratified sampling defaults to `prosody_style` for balanced emotional distribution
- Seed=42 for deterministic reproducibility (same inputs + same seed = same samples)
- Function-level imports for orchestration (import + call) instead of subprocess

## Deviations from Plan

**None - plan executed exactly as written**

## Issues Encountered

**1. [Rule 3 - Blocking] Fixed list initialization bug**

- **Found during:** Task 1 verification
- **Issue:** `seen_ids: set[str] = []` should be `seen_ids: set[str] = set()`
- **Fix:** Changed list literal to set constructor
- **Files modified:** `src/proactive_sat/data_pipeline/build_hf_dataset.py`
- **Commit:** `d5b0a07`

**2. [Rule 3 - Blocking] Fixed module import vs function import**

- **Found during:** Task 2 verification
- **Issue:** `from proactive_sat.data_pipeline import parse_data` imported module, not function
- **Fix:** Changed to `from proactive_sat.data_pipeline.parse_data import parse_data`
- **Files modified:** `src/proactive_sat/data_pipeline/run_pipeline.py`
- **Commit:** `a9b8081`

## Usage Examples

### Build HF dataset directly

```bash
# Default: 200 samples, seed=42, no stratification
uv run python -m proactive_sat.data_pipeline.build_hf_dataset

# With stratification for balanced prosody_style
uv run python -m proactive_sat.da  ta_pipeline.build_hf_dataset --stratify-by prosody_style

# Custom sample count and seed
uv run python -m proactive_sat.data_pipeline.build_hf_dataset --n 100 --seed 123
```

### Run full Phase 1 pipeline

```bash
# Default: 200 samples, rule_based neutralizer, no stratification
uv run python -m proactive_sat.data_pipeline.run_pipeline

# With all options
uv run python -m proactive_sat.data_pipeline.run_pipeline \
  --n 200 --seed 42 --neutralizer rule_based --stratify-by prosody_style
```

### Load the dataset in Python

```python
from datasets import load_from_disk

ds = load_from_disk(".data/proactive_sat/hf_dataset")
train_split = ds["train"]

print(f"Number of samples: {len(train_split)}")
print(f"Columns: {train_split.column_names}")
print(f"Prosody styles: {set(train_split['prosody_style'])}")
```

## User Setup Required

**None** - No external service configuration required.

## Next Phase Readiness

- **Ready:** `.data/proactive_sat/hf_dataset` (200 samples) for Phase 2 Speech Synthesis
- **Ready:** `build_hf_dataset.py` for dataset regeneration with different seeds/stratification
- **Ready:** `run_pipeline.py` for end-to-end pipeline execution
- **Note:** Dataset includes control/trigger instruction pairs for prosodic manipulation experiments in Phase 2

---

*Phase: 01-data-pipeline*
*Plan: 01-03*
*Completed: 2026-01-31*
