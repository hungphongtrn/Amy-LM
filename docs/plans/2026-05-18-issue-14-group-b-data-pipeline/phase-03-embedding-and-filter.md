# Phase 3: Embedding + Cosine Filter

> **Status:** STUB — will be detailed after Phase 2 completes.

## Phase Goal
Intermediate parquet with all provenance columns + cosine_similarity column, ready to feed into PreferenceDatasetProcessor.

## Files to Touch (Tentative)
- `scripts/filter_pairs.py`
- `tests/data/test_filter_pairs.py`

## High-Level Scope
- Load JSONL from `data/nvtts_pairs/pairs.jsonl`
- Embed chosen and rejected responses with `google/embeddinggemma-300m`
- Compute pairwise cosine similarity
- Output intermediate parquet: `data/nvtts_preference_pairs/nvtts_pairs.parquet`
- All samples retained (no hard filter — threshold applied manually)
- Output schema matches PreferenceDatasetProcessor input expectations

**Issue:** #23 (B3)
**Depends on:** Phase 2 (#22)
