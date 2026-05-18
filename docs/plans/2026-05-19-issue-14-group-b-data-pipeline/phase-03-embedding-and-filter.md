# Phase 3: Embedding + Cosine Filter

> **STUB** — Will be detailed after Phase 2 completes and learnings are incorporated.
> Do not implement this phase yet.

## Phase Goal

Intermediate parquet with all provenance columns + `cosine_similarity` column, feedable into `PreferenceDatasetProcessor.process_dataset()`.

**Issue:** #23 (B3)

**Depends on:** Phase 2 (#22) — pair generation at `data/nvtts_pairs/pairs.jsonl`

## Planned Approach

- Load pairs JSONL, merge with enriched dataset
- Embed chosen and rejected text with google/embeddinggemma-300m (local)
- Compute pairwise cosine similarity
- Store similarity as column (no hard filtering)
- Output parquet to `data/nvtts_preference_pairs/`
- Results feed into PreferenceDatasetProcessor
