# Phase 2: DeepSeek Pair Generation

> **STUB** — Will be detailed after Phase 1 completes and learnings are incorporated.
> Do not implement this phase yet.

## Phase Goal

All ~4K NVTTS samples have chosen/rejected response pairs saved to `data/nvtts_pairs/pairs.jsonl`. Resumable from partial failures via per-sample append checkpointing.

**Issue:** #22 (B2)

**Depends on:** Phase 1 Script 1 (#20) — enriched dataset at `data/nvtts_enriched/nvtts_enriched.parquet`

## Planned Approach

- Async batched DeepSeek V4 Flash API calls
- Configurable concurrency, exponential backoff, max 3 retries
- Per-sample JSONL append for checkpointing
- Resume by loading existing JSONL to get completed IDs
- Good prompt: rich speaker context + transcript with tags
- Bad prompt: bare transcript only (stripped of paralinguistic context)
- `response_format={'type': 'json_object'}` for structured output
