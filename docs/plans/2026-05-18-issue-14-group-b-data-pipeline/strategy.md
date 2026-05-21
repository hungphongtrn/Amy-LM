# Group B — Data Pipeline Strategy

## Goal
Build the full preference-pair data pipeline: enrich NVTTS with speaker context and NV tag mapping, generate chosen/rejected response pairs via DeepSeek V4 Flash, filter by cosine similarity, and encode through the existing FACodec preprocessing infrastructure into a final parquet ready for DPO training.

## Architecture

Three sequential pipeline scripts + one parallel processor:

```
NVTTS raw (deepvk/NonverbalTTS, 4045 samples)
  │
  ├─[Script 1: Enrichment]─► Enriched HF dataset
  │   └─ NV emoji→text mapping (#17)
  │   └─ Speaker context lookup (#18)
  │   └─ Output: data/nvtts_enriched/
  │
  ├─[Script 2: DeepSeek API]─► JSONL with chosen/rejected
  │   └─ Async batched DeepSeek V4 Flash calls
  │   └─ Good prompt (rich context) vs Bad prompt (stripped)
  │   └─ Checkpointed JSONL output
  │   └─ Output: data/nvtts_pairs/pairs.jsonl
  │
  ├─[Script 3: Embedding + Filter]─► Intermediate parquet
  │   └─ google/embeddinggemma-300m embeddings
  │   └─ Cosine similarity computed, stored (not hard-filtered)
  │   └─ Output: data/nvtts_preference_pairs/
  │
  └─[PreferenceDatasetProcessor]─► Final parquet (parallel to Scripts 2-3)
      └─ FACodec encode: prosody + timbre only (no content/acoustic)
      └─ Reads intermediate parquet from Script 3
      └─ Output: data/processed/nvtts-preference-processed/
```

## Tech Stack
- **HF Datasets** — load/store parquet datasets
- **FACodec (Amphion)** — prosody + timbre encoding in PreferenceDatasetProcessor
- **DeepSeek V4 Flash** — async HTTP API with `response_format={'type': 'json_object'}`
- **google/embeddinggemma-300m** — local text embeddings
- **Python asyncio/aiohttp** — batched API calls with retry/backoff

## Constraints & Assumptions
- NVTTS schema: `audio`, `Emotion`, `Initial text`, `Result`, `speaker_id`, `data_name`, `gender`, `dnsmos`, `duration`
- Script 1 must NOT modify audio — passes through raw
- Script 2 requires DEEPSEEK_API_KEY in environment
- Script 2 checkpointing via append-per-sample JSONL; resume by skipping already-processed IDs
- Script 3 embeds both chosen and rejected (2 calls per sample, ~8K total)
- PreferenceDatasetProcessor waits for Script 3 completion; reads intermediate parquet
- All scripts fail gracefully on missing dependencies with clear error messages
- DeepSeek API: retry 3x with exponential backoff, log failures, skip after max retries

## Phases (High-Level)

### Phase 1: Enrichment + Processor (Parallel Foundation)
**Outcome:** NVTTS enriched with speaker context + NV tags saved to parquet. PreferenceDatasetProcessor class built and tested (waits for Script 3 data to actually run).
**Rough scope:** Script 1 applies NV mapping and speaker lookup. PreferenceDatasetProcessor extends DatasetProcessor with preference pair schema, FACodec prosody+timbre encoding.
**Issues:** #20 (B1), #21 (B4)

### Phase 2: DeepSeek Pair Generation
**Outcome:** All ~4K samples have chosen/rejected response pairs in JSONL, resumable from partial failures.
**Rough scope:** Async batched API calls, checkpointing, JSON output parsing, error handling.
**Depends on:** Phase 1 Script 1 (#20)
**Issue:** #22 (B2)

### Phase 3: Embedding + Cosine Filter
**Outcome:** Intermediate parquet with all provenance columns + cosine_similarity, feedable into PreferenceDatasetProcessor.
**Rough scope:** EmbeddingGemma embedding, pairwise cosine similarity, parquet output with all columns preserved.
**Depends on:** Phase 2 (#22)
**Issue:** #23 (B3)

## Open Questions
- What is the exact DeepSeek V4 Flash rate limit? (determines optimal concurrency)
- How many samples survive cosine filtering at various thresholds? (determines final dataset size; manual decision after Script 3)
- Will FACodec encoding of NVTTS audio succeed without resampling? (NVTTS is 16kHz — should be compatible)
