# Group A Implementation — Strategy

## Goal
Build three foundational modules for the Issue #14 preference pair dataset pipeline: NV tag mapping (#17), speaker context lookup (#18), and the AmyLM HF model (#19).

## Architecture

```
Group A Foundation Layer
├── src/data/nv_tag_mapping.py          # Issue #17: emoji ↔ [Tag] mapping
├── src/data/speaker_cache.py           # Issue #18: cascading speaker lookup
├── data/speaker_lookup.json            # Issue #18: cached speaker metadata
├── src/models/amy_lm.py                # Issue #19: AmyLMConfig + AmyLM
└── src/models/amy_lm_config.py         # Issue #19: AmyLMConfig (may inline in amy_lm.py)
```

Issues #17 and #18 are pure data utilities — no GPU, no model dependencies. They build the dataset enrichment layer that feeds into the preference pair pipeline.

Issue #19 creates the training-ready HF model that consumes those enriched datasets during DPO.

## Tech Stack
- Python 3.10+, PyTorch, HuggingFace Transformers
- `datasets` library (HF) for NVTTS and VoxCeleb metadata
- `MossAudioModel` base class from `vendor/MOSS-Audio/src/modeling_moss_audio.py`
- Existing modules: `ProsodyEmbedding`, `TimbreProjection`, `TemporalPool`, `ResidualFusion`

## Constraints & Assumptions
- Tests run on CPU, reasonably fast. No GPU required for Group A.
- `MossAudioModel` must be importable (vendor path setup exists in `src/models/moss_audio.py`)
- FACodec modules exist in `src/models/embedding.py`, `src/models/fusion.py`, `src/models/pooling.py`
- Warm-start projector must become trainable (currently frozen at `src/models/embedding.py:48`)
- Speaker cache downloads from HF — needs network access for full build, but tests work offline with mocks

## Phases (High-Level)

### Phase 1: NV Tag Emoji Mapping — Issue #17
**Outcome:** `nv_tag_mapping.py` module with bidirectional emoji↔text mapping and transform function. All 10 NV types covered.
**Rough scope:** ~50 lines of code + test file. Pure dict + regex transform.

### Phase 2: Speaker Context Lookup Cache — Issue #18
**Outcome:** `speaker_cache.py` that downloads 3 VoxCeleb sources + Expresso, merges with cascading fallback, exports `data/speaker_lookup.json`.
**Rough scope:** ~150 lines of code + test file. HF datasets + CSV parsing + JSON cache.

### Phase 3: AmyLM Model + Config — Issue #19
**Outcome:** `AmyLMConfig` extending `MossAudioConfig` + `AmyLM` inheriting `MossAudioModel`. Enriches audio embeddings with prosody/timbre before masking. Trainable: projector, timbre projection, λ gates.
**Rough scope:** ~200 lines of model code + test file. PyTorch + HF integration.

## Open Questions
- None remaining — all design decisions resolved during the grilling session.
