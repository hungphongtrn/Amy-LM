# Amy LM Pilot Training — Strategy

## Goal
Implement the first Amy LM pilot: binary sarcasm classification on MUStARD using MOSS-Audio 4B as the frozen semantic backbone and FACodec Prosody + Timbre as residual extensions, with a fair baseline comparison.

## Architecture
MOSS-Audio 4B is loaded via `transformers` and its sub-modules (`audio_encoder`, `audio_adapter`, `language_model`) are extracted. The `AmyForProsodyClassification` model runs raw audio through the MOSS-Audio encoder/adapter to produce Semantic Stream S_t `[B, T, 2560]`. FACodec Prosody Embedding (warm-started from FACodec codebook vectors projected to D=2560) and Timbre Projection provide residual streams P_t and T_t. `ResidualFusion` sums: `LayerNorm(S_t + λ_p·P_t + λ_t·T_t)`, fed as `inputs_embeds` through the frozen Qwen3 language model. Mean-pool over frames + `Linear(2560→2)` classifier produces logits trained with `CrossEntropyLoss`.

## Tech Stack
- **MOSS-Audio 4B**: `transformers>=4.45.0`, model ID `OpenMOSS-Team/MOSS-Audio-4B-Thinking`
- **FACodec**: `vendor/Amphion/models/codec/ns3_codec`, checkpoints from HF `amphion/naturalspeech3_facodec`
- **Existing Amy modules**: `ProsodyEmbedding`, `TimbreProjection`, `TemporalPool`, `ResidualFusion` — all built and tested
- **Data**: `datasets` (HF), preprocessing pipeline in `src/preprocessing/`
- **Training**: Vanilla PyTorch, `wandb` for logging
- **Packaging**: `uv` for Python management, project config in `pyproject.toml`

## Constraints & Assumptions
- MUStARD audio has been downloaded and prepared via `scripts/prepare_mustard_dataset.py` (verified: `data/mustard_dataset/` exists)
- FACodec checkpoints must be downloaded from HuggingFace; not committed to the repo (742MB)
- MOSS-Audio 4B requires ~16GB GPU memory; training with batch_size=1-2 on single GPU
- Preprocessing (FACodec encoding of MUStARD) is a one-time step; training loads precomputed indices, not the FACodec encoder
- `transformers` and `wandb` dependencies must be added to `pyproject.toml`
- Stream Activation Config controls which streams are active (prosody=true, timbre=true; content=false, acoustic=false for this row)
- FACodec ProsodyEmbedding warm-start requires loading raw codebook vectors from the Amphion checkpoint — verify format and extraction before training
- If FACodec codebook vectors are unavailable, implementation must fail clearly; no silent random-init fallback for the main run

## Phases (High-Level)

### Phase 1: MOSS-Audio Backbone Integration — Foundation
**Outcome:** MOSS-Audio loads successfully, sub-modules extracted, semantic stream forward pass verified with shape checks.
**Rough scope:** Add `transformers` dependency, create `MossAudioWrapper` in `src/models/`, load model from HF Hub, extract `audio_encoder`/`audio_adapter`/`language_model` sub-modules, verify `[B, T_audio]` → `[B, T_frames, 2560]` output shape (12.5 Hz).

### Phase 2: Amy Model Assembly — Core Model
**Outcome:** `AmyForProsodyClassification` produces 2-class logits from audio + FACodec prosody indices + timbre vector.
**Rough scope:** Assemble MOSS-Audio wrapper + FACodec embeddings + ResidualFusion + classifier head. Stream activation config. Tests for full forward pass with mock FACodec data.
**Depends on:** Phase 1

### Phase 3: Data Pipeline — Training Data
**Outcome:** MUStARD FACodec-preprocessed parquet + PyTorch `Dataset`/`DataLoader` that yields correct batch dicts.
**Rough scope:** FACodec-encode MUStARD audio, create `MustardDataset` class loading parquet fields, `collate_fn` for variable-length batches, split logic (speaker-independent or random — see Open Questions).
**Depends on:** Phase 1 (MOSS-Audio for audio format verification during preprocessing)

### Phase 4: Training & Evaluation — Run It
**Outcome:** Baseline and Amy model trained, accuracy/F1 reported, W&B logs captured.
**Rough scope:** Training loop (vanilla PyTorch), baseline training (MOSS-Audio + Linear), Amy model training, evaluation on test split, W&B integration, results summary.
**Depends on:** Phase 2, Phase 3

## Open Questions
1. **MUStARD split strategy**: Speaker-independent 5-fold CV (original paper) vs simpler random 80/20? The issue mentions "mean ± std across folds when cross-validation is used" — confirm whether CV is required for this pilot or just a random split.
2. **FACodec codebook vector extraction**: What is the exact checkpoint key path for prosody codebook vectors? `facodec_prosody_codebook = load_facodec_prosody_codebook()` is specified but the actual Amphion state dict layout needs verification.
3. **Batch size**: What fits in GPU memory with MOSS-Audio 4B frozen + trainable embeddings? Start at 1-2, profile.
4. **MOSS-Audio output frame rate**: The Whisper encoder downsamples 200x (12.5 Hz). Verify actual frame count per audio duration matches expectation; if not 80→12.5 Hz alignment needs adjustment.
5. **W&B setup**: Project name, entity, run naming convention?
6. **Device strategy**: Single GPU is assumed. Is multi-GPU needed? DDP would complicate the vanilla PyTorch loop.
