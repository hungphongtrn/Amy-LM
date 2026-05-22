# DPO Collator + Training Script — Strategy

## Goal

Build a custom DPOCollator and AmyDPOTrainer (subclass of `trl.DPOTrainer`) that train AmyLM with QLoRA on NVTTS-FACodec preference pairs (`hungphongtrn/nvtts_facodec`), runnable on a single RTX 3060 12GB.

## Architecture

Three components, each in its own file:

1. **DPOCollator** (`src/training/dpo_collator.py`) — Custom HF data collator that:
   - Extracts mel spectrograms from raw waveforms via `MossAudioProcessor._extract_mel()`
   - Tokenizes system prompt + `<audio>` placeholder + chosen/rejected responses per sample
   - Bridges the gap between raw NVTTS-FACodec rows (with waveform arrays, FACodec codebook indices, text fields) and the concatenated batch format expected by TRL's `DPOTrainer`
   - Duplicates audio/prosody/timbre fields along the batch dimension (B → 2×B) since chosen and rejected share the same speech input

2. **AmyDPOTrainer** (`src/training/amy_dpo_trainer.py`) — Minimal subclass of `trl.DPOTrainer`:
   - Passes extra AmyLM kwargs (`audio_data`, `audio_data_seqlens`, `audio_input_mask`, `prosody_indices`, `timbre_vector`) through to `model()` via standard TRL model_kwargs passthrough — no `_compute_loss` override needed
   - Configuration: `precompute_ref_log_probs=True` by default
   - Handles reference model lambda-zeroing: precomputation runs at init when λ=0 (functionally equivalent to MOSS-Audio), so no reference model is kept in VRAM during training
   - Logs `lambda_p` and `lambda_t` alongside TRL's built-in DPO metrics

3. **Training script** (`scripts/train_amy_dpo.py`) — CLI entry point that:
   - Loads the NVTTS-FACodec dataset, filters by configurable cosine threshold
   - Initializes `AmyLM` with 4-bit quantization
   - Applies QLoRA adapters via PEFT config
   - Instantiates `DPOCollator` + `AmyDPOTrainer`
   - Runs training with W&B logging and HF-style checkpointing

## Key Design Insight: No `_compute_loss` Override

The current TRL `DPOTrainer._compute_loss` (post-refactor, 2026) builds `model_kwargs` by filtering out only three known keys (`completion_mask`, `ref_chosen_logps`, `ref_rejected_logps`). Everything else passes through to `model(**model_kwargs)`. Our DPOCollator outputs audio fields within the batch dict, so they automatically reach `AmyLM.forward()` via standard TRL passthrough. **No override of `_compute_loss` or `concatenated_forward` is needed.**

Similarly, reference log-prob precomputation runs the same `model(**model_kwargs)` with the same collator batches. Since λ starts at 0, the precomputed ref log-probs are correct without any lambda-zeroing trick.

## VRAM Budget (3060 12GB)

| Component | Size (approx) |
|-----------|---------------|
| 4-bit MOSS-Audio backbone | ~4GB |
| LoRA adapters (all linear layers) | ~200MB |
| FACodec modules (fp32) | ~12MB |
| Reference model | 0 (precomputed) |
| Activations (bf16, B=2, S=1024) | ~2GB |
| Optimizer states | ~1GB |
| **Total headroom** | ~4.7GB free |

## Phases

### Phase 1: Dependencies + DPOCollator
**Outcome:** DPOCollator unit tests pass — produces correctly-shaped concatenated batches with all AmyLM-specific fields.
**Rough scope:** Add trl/peft to pyproject.toml, implement DPOCollator class with mel extraction, tokenization, padding, batch-doubling. Unit tests with mock FACodec data.

### Phase 2: AmyDPOTrainer
**Outcome:** AmyDPOTrainer logs `rewards/margins`, `lambda_{p,t}`, entropy on mock preference data.
**Rough scope:** Subclass DPOTrainer with `precompute_ref_log_probs=True`, lambda logging, QLoRA config. Unit test verifying: (a) model_kwargs passthrough works, (b) trainable params are FACodec + LoRA only.

### Phase 3: Training Script
**Outcome:** `python scripts/train_amy_dpo.py` starts a training run on real NVTTS-FACodec data, logs to W&B.
**Rough scope:** CL arguments (model path, dataset, cosine threshold, hyperparameters), dataset loading + filtering, model init + QLoRA config, trainer instantiation, checkpointing.

### Phase 4: Integration Test
**Outcome:** End-to-end test: one training step on 4 mock preference pairs, verify gradient flow reaches `prosody_embedding.weight` and `residual_fusion.lambda_p`.
**Rough scope:** Integration test fixture with tiny AmyLM, synthetic data, verify nonzero grad in FACodec params after step.

## Open Questions

(none — all resolved in grilling session)
