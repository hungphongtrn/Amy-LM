# Decision Log — DPO Collator + Trainer

## 2026-05-22: No `_compute_loss` override — audio kwargs passthrough via collator

**Context:** TRL `DPOTrainer._compute_loss` (post-refactor) builds `model_kwargs` as `{k: v for k, v in inputs.items() if k not in _non_model_keys}` where `_non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}`. All other collator keys pass through to `model()`.

**Decision:** Do NOT override `_compute_loss`. Instead, have the custom `DPOCollator` output AmyLM-specific fields (`audio_data`, `audio_data_seqlens`, `audio_input_mask`, `prosody_indices`, `timbre_vector`) directly in the batch dict, already duplicated to `[2×B]` shape for the concatenated batch.

**Rationale:** This is zero-override — the standard TRL DPO flow handles everything. No fragile coupling to TRL internals that could break on library updates.

**Consequences:** The collator takes on more responsibility (mel extraction, tokenization, audio field duplication), but this is well-contained and testable.

## 2026-05-22: `precompute_ref_log_probs=True` avoids lambda-zeroing complexity

**Context:** The reference model should use λ=0 (MOSS-Audio equivalent). With PEFT QLoRA, the standard approach is to disable the LoRA adapter to get the base model. But FACodec modules (including λ gates) live outside the LoRA adapter — if adapter-disable is used for ref forward, the current (trained) λ values would leak in.

**Decision:** Set `precompute_ref_log_probs=True` on `AmyDPOTrainer`. TRL's precomputation runs at `__init__` time (before training starts), when λ=0. During training, the precomputed values are loaded from the dataset. No reference model lives in VRAM.

**Rationale:** Solves the lambda problem without:
- Overriding `_compute_loss` to zero λ temporarily (fragile)
- Loading a separate reference model instance (doubles 4-bit backbone memory)
- Freezing/saving/restoring lambda values in hooks (complex)

**Consequences:**
- Precomputation adds initial overhead (~10 min for 3641 samples at B=1 on 3060)
- Dataset columns include `ref_chosen_logps`, `ref_rejected_logps` (2 float32 per sample, negligible)
- Not compatible with `sync_ref_model=True` (TR-DPO), but we're not using that
- Resume from checkpoint would need re-precompute (or manual handling)

## 2026-05-22: Tokenization in collator (on-the-fly), not dataset pre-processing

**Context:** The NVTTS-FACodec dataset has raw waveform arrays + precomputed FACodec indices/vectors + chosen/rejected text. We could pre-tokenize in a dataset map step, or tokenize in the collator.

**Decision:** Tokenize in the collator. Each `__call__` batch: (1) extract mels from waveforms, (2) tokenize system prompt using `MossAudioProcessor._base_tokenizer`, (3) tokenize chosen/rejected texts.

**Rationale:**
- Dataset size is small (~4000 samples), so per-epoch tokenization cost is negligible
- Avoids separate pre-processing step and intermediate dataset storage
- Collator has access to the processor object for mel extraction anyway
- Simpler data pipeline: raw dataset → cosine filter → collator → trainer

**Consequences:**
- Collator must hold references to the processor and base tokenizer
- Tokenization happens on CPU; not a bottleneck at B=2 on 3060

## 2026-05-22: QLoRA config — LoRA on all Qwen3 linear layers

**Context:** MOSS-Audio uses Qwen3 as its LLM backbone (36 layers with `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj`). We need to freeze the 4-bit backbone and train only LoRA adapters + FACodec modules.

**Decision:** Apply `LoraConfig(target_modules="all-linear", ...)` via `peft.get_peft_model()`. This automatically targets all linear layers in Qwen3. Configurable rank (default 8), alpha (default 16), dropout (0.05).

**Rationale:**
- `"all-linear"` is the standard PEFT approach and covers all 7 linear projection types
- FACodec modules are trained separately in full precision (not quantized, not LoRA'd)
- TRL's `DPOTrainer` with QLoRA auto-handles: bf16 adapter weights, adapter disable for ref

**Consequences:**
- LoRA rank/alpha/dropout are hyperparameters exposed in training config
- Must verify `"all-linear"` does NOT target FACodec modules (they're on the base AmyLM, separate)

## 2026-05-22: File organization under `src/training/`

**Context:** New files needed for DPO implementation.

**Decision:**
- `src/training/dpo_collator.py` — DPOCollator class
- `src/training/amy_dpo_trainer.py` — AmyDPOTrainer class
- `scripts/train_amy_dpo.py` — training entry point
- `tests/training/test_dpo_collator.py` — collator unit tests
- `tests/training/test_amy_dpo_trainer.py` — trainer unit tests
- `tests/training/test_dpo_integration.py` — single-step integration test
