# Fix DPO Gradient Starvation — Strategy

## Goal
Force AmyLM to use audio streams for DPO preference learning by generating **adversarial preference pairs** whose chosen/rejected are text-indistinguishable, fixing the flat λ gradient starvation observed in Run A (literal pairs, #27).

## Architecture

Three components, none requiring changes to core model code:

1. **Pair Generator** (`scripts/generate_adversarial_pairs.py`): Single DeepSeek V4 Flash call returns `{strategy, chosen, rejected}` per sample, then two-phase quality filters discard low-quality pairs. JSONL output with resume checkpointing.

2. **Dataset Processor** (extend `src/preprocessing/preference_dataset_processor.py`): FACodec-encode adversarial pairs into the same schema as literal pairs (prosody + timbre streams only). Push to `hungphongtrn/nvtts_facodec_adversarial` on HF Hub.

3. **DPO Trainer Enhancement** (`src/training/amy_dpo_trainer.py`): Port `_register_lambda_grad_hooks()` pattern from `AmyTrainer` so λ gradient values are logged alongside λ magnitude at each logging step.

No changes to: `AmyMossLM.forward()`, `DPOCollator`, `DPOTrainingConfig`, `train_amy_dpo.py` (except dataset argument).

## Tech Stack
- DeepSeek V4 Flash (via `openai` `AsyncOpenAI`) — single-call pair generation
- `google/embeddinggemma-300m` (via `sentence-transformers`) — embedding similarity gate
- `nltk` — BLEU-1 lexical overlap
- FACodec encoder (existing `src/preprocessing/facodec_encoder.py`)
- HuggingFace `datasets` — dataset creation and push

## Constraints & Assumptions
- Source data: NVTTS concat all splits (4,046 samples)
- Target: 1,000 adversarial pairs, split 800/100/100 (train/dev/test)
- Inverse emotion mapping: `happy↔sad`, `angry→neutral`, `disgusted→neutral`, `fearful→sad`, `surprised→happy`, `neutral→sad`. `other` and `disgusted` are skipped from mapping but still generate pairs
- Retry: up to 3 generation attempts per sample with temperature jitter
- Quality filters are executed client-side (GPU machine), not in API
- JSON schema enforces structure; `strategy` field discarded after validation

## Phases (High-Level)

### Phase 1: Adversarial Pair Generation Pipeline
**Outcome:** 1,000+ high-quality adversarial pairs passing both filter phases, saved to JSONL with resume support.
**Rough scope:** Create `generate_adversarial_pairs.py` with DeepSeek API integration, two-phase quality filters, temperature-jittered retry, and JSONL checkpointing. Includes minimal local tests for filter functions and inverse emotion mapping.

### Phase 2: Dataset Encoding + Lambda Hooks
**Outcome:** `nvtts_facodec_adversarial` dataset on HF Hub with FACodec-encoded prosody+timbre streams. AmyDPOTrainer logs λ gradients. 50-sample text-blind sanity check passes.
**Rough scope:** Extend PreferenceDatasetProcessor for adversarial schema. Port lambda gradient hooks. Run encoding. Push to HF. Run text-blind sanity check.
**Depends on:** Phase 1

### Phase 3: Run B Training + Decision Gate
**Outcome:** 3-epoch DPO training on adversarial pairs complete. λ values and gradients logged. Decision gate evaluated.
**Rough scope:** Run `train_amy_dpo.py` against adversarial dataset. Monitor λ movement. Post final values. Apply decision gate.
**Depends on:** Phase 2

## Open Questions
1. What sentence-transformer model exactly? Plan uses `google/embeddinggemma-300m` (300M params). Verify this is the intended model ID — the CONEXT.md says "embeddinggemma-300m" but this may need exact HuggingFace model ID confirmation.
2. Is the DeepSeek API key already available in `.env`? The existing `generate_pairs_deepseek.py` loads from `DEEPSEEK_API_KEY` via `dotenv`.
3. The generator prompt passes no speaker context — confirmed in issue body. Confirm this is intentional and sufficient.
