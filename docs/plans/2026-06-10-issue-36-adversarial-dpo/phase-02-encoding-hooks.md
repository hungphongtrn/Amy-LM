# Phase 2: Dataset Encoding + Lambda Hooks

> **STUB — will be detailed after Phase 1 completes.**

## Phase Goal
FACodec-encode the adversarial JSONL pairs, push to `hungphongtrn/nvtts_facodec_adversarial` on HF Hub. Port lambda gradient hooks to `AmyDPOTrainer`. Run 50-sample text-blind sanity check.

## Rough Scope

### Task A: Extend PreferenceDatasetProcessor
- Add `process_adversarial_dataset()` method to `src/preprocessing/preference_dataset_processor.py`
- Maps adversarial pair fields (strategy, judge scores, inverse emotion) to the common FACodec encoding path
- Output schema: same as literal pairs (audio, prosody_codebooks_idx, timbre_vector, chosen, rejected) plus adversarial metadata

### Task B: FACodec Encode + Push
- Load adversarial JSONL
- Run through FACodec encoder (prosody + timbre only)
- Apply NVTTS train/dev/test split (800/100/100)
- Push to `hungphongtrn/nvtts_facodec_adversarial`

### Task C: Lambda Gradient Hooks
- Port `_register_lambda_grad_hooks()` from `AmyTrainer` to `AmyDPOTrainer`
- Use `self.accelerator.unwrap_model()` to reach base model (consistent with existing `log()`)
- Log `lambda_p_grad` / `lambda_t_grad` alongside `lambda_p` / `lambda_t` in `log()`
- Handle empty gradients on first call (no backward yet)

### Task D: Text-Blind Sanity Check
- Select 50 held-out pairs
- Strip all emotion context from prompt
- Present chosen and rejected (shuffled) to DeepSeek V4 Flash
- Accuracy must be ≤ 55% (near chance for binary choice)
- If accuracy > 55%, pairs are still text-distinguishable — regenerate with stricter filters

## Files to Touch
- `src/preprocessing/preference_dataset_processor.py` — Extend
- `src/training/amy_dpo_trainer.py` — Add hooks
- `scripts/encode_adversarial_pairs.py` or reuse existing — Encode + push
- Tests for processor extension and trainer hooks

## Depends on
- Phase 1 (adversarial JSONL)

## Phase Completion Criteria
- [ ] Adversarial dataset on HF Hub with FACodec streams
- [ ] Lambda gradient hooks functional in `AmyDPOTrainer`
- [ ] Text-blind sanity check passes (≤ 55% accuracy)
- [ ] Unit tests pass for new trainer hooks and processor method
