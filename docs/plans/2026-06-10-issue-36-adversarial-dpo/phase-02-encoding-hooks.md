# Phase 2: Dataset Encoding + Lambda Hooks

> **Status: Planning complete, implementation in progress.**

## Phase Goal

FACodec-encode the adversarial JSONL pairs, push to `hungphongtrn/nvtts_facodec_adversarial` on HF Hub. Port lambda gradient hooks to `AmyDPOTrainer`. Run 50-sample text-blind sanity check.

## Progress

| Task | Status | Output |
|------|--------|--------|
| A — Extend Processor | In progress | `src/preprocessing/preference_dataset_processor.py` |
| B — Encode + Push | In progress | `scripts/encode_adversarial_pairs.py` |
| C — Lambda Hooks | In progress | `src/training/amy_dpo_trainer.py` |
| D — Text-Blind Check | In progress | `scripts/text_blind_sanity_check.py` |

---

## Task A: Extend PreferenceDatasetProcessor

### Goal
Add `process_adversarial_dataset()` to join adversarial JSONL pairs with NVTTS audio, run through FACodec encoder, and produce an HF Dataset with adversarial metadata.

### Design

**Schema:** `ADVERSARIAL_FEATURES` — superset of `PREFERENCE_FEATURES` plus:

| New Field | Type | Source |
|-----------|------|--------|
| `inverse_emotion` | string | JSONL pair |
| `strategy` | string | JSONL pair |
| `judge_fidelity_chosen` | int32 | JSONL pair |
| `judge_fidelity_rejected` | int32 | JSONL pair |
| `judge_ambiguity_chosen` | int32 | JSONL pair |
| `judge_ambiguity_rejected` | int32 | JSONL pair |
| `generation_attempts` | int32 | JSONL pair |

**`cosine_similarity`** is set to `0.0` (always passes the `< 0.85` filter in `load_and_filter_dataset`). **`rationale_chosen` / `rationale_rejected`** are set to empty strings (not applicable to adversarial pairs).

**Method: `process_adversarial_dataset(jsonl_path, nvtts_dataset, dataset_tag)`**
1. Load JSONL → `pairs_by_id: dict[str, dict]`
2. Filter NVTTS to samples whose `id` has an adversarial pair
3. Batch-encode audio via `self.encoder.encode_batch()`
4. Build entries with `_build_adversarial_entry()`
5. Return `Dataset` with `ADVERSARIAL_FEATURES`

**`_build_adversarial_entry(row, pair, streams, dataset_tag)`** mirrors `_build_processed_entry()` but injects adversarial fields from `pair` and sets literal-only fields to defaults.

### Files
- `src/preprocessing/preference_dataset_processor.py` — Add schema, method, entry builder

### Dependencies
- Phase 1 JSONL output (`data/nvtts_adversarial/pairs.jsonl`)
- NVTTS enriched dataset (for audio + metadata by id)

---

## Task B: FACodec Encode + Push

### Goal
Run the full encoding pipeline: load JSONL → join with NVTTS → FACodec encode → split → push to HF Hub.

### Script: `scripts/encode_adversarial_pairs.py`

**CLI:**
```
uv run python scripts/encode_adversarial_pairs.py \
    --pairs data/nvtts_adversarial/pairs.jsonl \
    --nvtts data/nvtts_enriched/nvtts_enriched.parquet \
    --output hungphongtrn/nvtts_facodec_adversarial \
    --train-size 800 --dev-size 100 --test-size 100 \
    --batch-size 8
```

**Flow:**
1. Load NVTTS enriched parquet (audio + metadata)
2. Load adversarial JSONL pairs
3. Call `processor.process_adversarial_dataset(jsonl_path, nvtts_dataset)`
4. Shuffle with fixed seed (42)
5. Split: first N for train, next M for dev, next K for test
6. Push train/dev/test splits to HF Hub

**Edge cases:**
- If fewer pairs than requested split sizes, use all available (warn)
- If `--output` is a local path, save to parquet instead of pushing
- `--device cpu` for testing without GPU

### Dependencies
- Task A (processor extension)
- HF token in env (`HF_TOKEN`) for push
- FACodec checkpoints for encoding (or `--mock` flag for testing)

---

## Task C: Lambda Gradient Hooks

### Goal
Port `_register_lambda_grad_hooks()` pattern from `AmyTrainer` to `AmyDPOTrainer` so λ gradient magnitudes are logged alongside λ values at each logging step.

### Design

**New attributes:**
- `self._saved_lambda_grads: dict[str, float] = {}` — populated by backward hooks

**New method: `_register_lambda_grad_hooks(self)`**
```python
def _register_lambda_grad_hooks(self) -> None:
    try:
        model = self.accelerator.unwrap_model(self.model)
        base_model = getattr(model, "base_model", model)
        fusion = base_model.residual_fusion
        
        def _make_hook(name: str):
            def hook(grad: torch.Tensor) -> None:
                self._saved_lambda_grads[name] = grad.detach().cpu().item()
            return hook
        
        for name in ("lambda_p", "lambda_t"):
            param = getattr(fusion, name)
            param.register_hook(_make_hook(name))
    except AttributeError:
        pass
```

**Updated `log()` method** — injects `lambda_p_grad` / `lambda_t_grad` after existing lambda values:
```python
def log(self, logs: dict[str, float], *args, **kwargs) -> None:
    try:
        model = self.accelerator.unwrap_model(self.model)
        base_model = getattr(model, "base_model", model)
        fusion = base_model.residual_fusion
        logs["lambda_p"] = float(fusion.lambda_p.item())
        logs["lambda_t"] = float(fusion.lambda_t.item())
        if self._saved_lambda_grads:
            logs["lambda_p_grad"] = self._saved_lambda_grads.get("lambda_p", 0.0)
            logs["lambda_t_grad"] = self._saved_lambda_grads.get("lambda_t", 0.0)
    except AttributeError:
        pass
    super().log(logs, *args, **kwargs)
```

**PEFT compatibility:** The same `unwrap_model` → `getattr(_, "base_model", _)` chain works because:
1. `PeftModel.__getattr__` forwards to `.base_model` (LoraModel)
2. `LoraModel.__getattr__` forwards to `.model` (AmyMossLM)
3. `.residual_fusion` resolves on AmyMossLM
4. `.register_hook()` is called directly on the Parameter tensor (not affected by wrapping)

**Hook called from `__init__`** after `super().__init__()` (model is on device by then).

### Files
- `src/training/amy_dpo_trainer.py` — Add hooks, update `log()`

### Dependencies
- None (model architecture unchanged)

---

## Task D: Text-Blind Sanity Check

### Goal
Verify that adversarial pairs are genuinely text-indistinguishable. A text-only LLM should not be able to distinguish chosen from rejected at above-chance accuracy.

### Script: `scripts/text_blind_sanity_check.py`

**CLI:**
```
uv run python scripts/text_blind_sanity_check.py \
    --pairs data/nvtts_adversarial/pairs.jsonl \
    --num-samples 50
```

**Flow:**
1. Load JSONL pairs, select 50 random held-out samples (not used in encoding)
2. For each pair:
   - Create a text-blind prompt: transcript only, no emotion label, no inverse emotion
   - Shuffle chosen/rejected order (randomly swap positions)
   - Ask DeepSeek V4 Flash: "Which response reflects the speaker's actual emotion?"
3. Score accuracy: correct / total
4. Print results:
   - Overall accuracy (must be ≤ 55%)
   - Per-emotion breakdown
   - If > 55%: warn that pairs may be text-distinguishable

**Prompt (text-blind):**
```
A speaker said: "{transcript}"

Two responses were given:

Response A: "{response_a}"
Response B: "{response_b}"

Which response better reflects the speaker's tone and emotional state? Answer with just "A" or "B".
```

No emotion context, no inverse mapping, shuffled order. DeepSeek must guess from text alone.

**Edge cases:**
- Shuffle seed is fixed for reproducibility
- Stores per-sample results for analysis
- Respects API rate limits via semaphore (same pattern as generator)

### Files
- `scripts/text_blind_sanity_check.py` — Create

### Dependencies
- DeepSeek API key
- Phase 1 JSONL output (need at least 50 pairs beyond training split)

---

## Files to Touch Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/preprocessing/preference_dataset_processor.py` | Extend | Add adversarial schema + processor method |
| `scripts/encode_adversarial_pairs.py` | Create | Full encoding + push pipeline |
| `src/training/amy_dpo_trainer.py` | Extend | Lambda gradient hooks |
| `scripts/text_blind_sanity_check.py` | Create | Text-blind validation |
| `configs/dpo/adversarial.yaml` | Create | DPO config for adversarial training |
| `tests/preprocessing/test_preference_dataset_processor.py` | Extend | Adversarial processor tests |
| `tests/training/test_amy_dpo_trainer.py` | Extend | Lambda gradient hook tests |

## Phase Completion Criteria

- [ ] Adversarial dataset on HF Hub with FACodec streams
- [ ] Lambda gradient hooks functional in `AmyDPOTrainer`
- [ ] Text-blind sanity check passes (≤ 55% accuracy)
- [ ] Unit tests pass for new trainer hooks and processor method
