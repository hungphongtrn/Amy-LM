# Phase 3: Training Scripts and Tests

## Phase Goal
`train_amy_dpo.py`'s `init_model()` uses `AmyMossLM(moss=moss_4bit)` (no `_upgrade_from_moss`). LoRA `target_modules` uses explicit regex `^moss\..*`. All tests pass — old tests deleted, new tests for `AmyMossLM` pass.

**⚠️ Detailed tasks will be written after Phase 2 completes.** This is a stub.

## Files to Touch

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/train_amy_dpo.py` | **Edit** | Update `init_model()` |
| `tests/models/test_amy_lm.py` | **Rewrite** | New tests for `AmyMossLM` |
| `tests/training/test_amy_dpo_trainer.py` | **Edit** | Update `MockAmyLM` → `MockAmyMossLM` |

## High-Level Scope

1. **`train_amy_dpo.py` `init_model()`**: 
   - Load `MossAudioModel` with `load_in_4bit=True` (same as current)
   - Create `AmyMossLMConfig` (not `AmyLMConfig`)
   - Pass as `moss=moss_4bit` to `AmyMossLM(moss=moss_4bit, config=amy_config)`
   - LoRA `target_modules` changed from `"all-linear"` to `r"^moss\..*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)$"`
   - Import `AmyMossLM`/`AmyMossLMConfig` instead of `AmyLM`/`AmyLMConfig`

2. **`tests/models/test_amy_lm.py`**:
   - Delete all existing tests (352 lines)
   - Write new tests:
     - `TestAmyMossLMConfig` — config fields, model_type `"amy_moss_lm"`, to_dict roundtrip, nested `moss_config`
     - `TestAmyMossLMModelStructure` — composition (NOT inheritance) check, `hasattr(model, "moss")`, `isinstance(model.moss, MossAudioModel)`, FACodec modules on `self` (not `self.moss`), freeze logic on `self.moss.*`, `get_input_embeddings` delegates to `self.moss`
     - `TestAmyMossLMForward` — text-only, with audio, with prosody, with timbre, with both, backward-compatible, gradient flow through FACodec, frozen backbone no gradients
     - `TestAmyMossLMSaveLoad` — save_pretrained / from_pretrained roundtrip, `moss.*` key prefix
     - `TestAmyMossLMConstructorInjection` — pre-loaded moss kwarg works, no duplicate construction
     - `TestAmyMossLMPrepareBaseCheckpoint` — optionally, if feasible in test environment
     - `TestAmyMossLMAutoModel` — `AutoModelForCausalLM` registration

3. **`tests/training/test_amy_dpo_trainer.py`**:
   - Rename `MockAmyLM` → `MockAmyMossLM`
   - Add `moss` attribute (or mock it) if trainer code accesses `model.moss`
   - Keep existing mock logic, just updated naming

## Depends On
- Phase 2 (full `AmyMossLM` class must exist)
