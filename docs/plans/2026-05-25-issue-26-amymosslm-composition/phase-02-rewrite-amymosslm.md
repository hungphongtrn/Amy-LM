# Phase 2: Rewrite AmyMossLM

## Phase Goal
`amy_lm.py` is completely rewritten. `AmyMossLM(PreTrainedModel, GenerationMixin)` composes `self.moss = MossAudioModel(...)`, supports constructor injection, and includes `prepare_base_checkpoint()` for HuggingFace Hub bootstrap. No inheritance from `MossAudioModel`, no `__class__` mutation.

**⚠️ Detailed tasks will be written after Phase 1 completes.** This is a stub.

## Files to Touch

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models/amy_lm.py` | **Rewrite** | `AmyMossLMConfig` + `AmyMossLM` with composition |

## High-Level Scope

1. **`AmyMossLMConfig(PretrainedConfig)`**: No longer extends `MossAudioConfig`. Stores a nested `MossAudioConfig` as `self.moss_config`. Exposes FACodec fields directly. `model_type = "amy_moss_lm"`.

2. **`AmyMossLM(PreTrainedModel, GenerationMixin)`**:
   - `__init__(config, moss=None)`: If `moss` is provided (pre-loaded, e.g., 4-bit), use it directly. Otherwise construct `self.moss = MossAudioModel(config.moss_config)`.
   - `_add_facodec_modules()`: Same logic as current `AmyLM._add_facodec_modules()`, adds `prosody_embedding`, `timbre_projection`, `temporal_pool`, `residual_fusion` to `self`.
   - `_apply_freeze()`: Freezes `self.moss.audio_encoder`, `self.moss.audio_adapter`, `self.moss.language_model`.
   - `_enrich_audio_embeds()`: Same logic as current (delegates to `self.prosody_embedding`, `self.timbre_projection`, `self.temporal_pool`, `self.residual_fusion`).
   - `forward()`: Full orchestration ported from `MossAudioModel.forward()` with FACodec enrichment hook inserted between `self.moss.audio_adapter()` and `masked_scatter_()`. Delegates to `self.moss.*` for all heavy calls.
   - `prepare_inputs_for_generation()`: Ported from `MossAudioModel`.
   - `prepare_base_checkpoint(save_path)`: Class method that loads MOSS-Audio weights, wraps in `AmyMossLM`, saves full checkpoint.
   - `get_input_embeddings()`, `set_input_embeddings()`, `get_output_embeddings()`, `set_output_embeddings()`: Delegated to `self.moss`.
   - `_no_split_modules = ["Qwen3DecoderLayer", "WhisperEncoderLayer"]`
   - `supports_gradient_checkpointing = True`

3. **Key differences from inheritance version**:
   - `self.audio_encoder` → `self.moss.audio_encoder`
   - `self.audio_adapter` → `self.moss.audio_adapter`
   - `self.language_model` → `self.moss.language_model`
   - `self.lm_head` → `self.moss.lm_head`
   - `self.get_audio_features()` → `self.moss.get_audio_features()`
   - `self._register_llm_deepstack_hooks()` → `self.moss._register_llm_deepstack_hooks()`
   - `self.deepstack_audio_merger_list` → `self.moss.deepstack_audio_merger_list`
   - `self.config.language_config` → `self.config.moss_config.language_config`
   - `self.config.ignore_index` → `self.config.moss_config.ignore_index`

4. **Model registration**: `AmyMossLMConfig.register_for_auto_class()` and `AmyMossLM.register_for_auto_class("AutoModelForCausalLM")`

## Depends On
- Phase 1 (vendored `moss_audio_model.py` must be importable)
