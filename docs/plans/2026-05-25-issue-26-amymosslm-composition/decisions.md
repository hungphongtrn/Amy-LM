# Decision Log

## 2026-05-25: Chose composition over inheritance
**Context:** `AmyLM` inherited `MossAudioModel`, creating a `__class__` mutation hack (`_upgrade_from_moss`) to work around 4-bit loading of missing FACodec checkpoint keys.
**Decision:** `AmyMossLM(PreTrainedModel, GenerationMixin)` composes `self.moss = MossAudioModel(...)`. Constructor accepts optional `moss` kwarg for pre-loaded (e.g., 4-bit) backbones.
**Rationale:** Eliminates the `__class__` mutation hack. PyTorch composition gives `moss.*` key prefix for free (no manual state_dict remapping). Class-level Liger monkey-patches work regardless of architecture. Cleaner, more testable.
**Consequences:** `AmyMossLMConfig` no longer extends `MossAudioConfig` — it wraps `MossAudioConfig` as `self.config` (nested). `forward()` must replicate ~120 lines of MossAudioModel orchestration, delegating heavy calls to `self.moss`. No inheritance from `MossAudioPreTrainedModel` — `_no_split_modules`, gradient checkpointing support, etc. must be declared directly.

## 2026-05-25: MossAudio source vendoring location
**Context:** `vendor/MOSS-Audio/src/modeling_moss_audio.py` (606 lines) + `configuration_moss_audio.py` (126 lines) imported via `sys.path` hacks.
**Decision:** Concatenate both files into `src/models/moss_audio_model.py`. Rewrite internal import from `from configuration_moss_audio import ...` to `from .moss_audio_model import ...` (or keep relative within same file).
**Rationale:** Single file, no vendor path dependency at runtime. One import for everything MossAudio-related. Liger kernel patching and flash attention config can happen without touching vendor directory.
**Consequences:** `moss_audio.py` (`MossAudioWrapper`) is deleted. Tests and training scripts import from `src.models.moss_audio_model` instead of vendor path.

## 2026-05-25: LoRA target_modules regex
**Context:** `get_peft_model(model, target_modules="all-linear")` would accidentally target FACodec linear layers (prosody projector, timbre projection, temporal pool).
**Decision:** Explicit regex: `^moss\..*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)$`
**Rationale:** PyTorch composition gives `moss.*` key prefix automatically. Regex scoped to `moss.*` excludes FACodec modules. Explicit linear projection names avoid matching all linear layers in the model.
**Consequences:** Training script must use this regex instead of `"all-linear"`. Verified against MossAudioModel linear projection names.

## 2026-05-25: Liger kernel activation strategy
**Context:** `apply_liger_kernel_to_qwen3()` is available in liger-kernel 0.8.0, but only as a global class-level monkey-patch.
**Decision:** Call `apply_liger_kernel_to_qwen3()` at module level in `moss_audio_model.py` before any model construction.
**Rationale:** Works regardless of inheritance vs composition. No architecture impact. Call once at import time, applies to all Qwen3 instances thereafter.
**Consequences:** Must verify liger-kernel is available. If not installed, skip gracefully (warning, not error). Applies globally which matches current behavior.

## 2026-05-25: Flash attention configuration
**Context:** `MossAudioConfig` does not expose flash attention config; it's baked into `language_config` (a `Qwen3Config`).
**Decision:** Set `language_config["_attn_implementation"] = "flash_attention_2"` in `__init__` before constructing the language model.
**Rationale:** Flash attention is a performance requirement, not optional. Configured at the language model level via Qwen3's `_attn_implementation`.
**Consequences:** Must be done before `MossAudioModel.__init__()` constructs `self.language_model = Qwen3Model(config.language_config)`.

## 2026-05-25 (Phase 1 learnings): MossAudioWrapper must survive Phase 1

**Context:** Original plan said "Delete `src/models/moss_audio.py`" in Phase 1. Code quality review found `amy_classifier.py`, `baseline_classifier.py`, and `test_moss_audio.py` still import `MossAudioWrapper` from the deleted file.

**Decision:** Keep `src/models/moss_audio.py` but update its imports to use the vendored `.moss_audio_model` module instead of vendor `sys.path` hacks. `MossAudioProcessor` import from vendor preserved (processing code not yet vendored).

**Rationale:** The classifiers (Issue #8) are a separate concern from the AmyMossLM refactor. Deleting the wrapper breaks them. Keeping the file with updated imports eliminates the vendor path dependency for `MossAudioModel` while preserving backward compat for classifier consumers.

**Consequences:** `moss_audio.py` still imports `MossAudioProcessor` from vendor via `sys.path`. This is acceptable — the processor vendoring is a separate concern. Phase 1 successfully eliminated vendor dependency for the core `MossAudioModel`/`MossAudioConfig` imports used by `AmyMossLM`.

## 2026-05-25: In-place rename, no backward compat
**Context:** Renaming `AmyLM` → `AmyMossLM`, `AmyLMConfig` → `AmyMossLMConfig`.
**Decision:** Delete old names entirely. No aliases, no re-exports, no deprecation path.
**Rationale:** Branch `exp/amylm-facodec` is pre-production. No downstream consumers to break. Old tests are deleted and rewritten.
**Consequences:** `__init__.py` only exports `AmyMossLM`/`AmyMossLMConfig`. Tests fully rewritten. Training script imports updated.
