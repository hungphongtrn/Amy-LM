# AmyMossLM Composition Refactor — Strategy

## Goal
Refactor `AmyLM` (inheritance-based) into `AmyMossLM` (composition-based), replacing the `__class__` mutation hack with a clean HAS-A pattern.

## Architecture
`AmyMossLM` is a standalone `PreTrainedModel` + `GenerationMixin` that composes a `MossAudioModel` as `self.moss`. Constructor injection (`__init__(config, moss=None)`) supports pre-loaded (e.g., 4-bit) backbones. `forward()` replicates MossAudioModel orchestration, injecting FACodec enrichment between `audio_adapter()` and `masked_scatter_()`, delegating all heavy calls to `self.moss`.

MossAudio source (`modeling_moss_audio.py` + `configuration_moss_audio.py`) is vendored into `src/models/moss_audio_model.py`, eliminating vendor-path sys.path hacks at runtime.

## Tech Stack
- PyTorch (composition pattern)
- HuggingFace `PreTrainedModel`, `GenerationMixin`, `AutoModelForCausalLM`
- Liger kernel (`apply_liger_kernel_to_qwen3()` — global monkey-patch)
- Flash Attention 2
- PEFT/LoRA (scoped to `^moss\..*`)
- `bitsandbytes` (4-bit quantization)

## Constraints & Assumptions
- **No inheritance**: `AmyMossLM` does NOT extend `MossAudioModel` or `MossAudioPreTrainedModel`
- **No vendor path**: Virtually all runtime imports from `src.models`
- **No backward compat**: `AmyLM`/`AmyLMConfig` names deleted; no aliases
- **Existing modules preserved**: `ProsodyEmbedding`, `TimbreProjection`, `TemporalPool`, `ResidualFusion` unchanged
- **Liger kernel is class-level monkey-patch**: Works regardless of inheritance vs composition
- **Composition gives `moss.*` key prefix for free**: No manual state_dict key remapping needed
- **Flash attention**: Configured via `language_config["_attn_implementation"] = "flash_attention_2"` during construction
- **DPO training path**: `train_amy_dpo.py` is the only consumer that needs updating; `AmyForProsodyClassification` in `src/models/amy_classifier.py` uses `MossAudioWrapper` (separate concern, not in scope)

## Phases (High-Level)

### Phase 1: Vendor and Scaffold — Foundation
**Outcome:** `src/models/moss_audio_model.py` exists as vendored copy; `src/models/moss_audio.py` deleted; `__init__.py` exports updated to match new naming.
**Rough scope:** Copy `configuration_moss_audio.py` + `modeling_moss_audio.py` into a single vendored file, remove vendor path from `amy_lm.py`, update `__init__.py` imports, delete `moss_audio.py`.

### Phase 2: Rewrite AmyMossLM — Core Feature
**Outcome:** `AmyMossLM(PreTrainedModel, GenerationMixin)` composes `self.moss = MossAudioModel(...)`; `forward()` replicates orchestration with FACodec enrichment; `prepare_base_checkpoint()` bootstraps base checkpoint to HuggingFace Hub.
**Rough scope:** Rewrite `amy_lm.py` completely — new `AmyMossLMConfig(PretrainedConfig)`, new `AmyMossLM(PreTrainedModel, GenerationMixin)` with composition, Liger kernel activation, flash attention config, LoRA-friendly `target_modules`.
**Depends on:** Phase 1

### Phase 3: Training Scripts and Tests — Polish/Integration
**Outcome:** `train_amy_dpo.py` uses new composition-based `init_model()`, old tests deleted, new tests pass for `AmyMossLM`.
**Rough scope:** Update `init_model()` to use `AmyMossLM(moss=moss_4bit)`, rewrite `test_amy_lm.py` (composition assertions, no inheritance checks, moss.* key prefix, gradient flow), update `MockAmyMossLM` in `test_amy_dpo_trainer.py`.
**Depends on:** Phase 2

## Open Questions
None — all resolved in ADR #0001 grilling session.
