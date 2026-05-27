# ADR 0001: AmyMossLM — Composition over inheritance for MossAudio backbone

## Status

Proposed — 2025-05-25

## Context

`AmyLM` (issue #14) inherited `MossAudioModel` directly (IS-A). This forced a fragile `__class__` mutation hack (`moss.__class__ = AmyLM`) to work around HF's `_initialize_missing_keys` crash when loading with `load_in_4bit=True` — randomly-initialized FACodec modules absent from the checkpoint cause the quantized-load path to fail.

At the same time, we wanted to apply Liger kernel optimizations and flash attention to the Qwen3 backbone, and to stop depending on `vendor/MOSS-Audio/src/` at runtime.

## Decision

**Composition over inheritance.** `AmyMossLM` is a standalone `PreTrainedModel` + `GenerationMixin` that **composes** (HAS-A) a `MossAudioModel` as `self.moss`. FACodec enrichment modules (`ProsodyEmbedding`, `TimbreProjection`, `TemporalPool`, `ResidualFusion`) are sibling attributes.

Key design choices:
- `MossAudioModel` and `MossAudioConfig` source copied from `vendor/MOSS-Audio/src/` into `src/models/moss_audio_model.py` (Apache 2.0, same license)
- Constructor accepts optional `moss` parameter for pre-loaded backbones (e.g., 4-bit quantized)
- `forward()` replicates MossAudioModel's orchestration logic (~120 lines) but delegates all heavy calls (`get_audio_features`, `audio_adapter`, `language_model`, `lm_head`, `_register_llm_deepstack_hooks`) to `self.moss`
- Liger kernel applied globally via `apply_liger_kernel_to_qwen3()` at module level, before any model construction
- Flash attention configured via `language_config["_attn_implementation"] = "flash_attention_2"` in `AmyMossLMConfig`
- LoRA scoped to MossAudio backbone only via `target_modules` regex: `^moss\..*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)$`
- No `__class__` mutation, no `_upgrade_from_moss`, no vendor path dependency at runtime

### Base checkpoint bootstrap

`AmyMossLM.prepare_base_checkpoint(save_path)` loads MossAudio from `OpenMOSS-Team/MOSS-Audio-4B-Thinking`, wraps in `self.moss` composition (getting `moss.*` key prefix naturally via PyTorch child naming), adds zero-init FACodec modules, and saves the full model + processor to HF Hub as `hungphongtrn/amy-moss-lm-base`. This checkpoint is the foundation for all subsequent `from_pretrained` calls.

### `model_type`

`"amy_moss_lm"` — registered with `AutoModelForCausalLM` via `AutoModelForCausalLM.register(AmyMossLMConfig, AmyMossLM)`.

## Consequences

### Positive
- `load_in_4bit=True` path is natural: load quantized MossAudioModel → pass to constructor as `moss=moss_4bit`
- Clear ownership boundary: `self.moss` is the frozen/quantized backbone; everything else on `self` is trainable
- Liger kernel patches Qwen3 classes globally — works regardless of composition vs inheritance
- Vendor-free: all model source self-contained in `src/models/`
- `save_pretrained`/`from_pretrained` work via standard HF `PreTrainedModel` lifecycle

### Negative
- ~120 lines of forward orchestration replication (mitigated by clean delegation to `self.moss` methods)
- Full checkpoint is ~8GB (backbone + FACodec) — one-time storage cost on HF Hub
- Rename `AmyLM` → `AmyMossLM` breaks backward compat with existing code on this branch (acceptable — branch is exp/amylm-facodec, actively iterating)

## Alternatives considered

1. **Inheritance + `__class__` hack** (current): Rejected — fragile, mutation pattern, hard to debug.
2. **Architecture clone** (rewrite Whisper/Qwen3/GatedMLP into AmyMossLM): Rejected — ~400 lines of code duplication, diverges from upstream MossAudio fixes.
3. **`PyTorchModelHubMixin`** (plain `nn.Module`): Rejected — loses `generate()`, HF Trainer integration, and `GenerationMixin`.

## References

- Issue: [#26](https://github.com/hungphongtrn/Amy-LM/issues/26)
- CONTEXT.md: AmyMossLM, MOSS-Audio entries
- Source: `src/models/amy_lm.py` (AmyMossLM), `src/models/moss_audio_model.py` (vendored MossAudioModel)
