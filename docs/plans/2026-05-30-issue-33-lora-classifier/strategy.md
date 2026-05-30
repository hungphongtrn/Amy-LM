# LoRA Classifier — Strategy

## Goal
Add PEFT LoRA to the classifier training path to test H2: frozen MOSS-Audio backbone starves FACodec gradient. Let the backbone co-adapt to the enriched representation via LoRA adapters on audio_adapter and language_model projection layers.

## Architecture

```
AmyForProsodyClassification (PeftModel wrapper)
├── base_model.model.amy_moss (AmyMossLM)
│   ├── moss.audio_encoder          ← FROZEN (no LoRA)
│   ├── moss.audio_adapter (GatedMLP) ← LoRA on gate/up/down_proj ⬅ KEY ADDITION
│   ├── moss.language_model (Qwen3)   ← LoRA on q/k/v/o/up/down/gate_proj
│   ├── prosody_embedding           ← modules_to_save (trainable, fp32)
│   ├── timbre_projection           ← modules_to_save (trainable, fp32)
│   ├── temporal_pool               ← modules_to_save (trainable, fp32)
│   └── residual_fusion             ← modules_to_save (trainable, fp32)
└── base_model.model.classifier (nn.Linear) ← modules_to_save (trainable)
```

The LoRA target regex includes `audio_adapter.*` explicitly — the audio adapter is the direct bridge from FACodec-enriched audio to Qwen3 and a likely gradient choke-point per H2. The DPO pattern only targeted `language_model.*`; we extend it.

## Tech Stack
- `peft.LoraConfig` / `get_peft_model` (already a dependency)
- `TaskType.FEATURE_EXTRACTION` (classifier doesn't use LM head)
- Existing FACodec fp32 convention from DPO path (line 193 of `train_amy_dpo.py`)

## Constraints & Assumptions
- **LoRA ranks/config match DPO defaults** (r=8, alpha=16, dropout=0.05) as starting point
- **FACodec modules kept in fp32** (matching DPO convention); backbone stays bf16
- **audio_encoder stays frozen** — no LoRA on Whisper layer projections (too many params, not in scope for H2 test)
- **Lambda gradient hooks** must work through PEFT wrapper → `model.model.amy_moss.residual_fusion` (unwrap one level)
- **Checkpoint save/load** uses `PeftModel.save_pretrained()` style (saves adapter weights only, not full backbone)
- **Task type: FEATURE_EXTRACTION** — the classifier head is a separate `nn.Linear`, not the LM head. `TaskType.CAUSAL_LM` from DPO won't work here.

## Phases (High-Level)

### Phase 1: LoRA wrapping utility + init tests
**Outcome:** `wrap_classifier_with_lora()` function exists in `src/models/amy_classifier.py` (or dedicated utility). Static init test asserts LoRA adapters on audio_adapter.* and language_model.* projection layers; frozen backbone base weights; trainable FACodec modules + classifier via modules_to_save.
**Rough scope:** Create the utility function ported from DPO pattern, add two init tests (static + gradient flow). No trainer/training changes yet.

### Phase 2: Trainer PEFT adaptation
**Outcome:** `AmyTrainer.save_checkpoint` / `load_checkpoint` work with PEFT-wrapped model. Lambda gradient hooks (`_register_lambda_grad_hooks`, `_get_lambdas`) navigate the PEFT wrapper correctly. Checkpoint roundtrip test passes.
**Rough scope:** Fix path traversals, update state_dict filtering, add test.
**Depends on:** Phase 1

### Phase 3: CLI integration
**Outcome:** `--use-lora`, `--lora-r`, `--lora-alpha`, `--lora-dropout` flags on `train_amy_classifier.py` work. Model is conditionally wrapped with LoRA.
**Rough scope:** Add argparse flags, wire model construction, test that CLI produces trainable LoRA model.
**Depends on:** Phase 2

### Phase 4: Experiment script
**Outcome:** `scripts/train_33_lora_amy.py` launches with `--use-lora --mode amy --wandb`, trains to convergence on MUStARD. Lambda gradients and val_f1 logged for H2 validation.
**Rough scope:** Create experiment script with hardcoded config, launch command.
**Depends on:** Phase 3

## Open Questions
1. **Q: `TaskType.CAUSAL_LM` vs `TaskType.FEATURE_EXTRACTION`?** A: FEATURE_EXTRACTION — the classifier uses Qwen3 as encoder-only (no lm_head, no causal masking for loss). Set `TaskType.FEATURE_EXTRACTION` since the loss is computed on the classifier head, not the LM head.

2. **Q: Does `modules_to_save` on `classifier` work when classifier is a top-level attribute (not nested under amy_moss)?** A: Yes — PEFT registers it at the top level. The key will be `"classifier"` in the PeftModel, not `"amy_moss.xxx.classifier"`. The checkpoint should include both the adapter weights and the classifier/fusion weights.

3. **Q: Will the DPO pattern of casting FACodec modules to fp32 play well with LoRA?** A: Yes — this is a separate call before `get_peft_model`. LoRA adapters will be in the backbone's dtype (bf16), but `modules_to_save` will match the fp32 cast.
