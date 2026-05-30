# Phase 2: Trainer PEFT adaptation

## Phase Goal
`AmyTrainer.save_checkpoint` and `load_checkpoint` work with PEFT-wrapped model. Lambda gradient hooks (`_register_lambda_grad_hooks`, `_get_lambdas`) navigate the PEFT wrapper correctly. Checkpoint roundtrip test passes.

**Depends on:** Phase 1

## Files to Touch (preliminary)

| File | Action |
|------|--------|
| `src/training/trainer.py` | Modify save/load/hooks for PEFT-aware traversal |
| `tests/training/test_trainer.py` | Add PEFT checkpoint roundtrip test |

## Tasks (stub — to be detailed after Phase 1)

- Fix `self.model.amy_moss.residual_fusion` → `self.model.model.amy_moss.residual_fusion` (unwrap PeftModel)
- Fix `save_checkpoint`: filter only PEFT trainable params (use `peft_model.state_dict()` or manual key filtering)
- Fix `load_checkpoint`: load adapter weights with `strict=False`  
- Fix `_register_lambda_grad_hooks`: access fusion through unwrapped model
- Fix `_get_lambdas`: same unwrap
- Add `is_lora` flag to `AmyTrainer.__init__` to gate unwrap behavior
- Test: save/load roundtrip preserves LoRA weights and lambda values
