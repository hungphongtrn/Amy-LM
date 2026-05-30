# Phase 2: Trainer PEFT adaptation

## Phase Goal
`AmyTrainer.save_checkpoint` and `load_checkpoint` work with PEFT-wrapped model. Lambda gradient hooks navigate the `PeftModel` wrapper. Forward call uses keyword args for PEFT compatibility. Checkpoint roundtrip test passes.

**Depends on:** Phase 1

## Files to Touch

| File | Action |
|------|--------|
| `src/training/trainer.py` | Add `is_lora` flag, `_unwrap_model()`/`_unwrap_amy_moss()` helpers, fix forward args, fix save/load, fix lambda hooks |
| `tests/training/test_trainer.py` | Add LoRA checkpoint roundtrip test |

## Key changes needed

### 1. Add `is_lora` flag + unwrap helpers

PeftModel wraps the base model at `self.model.model`. All internal accesses to `self.model.amy_moss.*` must unwrap.

```python
@property
def _base_model(self) -> nn.Module:
    return self.model.model if self.is_lora else self.model
```

### 2. Forward call — use keyword args

`PeftModelForFeatureExtraction.forward` maps positional args to `input_ids`/`attention_mask`/`inputs_embeds`. Must pass kwargs matching the model's signature.

```python
# Line 68: change from positional to keyword
logits = self.model(audio=audio, prosody_indices=prosody, timbre_vector=timbre)
```

### 3. Lambda hooks — unwrap PEFT

```python
fusion = self._base_model.amy_moss.residual_fusion  # was: self.model.amy_moss...
```

### 4. save_checkpoint — PEFT-aware

With PEFT, `state_dict()` returns only trainable adapter weights + modules_to_save. No manual key filtering needed.

```python
def save_checkpoint(self, path: str) -> None:
    model_state = self.model.state_dict()  # PEFT returns trainable-only; non-PEFT needs filtering
    if not self.is_lora:
        model_state = {k: v for k, v in model_state.items() if not k.startswith("amy_moss.moss.")}
    facodec_state = self._base_model.amy_moss.facodec_state_dict() if not self.is_baseline else {}
    torch.save({...}, path)
```

### 5. load_checkpoint — PEFT-aware

PeftModel's `load_state_dict` handles adapter injection automatically.

```python
def load_checkpoint(self, path: str) -> None:
    ckpt = torch.load(path, map_location=self.device, weights_only=True)
    incompatible = self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Filter only non-moss frozen keys if not lora
    ...
```

## Tasks

### Task 1: Modify AmyTrainer for PEFT awareness

**File:** `src/training/trainer.py`

- [ ] **Step 1: Add `is_lora` parameter and `_base_model` property**

```python
def __init__(self, ..., is_lora: bool = False, ...):
    ...
    self.is_lora = is_lora

@property
def _base_model(self) -> nn.Module:
    return self.model.model if self.is_lora else self.model
```

- [ ] **Step 2: Fix forward call in `training_step`**

Line 68: Change `self.model(audio, prosody, timbre)` to `self.model(audio=audio, prosody_indices=prosody, timbre_vector=timbre)`

- [ ] **Step 3: Fix lambda hooks**

In `_register_lambda_grad_hooks` and `_get_lambdas`, change `self.model.amy_moss.residual_fusion` to `self._base_model.amy_moss.residual_fusion`

- [ ] **Step 4: Fix `save_checkpoint`**

```python
def save_checkpoint(self, path: str) -> None:
    if self.is_lora:
        model_state = self.model.state_dict()  # PEFT: trainable-only
    else:
        model_state = {k: v for k, v in self.model.state_dict().items()
                       if not k.startswith("amy_moss.moss.")}
    facodec_state = self._base_model.amy_moss.facodec_state_dict() if not self.is_baseline else {}
    torch.save({
        "model_state_dict": model_state,
        "facodec_state_dict": facodec_state,
        "optimizer_state_dict": self.optimizer.state_dict(),
        "epoch": self.current_epoch,
        "is_baseline": self.is_baseline,
        "is_lora": self.is_lora,
    }, path)
```

- [ ] **Step 5: Fix `load_checkpoint`**

```python
def load_checkpoint(self, path: str) -> None:
    ckpt = torch.load(path, map_location=self.device, weights_only=True)
    self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    self.current_epoch = ckpt["epoch"]
```

### Task 2: Add LoRA checkpoint roundtrip test

**File:** `tests/training/test_trainer.py`

- [ ] **Add test to existing `TestCheckpoint` class:**

```python
def test_save_load_roundtrip_lora(self, tmp_path, require_gpu, device):
    from src.models.amy_classifier import (
        AmyForProsodyClassification,
        wrap_classifier_with_lora,
    )
    from src.training.trainer import AmyTrainer

    vectors = torch.randn(1024, 2560, device=device)
    model = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)
    peft_model = wrap_classifier_with_lora(model, r=2, lora_alpha=4, lora_dropout=0.0)
    trainer = AmyTrainer(peft_model, device=device, is_baseline=False, is_lora=True)
    trainer.current_epoch = 5
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(path))

    # Load into fresh model
    model2 = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)
    peft_model2 = wrap_classifier_with_lora(model2, r=2, lora_alpha=4, lora_dropout=0.0)
    trainer2 = AmyTrainer(peft_model2, device=device, is_baseline=False, is_lora=True)
    trainer2.load_checkpoint(str(path))

    assert trainer2.current_epoch == 5
    # Verify LoRA weights match
    for (n1, p1), (n2, p2) in zip(
        trainer.model.named_parameters(), trainer2.model.named_parameters()
    ):
        if p1.requires_grad:
            assert torch.equal(p1, p2), f"Mismatch in {n1}"
```

### Task 3: Verify existing tests still pass

- [ ] Run non-LoRA trainer tests: `uv run python -m pytest tests/training/test_trainer.py -v`
- [ ] Run: `uv run python -m pytest tests/training/test_lora_classifier_init.py -v`

## Phase Completion Criteria
- [ ] `is_lora` flag on `AmyTrainer.__init__`
- [ ] `_base_model` property unwraps PEFT correctly
- [ ] Forward call uses keyword args (compatible with both PEFT and non-PEFT)
- [ ] Lambda hooks work through PEFT wrapper
- [ ] `save_checkpoint` / `load_checkpoint` roundtrip with LoRA model (GPU test)
- [ ] Existing trainer tests still pass
- [ ] Commit

## Handoff Notes
- The `_base_model` property returns `self.model.model` for LoRA, `self.model` otherwise
- PEFT's `state_dict()` returns only trainable parameters — no need for manual key filtering
- PEFT's `load_state_dict(strict=False)` handles adapter injection gracefully
- Positional args in `training_step` line 68 must be changed to keyword args — this works for both PEFT and non-PEFT models
