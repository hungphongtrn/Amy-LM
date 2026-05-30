# Phase 3: CLI integration

## Phase Goal
`--use-lora`, `--lora-r`, `--lora-alpha`, `--lora-dropout` flags on `train_amy_classifier.py` work end-to-end. Model is conditionally wrapped with LoRA at construction. Trainer receives `is_lora=True`.

**Depends on:** Phase 2

## Files to Touch

| File | Action |
|------|--------|
| `scripts/train_amy_classifier.py` | Add argparse flags, wire LoRA wrapping + trainer flag |

## Tasks

### Task 1: Add CLI flags and wire LoRA wrapping

**File:** `scripts/train_amy_classifier.py`

- [ ] **Step 1: Add argparse flags (after `--facodec-control` block, line 121)**

```python
p.add_argument("--use-lora", action="store_true", help="Wrap classifier with PEFT LoRA on audio_adapter + language_model")
p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8)")
p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)")
p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
```

- [ ] **Step 2: Wire LoRA wrapping in model construction (after line 177)**

After `model = model.to(device)`, add:

```python
# LoRA wrapping
use_lora = args.use_lora and not is_baseline
if use_lora:
    from src.models.amy_classifier import wrap_classifier_with_lora
    model = wrap_classifier_with_lora(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
```

- [ ] **Step 3: Pass `is_lora` to trainer (line 187)**

Change `AmyTrainer(...)` call to include `is_lora=use_lora`:

```python
trainer = AmyTrainer(
    model=model,
    device=device,
    lr=args.lr,
    weight_decay=args.weight_decay,
    grad_accum_steps=args.grad_accum,
    log_wandb=args.wandb,
    is_baseline=is_baseline,
    is_lora=use_lora,
    max_grad_norm=args.max_grad_norm,
)
```

### Task 2: Smoke test

- [ ] **Run a quick dry-run (1 epoch, no data):**

```bash
uv run python -c "
import torch
from src.models.amy_classifier import AmyForProsodyClassification, wrap_classifier_with_lora
vectors = torch.randn(1024, 2560)
model = AmyForProsodyClassification(warm_start_vectors=vectors, device='cpu')
peft = wrap_classifier_with_lora(model, r=2, lora_alpha=4, lora_dropout=0.0)
from src.training.trainer import AmyTrainer
t = AmyTrainer(peft, device='cpu', is_baseline=False, is_lora=True)
print('LoRA trainer init OK')
print('Epoch:', t.current_epoch)
print('Lambda values:', t._get_lambdas())
"
```

Expected: Lambda values printed without error.

## Phase Completion Criteria
- [ ] `--use-lora` flag on `train_amy_classifier.py` (with defaults for `--lora-r`, `--lora-alpha`, `--lora-dropout`)
- [ ] LoRA wrapping conditional on `not is_baseline`
- [ ] `is_lora=use_lora` passed to `AmyTrainer`
- [ ] Smoke test: LoRA trainer initializes and lambdas accessible
- [ ] Commit

## Handoff Notes
- `use_lora` is `False` for baseline mode — LoRA is only for Amy
- The LoRA wrapping happens after `model.to(device)` — PEFT wrapping doesn't change device
- The lambda hooks (`_register_lambda_grad_hooks`) fire during `__init__` — they use `_base_model` which properly unwraps for LoRA
