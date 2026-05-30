# Phase 4: Experiment script

## Phase Goal
`scripts/train_33_lora_amy.py` launches with `--use-lora --mode amy --wandb`, trains to convergence on MUStARD. Lambda gradients and val_f1 logged to W&B for H2 validation.

**Depends on:** Phase 3

## Files to Touch (preliminary)

| File | Action |
|------|--------|
| `scripts/train_33_lora_amy.py` | Create — experiment entry point |

## Tasks (stub — to be detailed after Phase 3)

- Create experiment script with hardcoded config matching prior Amy experiments (epochs=50, lr=1e-4, batch_size=1, grad_accum=8, seed=42)
- Launch command: `nohup python scripts/train_33_lora_amy.py --data-path data/processed/mustard-processed/train.parquet --mode amy --use-lora --wandb --epochs 50 > logs/train_33_lora_$(date +%Y%m%d_%H%M%S).log 2>&1 &`
- After convergence: compare λ movement, val_f1, train_loss vs baseline/Amy(shuffle)/Amy(aligned) from #32
- Document results on [issue #32](https://github.com/hungphongtrn/Amy-LM/issues/32)
