# Phase 4: Training & Evaluation

> **STUB** — Will be detailed after Phases 2+3 complete. Training parameters may change based on actual GPU memory profiling from Phase 2.

## Phase Goal
MOSS-Audio baseline and Amy model trained on MUStARD, accuracy/F1 reported, W&B logs captured.

## High-Level Tasks (TBD)
- Create `scripts/train_amy.py` — CLI entry point for training
- Vanilla PyTorch training loop (no Lightning, no HF Trainer)
- MOSS-Audio baseline training: frozen backbone + Linear(2560→2) classifier
- Amy model training: frozen backbone + FACodec streams + classifier
- W&B integration: log loss, accuracy, F1, lambda_p, lambda_t per epoch
- Evaluation on test split
- Optional diagnostic: evaluate Amy model with lambdas=0 to measure residual stream contribution
- Results summary output

## Dependencies
- Phase 2: Amy Model Assembly
- Phase 3: Data Pipeline
