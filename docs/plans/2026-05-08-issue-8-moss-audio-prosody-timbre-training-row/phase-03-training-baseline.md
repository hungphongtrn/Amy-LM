# Phase 3: Baseline And Amy Training

## Phase Goal

Train and evaluate the frozen MOSS-Audio baseline and Amy LM Prosody/Timbre row under the same binary sarcasm classification protocol.

## Current Status

Stub only. Detail this phase after Phase 2 exposes stable model and batch interfaces.

## Expected Scope

- Add a vanilla PyTorch runner, not Lightning and not HF Trainer.
- Load Phase 1 parquet/HF Dataset output and collate raw audio, labels, Prosody Stream indices, and Timbre Vectors.
- Support baseline mode with no FACodec residual streams and Amy mode with Prosody/Timbre enabled.
- Use `CrossEntropyLoss`, accuracy, and F1.
- Use identical splits/folds, optimizer settings, batch sizing, and classifier protocol for baseline and Amy rows.
- Log metrics and `lambda_p`/`lambda_t` to wandb per epoch.
- Use `nohup` for long-running training commands and preserve log paths.

## Completion Criteria

- Baseline and Amy commands run from documented CLI arguments.
- Metrics are reported for both rows under the same split protocol.
- Checkpoints are saved locally with enough metadata to reproduce the run.
