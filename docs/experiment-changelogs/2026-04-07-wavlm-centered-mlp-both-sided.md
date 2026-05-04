# Experiment: WavLM Centered MLP Both-Sided
**Date**: 2026-04-07
**Status**: Ready to Launch
**Branch/Commit**: Current working directory

## Summary
Prepare the next WavLM distillation run by combining two targeted changes: center both predicted and target WavLM features per clip, and replace the WavLM linear projection with a small MLP.

## Configuration
- Dataset: `data/Amy-LM-Dataset-Aligned`
- Model: MiMi 9-codebook semantic+prosody training
- Hyperparameters:
  - RVQ LR: `5e-5`
  - RVQ warmup: `100` steps
  - Projection LR: `3e-4`
  - `alpha_msspec=5.0`
  - `alpha_wavlm=5.0`
  - `alpha_llm=1.0`
- Duration: `max_steps=2000` with early inspection after `200-500` steps

## Results
- Key metrics: Pending
- Observations: Pending
- Audio quality: Pending

## Next Steps
1. Launch the run with the new entrypoint.
2. Inspect `train/wavlm` and `val/wavlm` within the first `200-500` steps.
3. If WavLM is still flat, switch to an isolation experiment rather than further tuning this setup.

## Planning Document
Link to detailed planning doc: [docs/planning/2026-04-07-wavlm-centered-mlp-both-sided.md](docs/planning/2026-04-07-wavlm-centered-mlp-both-sided.md)
