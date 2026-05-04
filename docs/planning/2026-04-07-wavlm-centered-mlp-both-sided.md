# WavLM Centered MLP Both-Sided

## Goal
Run the next WavLM-focused training experiment after the centered-target-only run plateaued around `train/wavlm ~= 0.94-0.98` while `llm` learned well.

## Hypothesis
Two remaining issues are worth testing together:

1. The WavLM objective still penalizes clip-level mean mismatch because only the target was centered.
2. The linear `wavlm_proj` may not have enough capacity to map Mimi latents into WavLM space.

## Changes

1. Center both predicted and target WavLM features per clip before cosine loss.
2. Replace `wavlm_proj` with:
   `Linear(512, 512) -> GELU -> LayerNorm -> Linear(512, 1024)`
3. Keep the previous improved setup unchanged:
   - RVQ LR `5e-5`
   - 100-step RVQ warmup
   - `alpha_msspec=5.0`
   - `alpha_wavlm=5.0`
   - `alpha_llm=1.0`

## Entry Point
`scripts/train_wavlm_centered_mlp.py`

## Run Command
`uv run python scripts/train_wavlm_centered_mlp.py`

## Success Criteria
1. `train/wavlm` should move materially below the previous plateau within the first `200-500` steps.
2. `val/wavlm` should follow the same direction.
3. `train/llm` should remain in the strong range already achieved by prior runs.

## Stop Criteria
Stop early if `train/wavlm` remains stuck near `0.9+` after `200-500` steps. In that case, move to a stricter isolation experiment rather than more weight tuning.
