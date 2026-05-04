# Experiment: WavLM-Centered with Improvements (Priority 2)

**Date**: 2026-04-06  
**Status**: Ready to Launch  
**Branch/Commit**: Current working directory

## Summary

After Priority 1 (standard semantic+prosody training) showed WavLM loss stuck at ~0.68 with no improvement for 7 epochs, we're launching Priority 2 with significant improvements.

## Key Improvements

### 1. Mean-Centering WavLM Features
```python
wavlm_feat_centered = wavlm_feat - wavlm_feat.mean(dim=1, keepdim=True)
```
- **Why**: Removes global speaker/timbre bias from WavLM
- **Expected outcome**: semantic+prosody can focus on content + prosody without speaker identity

### 2. Higher Learning Rate for RVQ Heads
- **Before**: 1e-5
- **After**: 5e-5 (5x increase)
- **Why**: Quantizers need to adapt faster to new target distribution

### 3. Learning Rate Warmup for RVQ Heads
```python
warmup_steps = 100
warmup_factor = (global_step + 1) / warmup_steps
```
- Gradual increase from 0 to 5e-5 over first 100 steps
- Stabilizes early training

### 4. Rebalanced Loss Weights
| Component | Before | After | Change |
|-----------|--------|-------|--------|
| alpha_msspec | 15.0 | **5.0** | Reduced (less reconstruction focus) |
| alpha_wavlm | 1.0 | **5.0** | Increased (more distillation focus) |
| alpha_llm | 1.0 | 1.0 | Unchanged |

- **Why**: Previous run had msspec dominating (15:1 ratio), overwhelming distillation signals

## Configuration

```python
trainer_config = CompressorTrainerConfig(
    semantic_prosody_only=True,
    alpha_msspec=5.0,      # Reduced
    alpha_wavlm=5.0,       # Increased
    alpha_llm=1.0,
    # ... other params
)
```

## Duration

- **Steps**: 2000 (~55 epochs)
- **Expected time**: ~2-3 hours on RTX 3060

## Success Criteria

| Metric | Priority 1 (Failed) | Priority 2 (Target) |
|--------|--------------------|--------------------|
| WavLM train loss | Stuck at 0.68 | **Drop to < 0.4** |
| WavLM val loss | 0.77 (worse) | **Drop to < 0.5** |
| LLM loss | 0.08 (good) | Maintain ~0.08 |
| msspec loss | 0.10 | Acceptable increase okay |

## Expected Outcomes

### If WavLM loss drops significantly:
- ✅ Validates that speaker bias was the problem
- ✅ semantic+prosody architecture is sound
- ✅ Can proceed to cross-synthesis testing (Priority 3)

### If WavLM loss still stuck:
- ❌ Problem is deeper (architecture, data, or training dynamics)
- ❌ May need: separate WavLM projections, different quantizer structure, or longer training

## Run Command

```bash
uv run python scripts/train_wavlm_centered.py
```

## Monitoring

View live metrics at: https://wandb.ai/yuuart/amy_wavlm_centered

Look for:
1. `train/wavlm` dropping from initial ~0.8-0.9 toward <0.4
2. `train/rvq_lr` showing warmup from 0 to 5e-5 over first 100 steps
3. `val/wavlm` following training trend (not diverging)
