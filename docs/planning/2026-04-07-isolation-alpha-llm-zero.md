# Isolation Experiment B: Alpha LLM = 0 Test

**Date**: 2026-04-07
**Purpose**: Test if WavLM projection works when LLM loss is completely disabled
**Status**: Planned

---

## Hypothesis

The WavLM projection fails because **LLM objective dominates the shared bottleneck**. By removing LLM entirely, WavLM should be able to learn effectively if the projection architecture is fundamentally sound.

**Test**: Set `alpha_llm = 0.0` and observe if WavLM loss drops without LLM competition.

---

## Methodology

### Changes from Priority 2B

Simply change one parameter:

```python
trainer_config = CompressorTrainerConfig(
    ...
    alpha_llm=0.0,  # REMOVE LLM loss entirely
    alpha_wavlm=5.0,  # Keep WavLM
    alpha_msspec=5.0,  # Keep reconstruction
    ...
)
```

Keep all other settings:
- MLP projection for WavLM (512→GELU→LayerNorm→1024)
- Both-sided centering for WavLM cosine loss
- RVQ LR=5e-5 with 100-step warmup

---

## Expected Outcomes

### If Hypothesis Correct (LLM was blocking WavLM)
- **WavLM loss**: Drops significantly to <0.50 within 200-500 steps
- **LLM metrics**: N/A (disabled)
- **MSSpec**: Should remain stable (reconstruction maintained)
- **Interpretation**: Confirms LLM was the blocker

### If Hypothesis Wrong (WavLM projection is the problem)
- **WavLM loss**: Remains stuck at ~0.93-0.95 even without LLM
- **MSSpec**: May improve (all capacity goes to reconstruction)
- **Interpretation**: WavLM projection architecture or target mismatch is the root cause

---

## Success Criteria

| Metric | Target | Decision |
|--------|--------|----------|
| train/wavlm_epoch < 0.50 | ✅ | WavLM works alone, need to fix integration |
| train/wavlm_epoch > 0.90 | ❌ | WavLM projection fundamentally broken |

---

## Time Budget

- **Max epochs**: 3-5 epochs (~100-180 steps)
- **Expected time**: ~1-2 hours
- **Stop criteria**: If no WavLM improvement by epoch 3, stop

---

## Implementation

**Script**: `scripts/train_alpha_llm_zero.py`
**WandB Project**: `amy_alpha_llm_zero`
**Checkpoints**: `checkpoints/alpha_llm_zero/`

---

## Decision Matrix

After running **both** Isolation A and B:

| Isolation A Result | Isolation B Result | Interpretation | Next Action |
|-------------------|-------------------|----------------|-------------|
| ✅ WavLM drops | ✅ WavLM drops | Conflict confirmed + WavLM works alone | Need architectural fix for integration |
| ✅ WavLM drops | ❌ WavLM stuck | Gradient conflict confirmed, but WavLM needs LLM signal | Complex integration needed |
| ❌ WavLM stuck | ✅ WavLM drops | Detach broke something, WavLM works alone | Check detach implementation |
| ❌ WavLM stuck | ❌ WavLM stuck | Problem is elsewhere (not conflict) | Deeper investigation needed |

---

## Related Documents

- Stop decision: `docs/experiment-changelogs/2026-04-07-wavlm-centered-mlp-STOPPED.md`
- Isolation A planning: `docs/planning/2026-04-07-isolation-semantic-detach.md`
