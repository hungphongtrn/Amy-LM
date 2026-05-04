# Isolation Experiment A: Semantic Detach Test

**Date**: 2026-04-07
**Purpose**: Test if WavLM can learn when LLM projection is fed from a detached pathway
**Status**: Planned

---

## Hypothesis

The WavLM projection fails because **LLM and WavLM objectives compete** for the same `rvq_first` bottleneck. The LLM objective dominates because it has a clearer gradient signal (next-token prediction vs. frame-level feature matching).

**Test**: Detach `rvq_first` before feeding to LLM projection, forcing gradient to flow only through WavLM projection for semantic learning.

---

## Methodology

### Changes from Priority 2B

In the `training_step`, modify line 152:

```python
# BEFORE (Priority 2B):
sem_latent = sem_res.x.transpose(1, 2)
sem_llm_proj = self.llm_proj(sem_latent)
loss_llm = 1.0 - F.cosine_similarity(sem_llm_proj, llm_feat, dim=-1).mean()

# AFTER (Isolation A):
sem_latent = sem_res.x.transpose(1, 2)
sem_latent_detached = sem_latent.detach()  # Stop gradient to rvq_first
sem_llm_proj = self.llm_proj(sem_latent_detached)
loss_llm = 1.0 - F.cosine_similarity(sem_llm_proj, llm_feat, dim=-1).mean()
```

This means:
- `loss_llm` backpropagates only to `llm_proj` parameters, NOT to `rvq_first`
- `loss_wavlm` backpropagates through both `wavlm_proj` AND `rvq_first`
- Semantic representation learning is driven **only** by WavLM objective

### Full Configuration

```python
trainer_config = CompressorTrainerConfig(
    orignal_filename=mimi_weights_path,
    mimi_config=mimi_config,
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_codebooks=9,
    wavlm_dim=1024,
    llm_dim=2048,
    adversarial_only=False,
    projections_only=False,
    semantic_prosody_only=True,
    alpha_adv=0.0,
    alpha_feat=0.0,
    alpha_msspec=5.0,
    alpha_wavlm=5.0,
    alpha_llm=1.0,  # Keep LLM loss (on detached pathway)
)
```

Keep all other settings:
- MLP projection for WavLM (512→GELU→LayerNorm→1024)
- Both-sided centering for WavLM cosine loss
- RVQ LR=5e-5 with 100-step warmup

---

## Expected Outcomes

### If Hypothesis Correct (Conflict exists)
- **WavLM loss**: Drops significantly to <0.50 within 200-500 steps
- **LLM loss**: Stays high (~0.7+) because no gradient flows to rvq_first
- **Interpretation**: Confirms that LLM was dominating the bottleneck

### If Hypothesis Wrong (No conflict)
- **WavLM loss**: Remains stuck at ~0.93-0.95
- **LLM loss**: Stays high (~0.7+) 
- **Interpretation**: Problem is elsewhere (e.g., WavLM projection architecture, target mismatch, etc.)

---

## Success Criteria

| Metric | Target | Decision |
|--------|--------|----------|
| train/wavlm_epoch < 0.50 | ✅ | Move to Isolation B or integration |
| train/wavlm_epoch > 0.90 | ❌ | Problem is elsewhere, need deeper fix |

---

## Time Budget

- **Max epochs**: 3-5 epochs (~100-180 steps)
- **Expected time**: ~1-2 hours
- **Stop criteria**: If no WavLM improvement by epoch 3, stop and run Isolation B

---

## Implementation

**Script**: `scripts/train_semantic_detach.py`
**WandB Project**: `amy_semantic_detach`
**Checkpoints**: `checkpoints/semantic_detach/`

---

## Next Steps After This Test

### If Succeeds
Run **Isolation Experiment B** (alpha_llm=0) to confirm WavLM works without LLM entirely.

### If Fails
Consider deeper architectural changes:
- Move WavLM projection to **before** RVQ (work on continuous latent)
- Increase projection capacity significantly
- Check WavLM target extraction pipeline

---

## Related Documents

- Stop decision: `docs/experiment-changelogs/2026-04-07-wavlm-centered-mlp-STOPPED.md`
- Priority 2B planning: `docs/planning/2026-04-07-wavlm-centered-mlp-both-sided.md`
