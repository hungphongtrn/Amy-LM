# Training Audit Report: WavLM Loss Stuck at High Values

**Date**: 2026-04-06  
**Training Run**: Priority 1 Clean Baseline (semantic_prosody_full)  
**Current Status**: Epoch 10/100, WavLM loss stuck at ~0.68-0.70

## Summary of Findings

### The Problem
The WavLM distillation loss is **stuck at approximately 0.68-0.70** (training) and **0.77-0.82** (validation). This is a **critical issue** because:

- **Loss interpretation**: `loss = 1.0 - cosine_similarity`
- **0.68 loss** = only ~32% cosine similarity between projected features and target WavLM features
- **Random projections** typically achieve ~0.5 cosine similarity (loss ~0.5)
- The model is performing **worse than random** on WavLM distillation!

### Historical Loss Progression
| Epoch | Train WavLM Loss | Notes |
|-------|------------------|-------|
| 0 | 0.826 | Initial (near random) |
| 1 | 0.712 | Some learning |
| 2 | 0.695 | Continued improvement |
| 3-10 | 0.68-0.70 | **STUCK** - No improvement for 7 epochs |

### Contrast with LLM Loss
| Loss Type | Value | Status |
|-----------|-------|--------|
| LLM (train) | ~0.08 | **Excellent** - 92% cosine similarity |
| LLM (val) | ~0.08 | Excellent |
| WavLM (train) | ~0.68 | **Terrible** - 32% similarity |
| WavLM (val) | ~0.77 | Even worse |

## Root Cause Analysis

### 1. **Architecture Mismatch** (Most Likely)
The current approach distills WavLM into **semantic + prosody combined**:
```python
sem_pros_latent = (sem_res.x + pros_res.x).transpose(1, 2)
sem_pros_wavlm_proj = self.wavlm_proj(sem_pros_latent)
```

**Problem**: WavLM contains **speaker identity information** that the frozen `rvq_rest` (acoustic) is supposed to capture. By freezing `rvq_rest`, we're asking the model to cram **speaker + prosody + acoustic details** into just semantic+prosody, which is fundamentally impossible.

### 2. **Learning Rate Imbalance**
Current configuration:
- RVQ heads (semantic + prosody): `lr = 1e-5`
- Projections: `lr = 3e-4`

The RVQ heads learn **very slowly** at 1e-5. After 10 epochs (360 steps), they may not have adapted enough.

### 3. **Loss Weighting Issues**
Current weights:
- `alpha_wavlm = 1.0`
- `alpha_msspec = 15.0`
- `alpha_llm = 1.0`

The **msspec loss dominates** (15x weight), potentially overwhelming the distillation signals.

### 4. **Projection Layer Capacity**
Simple Linear projection `nn.Linear(256, 1024)` may not have enough capacity to map from quantized representations to dense WavLM features.

## Evidence

From the training logs at Epoch 10:
```
train/wavlm_step: 0.691  # Step loss
train/wavlm_epoch: 0.678 # Epoch average
val/wavlm: 0.775         # Validation (even worse!)
```

The validation loss being higher than training suggests the model is **overfitting to poor local minima** rather than learning meaningful representations.

## Recommendations

### Immediate Actions

1. **Verify the architecture assumption**: Test if WavLM can actually be reconstructed from semantic+prosody without acoustic
   - If not, we need to include acoustic in WavLM distillation (but keep it frozen)
   - Or use a different target (e.g., WavLM minus speaker components)

2. **Increase learning rate for RVQ heads**:
   - Try `lr = 1e-4` or `5e-5` instead of `1e-5`
   - The quantizers need to adapt faster to capture the right information

3. **Add learning rate warmup**:
   - Current: No warmup
   - Suggested: 100-step warmup for RVQ heads

4. **Consider loss reweighting**:
   - Increase `alpha_wavlm` to 5.0 or 10.0
   - Or decrease `alpha_msspec` to allow more focus on distillation

### Alternative Approaches

If the above doesn't work:

1. **WavLM-Centered Training (Priority 2)**:
   - Use mean-centered WavLM features (removes speaker identity)
   - May be easier to distill since speaker info is stripped

2. **Progressive Training**:
   - Phase 1: Train only projections (frozen RVQ) - already done
   - Phase 2: Train only semantic RVQ head with LLM loss
   - Phase 3: Train prosody RVQ head with WavLM loss (freeze semantic)
   - Phase 4: Joint fine-tuning

3. **Architecture Change**:
   - Use separate projections for semantic→WavLM and prosody→WavLM
   - Combine them differently (e.g., weighted sum, attention)

## Conclusion

The training is **not converging** on the WavLM objective. After 10 epochs, we've made minimal progress and the validation loss is actually worse than training. This is a **showstopper** for Priority 1.

**Recommendation**: Stop the current training and address the root causes before continuing. The most promising path is to:
1. Try the WavLM-centered approach (Priority 2) which strips speaker information
2. Or increase learning rates and adjust loss weights in the current setup

Continuing the current training for 100 epochs is unlikely to yield meaningful improvements given the stagnation pattern.
