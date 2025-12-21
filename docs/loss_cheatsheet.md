# Amy-LM Training Loss Cheat Sheet

This cheat sheet helps interpret loss values during training of the Amy-LM neural audio codec. Use it to understand model performance and identify issues.

**Architecture Note**: Prosody is modeled as a residual of the semantic quantizer. Codebook 0 (semantic) is aligned with LLM features, while codebooks 0+1 combined (semantic+prosody) are aligned with WavLM features.

## Quick Reference Table

| Loss Metric | Good Range | Warning Range | Critical Range | What It Measures |
|-------------|------------|---------------|----------------|------------------|
| **llm_epoch** | < 0.1 | 0.1 - 0.3 | > 0.3 | Semantic alignment with Qwen LLM (1 - cosine similarity) |
| **wavlm_epoch** | < 0.2 | 0.2 - 0.5 | > 0.5 | Semantic+prosody alignment with WavLM features (1 - cosine similarity) |
| **msspec_epoch** | < 0.1 | 0.1 - 0.5 | > 0.5 | Multi-scale spectrogram reconstruction quality |
| **feat_epoch** | < 0.05 | 0.05 - 0.2 | > 0.2 | Feature matching between real/fake in discriminator |
| **adv_g_epoch** | 0.1 - 0.5 | 0.5 - 1.0 | > 1.0 | Generator's ability to fool discriminator |
| **loss_d_epoch** | 1.5 - 2.0 | 0.5 - 1.5 or 2.0 - 3.0 | < 0.5 or > 3.0 | Discriminator's ability to distinguish real/fake |
| **loss_g_epoch** | 1.0 - 3.0 | 3.0 - 5.0 | > 5.0 | Total generator loss (weighted sum) |

## Loss Component Details

### 1. Distillation Losses (Semantic & Prosody)

#### LLM Distillation (`llm_epoch`)
- **Formula**: `1.0 - cosine_similarity(codebook_0, qwen_features)`
- **Purpose**: Measures how well codebook 0 captures semantic content
- **Target**: Cosine similarity > 0.9 (loss < 0.1)
- **Good value**: ~0.086 (cosine similarity ~0.914)
- **Critical issue**: Loss > 0.3 means semantic content is not being captured

#### WavLM Distillation (`wavlm_epoch`)
- **Formula**: `1.0 - cosine_similarity(codebooks_0+1, wavlm_features)`
- **Purpose**: Measures how well the combined semantic+prosody representation (codebooks 0+1) matches WavLM features (which contain both semantic and prosodic information)
- **Target**: Cosine similarity > 0.9 (loss < 0.1)
- **Current challenge**: Often high (~0.693, similarity ~0.307)
- **Critical issue**: Loss > 0.5 indicates poor alignment of semantic+prosody with WavLM features, suggesting prosody modeling issues

### 2. Reconstruction Loss (`msspec_epoch`)
- **Purpose**: Multi-scale mel spectrogram reconstruction (5 scales: 64-1024 n_fft)
- **Characteristics**: L1 loss on linear spectrograms + weighted MSE on log spectrograms
- **During adaptation**: Should steadily decrease (high weight: alpha_msspec=15.0)
- **During adversarial fine-tuning**: May be disabled (`adversarial_only=True`)
- **Good value**: < 0.1 indicates accurate waveform reconstruction

### 3. Adversarial Losses

#### Generator Adversarial Loss (`adv_g_epoch`)
- **Purpose**: Generator's ability to fool the discriminator
- **Hinge loss formulation**: `-D(fake_samples).mean()`
- **Expected**: Should decrease as generator improves
- **Watch for**: Too low (< 0.1) may indicate discriminator is weak; too high (> 1.0) may indicate training instability

#### Discriminator Loss (`loss_d_epoch`)
- **Purpose**: Discriminator's ability to distinguish real from fake
- **Hinge loss formulation**: `max(0, 1 - D(real)).mean() + max(0, 1 + D(fake)).mean()`
- **Expected stable range**: 1.7-1.8 (not overpowering generator)
- **Warning signs**:
  - < 0.5: Discriminator too weak (mode collapse risk)
  - > 3.0: Discriminator too strong (vanishing gradients)

#### Feature Matching Loss (`feat_epoch`)
- **Purpose**: Aligns internal representations between real and fake samples in discriminator
- **Formula**: L1 distance between discriminator feature maps
- **Good value**: < 0.05 indicates good representation alignment
- **High value**: > 0.2 suggests poor internal feature matching

### 4. Total Losses

#### Total Generator Loss (`loss_g_epoch`)
- **Formula**: `alpha_adv*adv_g + alpha_feat*feat + alpha_wavlm*wavlm + alpha_llm*llm + alpha_msspec*msspec`
- **Default weights** (adaptation phase):
  - `alpha_adv = 1.0`
  - `alpha_feat = 4.0`
  - `alpha_msspec = 15.0` (high for adaptation)
  - `alpha_wavlm = 1.0`
  - `alpha_llm = 1.0`

## Training Phase Indicators

### Phase 1: Adaptation/Warmup
- `adversarial_only = False`
- `alpha_msspec = 15.0` (high)
- **Expected patterns**:
  - `msspec_epoch` decreasing steadily
  - `llm_epoch` and `wavlm_epoch` should improve
  - `loss_d_epoch` may be unstable initially

### Phase 2: Adversarial Fine-tuning
- `adversarial_only = True` (reconstruction disabled)
- **Expected patterns**:
  - `adv_g_epoch` and `loss_d_epoch` should stabilize
  - `feat_epoch` should remain low
  - Distillation losses continue to improve

## Example: Analyzing Your Training Log

Given:
```
train/adv_g_epoch:0.32181718945503235
train/feat_epoch:0.016863815486431122
train/llm_epoch:0.07964769005775452
train/loss_d_epoch:1.9732297658920288
train/loss_g_epoch:2.255605936050415
train/msspec_epoch:0.07455965131521225
train/wavlm_epoch:0.6682911515235901
```

### Analysis:
1. **✅ LLM distillation**: 0.079 (good, cosine similarity ~0.92)
2. **⚠️ WavLM distillation**: 0.668 (critical, cosine similarity ~0.33) - semantic+prosody alignment issue (prosody modeling)
3. **✅ Feature matching**: 0.017 (excellent alignment)
4. **✅ Discriminator**: 1.97 (stable hinge GAN range)
5. **✅ Generator adversarial**: 0.32 (reasonable)
6. **✅ Reconstruction**: 0.075 (good if in adaptation phase)
7. **Total generator loss**: 2.26 (reasonable given high WavLM loss)

### Diagnosis:
- **Primary issue**: Semantic+prosody alignment with WavLM features is failing (prosody capture)
- **Possible solutions**:
  1. Increase `alpha_wavlm` weight
  2. Improve WavLM projection architecture (MLP instead of linear)
  3. Ensure `adversarial_only=False` to maintain reconstruction guidance

## Common Issues & Solutions

### High WavLM Loss (> 0.5)
- **Cause**: Simple linear projection insufficient for complex WavLM features
- **Fix**: Replace `nn.Linear` with MLP (Linear→ReLU→Linear) in `wavlm_proj`
- **Alternative**: Increase `alpha_wavlm` to 2.0-5.0

### Discriminator Too Strong (loss_d < 0.5)
- **Cause**: Discriminator overpowering generator
- **Fix**: Reduce discriminator capacity or increase `alpha_adv`

### Discriminator Too Weak (loss_d > 3.0)
- **Cause**: Generator fooling discriminator too easily
- **Fix**: Increase discriminator capacity or learning rate

### LLM Loss Increasing
- **Cause**: Semantic content degradation
- **Fix**: Increase `alpha_llm` or reduce adversarial weight

### Feature Matching High (> 0.2)
- **Cause**: Poor internal representation alignment
- **Fix**: Ensure discriminator features are being properly extracted

## Performance Benchmarks

### Target Metrics (End of Training)
- **LLM cosine similarity**: > 0.95 (loss < 0.05)
- **WavLM cosine similarity**: > 0.90 (loss < 0.10)
- **Reconstruction loss**: < 0.05
- **Discriminator loss**: 1.5-2.0 (stable hinge GAN)
- **Audio quality**: WER < 5% on reconstructed speech

### Current Best (from training records)
- **LLM**: 0.086 loss (0.914 similarity) - Good
- **WavLM**: 0.693 loss (0.307 similarity) - Needs improvement
- **Discriminator**: 1.7-1.8 loss - Stable

## Monitoring Tips

1. **Track cosine similarities directly**: Convert losses to similarities: `similarity = 1 - loss`
2. **Visualize loss ratios**: Calculate `adv_g / loss_g` ratio (should be ~0.3-0.5)
3. **Monitor WavLM/LLM ratio**: `wavlm_loss / llm_loss` should approach 1.0
4. **Regular audio logging**: Essential when `msspec` is disabled
5. **Check discriminator balance**: Both real and fake accuracy should be ~50-70%

## Configuration Reference

### Loss Weights (train.py:51-56)
```python
alpha_adv=1.0      # Adversarial loss weight
alpha_feat=4.0     # Feature matching weight
alpha_msspec=15.0  # Reconstruction weight (adaptation phase)
alpha_wavlm=1.0    # WavLM distillation weight
alpha_llm=1.0      # LLM distillation weight
```

### Training Phases
```python
# Phase 1: Adaptation (reconstruction focused)
adversarial_only=False
alpha_msspec=15.0

# Phase 2: Adversarial fine-tuning (quality focused)
adversarial_only=True  # Disables msspec
alpha_msspec=0.0       # Or keep small weight (2.0-5.0)
```

---

*Last updated: 2025-12-21*
*Based on code analysis of Amy-LM commit 7f31ef6*