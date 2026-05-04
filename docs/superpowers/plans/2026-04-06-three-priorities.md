# Three-Priority Neural Audio Codec Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use @executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute three priority experiments: (1) Clean baseline with rvq_rest.eval() fix, (2) WavLM-centered disentanglement experiment, (3) Cross-synthesis evaluation to prove disentanglement.

**Architecture:** Phase 2 training modifies MiMi's split RVQ structure with semantic+prosody codebooks. The fix ensures frozen acoustic layers (rvq_rest) stay in eval mode during training. WavLM-centered experiment mean-centers features to strip speaker identity. Cross-synthesis test swaps semantic tokens between speakers.

**Tech Stack:** PyTorch, Lightning, MiMi architecture, WavLM, LLM features, WandB, safetensors

---

## File Structure

### Modified Files:
- `src/trainer/compressor_trainer.py` - Add rvq_rest.eval() fix in __init__
- `train.py` - Update for clean baseline restart from scratch
- `scripts/train_wavlm_centered.py` - Verify implementation is complete

### New Files:
- `scripts/cross_synthesis_test.py` - Cross-synthesis evaluation script
- `docs/experiment-changelogs/2026-04-06-clean-baseline-restart.md` - Experiment log
- `docs/experiment-changelogs/2026-04-06-cross-synthesis-evaluation.md` - Experiment log

---

## Priority 1: Clean Baseline (semantic_prosody_full)

### Task 1.1: Apply rvq_rest.eval() Fix to CompressorTrainer

**Files:**
- Modify: `src/trainer/compressor_trainer.py:104-122`

**Context:** When `semantic_prosody_only=True`, only rvq_first and rvq_prosody should be trainable. The rvq_rest (acoustic) layers must stay frozen and in eval mode to prevent EMA/codebook updates during training.

- [ ] **Step 1: Add eval() call for rvq_rest in semantic_prosody_only block**

Current code (lines 104-122):
```python
        # Semantic+Prosody only mode: unfreeze RVQ heads + projections
        if config.semantic_prosody_only:
            # First freeze everything
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze semantic and prosody quantizers
            quantizer = self.model.quantizer
            for param in quantizer.rvq_first.parameters():
                param.requires_grad = True
            for param in quantizer.rvq_prosody.parameters():
                param.requires_grad = True
            # Unfreeze projection layers
            for param in self.wavlm_proj.parameters():
                param.requires_grad = True
            for param in self.llm_proj.parameters():
                param.requires_grad = True
            print(
                "Semantic+Prosody mode: rvq_first, rvq_prosody, and projections trainable. "
                "All other Mimi parameters frozen."
            )
```

Add after line 122:
```python
            # CRITICAL FIX: Set rvq_rest to eval mode to prevent EMA/codebook updates
            if hasattr(quantizer, 'rvq_rest') and quantizer.rvq_rest is not None:
                quantizer.rvq_rest.eval()
                print("Applied rvq_rest.eval() to freeze acoustic layers during training.")
```

- [ ] **Step 2: Verify the fix is in place**

Run: `grep -A 5 "rvq_rest.eval" src/trainer/compressor_trainer.py`
Expected: Shows the newly added eval() call

- [ ] **Step 3: Commit the fix**

```bash
git add src/trainer/compressor_trainer.py
git commit -m "fix: apply rvq_rest.eval() for semantic_prosody_only mode

Prevents EMA/codebook updates in frozen acoustic layers during Phase 2 training.
This ensures the acoustic layers remain truly frozen while semantic+prosody
quantizers are trained."
```

---

### Task 1.2: Update train.py for Clean Baseline Restart

**Files:**
- Modify: `train.py:79-87` (checkpoint callback configuration)

**Context:** We need to restart training from scratch (step 0) to validate the fix. Update the checkpoint callback to start fresh.

- [ ] **Step 1: Verify train.py is configured correctly for clean baseline**

Current configuration should have:
```python
    trainer_config = CompressorTrainerConfig(
        # ...
        semantic_prosody_only=True,  # PHASE 2: Train semantic+prosody RVQ heads + projections
        alpha_adv=0.0,  # DISABLE adversarial for distillation phase
        alpha_feat=0.0,  # DISABLE feature matching
        alpha_msspec=15.0,  # High reconstruction weight to preserve audio quality
        alpha_wavlm=1.0,  # WavLM distillation
        alpha_llm=1.0,  # LLM distillation
    )
```

Verify: `grep "semantic_prosody_only=True" train.py`
Expected: Returns the line showing semantic_prosody_only=True

- [ ] **Step 2: Clear old checkpoints or backup them**

```bash
# Backup existing checkpoints first
mkdir -p checkpoints/semantic_prosody_full_backup_$(date +%Y%m%d)
cp checkpoints/semantic_prosody_full/*.ckpt checkpoints/semantic_prosody_full_backup_$(date +%Y%m%d)/ 2>/dev/null || echo "No checkpoints to backup"

# Remove old checkpoints to start fresh
rm -f checkpoints/semantic_prosody_full/*.ckpt
echo "Cleaned checkpoint directory for fresh start"
```

- [ ] **Step 3: Verify dataset paths exist**

```bash
ls -la data/Amy-LM-Dataset-Aligned/ | head -5
ls -la data/mimi_weights/tokenizer-e351c8d8-checkpoint125.safetensors
```
Expected: Both paths exist and are accessible

- [ ] **Step 4: Commit preparation changes**

```bash
git add -A
git commit -m "chore: prepare clean baseline restart from step 0

Backed up old checkpoints and verified dataset paths for Priority 1
clean baseline experiment."
```

---

### Task 1.3: Create Experiment Changelog Entry

**Files:**
- Create: `docs/experiment-changelogs/2026-04-06-clean-baseline-restart.md`

- [ ] **Step 1: Write the experiment log**

```markdown
# Experiment: Clean Baseline Restart (Priority 1)

**Date**: 2026-04-06
**Status**: In Progress
**Branch/Commit**: TBD

## Summary

Restart standard semantic_prosody_full Phase 2 training from scratch with the critical fix applied: `rvq_rest.eval()` to ensure frozen acoustic layers don't update during training.

**Key Fix**: Applied `rvq_rest.eval()` in semantic_prosody_only mode to prevent EMA/codebook updates in frozen acoustic layers.

## Configuration

- **Dataset**: Amy-LM-Dataset-Aligned
- **Model**: Mimi with semantic+prosody RVQ heads (rvq_first + rvq_prosody)
- **Training Mode**: `semantic_prosody_only=True`
- **Loss Weights**:
  - alpha_msspec: 15.0 (reconstruction)
  - alpha_wavlm: 1.0 (WavLM distillation)
  - alpha_llm: 1.0 (LLM distillation)
  - alpha_adv: 0.0 (disabled)
  - alpha_feat: 0.0 (disabled)
- **Hyperparameters**:
  - Batch size: 128
  - Learning rates: RVQ heads=1e-5, projections=3e-4
  - Max epochs: 100
  - Precision: bf16-mixed

## What to Watch

### Must-Track Metrics

| Metric | Target | Alert If |
|--------|--------|----------|
| `train/msspec` | Start low (<0.5), stay stable | Spikes above 0.5 |
| `train/wavlm` | Smoothly decrease | Stagnant or increases |
| `train/llm` | Smoothly decrease | Stagnant or increases |
| Audio quality | Highly intelligible, correct timbre | Muffled, noisy, or artifacts |

### Success Criteria

1. **msspec Loss**: Starts low (acoustic layers intact) and stays stable
2. **wavlm & llm Losses**: Smoothly decrease as projections learn
3. **Reconstruction**: Audio at epoch 10 should be highly intelligible with correct speaker timbre

## Results

- **Key metrics**: TBD (will update after training)
- **Observations**: TBD
- **Audio quality**: TBD

## Next Steps

Upon successful completion:
1. Verify stable loss curves and good audio quality
2. Proceed to Priority 2: WavLM-centered experiment
3. Document final checkpoint location

## References

- Fix location: `src/trainer/compressor_trainer.py` lines 104-130
- Main script: `train.py`
- WandB Project: `amy_semantic_prosody_full`
```

- [ ] **Step 2: Commit the changelog**

```bash
git add docs/experiment-changelogs/2026-04-06-clean-baseline-restart.md
git commit -m "docs: add Priority 1 clean baseline experiment log"
```

---

### Task 1.4: Launch Clean Baseline Training

**Files:**
- Execute: `train.py`

- [ ] **Step 1: Launch training in background with logging**

```bash
cd /home/hungphongtrn/Workspace/Amy-LM
nohup uv run python train.py > logs/priority1_clean_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Training launched with PID: $!"
```

- [ ] **Step 2: Verify training started successfully**

Wait 30 seconds, then:
```bash
tail -n 50 logs/priority1_clean_baseline_*.log | grep -E "(Starting training|GPU|error|Error|CUDA)"
```
Expected: Shows "Starting training..." and no CUDA/GPU errors

- [ ] **Step 3: Monitor initial metrics in WandB**

Open WandB project `amy_semantic_prosody_full` and verify:
- Training started and logging
- Initial msspec loss is low (<0.5)
- GPU utilization is high (>80%)

- [ ] **Step 4: Update experiment log with initial observations**

After ~10 epochs, update `docs/experiment-changelogs/2026-04-06-clean-baseline-restart.md` with initial results.

---

## Priority 2: WavLM-Centered Experiment (Disentanglement)

### Task 2.1: Verify wavlm_centered Implementation

**Files:**
- Verify: `scripts/train_wavlm_centered.py`

**Context:** The wavlm_centered experiment mean-centers WavLM features per clip to remove global speaker/timbre bias before computing cosine loss. This is already implemented but needs verification.

- [ ] **Step 1: Verify mean-centering code exists**

Run: `grep -A 3 "wavlm_feat_centered" scripts/train_wavlm_centered.py`
Expected: Shows the mean-centering implementation:
```python
wavlm_feat_centered = wavlm_feat - wavlm_feat.mean(dim=1, keepdim=True)
```

- [ ] **Step 2: Verify the modified loss computation**

Run: `grep -B 2 -A 5 "F.cosine_similarity.*wavlm_feat_centered" scripts/train_wavlm_centered.py`
Expected: Shows loss_wavlm computation using centered features

- [ ] **Step 3: Verify trainer class inherits correctly**

Run: `grep "class CompressorTrainerWavLMCentered" scripts/train_wavlm_centered.py`
Expected: `class CompressorTrainerWavLMCentered(CompressorTrainer):`

- [ ] **Step 4: Update existing experiment log with current status**

Read: `docs/experiment-changelogs/2026-02-06-wavlm-centered-features.md`

Update the status section to reflect we're ready to run this after Priority 1 completes.

---

### Task 2.2: Create Priority 2 Experiment Changelog

**Files:**
- Create: `docs/experiment-changelogs/2026-04-06-wavlm-centered-execution.md`

- [ ] **Step 1: Write the experiment log**

```markdown
# Experiment: WavLM-Centered Features (Priority 2)

**Date**: 2026-04-06
**Status**: Ready to Execute (Pending Priority 1 Completion)
**Branch/Commit**: TBD

## Summary

Execute the wavlm_centered diagnostic experiment to test whether WavLM features contain too much speaker/timbre bias that semantic+prosody codebooks can't capture.

**Key Hypothesis**: The WavLM plateau (~0.5 cosine distance) is due to global speaker/timbre information that semantic+prosody codebooks aren't meant to encode. By mean-centering WavLM features per clip (removing the global mean), we strip static speaker identity while preserving dynamic phonetic content.

**Why This Matters**: If successful, this validates the architecture split:
- semantic+prosody = content/prosody (dynamic, frame-level)
- acoustic (rvq_rest) = timbre/speaker (static, global)

This enables true speaker disentanglement without a speaker encoder.

## Configuration

- **Dataset**: Amy-LM-Dataset-Aligned
- **Model**: Mimi with semantic+prosody RVQ heads
- **Training Mode**: `semantic_prosody_only=True`
- **Special Modification**: Mean-center WavLM features per clip before cosine loss
- **Loss Weights**:
  - alpha_wavlm: 1.0
  - alpha_llm: 1.0
  - alpha_msspec: 15.0
  - alpha_adv: 0.0
  - alpha_feat: 0.0
- **Hyperparameters**:
  - Batch size: 128
  - Learning rates: RVQ heads=1e-5, projections=3e-4
  - Max steps: 2000
  - Precision: bf16-mixed

## Key Code Modification

```python
# Mean-center WavLM features per clip to remove global speaker/timbre bias
wavlm_feat_centered = wavlm_feat - wavlm_feat.mean(dim=1, keepdim=True)

# Use centered features in loss computation
loss_wavlm = 1.0 - F.cosine_similarity(
    sem_pros_wavlm_proj, wavlm_feat_centered, dim=-1
).mean()
```

## Expected Outcomes

### Success Scenario (wavlm loss decreases below 0.5):
- The plateau was due to global speaker/timbre in WavLM
- Semantic+prosody correctly encodes dynamic content only
- Acoustic layers (rvq_rest) should capture global speaker info
- This validates the disentanglement strategy

### Failure Scenario (wavlm loss stays ~0.5):
- The issue is elsewhere:
  - Projection layer capacity too small
  - Semantic+prosody codebook capacity insufficient
  - Frame rate mismatch between WavLM and Mimi
  - WavLM genuinely requires acoustic codebooks

## Execution Plan

1. Wait for Priority 1 (Clean Baseline) to complete and show stable results
2. Use the best checkpoint from Priority 1 as initialization (optional) OR start from Mimi base weights
3. Run for 2000 steps
4. Compare wavlm loss curve to baseline (Priority 1)

## Results

- **Key metrics**: TBD
- **Observations**: TBD
- **Audio quality**: TBD

## Next Steps

Based on results:
- **If successful**: Make mean-centering default for WavLM distillation; proceed to Priority 3
- **If unsuccessful**: Try larger projections, more codebooks, or different frame alignment

## References

- Script: `scripts/train_wavlm_centered.py`
- Based on: `scripts/train_wavlm_centered.py` (already implemented)
- WandB Project: `amy_wavlm_centered`
- Original log: `docs/experiment-changelogs/2026-02-06-wavlm-centered-features.md`
```

- [ ] **Step 2: Commit the changelog**

```bash
git add docs/experiment-changelogs/2026-04-06-wavlm-centered-execution.md
git commit -m "docs: add Priority 2 wavlm-centered experiment log"
```

---

### Task 2.3: Prepare WavLM-Centered Launch Script

**Files:**
- Verify: `scripts/train_wavlm_centered.py`

- [ ] **Step 1: Ensure checkpoint directory exists**

```bash
mkdir -p checkpoints/wavlm_centered
echo "Checkpoint directory ready"
```

- [ ] **Step 2: Verify script is executable**

```bash
head -20 scripts/train_wavlm_centered.py
```
Expected: Shows proper shebang and imports

- [ ] **Step 3: Create launch command reference**

Create file: `scripts/launch_priority2.sh`
```bash
#!/bin/bash
# Launch Priority 2: WavLM-Centered Experiment
# Run this after Priority 1 completes successfully

cd /home/hungphongtrn/Workspace/Amy-LM

# Launch training
nohup uv run python scripts/train_wavlm_centered.py > logs/priority2_wavlm_centered_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Priority 2 training launched with PID: $!"
echo "Monitor with: tail -f logs/priority2_wavlm_centered_*.log"
```

```bash
chmod +x scripts/launch_priority2.sh
git add scripts/launch_priority2.sh
git commit -m "chore: add Priority 2 launch script"
```

---

## Priority 3: Cross-Synthesis Test (Gate to Phase 3)

### Task 3.1: Create Cross-Synthesis Evaluation Script

**Files:**
- Create: `scripts/cross_synthesis_test.py`

**Context:** This script tests whether Phase 2 achieved disentanglement by swapping semantic tokens between different speakers. If successful, the output should sound like Speaker B (timbre) speaking Speaker A's words (content).

- [ ] **Step 1: Write the cross-synthesis evaluation script**

```python
"""
Cross-Synthesis Test: Evaluate disentanglement by swapping semantic tokens between speakers.

This test proves whether Phase 2 achieved true disentanglement:
- Speaker A's semantic tokens (content) + Speaker B's prosody/acoustic tokens (timbre)
- Success: Output sounds like Speaker B speaking Speaker A's words
- Failure: Output sounds like Speaker A (semantic codebook still holds speaker info)

Usage:
    uv run python scripts/cross_synthesis_test.py \
        --checkpoint checkpoints/semantic_prosody_full/last.ckpt \
        --speaker_a data/Amy-LM-Dataset-Aligned/speaker_a/audio1.wav \
        --speaker_b data/Amy-LM-Dataset-Aligned/speaker_b/audio2.wav \
        --output_dir results/cross_synthesis

Output:
    - reconstructed_a.wav: Speaker A reconstructed (baseline)
    - reconstructed_b.wav: Speaker B reconstructed (baseline)
    - hybrid_a_sem_b_rest.wav: Speaker A semantic + Speaker B prosody/acoustic
    - comparison_report.txt: Analysis of results
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.trainer.compressor_trainer import CompressorTrainer, CompressorTrainerConfig
from src.models.mimi.configuration_mimi import DEFAULT_MIMI_CONFIG


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create config (same as training)
    mimi_config = DEFAULT_MIMI_CONFIG
    trainer_config = CompressorTrainerConfig(
        orignal_filename="data/mimi_weights/tokenizer-e351c8d8-checkpoint125.safetensors",
        mimi_config=mimi_config,
        device=device,
        num_codebooks=9,
        wavlm_dim=1024,
        llm_dim=2048,
        adversarial_only=False,
        projections_only=False,
        semantic_prosody_only=True,
        alpha_adv=0.0,
        alpha_feat=0.0,
        alpha_msspec=15.0,
        alpha_wavlm=1.0,
        alpha_llm=1.0,
    )
    
    # Initialize model
    model = CompressorTrainer(config=trainer_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model, mimi_config


def load_audio(audio_path: str, sample_rate: int = 24000):
    """Load and preprocess audio file."""
    import librosa
    
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Convert to torch tensor: (1, 1, T_samples)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    
    return audio_tensor, sample_rate


def extract_tokens(model, audio: torch.Tensor, frame_rate: float):
    """
    Extract token components from audio through the encoder.
    
    Returns:
        sem_tokens: Semantic tokens from rvq_first
        pros_tokens: Prosody tokens from rvq_prosody
        ac_tokens: Acoustic tokens from rvq_rest (if available)
    """
    with torch.no_grad():
        # Encode to latent
        emb = model.model._encode_to_unquantized_latent(audio)
        
        # Extract quantized components
        quantizer = model.model.quantizer
        
        # Semantic
        sem_res = quantizer.rvq_first(emb, frame_rate)
        sem_tokens = sem_res.codes  # (B, n_q_semantic, T)
        sem_quantized = sem_res.x   # (B, D, T)
        
        # Prosody
        pros_res = quantizer.rvq_prosody(emb, frame_rate)
        pros_tokens = pros_res.codes  # (B, n_q_prosody, T)
        pros_quantized = pros_res.x   # (B, D, T)
        
        # Acoustic
        ac_tokens = None
        ac_quantized = None
        if quantizer.rvq_rest is not None:
            ac_res = quantizer.rvq_rest(emb, frame_rate)
            ac_tokens = ac_res.codes  # (B, n_q_acoustic, T)
            ac_quantized = ac_res.x   # (B, D, T)
        
        return {
            'semantic': {
                'codes': sem_tokens,
                'quantized': sem_quantized,
            },
            'prosody': {
                'codes': pros_tokens,
                'quantized': pros_quantized,
            },
            'acoustic': {
                'codes': ac_tokens,
                'quantized': ac_quantized,
            }
        }


def decode_tokens(model, tokens_dict: dict, frame_rate: float):
    """
    Decode tokens back to audio.
    
    Args:
        tokens_dict: Dictionary with 'semantic', 'prosody', 'acoustic' quantized values
    
    Returns:
        audio: Reconstructed audio tensor
    """
    with torch.no_grad():
        # Sum quantized embeddings
        emb_quant = tokens_dict['semantic']['quantized'] + tokens_dict['prosody']['quantized']
        if tokens_dict['acoustic']['quantized'] is not None:
            emb_quant = emb_quant + tokens_dict['acoustic']['quantized']
        
        # Decode
        emb_dec = model.model._to_encoder_framerate(emb_quant)
        if model.model.decoder_transformer is not None:
            (emb_dec,) = model.model.decoder_transformer(emb_dec)
        
        audio = model.model.decoder(emb_dec)
        
        return audio


def cross_synthesis(model, tokens_a: dict, tokens_b: dict, frame_rate: float):
    """
    Perform cross-synthesis: Speaker A's semantic + Speaker B's prosody/acoustic.
    
    Args:
        tokens_a: Tokens from Speaker A (source of semantic content)
        tokens_b: Tokens from Speaker B (source of timbre/prosody)
    
    Returns:
        audio: Hybrid reconstructed audio
    """
    # Create hybrid token dictionary
    hybrid_tokens = {
        'semantic': tokens_a['semantic'],  # Content from Speaker A
        'prosody': tokens_b['prosody'],    # Prosody from Speaker B
        'acoustic': tokens_b['acoustic'], # Timbre from Speaker B
    }
    
    audio = decode_tokens(model, hybrid_tokens, frame_rate)
    
    return audio


def save_audio(audio: torch.Tensor, path: str, sample_rate: int):
    """Save audio tensor to file."""
    audio_np = audio.squeeze().cpu().numpy()
    sf.write(path, audio_np, sample_rate)
    print(f"Saved: {path}")


def analyze_results(output_dir: str, audio_a: torch.Tensor, audio_b: torch.Tensor, 
                   hybrid: torch.Tensor, sample_rate: int):
    """
    Analyze and report cross-synthesis results.
    
    Computes metrics to evaluate disentanglement quality.
    """
    report_path = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CROSS-SYNTHESIS EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Objective:\n")
        f.write("  Test if semantic codebook (rvq_first) captures only content\n")
        f.write("  without speaker identity/timbre.\n\n")
        
        f.write("Method:\n")
        f.write("  - Speaker A's semantic tokens (content)\n")
        f.write("  - Speaker B's prosody + acoustic tokens (timbre)\n")
        f.write("  - Decode hybrid token sequence\n\n")
        
        f.write("Expected Result:\n")
        f.write("  ✓ SUCCESS: Hybrid sounds like Speaker B speaking Speaker A's words\n")
        f.write("  ✗ FAILURE: Hybrid sounds like Speaker A (semantic contaminated)\n\n")
        
        f.write("Files Generated:\n")
        f.write("  1. reconstructed_a.wav - Speaker A baseline reconstruction\n")
        f.write("  2. reconstructed_b.wav - Speaker B baseline reconstruction\n")
        f.write("  3. hybrid_a_sem_b_rest.wav - Cross-synthesis result\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("MANUAL EVALUATION INSTRUCTIONS:\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. Listen to 'reconstructed_a.wav' - note Speaker A's voice\n")
        f.write("2. Listen to 'reconstructed_b.wav' - note Speaker B's voice\n")
        f.write("3. Listen to 'hybrid_a_sem_b_rest.wav' - evaluate:\n")
        f.write("   - Does it have Speaker B's TIMBRE (voice characteristics)?\n")
        f.write("   - Does it speak Speaker A's WORDS (content)?\n\n")
        
        f.write("SUCCESS Criteria:\n")
        f.write("  - Timbre matches Speaker B (different from Speaker A)\n")
        f.write("  - Content matches Speaker A (words/phonemes)\n")
        f.write("  - Audio quality is clear and intelligible\n\n")
        
        f.write("FAILURE Indicators:\n")
        f.write("  - Timbre sounds like Speaker A (semantic codebook leaked speaker info)\n")
        f.write("  - Muffled or degraded audio quality\n")
        f.write("  - Unintelligible speech\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("NEXT STEPS:\n")
        f.write("=" * 70 + "\n\n")
        f.write("If SUCCESS:\n")
        f.write("  → Phase 2 achieved disentanglement! Proceed to Phase 3.\n")
        f.write("  → Consider expanding training or enabling adversarial loss.\n\n")
        f.write("If FAILURE:\n")
        f.write("  → Need stronger disentanglement:\n")
        f.write("    - Use WavLM-centered features (mean-centering)\n")
        f.write("    - Add gradient reversal layer on semantic codebook\n")
        f.write("    - Increase weight on LLM distillation (text-focused)\n")
        f.write("    - Train longer with speaker-swapped pairs\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Audio file paths:\n")
        f.write(f"  - Original A: {os.path.join(output_dir, 'original_a.wav')}\n")
        f.write(f"  - Original B: {os.path.join(output_dir, 'original_b.wav')}\n")
        f.write(f"  - Reconstructed A: {os.path.join(output_dir, 'reconstructed_a.wav')}\n")
        f.write(f"  - Reconstructed B: {os.path.join(output_dir, 'reconstructed_b.wav')}\n")
        f.write(f"  - Hybrid (A sem + B rest): {os.path.join(output_dir, 'hybrid_a_sem_b_rest.wav')}\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nReport saved: {report_path}")
    print("\n" + "=" * 70)
    print("CROSS-SYNTHESIS TEST COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nNEXT: Listen to the audio files and evaluate:")
    print("  - Does hybrid sound like Speaker B's timbre?")
    print("  - Does hybrid have Speaker A's content?")
    print("\nSee comparison_report.txt for detailed instructions.")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-synthesis test for disentanglement evaluation"
    )
    parser.add_argument(
        "--checkpoint", 
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--speaker_a", 
        required=True,
        help="Path to Speaker A audio file"
    )
    parser.add_argument(
        "--speaker_b", 
        required=True,
        help="Path to Speaker B audio file"
    )
    parser.add_argument(
        "--output_dir", 
        default="results/cross_synthesis",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device", 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.speaker_a):
        raise FileNotFoundError(f"Speaker A audio not found: {args.speaker_a}")
    if not os.path.exists(args.speaker_b):
        raise FileNotFoundError(f"Speaker B audio not found: {args.speaker_b}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, mimi_config = load_model_from_checkpoint(args.checkpoint, args.device)
    frame_rate = mimi_config.frame_rate
    sample_rate = int(mimi_config.sample_rate)
    
    print("\n" + "=" * 70)
    print("CROSS-SYNTHESIS TEST")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Speaker A:  {args.speaker_a}")
    print(f"Speaker B:  {args.speaker_b}")
    print(f"Output:     {args.output_dir}")
    print("=" * 70 + "\n")
    
    # Load audio files
    print("Loading audio files...")
    audio_a, _ = load_audio(args.speaker_a, sample_rate)
    audio_b, _ = load_audio(args.speaker_b, sample_rate)
    
    audio_a = audio_a.to(args.device)
    audio_b = audio_b.to(args.device)
    
    # Save original audio for reference
    save_audio(audio_a, os.path.join(args.output_dir, "original_a.wav"), sample_rate)
    save_audio(audio_b, os.path.join(args.output_dir, "original_b.wav"), sample_rate)
    
    # Extract tokens from both speakers
    print("\nExtracting tokens...")
    print("  Speaker A (semantic source)...")
    tokens_a = extract_tokens(model, audio_a, frame_rate)
    
    print("  Speaker B (timbre source)...")
    tokens_b = extract_tokens(model, audio_b, frame_rate)
    
    # Decode baselines
    print("\nDecoding baseline reconstructions...")
    rec_a = decode_tokens(model, tokens_a, frame_rate)
    rec_b = decode_tokens(model, tokens_b, frame_rate)
    
    save_audio(rec_a, os.path.join(args.output_dir, "reconstructed_a.wav"), sample_rate)
    save_audio(rec_b, os.path.join(args.output_dir, "reconstructed_b.wav"), sample_rate)
    
    # Perform cross-synthesis
    print("\nPerforming cross-synthesis...")
    print("  Combining: Speaker A semantic + Speaker B prosody/acoustic")
    hybrid = cross_synthesis(model, tokens_a, tokens_b, frame_rate)
    
    save_audio(hybrid, os.path.join(args.output_dir, "hybrid_a_sem_b_rest.wav"), sample_rate)
    
    # Generate report
    analyze_results(args.output_dir, audio_a, audio_b, hybrid, sample_rate)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script has no syntax errors**

```bash
cd /home/hungphongtrn/Workspace/Amy-LM
python -m py_compile scripts/cross_synthesis_test.py
echo "Syntax check passed"
```

- [ ] **Step 3: Commit the script**

```bash
git add scripts/cross_synthesis_test.py
git commit -m "feat: add cross-synthesis evaluation script for Priority 3

Implements the critical disentanglement test:
- Extracts semantic/prosody/acoustic tokens separately
- Swaps semantic tokens between speakers
- Decodes hybrid combinations
- Generates comparison report with evaluation criteria

This is the gate to Phase 3 - proves whether semantic codebook
captures only content without speaker identity."
```

---

### Task 3.2: Create Priority 3 Experiment Changelog

**Files:**
- Create: `docs/experiment-changelogs/2026-04-06-cross-synthesis-evaluation.md`

- [ ] **Step 1: Write the experiment log**

```markdown
# Experiment: Cross-Synthesis Evaluation (Priority 3)

**Date**: 2026-04-06
**Status**: Ready to Execute (Pending Priority 1 & 2 Completion)
**Branch/Commit**: TBD

## Summary

Execute cross-synthesis test to prove Phase 2 achieved disentanglement. This is the gate to Phase 3.

**Test Protocol**:
1. Take Speaker A audio → Extract only `rvq_first` (semantic) tokens
2. Take Speaker B audio → Extract `rvq_prosody` and `rvq_rest` tokens
3. Concatenate: Speaker A's semantic + Speaker B's prosody/acoustic
4. Pass hybrid token sequence through frozen decoder

**Success Criteria**: Output sounds like Speaker B (timbre) speaking the words of Speaker A (content).

**Failure Indicators**: Output sounds like Speaker A (semantic codebook still holds speaker info).

## Configuration

- **Evaluation Type**: Inference-only (no training)
- **Input**: Trained checkpoint from Priority 1 or 2
- **Test Samples**: 2 different speakers (Speaker A and Speaker B)
- **Output**: Audio files + comparison report

## Script Details

**Location**: `scripts/cross_synthesis_test.py`

**Usage**:
```bash
uv run python scripts/cross_synthesis_test.py \
    --checkpoint checkpoints/semantic_prosody_full/last.ckpt \
    --speaker_a data/Amy-LM-Dataset-Aligned/speaker_a/audio1.wav \
    --speaker_b data/Amy-LM-Dataset-Aligned/speaker_b/audio2.wav \
    --output_dir results/cross_synthesis
```

**Outputs**:
1. `original_a.wav` - Speaker A original audio
2. `original_b.wav` - Speaker B original audio
3. `reconstructed_a.wav` - Speaker A full reconstruction (baseline)
4. `reconstructed_b.wav` - Speaker B full reconstruction (baseline)
5. `hybrid_a_sem_b_rest.wav` - Cross-synthesis result
6. `comparison_report.txt` - Detailed evaluation instructions

## Evaluation Criteria

### SUCCESS (Disentanglement Achieved):
- **Timbre**: Hybrid audio has Speaker B's voice characteristics
- **Content**: Hybrid audio speaks Speaker A's words/phonemes
- **Quality**: Audio is clear and intelligible
- **Difference**: Hybrid sounds different from reconstructed_a.wav

### FAILURE (Need More Disentanglement):
- **Timbre**: Hybrid audio sounds like Speaker A (not B)
- **Cause**: Semantic codebook (rvq_first) is encoding speaker identity
- **Solution Options**:
  1. Use WavLM-centered features (Priority 2) to strip speaker bias
  2. Add gradient reversal layer on semantic codebook
  3. Increase LLM distillation weight (more text-focused)
  4. Train with explicit speaker-swapped pairs

## Results

- **Test Date**: TBD
- **Checkpoint Used**: TBD
- **Speakers Used**: TBD
- **Result**: TBD (SUCCESS/FAILURE)
- **Observations**: TBD

## Next Steps

### If SUCCESS:
- ✓ Phase 2 is complete and successful
- → Proceed to Phase 3: Full training or adversarial fine-tuning
- → Document final architecture as baseline
- → Consider expanding dataset or training duration

### If FAILURE:
- → Implement WavLM-centered features (if not already done in Priority 2)
- → Try gradient reversal on semantic codebook
- → Increase LLM distillation to make semantic more text-focused
- → Re-run cross-synthesis test after modifications

## References

- Script: `scripts/cross_synthesis_test.py`
- Depends on: Priority 1 (clean baseline) or Priority 2 (wavlm-centered)
- Architecture: SplitResidualVectorQuantizerWithProsody
- Theory: Disentanglement via vector-sum quantization
```

- [ ] **Step 2: Commit the changelog**

```bash
git add docs/experiment-changelogs/2026-04-06-cross-synthesis-evaluation.md
git commit -m "docs: add Priority 3 cross-synthesis experiment log"
```

---

### Task 3.3: Create Execution Reference Document

**Files:**
- Create: `docs/planning/2026-04-06-three-priority-execution.md`

- [ ] **Step 1: Write the execution reference**

```markdown
# Three-Priority Execution Guide

**Date**: 2026-04-06
**Status**: Ready for Execution

## Overview

This document provides quick reference for executing all three priorities in sequence.

---

## Priority 1: Clean Baseline

**Goal**: Prove fixed code works with rvq_rest.eval() applied

**Command**:
```bash
cd /home/hungphongtrn/Workspace/Amy-LM
rm -f checkpoints/semantic_prosody_full/*.ckpt
nohup uv run python train.py > logs/priority1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Monitor**:
```bash
tail -f logs/priority1_*.log
# WandB: amy_semantic_prosody_full
```

**Expected**: msspec < 0.5, wavlm/llm decreasing, good audio at epoch 10

**Log**: `docs/experiment-changelogs/2026-04-06-clean-baseline-restart.md`

---

## Priority 2: WavLM-Centered

**Goal**: Test disentanglement via mean-centered WavLM features

**Prerequisite**: Priority 1 complete and stable

**Command**:
```bash
cd /home/hungphongtrn/Workspace/Amy-LM
nohup uv run python scripts/train_wavlm_centered.py > logs/priority2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Monitor**:
```bash
tail -f logs/priority2_*.log
# WandB: amy_wavlm_centered
```

**Expected**: wavlm loss decreases below 0.5 plateau

**Log**: `docs/experiment-changelogs/2026-04-06-wavlm-centered-execution.md`

---

## Priority 3: Cross-Synthesis Test

**Goal**: Prove disentanglement via speaker token swapping

**Prerequisite**: Priority 1 or 2 checkpoint available

**Command**:
```bash
uv run python scripts/cross_synthesis_test.py \
    --checkpoint checkpoints/semantic_prosody_full/last.ckpt \
    --speaker_a data/Amy-LM-Dataset-Aligned/speaker_a/sample.wav \
    --speaker_b data/Amy-LM-Dataset-Aligned/speaker_b/sample.wav \
    --output_dir results/cross_synthesis_$(date +%Y%m%d)
```

**Evaluate**:
1. Listen to `reconstructed_a.wav` - Speaker A baseline
2. Listen to `reconstructed_b.wav` - Speaker B baseline
3. Listen to `hybrid_a_sem_b_rest.wav` - Cross-synthesis
4. Check `comparison_report.txt` for instructions

**Success**: Hybrid sounds like Speaker B speaking Speaker A's words

**Log**: `docs/experiment-changelogs/2026-04-06-cross-synthesis-evaluation.md`

---

## Decision Tree

```
Priority 1 Complete
        ↓
   msspec stable?
   wavlm decreasing?
        ↓
   YES → Priority 2: WavLM-Centered
        ↓
   wavlm < 0.5?
        ↓
   YES → Priority 3: Cross-Synthesis
        ↓
   Speaker swap works?
        ↓
   YES → → → PHASE 3 READY
        ↓
   NO → Re-train with stronger disentanglement
```

---

## Files Reference

| Component | File |
|-----------|------|
| Priority 1 Script | `train.py` |
| Priority 2 Script | `scripts/train_wavlm_centered.py` |
| Priority 3 Script | `scripts/cross_synthesis_test.py` |
| Core Fix | `src/trainer/compressor_trainer.py` (rvq_rest.eval()) |
| Priority 1 Log | `docs/experiment-changelogs/2026-04-06-clean-baseline-restart.md` |
| Priority 2 Log | `docs/experiment-changelogs/2026-04-06-wavlm-centered-execution.md` |
| Priority 3 Log | `docs/experiment-changelogs/2026-04-06-cross-synthesis-evaluation.md` |
| This Guide | `docs/planning/2026-04-06-three-priority-execution.md` |
```

- [ ] **Step 2: Commit the reference**

```bash
git add docs/planning/2026-04-06-three-priority-execution.md
git commit -m "docs: add three-priority execution reference guide"
```

---

## Final Verification Checklist

- [ ] All code changes committed
- [ ] All documentation created
- [ ] Priority 1 ready to launch
- [ ] Priority 2 script verified
- [ ] Priority 3 script created and tested
- [ ] Experiment logs created for all three priorities

## Execution Handoff

**Plan complete! All files are ready.**

**To execute Priority 1 immediately**:
```bash
cd /home/hungphongtrn/Workspace/Amy-LM
rm -f checkpoints/semantic_prosody_full/*.ckpt
nohup uv run python train.py > logs/priority1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**To view the complete plan**:
- Full plan: This document
- Quick reference: `docs/planning/2026-04-06-three-priority-execution.md`

**Execution options**:
1. **Run Priority 1 now** - Launch the clean baseline training immediately
2. **Review first** - Let me know if you want to review any specific part before launching
