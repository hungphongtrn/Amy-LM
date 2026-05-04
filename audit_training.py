#!/usr/bin/env python3
"""Quick script to audit the current training run."""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.trainer.compressor_data_loader import CompressorDataLoader, BatchInputData
from src.models.mimi.configuration_mimi import DEFAULT_MIMI_CONFIG
from src.models.mimi.modeling_mimi import (
    get_mimi_with_prosody_from_original_mimi_weights,
)

# Configuration
DATASET_PATH = "data/Amy-LM-Dataset-Aligned"
MIMI_WEIGHTS_PATH = "data/mimi_weights/tokenizer-e351c8d8-checkpoint125.safetensors"

print("=" * 60)
print("TRAINING AUDIT: Checking data and model")
print("=" * 60)

# 1. Check data module
data_module = CompressorDataLoader(
    data_path=DATASET_PATH,
    batch_size=4,  # Small batch for testing
    num_workers=0,  # No workers for debugging
    sample_rate=24000,
    fps=12.5,
    segment_duration=1.0,
)
data_module.setup()

print(f"\n[DATA]")
print(f"  Train set size: {len(data_module.train_ds)}")
print(f"  Val set size: {len(data_module.val_ds)}")

# Get one batch
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

print(f"\n[BATCH SHAPES]")
print(f"  Audio: {batch.audio.shape}")
print(f"  WavLM features: {batch.wavlm_feat.shape}")
print(f"  LLM features: {batch.llm_feat.shape}")

# 2. Check feature statistics
print(f"\n[WAVLM FEATURE STATISTICS]")
print(f"  Mean: {batch.wavlm_feat.mean():.4f}")
print(f"  Std: {batch.wavlm_feat.std():.4f}")
print(f"  Min: {batch.wavlm_feat.min():.4f}")
print(f"  Max: {batch.wavlm_feat.max():.4f}")

print(f"\n[LLM FEATURE STATISTICS]")
print(f"  Mean: {batch.llm_feat.mean():.4f}")
print(f"  Std: {batch.llm_feat.std():.4f}")
print(f"  Min: {batch.llm_feat.min():.4f}")
print(f"  Max: {batch.llm_feat.max():.4f}")

# 3. Check model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[MODEL - Device: {device}]")

model = get_mimi_with_prosody_from_original_mimi_weights(
    MIMI_WEIGHTS_PATH,
    DEFAULT_MIMI_CONFIG,
    device,
    num_codebooks=9,
)
model.eval()

# Move batch to device
audio = batch.audio.to(device)
wavlm_feat = batch.wavlm_feat.to(device)
llm_feat = batch.llm_feat.to(device)

# 4. Forward pass to check intermediate outputs
print(f"\n[FORWARD PASS - Baseline (no training)]")

with torch.no_grad():
    # Encode
    emb = model._encode_to_unquantized_latent(audio)
    print(f"  Encoded latent shape: {emb.shape}")

    # Get quantizer
    quantizer = model.quantizer

    # Get semantic output
    sem_res = quantizer.rvq_first(emb, 12.5)
    print(f"  Semantic output shape: {sem_res.x.shape}")
    print(f"  Semantic codes shape: {sem_res.codes.shape}")

    # Get prosody output
    pros_res = quantizer.rvq_prosody(emb, 12.5)
    print(f"  Prosody output shape: {pros_w.x.shape}")
    print(f"  Prosody codes shape: {pros_res.codes.shape}")

    # Check what projection would output
    wavlm_proj = torch.nn.Linear(model.dimension, 1024).to(device)
    llm_proj = torch.nn.Linear(model.dimension, 2048).to(device)

    # Project semantic
    sem_latent = sem_res.x.transpose(1, 2)  # (B, T, Dim)
    sem_llm_proj = llm_proj(sem_latent)

    # Project semantic + prosody
    sem_pros_latent = (sem_res.x + pros_res.x).transpose(1, 2)
    sem_pros_wavlm_proj = wavlm_proj(sem_pros_latent)

    print(f"\n[PROJECTION SHAPES]")
    print(f"  LLM proj output: {sem_llm_proj.shape}")
    print(f"  WavLM proj output: {sem_pros_wavlm_proj.shape}")

    # Compute initial losses (before training)
    import torch.nn.functional as F

    loss_llm = 1.0 - F.cosine_similarity(sem_llm_proj, llm_feat, dim=-1).mean()
    loss_wavlm = (
        1.0 - F.cosine_similarity(sem_pros_wavlm_proj, wavlm_feat, dim=-1).mean()
    )

    print(f"\n[INITIAL LOSSES (untrained projections)]")
    print(f"  LLM loss: {loss_llm.item():.4f}")
    print(f"  WavLM loss: {loss_wavlm.item():.4f}")
    print(f"  Note: 1.0 = completely dissimilar, 0.0 = identical")

    # Check norms
    print(f"\n[NORM COMPARISON]")
    print(f"  LLM target norm: {llm_feat.norm(dim=-1).mean():.4f}")
    print(f"  LLM proj norm: {sem_llm_proj.norm(dim=-1).mean():.4f}")
    print(f"  WavLM target norm: {wavlm_feat.norm(dim=-1).mean():.4f}")
    print(f"  WavLM proj norm: {sem_pros_wavlm_proj.norm(dim=-1).mean():.4f}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
