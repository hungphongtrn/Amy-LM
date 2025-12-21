# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Amy-LM** (Semantic Bridge: Disentangling Content and Prosody in Neural Audio Codecs) is a research project that modifies the Mimi neural audio codec architecture to disentangle semantic content (text) from prosody (acoustic features) and timbre. The goal is to create a 9-codebook model where:
- **Codebook 0**: Represents pure semantic content (aligned with LLM hidden states)
- **Codebook 1**: Represents prosody (captures residual prosodic information)
- **Codebooks 0+1 combined**: Aligned with WavLM features (which contain both semantic and prosodic information)
- **Codebooks 2-8**: Handle acoustic residuals and timbre reconstruction

## Architecture

### Core Components

1. **Modified Mimi Model** (`src/models/mimi/`):
   - 9-codebook architecture with hierarchical disentanglement
   - Codebook 0 locked to Qwen LLM hidden states (semantic anchor)
   - Codebooks 0+1 combined locked to WavLM features (prosody anchor)
   - Codebooks 2-8 for acoustic reconstruction

2. **Data Processing Pipeline** (`src/data/`):
   - Distributed across 4 machines for parallel processing
   - Uses Hugging Face Hub as synchronization layer
   - Pipeline: Alignment → WavLM extraction → LLM extraction → Merge/Upload

3. **Training System** (`src/trainer/`):
   - Two-phase training: reconstruction followed by adversarial fine-tuning
   - Multiple loss components: adversarial, feature matching, LLM/WavLM distillation, multi-scale spectrogram reconstruction
   - Curriculum learning with "acoustic dropout" to force dependency on semantic head

### Key Directories

- `src/` - Main source code
  - `data/` - Distributed data processing scripts
  - `models/mimi/` - Modified Mimi implementation
  - `trainer/` - Training utilities and loss functions
- `data/` - Processed datasets and audio files
- `checkpoints/` - Model checkpoints
- `docs/` - Research documentation and training records
- `notebooks/` - Jupyter notebooks for exploration
- `moshi/` - Forked Moshi codebase (speech-text foundation model)

## Development Setup

### Package Management
This project uses `uv` for fast Python dependency management:

```bash
# Install uv if not present
pip install uv

# Install core dependencies
uv sync

# Install optional dependency groups for distributed processing:
uv sync --extra alignment    # Machine A (NeMo alignment)
uv sync --extra wavlm        # Machine B (WavLM extraction)
uv sync --extra llm          # Machine C (LLM extraction)
uv sync --extra upload       # Machine D (dataset upload)
```

### Environment Variables
- `HF_TOKEN`: Hugging Face authentication token (required for dataset upload/download)
- Stored in `.env` file

## Common Commands

### Data Processing (Distributed)

The data pipeline is designed to run on 4 separate machines:

**Machine A: Alignment (NeMo)**
```bash
# System dependencies
sudo apt-get install ffmpeg libsndfile1 clang

# Run alignment
uv run src/data/1_extract_alignment.py
```

**Machine B: WavLM Extraction**
```bash
uv run src/data/2_extract_wavlm.py
```

**Machine C: LLM Extraction**
```bash
uv run src/data/3_extract_llm.py
```

**Machine D: Upload/Merge**
```bash
uv run src/data/push_to_hub.py
```

### Training

```bash
# Main training script
python train.py
```

The trainer uses PyTorch Lightning with WandB logging. Configuration is in `train.py` and uses `CompressorTrainerConfig` from `src/trainer/compressor_trainer.py`.

### Notebooks
- `notebooks/data_explore.ipynb` - Data exploration
- `notebooks/inference.ipynb` - Inference experiments
- `notebooks/mimi_frenkeinstein.ipynb` - Voice swapping experiments

## Research Context

### Key Concepts
- **Semantic Bridge**: Architecture that enforces hierarchical disentanglement of text, prosody, and timbre
- **Teacher-Student Distillation**: Codebook 0 predicts Qwen LLM hidden states, Codebooks 0+1 combined predict WavLM features
- **Acoustic Dropout**: Curriculum learning technique to force model dependency on semantic head

### Target Metrics
- Frame rate: 12.5 Hz (80ms per frame)
- Bitrate: ~1.2 kbps
- Intelligibility: WER < 5% on reconstructed audio using only Codebooks 0+1

### Capabilities
- Zero-shot voice transfer (mixing Codebook 0 from Source A with Codebooks 1-8 from Source B)
- Text-based audio editing
- Disentangled representations for downstream tasks

## Technical Stack

- **PyTorch** + **Lightning** for model training
- **Hugging Face** ecosystem for models and datasets
- **WandB** for experiment tracking
- **NeMo** for speech alignment
- **Transformers** for WavLM and Qwen models

## Notes for Development

1. **High Memory Requirements**: Data processing requires 512GB+ system RAM and 24GB+ VRAM
2. **Distributed Workflow**: Data pipeline is designed for parallel execution across multiple machines
3. **Research Focus**: This is an experimental research project - expect rapid architectural changes
4. **Moshi Integration**: The `moshi/` directory contains a forked version of the Moshi speech-text foundation model for reference/experimentation

## Recent Changes

Current branch: `exp/residual_prosody`
Recent modification: Added `.detach()` to real samples in feature matching loss (`src/trainer/adversarial_losses.py`)
Recent commit: "exp: Model prosody as residual of semantic quantizer"