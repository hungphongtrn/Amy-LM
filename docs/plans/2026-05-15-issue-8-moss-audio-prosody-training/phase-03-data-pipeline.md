# Phase 3: Data Pipeline

> **STUB** — Will be detailed after Phase 1 completes and we have verified MOSS-Audio audio format expectations.

## Phase Goal
MUStARD audio is FACodec-preprocessed to parquet, and a PyTorch `Dataset`/`DataLoader` yields correct batch dicts (audio, prosody indices, timbre vector, labels) for training.

## High-Level Tasks (TBD)
- Run `scripts/preprocess.py` on `data/mustard_dataset/` to produce FACodec-encoded parquet
- Create `src/data/mustard_dataset.py` — PyTorch `Dataset` class loading parquet
- Implement `collate_fn` for variable-length audio + padded FACodec streams
- Train/val/test split logic
- Test: dataset loading, batch shapes, label distribution

## Dependencies
- Phase 1: MOSS-Audio Backbone Integration (for audio format verification)
- Existing: `scripts/prepare_mustard_dataset.py` (already run, `data/mustard_dataset/` exists)
- Existing: `src/preprocessing/` pipeline (complete)
- Prerequisite: FACodec checkpoints downloaded to `checkpoints/facodec/`
