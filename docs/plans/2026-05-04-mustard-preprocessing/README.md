# Issue #6: MUStARD++ Preprocessing Pipeline

> Implementation plan for preprocessing MUStARD++ dataset with FACodec + MOSS-Audio encoders.

## Status

🔲 **Not Started**

## Quick Summary

This implements **Issue #6**: A preprocessing pipeline that downloads MUStARD++ dataset, runs FACodec and MOSS-Audio encoders offline, and saves aligned feature tuples as `.pt` files.

**Output per utterance:**
```python
{
    'semantic_frames': (N_sem, 2560),       # MOSS-Audio at 12.5 Hz
    'prosody_indices': (N_sem, 1),          # FACodec prosody (pooled)
    'prosody_indices_raw': (N_pros,),       # Original 80 Hz
    'timbre_vector': (256,),                # Global timbre
    'label': int,                           # Sarcasm: 0 or 1
    'alignment_info': {...},                # Frame counts, pooling ratio
}
```

## Start Here

Read **[phase-01-preprocessing.md](./phase-01-preprocessing.md)** for the 10-task implementation plan.

## What This Implements

- [ ] MUStARD++ dataset downloaded from HuggingFace
- [ ] FACodec encoder: prosody indices (80 Hz) + timbre vector (256-dim)
- [ ] MOSS-Audio encoder: semantic frames (12.5 Hz, 2560-dim)
- [ ] Temporal alignment via pooling (80 Hz → 12.5 Hz)
- [ ] `.pt` files per utterance with all features + labels
- [ ] Summary report with statistics and failures

## Key Decisions

See [decisions.md](./decisions.md) for:
- Offline preprocessing (encoders run once, not in training loop)
- Pooling strategy (average pooling 6:1 for 80 Hz → 12.5 Hz)
- Mock fallbacks (tests work without Amphion/transformers installed)

## Execution

Once ready, implement Phase 1 tasks sequentially (test-first):

```bash
# Task order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10
# Each task: write failing test → implement → verify pass → commit
```

Validation:
```bash
python scripts/preprocess_mustard.py --split test --max-utterances 5
```

---

*This plan implements GitHub Issue #6: MUStARD++ Preprocessing Pipeline*
