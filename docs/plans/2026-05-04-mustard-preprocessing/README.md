# MUStARD++ Preprocessing — Implementation Plan

> **For agentic workers:** Use subagent-driven-development or executing-plans. Start with Phase 1 — don't read ahead to Phase 2/3 until Phase 1 completes.

## Quick Status

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 - Preprocessing | 🔲 **Not Started** | Preprocessed `.pt` files ready | [phase-01-preprocessing.md](./phase-01-preprocessing.md) |
| 2 - Embeddings & Fusion | 🔲 **Stub Only** | Embedding tables + λ gate working | [phase-02-embeddings-fusion.md](./phase-02-embeddings-fusion.md) |
| 3 - Training & Matrix | 🔲 **Stub Only** | MUStARD++ training + 8 benchmarks | [phase-03-training-matrix.md](./phase-03-training-matrix.md) |

## Start Here

New implementer? Read in this order:

1. **[strategy.md](./strategy.md)** — Understand the big picture (5 min)
   - Goal: Validate Extension Architecture hypothesis
   - Architecture: FACodec + MOSS-Audio with residual fusion
   - Phases: Preprocessing → Embeddings/Fusion → Training/Matrix

2. **[phase-01-preprocessing.md](./phase-01-preprocessing.md)** — Only the current phase (30 min)
   - 10 tasks from setup → integration tests
   - Each task: failing test → implementation → passing test → commit
   - Full implementation details in `OLD-full-plan.md` (lines 117-1865)

3. **[decisions.md](./decisions.md)** — Context on choices made (optional, 5 min)

**Do NOT read Phase 2 or 3 yet.** They're intentionally stubbed and will be detailed after Phase 1 completes and we incorporate learnings.

## What This Plan Implements

This is **Issue #5: Amy LM Pilot Validation** — Phase 1 only.

Phase 1 delivers a preprocessing pipeline that:
1. Downloads MUStARD++ dataset from HuggingFace
2. Runs FACodec offline → prosody indices (80 Hz) + timbre vectors (256-dim)
3. Runs MOSS-Audio offline → semantic frames (12.5 Hz, 2560-dim)
4. Aligns temporal rates via pooling (80 Hz → 12.5 Hz)
5. Saves `.pt` files per utterance with all features + labels

**Output format:**
```python
{
    'utterance_id': str,
    'semantic_frames': (N_sem, 2560),      # MOSS-Audio at 12.5 Hz
    'prosody_indices': (N_sem, 1),          # FACodec prosody pooled
    'prosody_indices_raw': (N_pros,),        # Original 80 Hz
    'timbre_vector': (256,),                # Global timbre
    'label': int,                           # 0 or 1 (sarcasm)
    'duration_sec': float,
    'alignment_info': {'prosody_frames', 'semantic_frames', 'pooling_ratio'},
    'metadata': dict
}
```

## Key Decisions (so far)

See [decisions.md](./decisions.md) for full rationale on:
- Offline preprocessing (encoders run once, not in training loop)
- Pooling strategy (average pooling 6:1 for 80 Hz → 12.5 Hz)
- Mock fallbacks (tests work without Amphion/transformers installed)

## Execution Options

Once you're ready to implement Phase 1:

**Option A: Subagent-Driven (recommended)**
- I dispatch a fresh subagent per task
- Review between tasks
- Good for complex or unfamiliar tasks

**Option B: Inline Execution**
- Execute tasks in this session
- Faster for straightforward tasks
- Good when you're already familiar with the codebase

**Which approach do you prefer for Phase 1?**

---

*Plan initialized with progressive disclosure. Strategy and Phase 1 are detailed; Phases 2+ will be expanded after Phase 1 completion.*
