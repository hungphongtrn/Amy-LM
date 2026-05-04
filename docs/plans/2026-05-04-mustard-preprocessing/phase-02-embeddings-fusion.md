# Phase 2: Embedding Tables & Residual Fusion

## Phase Goal

Trainable embedding tables with warm-start initialization, residual fusion module with λ gate, end-to-end forward pass working on preprocessed data.

**Success criteria:**
- ✓ Embedding lookup tables for FACodec prosody (1024 vocab → 2560-dim)
- ✓ Three init strategies implemented: random, FACodec warm-start via projection, continuous projector
- ✓ Temporal pooling module (swappable strategies) 
- ✓ Residual fusion: summation + λ gate + LayerNorm
- ✓ λ = 0 at init produces identity with MOSS-Audio input
- ✓ Integration tests pass

**Depends on:** Phase 1 completion (preprocessed `.pt` files)

---

## Stub — To Be Detailed After Phase 1

This phase will be fully detailed once Phase 1 completes and learnings are incorporated.

### Rough Scope

1. **Embedding Tables** (deep module)
   - Prosody embedding: 1024 vocab → 2560-dim
   - Three init strategies:
     - Random: N(0, 0.02)
     - FACodec warm-start: project FACodec codebook vectors 256→2560
     - Continuous projector: trainable Linear(256, 2560) mapping on-the-fly
   - Warm-start projection applied once during init

2. **Temporal Pooling** (deep module)
   - Interface: `pool(prosody_embedded_80hz, target_frames) -> pooled_12_5hz`
   - Default: average pooling 6:1
   - Swappable: max pooling, attention-based pooling

3. **Timbre Projection** (deep module)
   - Linear(256, 2560) or small MLP
   - Broadcast to all frames

4. **Residual Fusion** (deep module)
   - Formula: `H = LayerNorm(S + λ * (P + T))`
   - λ: zero-init learnable scalar
   - S: semantic (2560-dim), P: prosody (2560-dim), T: timbre (2560-dim)

5. **Integration Tests**
   - λ=0 produces identity (MOSS-Audio output unchanged)
   - Known inputs produce expected fused output within tolerance
   - Embedding init verification (warm-start vectors match projected FACodec)

### Files to Touch (anticipated)

- `src/amy_lm/embeddings.py` - Embedding tables with warm-start init
- `src/amy_lm/pooling.py` - Temporal pooling strategies
- `src/amy_lm/fusion.py` - Residual fusion with λ gate
- `src/amy_lm/model.py` - End-to-end Amy LM model
- `tests/amy_lm/test_*.py` - Unit tests for each module

### Open Questions to Resolve

1. Should FACodec warm-start embeddings be frozen or trainable?
2. Is a small MLP better than Linear for timbre projection?
3. Should pooling happen in DataLoader or forward pass?

---

*This document will be expanded with full task details after Phase 1 completion.*
