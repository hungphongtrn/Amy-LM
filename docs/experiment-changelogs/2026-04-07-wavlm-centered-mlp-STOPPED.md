# STOP DECISION: WavLM-Centered-MLP Experiment

**Date**: 2026-04-07
**Status**: STOPPED (Failed stop criteria)
**WandB Run**: https://wandb.ai/pedestrian313/amy_wavlm_centered_mlp/runs/sg6lr95w
**Checkpoint**: `checkpoints/wavlm_centered_mlp/amy-wavlm-centered-mlp-epoch=04-step=000180.ckpt`

---

## Summary

The **Priority 2B** experiment (both-sided WavLM centering + MLP projection) has been running for **5 epochs** (~7.5 hours) and has **failed to break the WavLM plateau**.

Per the stop criteria from the planning document, we must stop the run and pivot to **isolation experiments**.

---

## Key Metrics at Epoch 5

| Metric | Initial (Epoch 0) | Current (Epoch 5) | Target | Status |
|--------|------------------|-------------------|--------|--------|
| train/wavlm_epoch | ~0.97 | **0.935** | <0.50 | ❌ **FAIL** |
| train/llm_epoch | ~0.74 | **0.104** | <0.50 | ✅ PASS |
| train/msspec_epoch | ~0.12 | **0.099** | <0.10 | ⚠️ Borderline |
| val/wavlm | ~0.98 | **0.967** | <0.50 | ❌ **FAIL** |
| val/llm | ~0.77 | **0.098** | <0.50 | ✅ PASS |

---

## Failure Analysis

### What We Tried
The Priority 2B experiment implemented the following fixes:
1. ✅ Both-sided centering (center both predicted AND target WavLM)
2. ✅ MLP projection (512→GELU→LayerNorm→1024) instead of Linear
3. ✅ RVQ LR=5e-5 with 100-step warmup
4. ✅ alpha_msspec=5.0, alpha_wavlm=5.0, alpha_llm=1.0

### What Happened
The same failure pattern as before:
- **WavLM**: Stuck at ~0.935 (only 3.5% improvement from initial ~0.97)
- **LLM**: Dropped to 0.104 (86% improvement from initial ~0.74)

### Diagnosis
Both-sided centering + MLP projection provided **marginal improvement** (~0.01-0.02 loss reduction) but fundamentally **did NOT solve the WavLM learning problem**. The issue is **deeper than representation normalization**.

### Root Cause Hypothesis
Per the planning document analysis, the problem is likely:
> **RVQ-first is already capturing enough semantic information to satisfy the LLM projection, making WavLM projection redundant or conflicting.**

When both objectives compete for the same bottleneck representation (rvq_first), the LLM objective dominates because:
1. LLM target is a direct next-token prediction task (clearer gradient signal)
2. WavLM target is a frame-level feature matching task (more nuanced)
3. The architecture shares the same rvq_first → projection pathway

---

## Stop Criteria Triggered

From `docs/planning/2026-04-07-wavlm-centered-mlp-both-sided.md`:

> **Stop Criteria**: 
> - If WavLM remains stuck near 0.9+ after 200-500 steps, stop and move to isolation experiments

**Epoch 5 = ~180 steps**, WavLM = 0.935, showing no meaningful downward trend.

**Decision**: STOP immediately and pivot to isolation experiments.

---

## Next Actions

Per the planning document, we will run **two isolation experiments** to confirm the conflict:

### Isolation Experiment A: Semantic Detach Test
**Goal**: Test if WavLM can learn when LLM projection is fed from a detached (non-differentiated) pathway.

**Changes**:
- Detach `rvq_first` before feeding to LLM projection
- Gradient only flows: rvq_first → wavlm_proj → wavlm_loss
- LLM projection becomes a "read-only" branch

**Expected outcome if hypothesis correct**:
- WavLM loss should drop significantly (<0.5)
- LLM loss will stay high (no gradient flow), which is acceptable for this diagnostic

### Isolation Experiment B: Alpha LLM = 0 Test
**Goal**: Test if WavLM projection works when LLM loss is completely disabled.

**Changes**:
- Set `alpha_llm = 0.0` (remove LLM loss entirely)
- Keep WavLM + MSSpec + Reconstruction losses

**Expected outcome if hypothesis correct**:
- WavLM loss should drop to <0.5
- Reconstruction quality should be maintained

### Decision Tree
- **If both isolation tests fail**: Architecture problem → consider moving WavLM before RVQ
- **If isolation test A succeeds but B fails**: Gradient conflict confirmed → need separate pathways
- **If both succeed**: Re-integrate with architectural changes

---

## Time Invested

- Priority 2 (centered only): ~8 epochs → WavLM ~0.94
- Priority 2B (MLP + both-sided): ~5 epochs → WavLM ~0.935

**Total time on WavLM centering approach**: ~13 epochs, ~16 hours

**Lesson**: The representation-level fixes were insufficient. We need architectural-level investigation.

---

## Planning Document

Full analysis and next steps:
- `docs/planning/2026-04-07-wavlm-centered-mlp-both-sided.md`

Related experiments:
- `docs/experiment-changelogs/2026-04-06-wavlm-centered-improved.md`
- `docs/experiment-changelogs/2026-04-06-training-audit-wavlm-stuck.md`
