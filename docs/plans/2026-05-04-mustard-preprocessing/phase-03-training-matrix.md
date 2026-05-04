# Phase 3: Training Loop & Hypothesis Matrix

## Phase Goal

MUStARD++ training with observable λ growth, hypothesis matrix execution from simplest to complex, benchmark evaluation across 8 datasets.

**Success criteria:**
- ✓ Training loop with pluggable fusion into MOSS-Audio forward pass
- ✓ Three loss modes: classification head, LM loss, combined
- ✓ Three training modes: frozen MOSS-Audio, LoRA (rank 64), full fine-tune
- ✓ Hypothesis matrix runner: 3×3×3 with early stopping at diminishing returns
- ✓ Benchmark evaluator for all 8 datasets
- ✓ λ growth observable and logged
- ✓ In-domain (MUStARD++) and out-of-domain (8 benchmarks) metrics tracked

**Depends on:** Phase 2 completion (embedding tables + fusion working)

---

## Stub — To Be Detailed After Phase 2

This phase will be fully detailed once Phase 2 completes and learnings are incorporated.

### Rough Scope

1. **Training Loop** (shallow module)
   - Load preprocessed `.pt` files
   - Plug fusion module into MOSS-Audio forward pass
   - Support three loss objectives:
     - Classification head (Linear on pooled hidden states + CE)
     - LM loss (prompt + next-token prediction)
     - Combined (weighted sum)
   - Support three training strategies:
     - Frozen: only embeddings + λ trained
     - LoRA: rank 64 on all LLM layers
     - Full fine-tune: all parameters

2. **Hypothesis Matrix Runner** (shallow module)
   - 3 embedding init × 3 training × 3 loss = 27 combinations
   - Execution order: simplest → most complex
   - Stopping criteria: λ stays near zero (falsified) or diminishing returns
   - Config logging and metrics per run

3. **Benchmark Evaluator** (shallow module)
   - Load preprocessed data for each benchmark
   - Run inference through Amy LM
   - Compute task-specific metrics
   - 8 benchmarks: MUStARD++, MCR-Bench, MSP-Podcast, MSPB, SpeechWellness, SUPERB-prosody, NonVerbalSpeech-38K, Vox-Profile

4. **Metrics & Logging**
   - λ value tracked per epoch
   - In-domain: MUStARD++ accuracy/F1
   - Out-of-domain: benchmark suite average
   - Comparison: Amy LM vs MOSS-Audio baseline

### Files to Touch (anticipated)

- `src/amy_lm/trainer.py` - Training loop with strategy/loss switching
- `src/amy_lm/matrix_runner.py` - Hypothesis matrix execution
- `src/amy_lm/evaluator.py` - Benchmark evaluation
- `scripts/train_amy.py` - Training CLI
- `scripts/run_matrix.py` - Matrix runner CLI
- `scripts/evaluate_benchmarks.py` - Evaluation CLI

### Execution Order (Progressive)

1. **Simplest first:** Random init + Frozen + Classification head
2. **If λ grows:** Try warm-start init + Frozen + Classification head
3. **If still growing:** Try warm-start + LoRA + Classification head
4. **Continue only if returns justify:** Full fine-tune, LM loss, etc.
5. **Stop when:** λ ≈ 0 (falsified) or improvements < threshold

### Open Questions to Resolve

1. What λ threshold indicates "learning" vs "not learning"?
2. At what accuracy delta do we declare "diminishing returns"?
3. How many epochs per matrix cell?
4. Can we parallelize matrix cells across GPUs?

---

*This document will be expanded with full task details after Phase 2 completion.*
