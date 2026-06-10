# Phase 3: Run B Training + Decision Gate

> **Status:** Detailed. Ready for execution after Phase 2 completes.

## Phase Goal

Run 3-epoch DPO training on the adversarial dataset (`hungphongtrn/nvtts_facodec_adversarial`). Monitor λ_p and λ_t values and gradients at every logging step. Apply the decision gate: does λ move, confirming that adversarial pairs force the model to consult audio streams?

## Context

Run A (baseline DPO on literal pairs, #27) showed flat λ throughout training (λ_p: 1.012, λ_t: 1.002). The hypothesis: literal pairs are semantically distinguishable from text alone, so the LLM solves DPO without touching the FACodec enrichment path.

Run B replaces the literal pairs with adversarial pairs — chosen/rejected are semantically similar (emotion-mirrored) but differ in prosody/timbre representation in FACodec code space. If λ moves, the model is consulting the audio streams.

## Pre-Flight Checks

Before starting Phase 3, verify Phase 1 and 2 outputs exist:

```bash
# Verify adversarial pairs exist (1,000+ lines)
wc -l data/nvtts_adversarial/pairs.jsonl

# Verify text-blind sanity check passed (≤55% accuracy acceptable)
uv run python scripts/text_blind_sanity_check.py \
  --pairs data/nvtts_adversarial/pairs.jsonl \
  --num-samples 50

# Verify HF Hub dataset is accessible
uv run python -c "from datasets import load_dataset; ds = load_dataset('hungphongtrn/nvtts_facodec_adversarial', split='train'); print(f'Ready: {len(ds)} train samples')"
```

## Task A: Smoke Test (Debug Run)

Before the full 3-epoch run, do a 10-sample smoke test to verify the training loop handles adversarial pairs:

```bash
mkdir -p logs
uv run python scripts/train_amy_dpo.py \
  --config configs/dpo/adversarial.yaml \
  --num-samples 10 \
  --num-epochs 0.1 \
  --no-wandb \
  2>&1 | head -80
```

**Verify:**
- Model initializes without errors
- `lambda_p` / `lambda_t` appear in log output at ~1.0
- No CUDA OOM or NaN loss
- `cosine_similarity` filter passes all 10 samples (all set to 0.0)

## Task B: Run B DPO Training (Full)

```bash
mkdir -p logs output/amy_dpo_adversarial
nohup uv run python scripts/train_amy_dpo.py \
  --config configs/dpo/adversarial.yaml \
  > logs/adversarial_dpo_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "PID: $!"
echo "Monitor: tail -f logs/adversarial_dpo_*.log"
echo "W&B: https://wandb.ai/hungphongtrn/amy-lm-dpo (run: adversarial-run-b)"
```

**Config details (from `configs/dpo/adversarial.yaml`):**

| Parameter | Value | Note |
|-----------|-------|------|
| dataset | `hungphongtrn/nvtts_facodec_adversarial` | Phase 2 output |
| cosine_threshold | 0.85 | No-op (all pairs have cosine=0.0) |
| beta | 0.1 | Same as Run A |
| lr | 5e-5 | Same as Run A |
| epochs | 3.0 | Same as Run A |
| logging_steps | 10 | λ/grad logged every 10 steps |
| output_dir | `./output/amy_dpo_adversarial` | Checkpoints here |
| wandb_run_name | `adversarial-run-b` | W&B run identifier |

**Approximate runtime:** 
- ~800 train samples, effective batch size 4 (1×4 grad accum)
- 200 steps/epoch × 3 epochs = ~600 steps
- Expect 3–8 hours on a single GPU (RTX 3060 12GB or better)

## Task C: Monitor λ During Training

### Real-time monitoring

```bash
# Watch logs for λ values
tail -f logs/adversarial_dpo_*.log | grep -E "lambda|loss"

# Or use W&B dashboard
# https://wandb.ai/hungphongtrn/amy-lm-dpo (run: adversarial-run-b)
```

### Metrics to track

| Metric | Expected in Run B | Run A Baseline |
|--------|------------------|----------------|
| `lambda_p` | Should move from 1.0 | Flat at ~1.012 |
| `lambda_t` | Should move from 1.0 | Flat at ~1.002 |
| `lambda_p_grad` | Non-zero if moving | ~0.000 (flat) |
| `lambda_t_grad` | Non-zero if moving | ~0.000 (flat) |
| `loss` | Should decrease | Decreased normally |

### What to look for

- **Good sign**: `lambda_p_grad` or `lambda_t_grad` is consistently non-zero during early steps, meaning the DPO loss is flowing gradients into the enrichment modules.
- **Good sign**: λ values start drifting away from 1.0 by epoch 2.
- **Bad sign**: Both `lambda_p_grad` and `lambda_t_grad` stay near-zero after the first few logging steps (similar to Run A).
- **Neutral**: Only one of the two λ moves. The gate considers combined movement; even partial signal is useful information.

### Step count reference

| Step | Epoch | Expected behavior |
|------|-------|-------------------|
| 0 | 0.0 | λ_p=1.0, λ_t=1.0, reference model precomputed |
| 10–50 | 0.05–0.25 | Gradients should appear if architecture is engaging |
| 100 | 0.5 | λ should start diverging from 1.0 |
| 200 | 1.0 | Significant λ movement expected by end of epoch 1 |
| 400 | 2.0 | λ trajectory should be clear |
| 600 | 3.0 | Final λ values available |

## Task D: Decision Gate

After training completes, check final λ values from the W&B run or training log:

```bash
# Extract final λ values from W&B (preferred)
# OR from the log file:
grep -E "lambda_[pt]\"?:" logs/adversarial_dpo_*.log | tail -20
```

### Gate criteria

| λ Movement (Δλ) | Verdict | Action |
|---|---|---|
| Δλ ≥ 0.2 (e.g., λ_p=0.75) | **PASS** — Architecture works. Adversarial pairs force audio-stream consultation. | Close #36 (#37 is moot). |
| 0.02 < Δλ < 0.2 | **GRAY** — Partial signal. Architecture is engaging but weakly. | Proceed to #37 (intermediate prosody loss) to amplify. |
| Δλ ≤ 0.02 (flat) | **FAIL** — Genuine modality dominance. Text alone is sufficient even with adversarial pairs. | Proceed to #37 (intermediate prosody loss) as next contingency. |

Δλ is defined as `max(|final(λ_p) - 1.0|, |final(λ_t) - 1.0|)` — the larger absolute deviation between the two lambdas.

### Gray-zone analysis

If λ moves between 0.02 and 0.2, also check:
1. **Direction**: Is λ decreasing (expected) or increasing? Decreasing λ means the enrichment modules are being turned *down* — the model may be learning that raw LLM text is sufficient.
2. **Trajectory shape**: Does λ plateau early or continue drifting?
3. **Gradient magnitude**: Are λ gradients consistently non-zero even if λ moves slowly?

If trajectory shows continued drift (not plateau), consider extending to 5 epochs.

## Task E: Post Results

After training completes:

1. **Capture final λ values** from W&B or log:
   ```
   Epoch 3 complete:
   lambda_p = X.XXX (Δ = |X.XXX - 1.0| = Y.YYY)
   lambda_t = X.XXX (Δ = |X.XXX - 1.0| = Y.YYY)
   ```

2. **Post to issue #36** with:
   - Final λ_p, λ_t values and their deltas
   - λ_grad_p, λ_grad_t trajectory (attach W&B screenshot)
   - Decision gate outcome (PASS/GRAY/FAIL)
   - Final validation loss
   - Path to best checkpoint: `output/amy_dpo_adversarial/checkpoint-XXX/`

3. **Based on gate outcome:**
   - **PASS**: Comment on #36 with results, close #36 and #37. AmyLM architecture validated.
   - **GRAY/FAIL**: Link to #37 for next phase. Keep #36 open for tracking.

## Known Limitations

1. **No eval split in training loop.** `load_and_filter_dataset()` uses hardcoded NVTTS split indices (3641/3687). For the adversarial dataset (800 train samples), `eval_dataset` will be `None`. Evaluation metrics won't be logged during training. The HF Hub dev/test splits (100 each) can be evaluated offline post-hoc.

2. **Small dataset (800 train).** At 3 epochs with effective batch size 4, total updates are ~600. This is sufficient for λ monitoring but may not be enough for convergence. λ movement in a small-dataset regime is a conservative test — if it moves here, it will move in larger regimes.

3. **λ measured at log time, not step time.** λ values reflect the state at `logging_steps` intervals. Rapid λ changes between steps may be missed, but the gradient values capture whether backprop is flowing.

## Files

| File | Purpose |
|------|---------|
| `configs/dpo/adversarial.yaml` | Run B config (already exists) |
| `scripts/train_amy_dpo.py` | Training entry point (unchanged) |
| `src/training/amy_dpo_trainer.py` | λ logging + gradient hooks (Phase 2) |
| `output/amy_dpo_adversarial/` | Checkpoints and final model |

## Phase Completion Criteria

- [ ] Smoke test passes (10 samples, 0.1 epochs)
- [ ] Run B completed (3 epochs on adversarial dataset)
- [ ] λ_p, λ_t logged at every logging step (verified in W&B)
- [ ] λ_p_grad, λ_t_grad logged (non-zero confirms gradient flow)
- [ ] Decision gate evaluated (PASS/GRAY/FAIL)
- [ ] Results posted to issue #36
- [ ] Next step determined (#37 or close #36)
