# Phase 3: Run B Training + Decision Gate

> **STUB — will be detailed after Phase 2 completes.**

## Phase Goal
Run 3-epoch DPO training on adversarial dataset (`nvtts_facodec_adversarial`). Monitor λ_p and λ_t values and gradients. Apply decision gate to determine if adversarial pairs resolve the gradient starvation.

## Rough Scope

### Task A: Run B DPO Training
- Run `scripts/train_amy_dpo.py` with `--dataset hungphongtrn/nvtts_facodec_adversarial`
- 3 epochs, same hyperparameters as Run A
- λ values and gradients logged at each `logging_steps`

### Task B: Monitor λ Movement
- Track `lambda_p`, `lambda_t`, `lambda_p_grad`, `lambda_t_grad` over training
- Plot λ vs. step (W&B or local)
- Compare with Run A baseline (flat λ at ~1.0)

### Task C: Decision Gate
| Outcome | Meaning | Next Step |
|---|---|---|
| λ moves ≥ 0.2 | Architecture works — adversarial pairs close the text shortcut | Paper proceeds. Close #37. |
| λ stays flat (< 0.02) | Genuine modality dominance | Implement intermediate prosody loss (#37). |

### Task D: Post Results
- Post final λ values and evaluation results to issue #36
- If gate passes: close #36, proceed with paper
- If gate fails: link to #37 for contingency implementation

## Files to Touch
- `scripts/train_amy_dpo.py` — Change dataset argument (or config YAML)
- `configs/dpo/adversarial.yaml` — Create (same as baseline but with adversarial dataset)
- Training output directory — Checkpoint, logs, W&B run

## Depends on
- Phase 2 (adversarial dataset on HF Hub + lambda hooks)

## Phase Completion Criteria
- [ ] Run B completed (3 epochs on adversarial dataset)
- [ ] λ values and gradients logged and plotted
- [ ] Decision gate evaluated and posted to issue #36
- [ ] Next step determined (#37 or close)
