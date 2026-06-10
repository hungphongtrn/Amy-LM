# Decision Log

## 2026-06-10: Plan initialized from grilled issue #36
**Context:** Issue was pre-grilled via `grill-with-docs`. All design decisions (file architecture, generation pipeline, quality filters, training strategy, decision gate) were resolved before planning.
**Decision:** Adopted the issue body as the source of truth. Plan translates those decisions into executable phases without re-litigating.
**Rationale:** The grilling session produced a complete design. Re-deciding would waste time.
**Consequences:** Plan follows the issue body exactly. No new architectural choices needed.

## 2026-06-10: Three-phase structure chosen
**Context:** Issue covers (a) pair generation with quality filters, (b) FACodec encoding + dataset push, (c) lambda gradient hooks, and (d) training run + evaluation.
**Decision:** Split into 3 phases: (1) generation, (2) encoding + hooks, (3) training + gate.
**Rationale:** Generation is the riskiest and most independent phase. Encoding + hooks are both prerequisites for training but independent of each other. Training is the final integration. This allows early validation of pair quality before spending GPU time.
**Consequences:** Phase 2 bundles two independent tasks (encoding + hooks) for convenience since both are quick and both gate Phase 3.

## 2026-06-10: PreferenceDatasetProcessor extension vs. new processor
**Context:** Adversarial pairs have a different input schema (additional fields: strategy, inverse_emotion, judge_fidelity_chosen, etc.) compared to literal pairs.
**Decision:** Extend existing `PreferenceDatasetProcessor` with a new method rather than creating a separate processor class.
**Rationale:** The FACodec encoding path is identical. Only the input field mapping differs. A single class with a flag preserves the DRY encoding logic.
**Consequences:** `PreferenceDatasetProcessor.process_adversarial_dataset()` added alongside existing `process_dataset()`.

## 2026-06-10: Lambda gradient hooks ported from AmyTrainer, not redesigned
**Context:** `AmyTrainer._register_lambda_grad_hooks()` uses `register_hook()` on `lambda_p`/`lambda_t` parameters and stores gradients in `self._saved_lambda_grads`.
**Decision:** Port the same pattern to `AmyDPOTrainer`, using `self.accelerator.unwrap_model()` to reach the base model (consistent with existing `log()` method).
**Rationale:** The pattern is proven and 1:1 applicable. The only difference is PEFT unwrapping path (`unwrap_model` vs. `_base_model`).
**Consequences:** Gradients logged at `logging_steps` frequency (not a separate counter). Empty on first call (no backward yet).
