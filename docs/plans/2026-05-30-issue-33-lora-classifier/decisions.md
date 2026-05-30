# Decision Log

## 2026-05-30: audio_adapter LoRA targets added to DPO reference pattern

**Context:** The DPO LoRA pattern (`scripts/train_amy_dpo.py:199-213`) only targets `moss.language_model.*` projection layers. For the classifier path, H2 hypothesizes that the frozen audio_adapter (GatedMLP bridging FACodec-enriched audio to Qwen3) is a gradient choke-point.

**Decision:** Add `moss.audio_adapter.*` to `target_modules` regex alongside `moss.language_model.*`. The audio_adapter is a GatedMLP with gate_proj/up_proj/down_proj — same linear projection names as Qwen3 attention/FFN layers, so they can be covered by a single regex.

**Rationale:** The audio_adapter sits directly between FACodec-enriched audio embeddings and Qwen3's input. If the hypothesis is correct that frozen backbone starves FACodec gradient, unfreezing this bridge via LoRA should show measurable λ gradient improvement.

**Consequences:**
- More trainable parameters (~1M extra for audio_adapter LoRA), but still far fewer than full fine-tuning
- Regex becomes: `r"^(?:moss\.audio_adapter|moss\.language_model)\..*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)$"` — slightly more complex but stays as a single regex

## 2026-05-30: TaskType FEATURE_EXTRACTION (not CAUSAL_LM)

**Context:** The DPO path uses `TaskType.CAUSAL_LM` because DPO trains the LM head with next-token prediction loss. The classifier path uses Qwen3 purely as encoder — no LM head, loss is on classifier logits.

**Decision:** Use `TaskType.FEATURE_EXTRACTION`. This tells PEFT that the model is a feature extractor, which matches how we use it (pool Qwen3 last_hidden_state → classifier head).

**Rationale:** `TaskType.CAUSAL_LM` would add unnecessary PEFT modifications. `TaskType.SEQ_CLS` would expect a specific HuggingFace classifier head interface. `FEATURE_EXTRACTION` is the most minimal and correct.

## 2026-05-30: classifier in modules_to_save (not lora target)

**Context:** The classifier head is a simple `nn.Linear(2560, 2)`. It should be fully trainable, not LoRA-adapted.

**Decision:** Add `"classifier"` to `modules_to_save` alongside FACodec modules.

**Rationale:** The classifier is newly initialized each run — there are no pre-trained weights to preserve with LoRA. Full training is appropriate and simpler. Also, `TaskType.FEATURE_EXTRACTION` PEFT may not support LoRA on arbitrary linear layers outside the transformer blocks.

## 2026-05-30: Lambda gradient hooks — unwrap via model.model

**Context:** With `get_peft_model`, the PeftModel wrapper introduces an additional layer: `peft_model.model` or `peft_model.base_model.model` to access the wrapped module. The current lambda hooks access `self.model.amy_moss.residual_fusion` which will break.

**Decision:** Use `model.model` to unwrap one level (this is the standard PEFT accessor for `PeftModel.model` which returns the underlying base model). The path becomes `trainer.model.model.amy_moss.residual_fusion` for hook registration and `trainer.model.model.amy_moss` for FACodec module access.

**Rationale:** `PeftModel.model` is the canonical way to access the wrapped model (confirmed in PEFT docs and wild code). `base_model` is an internal attribute and may vary across PEFT versions.

**Consequences:** The trainer needs an `is_lora` flag to conditionally unwrap. The `_register_lambda_grad_hooks`, `_get_lambdas`, `save_checkpoint`, and `load_checkpoint` methods all need this path branching.

## 2026-05-30: Checkpoint format — PeftModel.save_pretrained style

**Context:** The current checkpoint format filters out `amy_moss.moss.*` keys and saves only trainable parameters + optimizer + epoch. With PEFT, the saved state_dict should include LoRA adapter weights + modules_to_save weights but NOT base model weights (they're frozen and recoverable from pretrained).

**Decision:** Use `model.state_dict()` (which PEFT overrides to only return trainable adapter + modules_to_save weights). This is simpler and already filtered. For loading, use `model.load_state_dict(strict=False)`.

**Rationale:** PEFT's `state_dict()` already knows which parameters are trainable. Using it avoids manual key filtering. `strict=False` handles any PEFT-internal keys gracefully.
