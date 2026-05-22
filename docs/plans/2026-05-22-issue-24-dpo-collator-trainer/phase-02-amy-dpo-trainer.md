# Phase 2: AmyDPOTrainer

## Phase Goal

`AmyDPOTrainer` subclass successfully initializes, runs a single training step on mock data, and logs AmyLM-specific metrics (λ_p, λ_t) alongside standard TRL DPO metrics.

## Files to Touch

- `src/training/amy_dpo_trainer.py` — **Create**: AmyDPOTrainer class
- `tests/training/test_amy_dpo_trainer.py` — **Create**: Unit tests

## Design

### Key insight: zero `_compute_loss` override

TRL's `_compute_loss` builds `model_kwargs` by filtering only three keys:

```python
_non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
```

Our collator produces `audio_data`, `audio_data_seqlens`, `audio_input_mask`, `prosody_indices`, `timbre_vector` — all of which pass through to `model(**model_kwargs)` → `AmyLM.forward()`. No override needed.

### Reference model: `precompute_ref_log_probs=True`

The precomputation runs in `DPOTrainer.__init__` before training, when λ=0. During training, TRL loads cached reference log-probs from the dataset. This sidesteps the lambda-zeroing problem entirely.

### QLoRA setup

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(amy_lm, lora_config)
```

TRL's `DPOTrainer` with PEFT auto-handles:
- Adapter disable for reference precomputation
- bf16 adapter weights for QLoRA
- Separate optimizer group for LoRA params

### Trainable params

| Group | Module | Requires Grad | Precision | Optimizer |
|-------|--------|:---:|:---:|------|
| LoRA adapters | Qwen3 linear layers | ✓ | bf16 | paged_adamw_8bit |
| FACodec | `prosody_embedding.*` | ✓ | fp32 | AdamW |
| FACodec | `timbre_projection.*` | ✓ | fp32 | AdamW |
| FACodec | `residual_fusion.lambda_{p,t}` | ✓ | fp32 | AdamW |
| FACodec | `temporal_pool` | ✗ (no params) | — | — |
| Backbone | audio_encoder, audio_adapter, lm_head | ✗ | 4-bit | — |
| Backbone | Qwen3 (frozen) | ✗ | 4-bit | — |

## Tasks

### Task 1: Create AmyDPOTrainer class

**Files:**
- Create: `src/training/amy_dpo_trainer.py`

- [ ] **Step 1: Implement minimal subclass**

```python
"""AmyDPOTrainer — TRL DPOTrainer subclass for AmyLM DPO with FACodec."""

from __future__ import annotations

import math
from typing import Any

import torch
from trl import DPOTrainer, DPOConfig
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from datasets import Dataset, IterableDataset


class AmyDPOTrainer(DPOTrainer):
    """DPOTrainer subclass for AmyLM DPO training.

    Key differences from standard DPOTrainer:
     - precompute_ref_log_probs=True by default (lambda-zero at init)
     - Logs lambda_p and lambda_t from AmyLM's ResidualFusion gates
     - Passes AmyLM-specific kwargs (audio_data, prosody_indices, etc.)
       through to model.forward() automatically via TRL's model_kwargs passthrough

    Usage:
        trainer = AmyDPOTrainer(
            model=amy_lm_model,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
            args=dpo_config,
        )
        trainer.train()
    """

    def __init__(
        self,
        model,
        ref_model=None,
        args: DPOConfig | None = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        **kwargs,
    ):
        # Force precompute_ref_log_probs by default
        if args is None:
            args = DPOConfig(
                output_dir="./amy_dpo_output",
                precompute_ref_log_probs=True,
                beta=0.1,
                learning_rate=0.0,  # Will be set in training script with optimizer params
                bf16=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,
                max_length=1024,
                logging_steps=10,
                save_steps=500,
                eval_steps=500,
                num_train_epochs=3,
                loss_type=["sigmoid"],
            )
        elif not args.precompute_ref_log_probs:
            raise ValueError(
                "AmyDPOTrainer requires precompute_ref_log_probs=True. "
                "This ensures the reference model (lambda=0) is computed "
                "before training starts, when FACodec gates are zero-initialized."
            )

        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs,
        )
```

- [ ] **Step 2: Add lambda logging**

```python
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        """Inject lambda_p and lambda_t into TRL's log output."""
        try:
            model = self.accelerator.unwrap_model(self.model)
            # Handle PEFT wrapper
            base_model = getattr(model, "base_model", model)
            fusion = base_model.residual_fusion
            logs["lambda_p"] = fusion.lambda_p.item()
            logs["lambda_t"] = fusion.lambda_t.item()
        except Exception:
            pass
        super().log(logs, *args, **kwargs)
```

### Task 2: Write unit tests for AmyDPOTrainer

**Files:**
- Create: `tests/training/test_amy_dpo_trainer.py`

- [ ] **Step 1: Test with mock model (init + lambda logging)**

```python
"""Unit tests for AmyDPOTrainer."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


class MockResidualFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_p = nn.Parameter(torch.tensor(0.0))
        self.lambda_t = nn.Parameter(torch.tensor(0.0))

    def forward(self, *args, **kwargs):
        return args[0] if args else None


class MockAmyLM(nn.Module):
    """Minimal AmyLM for trainer init tests."""
    def __init__(self):
        super().__init__()
        self.residual_fusion = MockResidualFusion()
        self.config = MagicMock()

    def forward(self, **kwargs):
        B = kwargs["input_ids"].shape[0]
        S = kwargs["input_ids"].shape[1]
        V = 152064  # vocab size
        return type("MockOutput", (), {
            "logits": torch.randn(B, S, V),
            "loss": torch.tensor(0.0),
        })()


class TestAmyDPOTrainer:
    def test_precompute_ref_log_probs_required(self):
        """AmyDPOTrainer requires precompute_ref_log_probs=True."""
        from trl import DPOConfig
        from src.training.amy_dpo_trainer import AmyDPOTrainer

        config = DPOConfig(
            output_dir="/tmp/test_dpo",
            precompute_ref_log_probs=False,
        )
        with pytest.raises(ValueError, match="precompute_ref_log_probs=True"):
            AmyDPOTrainer(
                model=MockAmyLM(),
                train_dataset=MagicMock(),
                args=config,
            )

    def test_default_config_sets_precompute(self):
        """Default config precompute_ref_log_probs is True."""
        from src.training.amy_dpo_trainer import AmyDPOTrainer

        trainer = AmyDPOTrainer(
            model=MockAmyLM(),
            train_dataset=MagicMock(),
        )
        assert trainer.args.precompute_ref_log_probs is True

    def test_lambda_logging(self):
        """log() injects lambda_p and lambda_t."""
        from src.training.amy_dpo_trainer import AmyDPOTrainer

        trainer = AmyDPOTrainer(
            model=MockAmyLM(),
            train_dataset=MagicMock(),
        )

        # Manually set lambda values
        trainer.accelerator = MagicMock()
        trainer.accelerator.unwrap_model.return_value = MockAmyLM()
        trainer.model = MockAmyLM()
        trainer.model.residual_fusion.lambda_p.data = torch.tensor(0.5)
        trainer.model.residual_fusion.lambda_t.data = torch.tensor(0.3)

        logs = {"loss": 0.5, "rewards/margins": 0.1}
        trainer.log(logs)
        assert "lambda_p" in logs
        assert "lambda_t" in logs
        assert abs(logs["lambda_p"] - 0.5) < 1e-6
        assert abs(logs["lambda_t"] - 0.3) < 1e-6
```

### Task 3: Run tests and commit

- [ ] **Step 1: Run tests**

```bash
uv run python -m pytest tests/training/test_amy_dpo_trainer.py -v
```

- [ ] **Step 2: Commit**

```bash
git add src/training/amy_dpo_trainer.py tests/training/test_amy_dpo_trainer.py
git commit -m "feat: add AmyDPOTrainer with precompute_ref_log_probs and lambda logging"
```

## Phase Completion Criteria
- [ ] `AmyDPOTrainer.__init__` enforces `precompute_ref_log_probs=True`
- [ ] Lambda values logged alongside TRL metrics
- [ ] All unit tests pass
- [ ] Class does NOT override `_compute_loss`
