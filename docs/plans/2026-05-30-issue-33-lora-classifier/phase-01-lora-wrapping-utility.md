# Phase 1: LoRA wrapping utility + init tests

## Phase Goal
`wrap_classifier_with_lora()` function exists in `src/models/amy_classifier.py`, produces a PEFT-wrapped `AmyForProsodyClassification` with correct adapter placement and parameter partitioning. Two tests: static init (asserts adapter existence + frozen/trainable split) and gradient flow (GPU, asserts non-zero LoRA grads in forward+backward).

## Files to Touch

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models/amy_classifier.py` | Modify | Add `wrap_classifier_with_lora()` utility |
| `tests/training/test_lora_classifier_init.py` | Create | Static init test + gradient flow test |

## Reference Code

**DPO LoRA config** (`scripts/train_amy_dpo.py:199-213`):
```python
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=r"^moss\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)$",
    modules_to_save=["prosody_embedding", "timbre_projection", "temporal_pool", "residual_fusion"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

**DPO fp32 cast** (`scripts/train_amy_dpo.py:189-193`):
```python
for module in (model.prosody_embedding, model.timbre_projection, model.temporal_pool, model.residual_fusion):
    module.to(dtype=torch.float32)
```

**DPO gradient flow test** (`tests/training/test_train_amy_dpo_init_model.py:72-178`) — full pattern to mirror.

## Tasks

### Task 1: Create `wrap_classifier_with_lora()` utility

**Files:**
- Modify: `src/models/amy_classifier.py`

- [ ] **Step 1: Add imports and function to amy_classifier.py**

Insert after the class definition (before the module end):

```python
def wrap_classifier_with_lora(
    model: AmyForProsodyClassification,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> "PeftModel":
    """Wrap AmyForProsodyClassification with LoRA for H2 gradient-flow test.

    LoRA targets (via regex):
      - moss.audio_adapter.* (gate_proj, up_proj, down_proj)  [GatedMLP]
      - moss.language_model.* (q_proj, k_proj, v_proj, o_proj,   [Qwen3]
                               up_proj, down_proj, gate_proj)

    modules_to_save (fully trainable, fp32):
      - amy_moss.prosody_embedding
      - amy_moss.timbre_projection
      - amy_moss.temporal_pool
      - amy_moss.residual_fusion
      - classifier

    Audio encoder stays frozen (no LoRA, no modules_to_save).

    Args:
        model: AmyForProsodyClassification instance (frozen backbone, trainable FACodec+head).
        r: LoRA rank (default: 8, matches DPO).
        lora_alpha: LoRA alpha (default: 16, matches DPO).
        lora_dropout: LoRA dropout (default: 0.05, matches DPO).

    Returns:
        PeftModel wrapping the classifier.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    for module in (
        model.amy_moss.prosody_embedding,
        model.amy_moss.timbre_projection,
        model.amy_moss.temporal_pool,
        model.amy_moss.residual_fusion,
    ):
        module.to(dtype=torch.float32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Pre-LoRA trainable: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=(
            r"^(?:moss\.audio_adapter|moss\.language_model)\."
            r".*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)$"
        ),
        modules_to_save=[
            "amy_moss.prosody_embedding",
            "amy_moss.timbre_projection",
            "amy_moss.temporal_pool",
            "amy_moss.residual_fusion",
            "classifier",
        ],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from src.models.amy_classifier import wrap_classifier_with_lora; print('OK')"`
Expected: "OK" (no import errors)

### Task 2: Write static init test

**Files:**
- Create: `tests/training/test_lora_classifier_init.py`

- [ ] **Step 1: Create test file with tiny model helper**

```python
"""Tests for wrap_classifier_with_lora() — LoRA classifier wrapping."""

from __future__ import annotations

import pytest
import torch


def _tiny_moss_config():
    from src.models.moss_audio_model import MossAudioConfig
    return MossAudioConfig(
        language_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
        audio_config={
            "d_model": 32,
            "output_dim": 32,
            "num_mel_bins": 128,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64,
            "downsample_hidden_size": 8,
            "deepstack_encoder_layer_indexes": [],
        },
        adapter_hidden_size=32,
    )


class TestLoraClassifierStaticInit:
    """Verify LoRA wrapper produces correct parameter partitioning (no forward)."""

    def test_lora_adapters_on_audio_adapter_and_language_model(self, device, monkeypatch):
        from src.models.moss_audio_model import MossAudioModel
        from src.models.amy_lm import AmyMossLMConfig
        from src.models.amy_classifier import (
            AmyForProsodyClassification,
            wrap_classifier_with_lora,
        )

        tiny = MossAudioModel(_tiny_moss_config()).to(device)
        hidden_dim = tiny.config.language_config.hidden_size  # 32

        monkeypatch.setattr(MossAudioModel, "from_pretrained", lambda *a, **kw: tiny)

        original_init = AmyMossLMConfig.__init__
        def patched_init(self, moss_config=None, hidden_dim=hidden_dim, **kwargs):
            if moss_config is not None and hidden_dim is None:
                hidden_dim = moss_config.language_config.hidden_size
            return original_init(self, moss_config=moss_config, hidden_dim=hidden_dim, **kwargs)
        monkeypatch.setattr(AmyMossLMConfig, "__init__", patched_init)

        vectors = torch.randn(1024, hidden_dim, device=device)
        model = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)
        peft_model = wrap_classifier_with_lora(model, r=2, lora_alpha=4, lora_dropout=0.0)

        trainable = [name for name, param in peft_model.named_parameters() if param.requires_grad]

        assert any("moss.audio_adapter" in name and "lora_" in name for name in trainable), (
            "Missing LoRA on audio_adapter"
        )
        assert any("moss.language_model" in name and "lora_" in name for name in trainable), (
            "Missing LoRA on language_model"
        )
        assert any("prosody_embedding" in name for name in trainable), "prosody_embedding not trainable"
        assert any("timbre_projection" in name for name in trainable), "timbre_projection not trainable"
        assert any("temporal_pool" in name for name in trainable), "temporal_pool not trainable"
        assert any("residual_fusion" in name for name in trainable), "residual_fusion not trainable"
        assert any("classifier" in name for name in trainable), "classifier not trainable"

        assert not any("audio_encoder" in name and "lora_" in name for name in trainable), (
            "LoRA on audio_encoder (should be frozen)"
        )
        for name, param in peft_model.named_parameters():
            if "moss.language_model" in name and "lora_" not in name:
                assert not param.requires_grad, f"LM base param {name} should be frozen"
            if "moss.audio_encoder" in name:
                assert not param.requires_grad, f"Audio encoder param {name} should be frozen"

    def test_lora_model_preserves_forward_signature(self, device, monkeypatch):
        """PEFT-wrapped model can run forward with (audio, prosody, timbre)."""
        from src.models.moss_audio_model import MossAudioModel
        from src.models.amy_lm import AmyMossLMConfig
        from src.models.amy_classifier import (
            AmyForProsodyClassification,
            wrap_classifier_with_lora,
        )

        tiny = MossAudioModel(_tiny_moss_config()).to(device)
        hidden_dim = tiny.config.language_config.hidden_size
        monkeypatch.setattr(MossAudioModel, "from_pretrained", lambda *a, **kw: tiny)

        original_init = AmyMossLMConfig.__init__
        monkeypatch.setattr(AmyMossLMConfig, "__init__",
            lambda self, moss_config=None, hidden_dim=hidden_dim, **kw:
                original_init(self, moss_config=moss_config, hidden_dim=hidden_dim, **kw))

        vectors = torch.randn(1024, hidden_dim, device=device)
        model = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)
        peft_model = wrap_classifier_with_lora(model, r=2, lora_alpha=4, lora_dropout=0.0)

        B = 1
        audio = torch.randn(B, 16000, device=device)
        prosody = torch.randint(0, 1024, (B, 1, 80), device=device)
        timbre = torch.randn(B, 256, device=device)

        peft_model.eval()
        with torch.no_grad():
            logits = peft_model(audio, prosody, timbre)
        assert logits.shape == (B, 2), f"Expected [1,2] logits, got {logits.shape}"
```

- [ ] **Step 2: Run the static init test**

Run: `uv run python -m pytest tests/training/test_lora_classifier_init.py::TestLoraClassifierStaticInit -v`
Expected: 2 tests PASS

- [ ] **Step 3: Run the forward signature test (needs `require_gpu` or `device`)**

Run: `uv run python -m pytest tests/training/test_lora_classifier_init.py::TestLoraClassifierStaticInit::test_lora_model_preserves_forward_signature -v`
Expected: PASS (may be slow on CPU due to WhisperEncoder forward, but tiny model should be manageable)

### Task 3: Write gradient flow test (GPU)

**Files:**
- Modify: `tests/training/test_lora_classifier_init.py`

- [ ] **Step 1: Add gradient flow test class**

```python
class TestLoraClassifierGradientFlow:
    """GPU tests: forward+backward produces non-zero LoRA/FACodec gradients."""

    def test_lora_and_facodec_gradients_nonzero(self, device, require_gpu):
        """LoRA adapters and modules_to_save get non-zero grads; frozen params get None."""
        from src.models.amy_classifier import (
            AmyForProsodyClassification,
            wrap_classifier_with_lora,
        )
        import torch.nn as nn

        vectors = torch.randn(1024, 2560, device=device)
        model = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)
        peft_model = wrap_classifier_with_lora(model, r=2, lora_alpha=4, lora_dropout=0.0)
        peft_model.train()

        B = 1
        audio = torch.randn(B, 24000, device=device)
        prosody = torch.randint(0, 1024, (B, 1, 120), device=device)
        timbre = torch.randn(B, 256, device=device)
        labels = torch.tensor([0], device=device)

        logits = peft_model(audio, prosody, timbre)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        # LoRA adapters have non-zero grads
        lora_params = [(n, p) for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad]
        assert len(lora_params) > 0, "No LoRA parameters found"
        for name, param in lora_params:
            assert param.grad is not None, f"LoRA {name} has None grad"
            assert param.grad.abs().sum().item() > 0, f"LoRA {name} has zero grad"

        # modules_to_save have non-zero grads
        save_params = [
            (n, p) for n, p in peft_model.named_parameters()
            if p.requires_grad and "lora_" not in n
            and not n.startswith("base_model.model.moss.audio_encoder")
        ]
        assert len(save_params) > 0, "No modules_to_save params found"
        for name, param in save_params:
            assert param.grad is not None, f"{name} has None grad"
            assert param.grad.abs().sum().item() > 0, f"{name} has zero grad"

        # Frozen audio_encoder has None grads
        for name, param in peft_model.named_parameters():
            if "audio_encoder" in name:
                assert param.grad is None, f"Frozen encoder {name} has grad"
```

- [ ] **Step 2: Run gradient flow test (GPU only)**

Run: `uv run python -m pytest tests/training/test_lora_classifier_init.py::TestLoraClassifierGradientFlow -v`
Expected: 1 test PASS (GPU)

## Phase Completion Criteria
- [ ] `wrap_classifier_with_lora()` in `src/models/amy_classifier.py` — importable
- [ ] `test_lora_adapters_on_audio_adapter_and_language_model` — PASS
- [ ] `test_lora_model_preserves_forward_signature` — PASS
- [ ] `test_lora_and_facodec_gradients_nonzero` — PASS (GPU)
- [ ] Commit with message: `feat: add wrap_classifier_with_lora() utility + init tests (#33)`

## Handoff Notes
- The `PeftModel` wraps the base model: `peft_model.model` gives access to the underlying `AmyForProsodyClassification`
- FACodec module path: `peft_model.model.amy_moss.residual_fusion.lambda_p`
- Phase 2 trainer adaptation will need to unwrap via `model.model`
- Monkeypatch for `AmyMossLMConfig` is needed because the classifier constructor uses default `hidden_dim=2560` but tiny test model has `hidden_size=32` — the patch forces the config to match the tiny model's dimension
