# Phase 3: AmyLM Model + Config — Issue #19

## Phase Goal
`AmyLMConfig` extending `MossAudioConfig` + `AmyLM` inheriting `MossAudioModel`. Enriches audio embeddings with prosody/timbre before scattering into `inputs_embeds`. Trainable: FACodec modules. Backward-compatible with base model.

## Files to Touch

| File | Action | Purpose |
|------|--------|---------|
| `src/models/amy_lm.py` | Create | `AmyLMConfig` + `AmyLM` class |
| `src/models/embedding.py` | Edit | Make warm_start projector trainable (line 48: `requires_grad = True`) |
| `tests/models/test_embedding.py` | Edit | Update `test_warm_start_projector_is_frozen` → `test_warm_start_projector_is_trainable` |
| `tests/models/test_amy_lm.py` | Create | Unit tests for AmyLM |
| `src/models/__init__.py` | Edit | Export `AmyLMConfig`, `AmyLM` |

## Architecture

```python
# src/models/amy_lm.py

from dataclasses import dataclass, field
from typing import Optional, List
import torch
import torch.nn as nn

# Path setup (same as moss_audio.py)
import os, sys
_VENDOR_MOSS_AUDIO_SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "MOSS-Audio", "src")
if os.path.isdir(_VENDOR_MOSS_AUDIO_SRC_PATH) and _VENDOR_MOSS_AUDIO_SRC_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_MOSS_AUDIO_SRC_PATH)

from configuration_moss_audio import MossAudioConfig, MossAudioEncoderConfig
from modeling_moss_audio import MossAudioModel

from .embedding import ProsodyEmbedding, TimbreProjection
from .pooling import TemporalPool
from .fusion import ResidualFusion


class AmyLMConfig(MossAudioConfig):
    model_type = "amy_lm"
    
    def __init__(
        self,
        audio_config=None,
        language_config=None,
        adapter_hidden_size=8192,
        ignore_index=-100,
        deepstack_num_inject_layers=None,
        # FACodec stream config
        prosody_vocab_size: int = 1024,
        prosody_init_strategy: str = "random",
        prosody_init_std: float = 0.02,
        prosody_input_rate: float = 80.0,
        prosody_output_rate: float = 12.5,
        timbre_dim: int = 256,
        hidden_dim: int = 2560,
        # Freeze control
        freeze_audio_encoder: bool = True,
        freeze_audio_adapter: bool = True,
        freeze_llm: bool = True,
        **kwargs,
    ):
        super().__init__(
            audio_config=audio_config,
            language_config=language_config,
            adapter_hidden_size=adapter_hidden_size,
            ignore_index=ignore_index,
            deepstack_num_inject_layers=deepstack_num_inject_layers,
            **kwargs,
        )
        self.prosody_vocab_size = prosody_vocab_size
        self.prosody_init_strategy = prosody_init_strategy
        self.prosody_init_std = prosody_init_std
        self.prosody_input_rate = prosody_input_rate
        self.prosody_output_rate = prosody_output_rate
        self.timbre_dim = timbre_dim
        self.hidden_dim = hidden_dim
        self.freeze_audio_encoder = freeze_audio_encoder
        self.freeze_audio_adapter = freeze_audio_adapter
        self.freeze_llm = freeze_llm


class AmyLM(MossAudioModel):
    config_class = AmyLMConfig
    
    def __init__(self, config: AmyLMConfig):
        super().__init__(config)
        
        # FACodec enrichment modules
        self.prosody_embedding = ProsodyEmbedding(
            vocab_size=config.prosody_vocab_size,
            embed_dim=config.hidden_dim,
            init_strategy=config.prosody_init_strategy,
            init_std=config.prosody_init_std,
        )
        self.timbre_projection = TimbreProjection(
            timbre_dim=config.timbre_dim,
            output_dim=config.hidden_dim,
        )
        self.temporal_pool = TemporalPool(
            input_rate=config.prosody_input_rate,
            output_rate=config.prosody_output_rate,
        )
        self.residual_fusion = ResidualFusion(hidden_dim=config.hidden_dim)
        
        # Apply freeze configuration
        self._apply_freeze(config)
        
        self.post_init()
    
    def _apply_freeze(self, config: AmyLMConfig):
        if config.freeze_audio_encoder:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
        if config.freeze_audio_adapter:
            for p in self.audio_adapter.parameters():
                p.requires_grad = False
        if config.freeze_llm:
            for p in self.language_model.parameters():
                p.requires_grad = False
    
    def forward(self, **kwargs):
        """Override MossAudioModel.forward to add FACodec enrichment.
        
        Extracts prosody_indices and timbre_vector from kwargs,
        enriches audio_embeds after audio_adapter but before masked_scatter_.
        Without prosody/timbre, behaves identically to MossAudioModel.
        """
        # Extract FACodec inputs from kwargs (HF-compatible passthrough)
        prosody_indices = kwargs.pop("prosody_indices", None)
        timbre_vector = kwargs.pop("timbre_vector", None)
        
        # ... (full MossAudioModel.forward logic with enrichment step)
        # See detailed task spec below for exact implementation
```

### forward() logic (detailed)

The `forward()` must replicate `MossAudioModel.forward()` lines 437–542, adding enrichment between line 470 (`audio_embeds = self.audio_adapter(audio_embeds)`) and line 479 (`mask_expanded = ...`):

```python
# After: audio_embeds = self.audio_adapter(audio_embeds)
# NEW: Enrich audio embeddings with FACodec prosody + timbre
if prosody_indices is not None or timbre_vector is not None:
    streams = {}
    
    if prosody_indices is not None:
        # prosody_indices: [B, 1, T80] int64
        p_emb = self.prosody_embedding(prosody_indices)  # [B, T80, D]
        p_emb = self.temporal_pool(p_emb)                 # [B, T12, D]
        # Truncate/pad to match audio_embeds length
        if p_emb.shape[1] < audio_embeds.shape[1]:
            pad = torch.zeros(p_emb.shape[0], audio_embeds.shape[1] - p_emb.shape[1], p_emb.shape[2],
                            device=p_emb.device, dtype=p_emb.dtype)
            p_emb = torch.cat([p_emb, pad], dim=1)
        elif p_emb.shape[1] > audio_embeds.shape[1]:
            p_emb = p_emb[:, :audio_embeds.shape[1], :]
        streams["prosody"] = p_emb
    
    if timbre_vector is not None:
        # timbre_vector: [B, 256] float32
        t_emb = self.timbre_projection(timbre_vector)    # [B, D]
        t_emb = t_emb.unsqueeze(1).expand(-1, audio_embeds.shape[1], -1)  # [B, T, D]
        streams["timbre"] = t_emb
    
    audio_embeds = self.residual_fusion(
        audio_embeds,
        prosody=streams.get("prosody"),
        content=None,
        acoustic=None,
        timbre=streams.get("timbre"),
    )
# END enrichment
```

## Tasks

### Task 3.0: Unfreeze warm_start projector (prerequisite)

**Files:** `src/models/embedding.py`, `tests/models/test_embedding.py`

- [ ] **Step 0.1: Make projector trainable**

In `src/models/embedding.py:48`, change:
```python
param.requires_grad = False
```
→
```python
param.requires_grad = True
```

- [ ] **Step 0.2: Update test**

In `tests/models/test_embedding.py:70-82`, rename and invert the assertion:
```python
def test_warm_start_projector_is_trainable(self):
    """Warm-start projector is trainable for DPO."""
    codebook = torch.randn(PROSODY_VOCAB * 2, 32)
    emb = ProsodyEmbedding(
        vocab_size=PROSODY_VOCAB,
        embed_dim=EMBED_DIM,
        init_strategy="warm_start",
        warm_start_vectors=codebook,
    )
    assert hasattr(emb, '_projector'), "projector should exist in warm_start mode"
    for name, param in emb.named_parameters():
        assert param.requires_grad, f"{name} should be trainable"
```

- [ ] **Step 0.3: Run tests**

```bash
uv run python -m pytest tests/models/test_embedding.py -v
```

- [ ] **Step 0.4: Commit**

```bash
git add src/models/embedding.py tests/models/test_embedding.py
git commit -m "fix: make warm_start projector trainable for DPO (#19)"
```

### Task 3.1: Write AmyLM test (red)

**Files:**
- Create: `tests/models/test_amy_lm.py`

```python
"""Tests for AmyLM — Issue #19."""
import os
import tempfile
import pytest
import torch

# Ensure MOSS-Audio path is set up for imports
import sys
_vendor_src = os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "MOSS-Audio", "src")
if os.path.isdir(_vendor_src) and _vendor_src not in sys.path:
    sys.path.insert(0, _vendor_src)

from configuration_moss_audio import MossAudioConfig
from src.models.amy_lm import AmyLMConfig, AmyLM


class TestAmyLMConfig:
    
    def test_config_extends_moss_audio_config(self):
        """AmyLMConfig is a subclass of MossAudioConfig."""
        config = AmyLMConfig()
        assert isinstance(config, MossAudioConfig)
    
    def test_config_model_type(self):
        """model_type is 'amy_lm'."""
        config = AmyLMConfig()
        assert config.model_type == "amy_lm"
    
    def test_config_facodec_fields(self):
        """Config includes FACodec-specific fields with defaults."""
        config = AmyLMConfig()
        assert config.prosody_vocab_size == 1024
        assert config.prosody_init_strategy == "random"
        assert config.timbre_dim == 256
        assert config.hidden_dim == 2560
    
    def test_config_freeze_defaults(self):
        """Default freeze: encoder, adapter, LLM frozen."""
        config = AmyLMConfig()
        assert config.freeze_audio_encoder is True
        assert config.freeze_audio_adapter is True
        assert config.freeze_llm is True
    
    def test_config_serialization(self):
        """Config can be serialized to dict and back."""
        config = AmyLMConfig(prosody_vocab_size=512, timbre_dim=128)
        d = config.to_dict()
        assert d["prosody_vocab_size"] == 512
        assert d["timbre_dim"] == 128


class TestAmyLMModelStructure:
    
    @pytest.fixture
    def config(self):
        return AmyLMConfig(
            language_config={"vocab_size": 1000, "hidden_size": 2560},
        )
    
    @pytest.fixture
    def model(self, config):
        return AmyLM(config)
    
    def test_model_inherits_moss_audio(self, model):
        """AmyLM is a MossAudioModel."""
        from modeling_moss_audio import MossAudioModel
        assert isinstance(model, MossAudioModel)
    
    def test_model_has_facodec_modules(self, model):
        """Model has prosody, timbre, pooling, and fusion modules."""
        assert hasattr(model, "prosody_embedding")
        assert hasattr(model, "timbre_projection")
        assert hasattr(model, "temporal_pool")
        assert hasattr(model, "residual_fusion")
    
    def test_model_has_audio_modules(self, model):
        """Model has audio encoder, adapter, and language model."""
        assert hasattr(model, "audio_encoder")
        assert hasattr(model, "audio_adapter")
        assert hasattr(model, "language_model")
    
    def test_model_has_lm_head(self, model):
        """Model has lm_head for text generation."""
        assert hasattr(model, "lm_head")
    
    def test_freeze_applies_correctly(self, model):
        """Audio encoder, adapter, LLM are frozen by default."""
        for p in model.audio_encoder.parameters():
            assert not p.requires_grad
        for p in model.audio_adapter.parameters():
            assert not p.requires_grad
        for p in model.language_model.parameters():
            assert not p.requires_grad
    
    def test_facodec_modules_are_trainable(self, model):
        """FACodec modules are trainable by default."""
        # Prosody embedding (random init mode)
        trainable_params = sum(1 for p in model.prosody_embedding.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.prosody_embedding.parameters())
        assert trainable_params == total_params, "All prosody_embedding params should be trainable"
        
        # Timbre projection
        trainable_params = sum(1 for p in model.timbre_projection.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.timbre_projection.parameters())
        assert trainable_params == total_params, "All timbre_projection params should be trainable"
        
        # Lambda gates in fusion
        for name, p in model.residual_fusion.named_parameters():
            assert p.requires_grad, f"{name} should be trainable"
    
    def test_get_input_embeddings(self, model):
        """get_input_embeddings returns token embeddings."""
        emb = model.get_input_embeddings()
        assert emb is not None


class TestAmyLMForward:
    
    @pytest.fixture
    def config(self):
        return AmyLMConfig(
            language_config={"vocab_size": 1000, "hidden_size": 2560},
        )
    
    @pytest.fixture
    def model(self, config):
        return AmyLM(config)
    
    def test_forward_no_audio_no_prosody(self, model):
        """Forward with just text tokens (no audio, no FACodec)."""
        batch, seq = 2, 10
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(output, "logits")
        assert output.logits.shape[:2] == (batch, seq)
    
    def test_forward_with_prosody_indices(self):
        """Forward with prosody_indices enriches audio embeddings."""
        batch, seq = 2, 20
        config = AmyLMConfig(
            language_config={"vocab_size": 1000, "hidden_size": 2560},
        )
        model = AmyLM(config)
        
        # Create dummy input_ids with placeholder tokens
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        
        # Simulate audio features
        audio_data = torch.randn(batch, 128, 3000)  # [B, 128, T_mel]
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :20] = True  # First 20 tokens are audio
        
        # Prosody indices: [B, 1, T80]
        prosody_indices = torch.randint(0, 1024, (batch, 1, 80))
        
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
            prosody_indices=prosody_indices,
        )
        assert hasattr(output, "logits")
        assert output.logits.shape[:2] == (batch, seq)
    
    def test_forward_with_timbre_vector(self):
        """Forward with timbre_vector enriches audio embeddings."""
        batch, seq = 2, 20
        config = AmyLMConfig(
            language_config={"vocab_size": 1000, "hidden_size": 2560},
        )
        model = AmyLM(config)
        
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        audio_data = torch.randn(batch, 128, 3000)
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :20] = True
        
        timbre_vector = torch.randn(batch, 256)
        
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
            timbre_vector=timbre_vector,
        )
        assert hasattr(output, "logits")
        assert output.logits.shape[:2] == (batch, seq)
    
    def test_backward_compatible_no_facodec_inputs(self):
        """Without prosody/timbre, output matches base MossAudioModel behavior."""
        batch, seq = 2, 15
        config = AmyLMConfig(
            language_config={"vocab_size": 1000, "hidden_size": 2560},
        )
        model = AmyLM(config)
        
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        audio_data = torch.randn(batch, 128, 3000)
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :15] = True
        
        # No prosody or timbre
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
        )
        assert output.logits.shape[:2] == (batch, seq)
        # At lambda=0, output should match base model (identity through fusion)
        # This is verified by the ResidualFusion identity test
    
    def test_gradient_flows_through_facodec_modules(self):
        """Gradient flows through prosody embedding and timbre projection."""
        batch, seq = 2, 20
        config = AmyLMConfig(
            language_config={"vocab_size": 1000, "hidden_size": 2560},
        )
        model = AmyLM(config)
        
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        audio_data = torch.randn(batch, 128, 3000)
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :20] = True
        
        prosody_indices = torch.randint(0, 1024, (batch, 1, 80))
        timbre_vector = torch.randn(batch, 256)
        
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
            prosody_indices=prosody_indices,
            timbre_vector=timbre_vector,
        )
        
        loss = output.logits.sum()
        loss.backward()
        
        # FACodec modules should have gradients
        assert model.prosody_embedding.embedding.weight.grad is not None
        assert model.timbre_projection.linear.weight.grad is not None
        
        # Lambda gates should have gradients
        assert model.residual_fusion.lambda_p.grad is not None
        assert model.residual_fusion.lambda_t.grad is not None
        
        # Frozen modules should NOT have gradients
        for name, p in model.audio_encoder.named_parameters():
            assert p.grad is None, f"audio_encoder.{name} should have no grad"
```

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/models/test_amy_lm.py -v
```
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

### Task 3.2: Implement AmyLMConfig (green)

- [ ] **Step 3.2.1: Create `src/models/amy_lm.py`**

Implement `AmyLMConfig` with all FACodec fields. Ensure `to_dict()` and `from_pretrained()` work with the extra fields.

- [ ] **Step 3.2.2: Implement `AmyLM.__init__()`**

Instantiate all FACodec modules. Apply freeze. Call `post_init()`.

- [ ] **Step 3.2.3: Implement `AmyLM.forward()`**

Full override of `MossAudioModel.forward()` (lines 437–542) with enrichment step inserted. The method must:
1. Extract `prosody_indices` and `timbre_vector` from `**kwargs`
2. Compute `inputs_embeds` from `input_ids`
3. If `audio_data` provided: encode → adapt → enrich with FACodec streams → `masked_scatter_`
4. DeepStack hooks (preserved from parent)
5. Run `language_model` forward
6. Compute `lm_head` logits and optional loss

- [ ] **Step 3.2.4: Update `src/models/__init__.py`**

Add `AmyLMConfig`, `AmyLM` to exports.

- [ ] **Step 4: Run tests**

```bash
uv run python -m pytest tests/models/test_amy_lm.py -v
```

- [ ] **Step 5: Run full test suite**

```bash
uv run python -m pytest tests/ -x -q --ignore=tests/training -k "not (train_epoch or evaluate or save_load_roundtrip or training_step)"
```

- [ ] **Step 6: Commit**

```bash
git add src/models/amy_lm.py src/models/__init__.py tests/models/test_amy_lm.py
git commit -m "feat: AmyLM model + config — HF speech LM with FACodec enrichment (#19)"
```

## Gotchas

1. **Path setup**: `amy_lm.py` must replicate the vendor path injection from `moss_audio.py`. Import `MossAudioModel` and `MossAudioConfig` from the vendor path.

2. **kwargs extraction**: `pop()` the FACodec kwargs so the base model's `**kwargs` doesn't receive them (they'd be unrecognized by `language_model`).

3. **Sequence length alignment**: Prosody after `TemporalPool` may not match `audio_embeds` length exactly (80 Hz → 12.5 Hz vs. mel spectrogram downsampling). Truncate or zero-pad to align.

4. **DeepStack**: The parent model's DeepStack hooks must be preserved. Copy the exact DeepStack logic from `MossAudioModel.forward()`.

5. **HF compatibility**: `AmyLMConfig` must be serializable via `to_dict()` and reconstructable via `from_dict()`. The parent `MossAudioConfig.to_dict()` already handles `audio_config` and `language_config` — add FACodec fields to the output dict.

6. **`model_type`**: Must be `"amy_lm"` so HF can auto-resolve the model class from config. This means we may need a `register_for_auto_class()` call or `AutoModel` registration. For now, explicit class usage is fine.

7. **CPU testing**: These tests create a lightweight config (vocab=1000, hidden=2560) — no 4B weights loaded. Tests verify architectural correctness only. Full model loading is GPU-only.

## Phase Completion Criteria
- [ ] `AmyLMConfig` extends `MossAudioConfig` with FACodec fields
- [ ] `AmyLM` inherits `MossAudioModel` with FACodec enrichment modules
- [ ] `forward()` extracts `prosody_indices`/`timbre_vector` from `**kwargs`
- [ ] Without FACodec inputs, model is backward compatible (output shapes match)
- [ ] Gradient flows through FACodec modules (prosody, timbre, λ gates)
- [ ] Frozen backbone parameters (encoder, adapter, LLM) have no gradients
- [ ] Warm_start projector is trainable (verified by updated test)
- [ ] All tests pass
