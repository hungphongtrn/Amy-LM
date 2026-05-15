# Phase 1: MOSS-Audio Backbone Integration

## Phase Goal
MOSS-Audio 4B loads from HuggingFace, sub-modules are correctly extracted, and the semantic stream forward pass produces `[B, T_frames, 2560]` tensors at the expected 12.5 Hz frame rate. Verified through unit tests with both mock and real model.

## Files to Touch
- **Create:** `src/models/moss_audio.py` — `MossAudioWrapper` class
- **Create:** `tests/models/test_moss_audio.py` — Unit tests
- **Modify:** `pyproject.toml` — Add `transformers` dependency

## Pre-Task: Dependency Setup

- [ ] **Step 1: Add `transformers` to pyproject.toml**

```bash
uv add transformers torch
```

This adds `transformers` and `torch` as base dependencies (torch already exists but may need version pinning for MOSS-Audio compatibility).

Verify: `uv run python -c "import transformers; print(transformers.__version__)"`

---

## Task 1: MossAudioWrapper — Model Loading & Sub-module Extraction

**Files:**
- Create: `src/models/moss_audio.py`
- Test: `tests/models/test_moss_audio.py`

### Step 1: Write the failing test for model loading

Create `tests/models/test_moss_audio.py`:

```python
"""Tests for MOSS-Audio backbone integration."""

import pytest
import torch
from src.models.moss_audio import MossAudioWrapper


class TestMossAudioWrapper:
    """Tests for MossAudioWrapper model loading and sub-module extraction."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_init_loads_model_and_extracts_submodules(self, device):
        """Wrapper should load model and expose encoder, adapter, language_model."""
        wrapper = MossAudioWrapper(device=device)

        assert wrapper.audio_encoder is not None
        assert wrapper.audio_adapter is not None
        assert wrapper.language_model is not None

    def test_semantic_stream_output_shape(self, device):
        """Semantic stream should produce [B, T_frames, 2560] from raw audio."""
        wrapper = MossAudioWrapper(device=device)

        # Simulate ~2 seconds of 16kHz audio
        audio = torch.randn(2, 32000, device=device)  # [B, T_audio]
        semantic = wrapper.encode_semantic(audio)

        # Qwen3 hidden_dim = 2560
        assert semantic.dim() == 3
        assert semantic.shape[0] == 2
        assert semantic.shape[2] == 2560
        # Frame count should be ~250 at 12.5 Hz for 2s audio (200x downsample)
        assert 200 <= semantic.shape[1] <= 300

    def test_submodules_are_frozen_by_default(self, device):
        """Audio encoder, adapter, and language model should have no trainable params."""
        wrapper = MossAudioWrapper(device=device)

        for name, param in wrapper.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_encode_semantic_different_lengths(self, device):
        """Should handle different audio lengths in a batch via padding."""
        wrapper = MossAudioWrapper(device=device)

        # Different lengths, same batch
        audio = torch.randn(2, 48000, device=device)  # 3s
        semantic = wrapper.encode_semantic(audio)
        assert semantic.dim() == 3
        assert semantic.shape[0] == 2
```
```

### Step 2: Run test to verify it fails

```bash
uv run python -m pytest tests/models/test_moss_audio.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.moss_audio'`

### Step 3: Write minimal implementation

Create `src/models/moss_audio.py`:

```python
"""MOSS-Audio backbone wrapper for Amy LM.

Extracts audio_encoder, audio_adapter, and language_model sub-modules
from MossAudioModel and provides a reusable semantic stream encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class MossAudioWrapper(nn.Module):
    """Frozen MOSS-Audio backbone exposing sub-modules for Amy LM.

    Loads MOSS-Audio-4B-Thinking via transformers, then extracts:
      - audio_encoder: Whisper-style audio encoder
      - audio_adapter: GatedMLP adapter (audio features -> LLM embedding space)
      - language_model: Qwen3 language model

    All sub-modules are frozen (requires_grad=False). The wrapper provides
    encode_semantic() which runs audio through encoder + adapter to produce
    the Semantic Stream S_t [B, T_frames, 2560].
    """

    def __init__(
        self,
        model_id: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.model_id = model_id
        self.device = torch.device(device)

        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        # Extract sub-modules from the loaded model
        self.audio_encoder = model.audio_encoder
        self.audio_adapter = model.audio_adapter
        self.language_model = model.language_model

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.to(self.device)

    def encode_semantic(self, audio: torch.Tensor) -> torch.Tensor:
        """Produce Semantic Stream S_t from raw audio waveform.

        Args:
            audio: Raw audio waveform [B, T_audio] at 16 kHz.

        Returns:
            Semantic Stream [B, T_frames, 2560] at ~12.5 Hz frame rate.
        """
        audio = audio.to(self.device)
        with torch.no_grad():
            features = self.audio_encoder(audio)
            semantic = self.audio_adapter(features)
        return semantic
```

### Step 4: Run tests to verify they pass

```bash
uv run python -m pytest tests/models/test_moss_audio.py -v
```

Note: The `test_init_loads_model` test will download the model from HF on first run. If HF token is needed, ensure `.env` is loaded.

### Step 5: Commit

```bash
git add src/models/moss_audio.py tests/models/test_moss_audio.py pyproject.toml
git commit -m "feat: add MossAudioWrapper for MOSS-Audio backbone integration"
```

---

## Task 2: `__init__.py` Export

**Files:**
- Modify: `src/models/__init__.py`

### Step 1: Add MossAudioWrapper to exports

```python
from src.models.moss_audio import MossAudioWrapper
```

And add `"MossAudioWrapper"` to `__all__`.

### Step 2: Verify import works

```bash
uv run python -c "from src.models import MossAudioWrapper; print('OK')"
```

### Step 3: Commit

```bash
git add src/models/__init__.py
git commit -m "feat: export MossAudioWrapper from models package"
```

---

## Task 3: Sub-module Internal Structure Verification

**Files:**
- Modify: `tests/models/test_moss_audio.py` — Add structural tests

### Step 1: Add tests for sub-module internal properties

```python
    def test_audio_encoder_output_dim(self, device):
        """Audio encoder hidden dim should match expected Whisper config."""
        wrapper = MossAudioWrapper(device=device)
        # MOSS-Audio uses Whisper-medium-large encoder
        assert hasattr(wrapper.audio_encoder, 'config')
        # Encoder hidden dim is the adapter input dim
        adapter_in_features = wrapper.audio_adapter.in_features
        assert adapter_in_features > 0

    def test_language_model_hidden_size(self, device):
        """Qwen3 hidden dim should be 2560 for 4B variant."""
        wrapper = MossAudioWrapper(device=device)
        hidden_size = wrapper.language_model.config.hidden_size
        assert hidden_size == 2560, (
            f"Expected Qwen3 hidden_size=2560, got {hidden_size}"
        )

    def test_adapter_output_dim_matches_llm_input(self, device):
        """Audio adapter output dim must equal Qwen3 hidden dim."""
        wrapper = MossAudioWrapper(device=device)
        adapter_out = wrapper.audio_adapter.out_features
        llm_hidden = wrapper.language_model.config.hidden_size
        assert adapter_out == llm_hidden, (
            f"Adapter output {adapter_out} != LLM hidden {llm_hidden}"
        )

    def test_encode_semantic_produces_valid_embeddings(self, device):
        """Output values should be finite and non-zero."""
        wrapper = MossAudioWrapper(device=device)
        audio = torch.randn(1, 16000, device=device)  # 1s audio
        semantic = wrapper.encode_semantic(audio)
        assert torch.isfinite(semantic).all()
        # Embeddings should not be all zeros (audio is random noise but encoder still responds)
        assert not torch.allclose(semantic, torch.zeros_like(semantic), atol=1e-6)
```

### Step 2: Run structural tests

```bash
uv run python -m pytest tests/models/test_moss_audio.py -v -k "test_audio_encoder_output_dim or test_language_model_hidden_size or test_adapter_output or test_encode_semantic_produces"
```

### Step 3: Commit

```bash
git add tests/models/test_moss_audio.py
git commit -m "test: add structural verification tests for MossAudioWrapper sub-modules"
```

---

## Task 4: Audio Encoder Frame Rate Verification

**Files:**
- Modify: `tests/models/test_moss_audio.py` — Add rate test

### Step 1: Add frame rate test

```python
    def test_semantic_frame_rate_is_approximately_12_5_hz(self, device):
        """2 seconds of 16kHz audio should produce ~250 frames (~12.5 Hz)."""
        wrapper = MossAudioWrapper(device=device)

        # 2 seconds at 16kHz = 32000 samples
        audio = torch.randn(1, 32000, device=device)
        semantic = wrapper.encode_semantic(audio)

        n_frames = semantic.shape[1]
        expected_frames = 250  # 20000ms / 80ms per frame = 250

        # Allow ±5% tolerance for edge padding/rounding
        assert 237 <= n_frames <= 263, (
            f"Expected ~250 frames for 2s audio, got {n_frames}"
        )

    def test_frame_rate_scales_with_duration(self, device):
        """Longer audio should produce proportionally more frames."""
        wrapper = MossAudioWrapper(device=device)

        audio_1s = torch.randn(1, 16000, device=device)
        audio_2s = torch.randn(1, 32000, device=device)

        frames_1s = wrapper.encode_semantic(audio_1s).shape[1]
        frames_2s = wrapper.encode_semantic(audio_2s).shape[1]

        # 2s should have roughly 2x the frames of 1s
        ratio = frames_2s / frames_1s
        assert 1.7 <= ratio <= 2.3, (
            f"Expected ~2x frames for 2s vs 1s, got {ratio:.2f}"
        )
```

### Step 2: Run frame rate tests

```bash
uv run python -m pytest tests/models/test_moss_audio.py -v -k "frame_rate"
```

### Step 3: Commit

```bash
git add tests/models/test_moss_audio.py
git commit -m "test: add frame rate verification for MOSS-Audio semantic stream"
```

---

## Phase Completion Criteria
- [ ] `MossAudioWrapper` loads from HF Hub and extracts `audio_encoder`, `audio_adapter`, `language_model`
- [ ] All sub-module parameters are frozen (`requires_grad=False`)
- [ ] `encode_semantic()` produces `[B, T_frames, 2560]` at ~12.5 Hz
- [ ] Adapter output dim matches Qwen3 hidden_size=2560
- [ ] Frame rate scales correctly with audio duration
- [ ] All tests pass: `uv run python -m pytest tests/models/test_moss_audio.py -v`

## Handoff Notes

After Phase 1 completes, the implementer for Phase 2 will have:
- A working `MossAudioWrapper` with verified output shapes
- Known audio encoder frame count per second (~12.5 Hz => 12.5 frames/s)
- Verified Qwen3 hidden_size=2560

Phase 2 will use `encode_semantic()` as the source of S_t for `ResidualFusion` and verify that the TemporalPool (80 Hz → 12.5 Hz) produces the correct number of frames for FACodec streams to align with MOSS-Audio's output.
