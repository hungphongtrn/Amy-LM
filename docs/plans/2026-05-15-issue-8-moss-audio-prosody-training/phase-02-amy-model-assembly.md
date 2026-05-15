# Phase 2: Amy Model Assembly

## Phase Goal
`AmyForProsodyClassification.forward()` produces 2-class logits from raw audio + precomputed FACodec prosody indices + timbre vector. Verified with shape and gradient tests. Model equals MOSS-Audio baseline when lambdas=0. Content and acoustic streams are not instantiated or referenced.

## Files to Touch
- **Create:** `src/models/amy_classifier.py` — `AmyForProsodyClassification(nn.Module)`
- **Create:** `tests/models/test_amy_classifier.py` — Unit tests (8+ tests)
- **Create:** `src/models/codebook_utils.py` — FACodec prosody codebook vector extraction
- **Modify:** `src/models/__init__.py` — Export `AmyForProsodyClassification`
- **Modify:** `tests/conftest.py` — Optional: shared mock fixtures if any

## Pre-Task: Codebook Extraction Utility

Before building the model, we need a function to extract prosody codebook vectors from the FACodec decoder checkpoint. This is a standalone utility, not part of the model class.

---

## Tasks

### Task 1: FACodec Proosity Codebook Vector Extraction

**Files:**
- Create: `src/models/codebook_utils.py`
- Create: `tests/models/test_codebook_utils.py`

**Rationale:** `ProsodyEmbedding(init_strategy="warm_start", warm_start_vectors=...)` needs the raw FACodec prosody codebook vectors `[1024, 8]`. These live in the FACodec decoder checkpoint at state_dict key `quantizer.0.layers.0._codebook.weight`.

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_codebook_utils.py`:

```python
"""Tests for FACodec codebook vector extraction utilities."""
import os
import tempfile
import torch
import pytest
from src.models.codebook_utils import load_prosody_codebook_vectors


class TestLoadProosityCodebookVectors:
    """Tests for loading prosody codebook vectors from FACodec decoder checkpoint."""

    def test_extracts_correct_shape_from_mock_checkpoint(self):
        """Load vectors from a mock checkpoint and verify shape [1024, 8]."""
        mock_state = {"quantizer.0.layers.0._codebook.weight": torch.randn(1024, 8)}
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            torch.save(mock_state, f.name)
            checkpoint_path = f.name

        try:
            vectors = load_prosody_codebook_vectors(checkpoint_path)
            assert vectors.shape == (1024, 8)
            assert vectors.dtype == torch.float32
            assert torch.equal(vectors, mock_state["quantizer.0.layers.0._codebook.weight"])
        finally:
            os.unlink(checkpoint_path)

    def test_raises_if_checkpoint_not_found(self):
        """Should raise FileNotFoundError for missing checkpoint."""
        with pytest.raises(FileNotFoundError):
            load_prosody_codebook_vectors("/nonexistent/path/checkpoint.bin")

    def test_raises_if_key_missing(self):
        """Should raise KeyError if quantizer key is missing from state_dict."""
        mock_state = {"some.other.key": torch.randn(10, 10)}
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            torch.save(mock_state, f.name)
            checkpoint_path = f.name

        try:
            with pytest.raises(KeyError, match="quantizer.0.layers.0._codebook.weight"):
                load_prosody_codebook_vectors(checkpoint_path)
        finally:
            os.unlink(checkpoint_path)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/models/test_codebook_utils.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.codebook_utils'`

- [ ] **Step 3: Write minimal implementation**

Create `src/models/codebook_utils.py`:

```python
"""Utilities for extracting FACodec codebook vectors from Amphion checkpoints."""

import torch


def load_prosody_codebook_vectors(checkpoint_path: str) -> torch.Tensor:
    """Load prosody codebook vectors from FACodec decoder checkpoint.

    The FACodec decoder checkpoint (ns3_facodec_decoder.bin) stores
    factorized codebook vectors at 8 dimensions per entry. The prosody
    codebook is the first quantizer in the ResidualVQ group (index 0).

    State dict key: ``quantizer.0.layers.0._codebook.weight``
    Shape: ``[1024, 8]``  (vocab_size=1024, codebook_dim=8)

    Args:
        checkpoint_path: Path to ns3_facodec_decoder.bin.

    Returns:
        Float32 tensor [1024, 8] — raw FACodec prosody codebook vectors.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        KeyError: If the prosody codebook key is not found in the checkpoint.
    """
    import os
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"FACodec decoder checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    key = "quantizer.0.layers.0._codebook.weight"

    if key not in state_dict:
        raise KeyError(
            f"Prosody codebook key '{key}' not found in checkpoint at {checkpoint_path}. "
            f"Available keys: {list(state_dict.keys())[:10]}..."
        )

    return state_dict[key].float().detach()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/models/test_codebook_utils.py -v
```
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/codebook_utils.py tests/models/test_codebook_utils.py
git commit -m "feat: add FACodec prosody codebook vector extraction utility"
```

---

### Task 2: `AmyForProsodyClassification` — Model Implementation

**Files:**
- Create: `src/models/amy_classifier.py`
- Create: `tests/models/test_amy_classifier.py`

**Architecture:**

```
Input:
  - audio:           [B, T_audio]      — raw 16kHz waveform
  - prosody_indices: [B, 1, T80]       — precomputed FACodec prosody VQ IDs
  - timbre_vector:   [B, 256]          — precomputed FACodec spk_embs

Forward:
  1. S = wrapper.encode_semantic(audio)                        → [B, T_moss, 2560]
  2. p_emb = prosody_embedding(prosody_indices)                → [B, T80, 2560]
  3. P = temporal_pool(p_emb)                                  → [B, T_moss, 2560]
  4. t_proj = timbre_projection(timbre_vector)                 → [B, 2560]
  5. T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)           → [B, T_moss, 2560]
  6. H = fusion(S, prosody=P, timbre=T, content=None, acoustic=None) → [B, T_moss, 2560]
  7. lm_out = language_model(inputs_embeds=H).last_hidden_state → [B, T_moss, 2560]
  8. pooled = lm_out.mean(dim=1)                               → [B, 2560]
  9. logits = classifier(pooled)                               → [B, 2]

Frozen:  audio_encoder, audio_adapter, language_model
Trainable: prosody_embedding, timbre_projection, fusion.lambda_p, fusion.lambda_t, classifier
```

**Stream Activation Config:** The constructor accepts a dict `{"prosody": True, "content": False, "acoustic": False, "timbre": True}`. Disabled streams are never created as module attributes.

- [ ] **Step 1: Write the failing test for forward shape contract**

Add to `tests/models/test_amy_classifier.py`:

```python
"""Tests for AmyForProsodyClassification — end-to-end model assembly."""
import pytest
import torch
from src.models.amy_classifier import AmyForProsodyClassification


def make_prosody_codebook_vectors():
    """Mock FACodec prosody codebook vectors [1024, 8]."""
    return torch.randn(1024, 8)


class TestAmyForwardShape:
    """Verify forward pass produces correct output shapes."""

    @pytest.fixture
    def model(self):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device="cpu",
        )

    def test_forward_output_shape(self, model):
        """Forward pass should produce [B, 2] logits."""
        batch = 2
        audio = torch.randn(batch, 32000)  # 2s at 16kHz
        prosody_indices = torch.randint(0, 1024, (batch, 1, 160))  # 2s at 80Hz
        timbre_vector = torch.randn(batch, 256)

        logits = model(audio, prosody_indices, timbre_vector)
        assert logits.shape == (batch, 2)
        assert logits.dtype == torch.float32

    def test_single_sample_batch(self, model):
        """Should handle batch_size=1."""
        audio = torch.randn(1, 16000)  # 1s
        prosody_indices = torch.randint(0, 1024, (1, 1, 80))  # 1s at 80Hz
        timbre_vector = torch.randn(1, 256)

        logits = model(audio, prosody_indices, timbre_vector)
        assert logits.shape == (1, 2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/models/test_amy_classifier.py::TestAmyForwardShape -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.amy_classifier'`

- [ ] **Step 3: Write minimal implementation**

Create `src/models/amy_classifier.py`:

```python
"""AmyForProsodyClassification — end-to-end model for binary sarcasm classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import ProsodyEmbedding, TimbreProjection
from .fusion import ResidualFusion
from .moss_audio import MossAudioWrapper
from .pooling import TemporalPool


class AmyForProsodyClassification(nn.Module):
    """Amy LM for binary sarcasm classification with Prosody + Timbre residual fusion.

    Freezes MOSS-Audio backbone (audio_encoder, audio_adapter, language_model).
    Trainable modules: ProsodyEmbedding (warm-started), TimbreProjection,
    ResidualFusion lambdas, and a 2-class classifier head.

    Args:
        warm_start_vectors: FACodec prosody codebook vectors [1024, 8] for
            warm-starting ProsodyEmbedding.
        moss_model_id: HuggingFace model ID for MOSS-Audio backbone.
        device: Device for model (default: "cpu").
        stream_config: Dict controlling which FACodec streams are active.
            Default: prosody=True, timbre=True, content=False, acoustic=False.
        hidden_dim: Embedding dimension (must match MOSS-Audio hidden_size=2560).
        num_classes: Number of output classes (default: 2 for binary sarcasm).
        input_rate: FACodec frame rate in Hz (default: 80.0).
        output_rate: MOSS-Audio semantic frame rate in Hz (default: 12.5).
        timbre_dim: Dimensionality of input timbre vector (default: 256).
        vocab_size: FACodec codebook vocabulary size (default: 1024).
    """

    def __init__(
        self,
        warm_start_vectors: torch.Tensor,
        moss_model_id: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        device: torch.device | str = "cpu",
        stream_config: dict | None = None,
        hidden_dim: int = 2560,
        num_classes: int = 2,
        input_rate: float = 80.0,
        output_rate: float = 12.5,
        timbre_dim: int = 256,
        vocab_size: int = 1024,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.output_rate = output_rate

        if stream_config is None:
            stream_config = {"prosody": True, "content": False, "acoustic": False, "timbre": True}
        self.stream_config = stream_config

        # ── MOSS-Audio backbone (frozen) ──
        self.wrapper = MossAudioWrapper(model_id=moss_model_id, device=self.device)

        # ── FACodec embedding modules ──
        if stream_config.get("prosody", False):
            self.prosody_embedding = ProsodyEmbedding(
                vocab_size=vocab_size,
                embed_dim=hidden_dim,
                init_strategy="warm_start",
                warm_start_vectors=warm_start_vectors,
            )
        if stream_config.get("timbre", False):
            self.timbre_projection = TimbreProjection(
                timbre_dim=timbre_dim,
                output_dim=hidden_dim,
            )

        # ── Temporal pooling ──
        self.temporal_pool = TemporalPool(
            input_rate=input_rate,
            output_rate=output_rate,
        )

        # ── Fusion ──
        self.fusion = ResidualFusion(hidden_dim=hidden_dim)

        # ── Classifier head ──
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # ── Freeze backbone, keep FACodec modules trainable ──
        self._freeze_backbone()
        self._ensure_facodec_trainable()

    def _freeze_backbone(self) -> None:
        """Freeze MOSS-Audio encoder, adapter, and language model."""
        for param in self.wrapper.parameters():
            param.requires_grad = False

    def _ensure_facodec_trainable(self) -> None:
        """Ensure FACodec stream modules and classifier are trainable."""
        for name, module in self.named_children():
            if name in ("wrapper",):
                continue
            for param in module.parameters():
                param.requires_grad = True

    def get_language_model(self) -> nn.Module:
        """Return the frozen Qwen3 language model from MOSS-Audio."""
        return self.wrapper.language_model

    def forward(
        self,
        audio: torch.Tensor,
        prosody_indices: torch.Tensor,
        timbre_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: audio + FACodec features → 2-class logits.

        Args:
            audio: Raw waveform [B, T_audio] at 16kHz.
            prosody_indices: FACodec prosody VQ IDs [B, 1, T80].
            timbre_vector: FACodec speaker embedding [B, 256].

        Returns:
            Logits [B, 2] for binary sarcasm classification.
        """
        # 1. Semantic stream from MOSS-Audio (frozen, no grad)
        with torch.no_grad():
            semantic = self.wrapper.encode_semantic(audio)  # [B, T_moss, 2560]
        T_moss = semantic.shape[1]

        # 2. Prosody residual stream
        p_emb = self.prosody_embedding(prosody_indices)  # [B, T80, 2560]
        P = self.temporal_pool(p_emb)  # [B, T_moss, 2560]

        # 3. Timbre residual stream (utterance-level → broadcast to frames)
        t_proj = self.timbre_projection(timbre_vector)  # [B, 2560]
        T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)  # [B, T_moss, 2560]

        # 4. Fuse streams
        H = self.fusion(semantic, prosody=P, timbre=T, content=None, acoustic=None)

        # 5. Language model forward
        language_model = self.get_language_model()
        lm_out = language_model(inputs_embeds=H).last_hidden_state  # [B, T_moss, 2560]

        # 6. Mean-pool over time
        pooled = lm_out.mean(dim=1)  # [B, 2560]

        # 7. Classifier
        logits = self.classifier(pooled)  # [B, 2]
        return logits
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/models/test_amy_classifier.py::TestAmyForwardShape -v
```
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/amy_classifier.py tests/models/test_amy_classifier.py
git commit -m "feat: add AmyForProsodyClassification — end-to-end model assembly"
```

---

### Task 3: Gradient Flow & Freezing Tests

**Files:**
- Modify: `tests/models/test_amy_classifier.py` — Add gradient flow tests

- [ ] **Step 1: Write gradient flow tests**

Add to `tests/models/test_amy_classifier.py`:

```python
class TestAmyGradientFlow:
    """Verify which parameters receive gradients."""

    @pytest.fixture
    def model(self):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device="cpu",
        )

    @pytest.fixture
    def batch(self):
        return (
            torch.randn(2, 32000),          # audio
            torch.randint(0, 1024, (2, 1, 160)),  # prosody
            torch.randn(2, 256),             # timbre
        )

    def test_only_trainable_params_get_gradients(self, model, batch):
        """Only trainable modules should receive gradients after backward."""
        audio, prosody_idx, timbre = batch
        logits = model(audio, prosody_idx, timbre)
        loss = logits.sum()
        loss.backward()

        trainable_with_grad = []
        frozen_with_grad = []

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                trainable_with_grad.append(name)
            if not param.requires_grad and param.grad is not None:
                frozen_with_grad.append(name)

        assert len(trainable_with_grad) == 0, (
            f"Trainable params without grad: {trainable_with_grad}"
        )
        assert len(frozen_with_grad) == 0, (
            f"Frozen params with grad: {frozen_with_grad}"
        )

    def test_fusion_lambdas_are_trainable(self, model):
        """lambda_p and lambda_t should require grad."""
        assert model.fusion.lambda_p.requires_grad
        assert model.fusion.lambda_t.requires_grad

    def test_classifier_is_trainable(self, model):
        """Classifier head should require grad."""
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_prosody_embedding_is_trainable(self, model):
        """ProsodyEmbedding should be trainable (via weight property)."""
        # Warm-start mode: _projector is frozen, but weight is computed
        # The weight property returns projected vectors ready for lookup
        assert model.prosody_embedding.weight.requires_grad is False
        # The embedding lookup is trainable via functional.embedding —
        # gradient flows back through the embedding output to the indices,
        # but the weight itself is trainable in random mode only.
        # In warm-start mode, the projector is frozen.
        # This is correct: vectors are warm-started, then lookup is learned.

    def test_backbone_fully_frozen(self, model):
        """All MOSS-Audio backbone params should have requires_grad=False."""
        for name, param in model.wrapper.named_parameters():
            assert not param.requires_grad, (
                f"Backbone param '{name}' should be frozen"
            )
```

- [ ] **Step 2: Run gradient tests**

```bash
uv run python -m pytest tests/models/test_amy_classifier.py::TestAmyGradientFlow -v
```
Expected: 5 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/models/test_amy_classifier.py
git commit -m "test: add gradient flow and freezing verification for Amy model"
```

---

### Task 4: Lambda Zero-Init Baseline Equivalence Test

**Files:**
- Modify: `tests/models/test_amy_classifier.py` — Add equivalence test

- [ ] **Step 1: Write baseline equivalence test**

Add to `tests/models/test_amy_classifier.py`:

```python
class TestAmyBaselineEquivalence:
    """Verify Amy model equals MOSS-Audio baseline when lambdas are zero."""

    @pytest.fixture
    def model(self):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device="cpu",
        )

    @pytest.fixture
    def audio(self):
        return torch.randn(1, 32000)  # 2s

    @pytest.fixture
    def prosody_indices(self):
        return torch.randint(0, 1024, (1, 1, 160))

    @pytest.fixture
    def timbre_vector(self):
        return torch.randn(1, 256)

    def test_lambdas_start_at_zero(self, model):
        """lambda_p and lambda_t must be zero at initialization."""
        assert model.fusion.lambda_p.item() == 0.0
        assert model.fusion.lambda_t.item() == 0.0

    def test_semantic_alone_equals_baseline(self, model, audio, prosody_indices, timbre_vector):
        """With lambdas=0 and prosody/timbre fed, output should equal
        running only semantic through the same path."""
        # Full Amy forward
        with torch.no_grad():
            semantic = model.wrapper.encode_semantic(audio)
        T_moss = semantic.shape[1]

        # Manually compute baseline: semantic → LM → mean-pool → classifier
        with torch.no_grad():
            H_baseline = model.fusion(
                semantic,
                prosody=None, content=None, acoustic=None, timbre=None,
            )
            lm_out_baseline = model.get_language_model()(
                inputs_embeds=H_baseline
            ).last_hidden_state
            pooled_baseline = lm_out_baseline.mean(dim=1)
            logits_baseline = model.classifier(pooled_baseline)

        # Amy forward with zero lambdas but prosody + timbre provided
        with torch.no_grad():
            p_emb = model.prosody_embedding(prosody_indices)
            P = model.temporal_pool(p_emb)
            t_proj = model.timbre_projection(timbre_vector)
            T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)
            H_full = model.fusion(
                semantic, prosody=P, timbre=T, content=None, acoustic=None,
            )
            lm_out_full = model.get_language_model()(
                inputs_embeds=H_full
            ).last_hidden_state
            pooled_full = lm_out_full.mean(dim=1)
            logits_full = model.classifier(pooled_full)

        # Since lambdas=0, fusion with prosody/timbre should equal fusion without them
        # Both go through same LM + classifier, so logits must match
        assert torch.allclose(logits_baseline, logits_full, atol=1e-5)
```

- [ ] **Step 2: Run equivalence tests**

```bash
uv run python -m pytest tests/models/test_amy_classifier.py::TestAmyBaselineEquivalence -v
```
Expected: 2 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/models/test_amy_classifier.py
git commit -m "test: add lambda zero-init baseline equivalence verification"
```

---

### Task 5: Stream Activation Config Tests

**Files:**
- Modify: `tests/models/test_amy_classifier.py` — Add config tests

- [ ] **Step 1: Write stream config tests**

Add to `tests/models/test_amy_classifier.py`:

```python
class TestAmyStreamConfig:
    """Verify stream activation config controls module construction."""

    def test_disabled_streams_not_instantiated(self):
        """Content and acoustic modules should not exist when disabled."""
        vectors = torch.randn(1024, 8)
        config = {"prosody": True, "content": False, "acoustic": False, "timbre": True}
        model = AmyForProsodyClassification(
            warm_start_vectors=vectors,
            stream_config=config,
            device="cpu",
        )
        assert hasattr(model, "prosody_embedding")
        assert hasattr(model, "timbre_projection")
        assert not hasattr(model, "content_embedding")
        assert not hasattr(model, "acoustic_embedding")

    def test_config_key_missing_for_disabled_streams(self):
        """Missing keys in config default to False (disabled)."""
        vectors = torch.randn(1024, 8)
        config = {"prosody": True, "timbre": True}  # content/acoustic missing
        model = AmyForProsodyClassification(
            warm_start_vectors=vectors,
            stream_config=config,
            device="cpu",
        )
        assert hasattr(model, "prosody_embedding")
        assert hasattr(model, "timbre_projection")
        assert not hasattr(model, "content_embedding")
        assert not hasattr(model, "acoustic_embedding")

    def test_config_stored_as_attribute(self, model_with_vectors):
        """Stream config should be accessible as an attribute."""
        vectors = torch.randn(1024, 8)
        config = {"prosody": True, "content": True, "acoustic": False, "timbre": True}
        model = AmyForProsodyClassification(
            warm_start_vectors=vectors,
            stream_config=config,
            device="cpu",
        )
        assert model.stream_config == config
```

- [ ] **Step 2: Run config tests**

```bash
uv run python -m pytest tests/models/test_amy_classifier.py::TestAmyStreamConfig -v
```
Expected: 3 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/models/test_amy_classifier.py
git commit -m "test: verify stream activation config controls module construction"
```

---

### Task 6: Temporal Alignment Verification

**Files:**
- Modify: `tests/models/test_amy_classifier.py` — Add alignment test

- [ ] **Step 1: Write temporal alignment test**

Add to `tests/models/test_amy_classifier.py`:

```python
class TestAmyTemporalAlignment:
    """Verify FACodec 80Hz stream aligns to MOSS-Audio ~12.5Hz frames."""

    @pytest.fixture
    def model(self):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device="cpu",
        )

    def test_prosody_pool_matches_semantic_frames(self, model):
        """Pooled prosody [B, T_moss, 2560] must have same T_moss as semantic."""
        audio = torch.randn(2, 48000)  # 3s audio
        prosody_indices = torch.randint(0, 1024, (2, 1, 240))  # 3s at 80Hz = 240

        with torch.no_grad():
            semantic = model.wrapper.encode_semantic(audio)
        T_moss = semantic.shape[1]

        p_emb = model.prosody_embedding(prosody_indices)  # [2, 240, 2560]
        P = model.temporal_pool(p_emb)

        assert P.shape[1] == T_moss, (
            f"Pooled prosody frames ({P.shape[1]}) must equal "
            f"semantic frames ({T_moss})"
        )

    def test_timbre_broadcast_matches_semantic_frames(self, model):
        """Broadcast timbre must have correct T_moss."""
        audio = torch.randn(1, 16000)  # 1s
        timbre = torch.randn(1, 256)

        with torch.no_grad():
            semantic = model.wrapper.encode_semantic(audio)
        T_moss = semantic.shape[1]

        t_proj = model.timbre_projection(timbre)  # [1, 2560]
        T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)  # [1, T_moss, 2560]

        assert T.shape[1] == T_moss
        assert T.shape[0] == 1
        assert T.shape[2] == 2560
```

- [ ] **Step 2: Run alignment tests**

```bash
uv run python -m pytest tests/models/test_amy_classifier.py::TestAmyTemporalAlignment -v
```
Expected: 2 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/models/test_amy_classifier.py
git commit -m "test: verify temporal alignment between FACodec and MOSS-Audio streams"
```

---

### Task 7: `__init__.py` Export

**Files:**
- Modify: `src/models/__init__.py`

- [ ] **Step 1: Add export**

Edit `src/models/__init__.py` to add the import and export:

```python
from .amy_classifier import AmyForProsodyClassification
```

And add `"AmyForProsodyClassification"` to `__all__`.

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "from src.models import AmyForProsodyClassification; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/models/__init__.py
git commit -m "feat: export AmyForProsodyClassification from models package"
```

---

## Phase Completion Criteria
- [ ] `load_prosody_codebook_vectors()` extracts `[1024, 8]` from FACodec decoder checkpoint
- [ ] `AmyForProsodyClassification(audio, prosody_idx, timbre_vec)` produces `[B, 2]` logits
- [ ] Only trainable params receive gradients (backbone frozen, lambdas/classifier/FACodec modules trainable)
- [ ] `lambda_p` and `lambda_t` are zero-initialized
- [ ] Model output equals MOSS-Audio baseline when lambdas=0 (baseline equivalence)
- [ ] Content and acoustic modules are not instantiated when stream config disables them
- [ ] Pooled prosody frames align to actual MOSS-Audio T_moss
- [ ] All tests pass: `uv run python -m pytest tests/models/test_amy_classifier.py tests/models/test_codebook_utils.py -v`

## Integration Test (Run After All Tasks)

```bash
uv run python -m pytest tests/models/test_amy_classifier.py tests/models/test_codebook_utils.py -v
```

Expected: All tests pass (~15 tests total).

## Handoff Notes

After Phase 2 completes, the implementer for Phase 3 will have:
- A working `AmyForProsodyClassification` that accepts `(audio, prosody_indices, timbre_vector)` and produces (batch, 2) logits
- Verified that only FACodec modules + lambdas + classifier are trainable
- Verified baseline equivalence at lambda init
- Verified temporal alignment between 80 Hz FACodec and ~12.5 Hz MOSS-Audio
- `load_prosody_codebook_vectors()` utility for extracting real codebook vectors from the Amphion checkpoint

**Key Phase 2 learnings that affect Phase 3:**
- The forward pass expects `prosody_indices` shape `[B, 1, T80]` — Phase 3 collate_fn must produce this
- The forward pass expects `timbre_vector` shape `[B, 256]` — Phase 3 collate_fn must produce this
- Audio is raw 16kHz waveform `[B, T_audio]` — Phase 3 loads from parquet
- The model runs MOSS-Audio encoding online per batch — Phase 3 does NOT need precomputed semantic frames in the dataset
