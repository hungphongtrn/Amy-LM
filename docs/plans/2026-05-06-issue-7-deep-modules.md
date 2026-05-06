# Deep Modules: Embedding Tables, Temporal Pooling, Residual Fusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use @subagent-driven-development (recommended) or @executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three deep neural modules (Embedding Tables, Temporal Pooling, Residual Fusion) with comprehensive unit tests for the FACodec Residual Extension in Amy-LM.

**Architecture:** Three standalone `nn.Module`s with clean interfaces. `ProsodyEmbedding` and `TimbreEmbedding` map discrete codebook indices to continuous embeddings (1024 → 2560 and 256 → 2560 respectively), supporting both random init and warm-start from FACodec codebook vectors. `TemporalPool` compresses embedded prosody from FACodec's native 80 Hz to Amy-LM's 12.5 Hz using `nn.AdaptiveAvgPool1d` (handles the 6.4:1 non-integer ratio). `ResidualFusion` combines semantic, prosody, and timbre streams via `LayerNorm(semantic + λ · (prosody + timbre))` where λ is a learnable scalar initialized at zero.

**Tech Stack:** PyTorch 2.x, pytest

---

## File Structure

### New Files:
| File | Responsibility |
|------|---------------|
| `src/models/embedding.py` | `ProsodyEmbedding`, `TimbreEmbedding` — embedding tables with init strategies |
| `src/models/pooling.py` | `TemporalPool` — adaptive average pooling 80 Hz → 12.5 Hz |
| `src/models/fusion.py` | `ResidualFusion` — learnable residual combination |
| `src/models/__init__.py` | Public exports for the three modules |
| `tests/models/__init__.py` | Empty init for test package |
| `tests/models/test_embedding.py` | Unit tests for embedding tables |
| `tests/models/test_pooling.py` | Unit tests for temporal pooling |
| `tests/models/test_fusion.py` | Unit tests for residual fusion |

### No Existing Files Modified.

---

## Context: Frame Rate Clarification

FACodec outputs at **80 Hz** (hop_size=200 at 16kHz, confirmed at `src/preprocessing/facodec_encoder.py:52-53`). Amy-LM operates at **12.5 Hz**. The ratio is **80/12.5 = 6.4:1** (not 6:1 as originally scoped).

`nn.AdaptiveAvgPool1d` solves the non-integer ratio: you specify target output length; it automatically sizes windows so the overall ratio is exact. Most frames pool 6 inputs, roughly every 2.5th frame pools 7 inputs to absorb the 0.4 remainder.

---

## Task 1: Embedding Tables (`ProsodyEmbedding`, `TimbreEmbedding`)

### Task 1.1: Write failing tests for embedding tables

**Files:**
- Create: `tests/models/test_embedding.py`

- [ ] **Step 1: Create test file with all test cases**

```python
import pytest
import torch
from src.models.embedding import ProsodyEmbedding, TimbreEmbedding

PROSODY_VOCAB = 1024
TIMBRE_VOCAB = 256
EMBED_DIM = 2560


def make_prosody_indices(batch=2, seq=25):
    return torch.randint(0, PROSODY_VOCAB, (batch, seq))


def make_timbre_indices(batch=2):
    return torch.randint(0, TIMBRE_VOCAB, (batch,))


class TestProsodyEmbedding:
    @pytest.fixture
    def random_emb(self):
        return ProsodyEmbedding(
            vocab_size=PROSODY_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="random",
            init_std=0.02,
        )

    def test_random_init_output_shape(self, random_emb):
        indices = make_prosody_indices(batch=2, seq=25)
        out = random_emb(indices)
        assert out.shape == (2, 25, EMBED_DIM)

    def test_random_init_weight_stats(self, random_emb):
        weight = random_emb.weight
        mean = weight.mean().item()
        std = weight.std().item()
        assert abs(mean) < 0.1
        assert 0.01 < std < 0.03

    def test_random_init_same_index_same_vector(self, random_emb):
        indices = torch.tensor([7, 7])
        out = random_emb(indices)
        assert torch.allclose(out[0], out[1])

    def test_warm_start_produces_correct_shape(self):
        codebook = torch.randn(PROSODY_VOCAB * 2, 32)  # mimic FACodec raw codebook
        emb = ProsodyEmbedding(
            vocab_size=PROSODY_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="warm_start",
            warm_start_vectors=codebook,
        )
        indices = make_prosody_indices()
        out = emb(indices)
        assert out.shape == (2, 25, EMBED_DIM)

    def test_warm_start_gradients_flow(self):
        codebook = torch.randn(PROSODY_VOCAB * 2, 32)
        emb = ProsodyEmbedding(
            vocab_size=PROSODY_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="warm_start",
            warm_start_vectors=codebook,
        )
        indices = make_prosody_indices()
        out = emb(indices)
        loss = out.sum()
        loss.backward()
        for name, param in emb.named_parameters():
            assert param.grad is not None, f"{name} should have grad"

    def test_invalid_init_strategy_raises(self):
        with pytest.raises(ValueError, match="init_strategy"):
            ProsodyEmbedding(
                vocab_size=PROSODY_VOCAB,
                embed_dim=EMBED_DIM,
                init_strategy="nonexistent",
            )

    def test_warm_start_without_vectors_raises(self):
        with pytest.raises(ValueError, match="warm_start_vectors"):
            ProsodyEmbedding(
                vocab_size=PROSODY_VOCAB,
                embed_dim=EMBED_DIM,
                init_strategy="warm_start",
            )


class TestTimbreEmbedding:
    @pytest.fixture
    def random_emb(self):
        return TimbreEmbedding(
            vocab_size=TIMBRE_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="random",
        )

    def test_random_init_output_shape(self, random_emb):
        indices = make_timbre_indices(batch=4)
        out = random_emb(indices)
        assert out.shape == (4, EMBED_DIM)

    def test_per_sample_constant_vector(self, random_emb):
        """Timbre embedding is per-sample: same index repeated should give same vector."""
        indices = torch.tensor([42, 42])
        out = random_emb(indices)
        assert torch.allclose(out[0], out[1])

    def test_warm_start_shape(self):
        codebook = torch.randn(TIMBRE_VOCAB, 320)
        emb = TimbreEmbedding(
            vocab_size=TIMBRE_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="warm_start",
            warm_start_vectors=codebook,
        )
        indices = make_timbre_indices()
        out = emb(indices)
        assert out.shape == (2, EMBED_DIM)

    def test_broadcastable_to_sequence_dim(self, random_emb):
        """Timbre vector should be broadcastable to (B, T, D) for downstream fusion."""
        indices = make_timbre_indices(batch=4)
        out = random_emb(indices)
        out_expanded = out.unsqueeze(1).expand(4, 30, EMBED_DIM)
        assert out_expanded.shape == (4, 30, EMBED_DIM)
        for b in range(4):
            for t in range(30):
                assert torch.equal(out_expanded[b, t], out[b])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
mkdir -p tests/models && touch tests/models/__init__.py
uv run python -m pytest tests/models/test_embedding.py -v 2>&1 | tail -20
```

Expected: All FAIL with `ModuleNotFoundError: No module named 'src.models.embedding'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/models/__init__.py tests/models/test_embedding.py
git commit -m "test: add failing tests for ProsodyEmbedding and TimbreEmbedding"
```

---

### Task 1.2: Implement `ProsodyEmbedding` and `TimbreEmbedding`

**Files:**
- Create: `src/models/embedding.py`

- [ ] **Step 1: Implement embedding tables**

```python
"""Embedding tables for prosody and timbre discrete codebook indices."""
from typing import Optional
import torch
import torch.nn as nn


class ProsodyEmbedding(nn.Module):
    """Embedding table for prosody codebook indices (vocab 1024 → 2560 dims).
    
    Supports two initialization strategies:
    - ``"random"``: Normal(0, 0.02), standard PyTorch embedding init.
    - ``"warm_start"``: Loads raw FACodec codebook vectors via a static Linear
      projector. The projector is frozen (not trainable) after init.
    
    Args:
        vocab_size: Number of entries in the prosody codebook (1024 for FACodec).
        embed_dim: Output embedding dimension (2560).
        init_strategy: ``"random"`` or ``"warm_start"``.
        init_std: Std for random init (default 0.02).
        warm_start_vectors: Raw FACodec codebook weights, shape 
            ``(num_entries * codebook_dim,)``. Required when ``init_strategy="warm_start"``.
    """
    
    def __init__(
        self,
        vocab_size: int = 1024,
        embed_dim: int = 2560,
        init_strategy: str = "random",
        init_std: float = 0.02,
        warm_start_vectors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if init_strategy not in ("random", "warm_start"):
            raise ValueError(
                f"init_strategy must be 'random' or 'warm_start', got '{init_strategy}'"
            )
        if init_strategy == "warm_start" and warm_start_vectors is None:
            raise ValueError(
                "warm_start_vectors is required when init_strategy='warm_start'"
            )
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init_strategy = init_strategy
        
        if init_strategy == "random":
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        elif init_strategy == "warm_start":
            codebook_dim = warm_start_vectors.shape[-1]
            self._projector = nn.Linear(codebook_dim, embed_dim, bias=False)
            self._embedding: Optional[nn.Embedding] = None
            self._init_from_facodec(warm_start_vectors)
    
    def _init_from_facodec(self, warm_start_vectors: torch.Tensor):
        """Initialize embedding from FACodec codebook via static Linear projector."""
        weight = warm_start_vectors.detach()
        with torch.no_grad():
            self._projector.weight.data.normal_(mean=0.0, std=0.02)
        for param in self._projector.parameters():
            param.requires_grad = False
        self.register_buffer("_facodec_weights", weight)
    
    @property
    def weight(self) -> torch.Tensor:
        if self.init_strategy == "random":
            return self.embedding.weight
        projected = self._projector(self._facodec_weights)
        num_entries = self._facodec_weights.shape[0]
        target_entries = self.vocab_size
        if num_entries >= target_entries:
            return projected[:target_entries]
        repeats = (target_entries + num_entries - 1) // num_entries
        tiled = projected.repeat(repeats, 1)
        return tiled[:target_entries]
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        w = self.weight
        return nn.functional.embedding(indices, w)


class TimbreEmbedding(nn.Module):
    """Embedding table for timbre codebook indices (vocab 256 → 2560 dims).
    
    Output is per-sample (no sequence dimension). This represents speaker identity
    as a single global vector per utterance.
    
    Args:
        vocab_size: Number of entries in the timbre codebook (256 for FACodec).
        embed_dim: Output embedding dimension (2560).
        init_strategy: ``"random"`` or ``"warm_start"``.
        init_std: Std for random init (default 0.02).
        warm_start_vectors: Raw FACodec codebook weights. Required for warm-start.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 2560,
        init_strategy: str = "random",
        init_std: float = 0.02,
        warm_start_vectors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if init_strategy not in ("random", "warm_start"):
            raise ValueError(
                f"init_strategy must be 'random' or 'warm_start', got '{init_strategy}'"
            )
        if init_strategy == "warm_start" and warm_start_vectors is None:
            raise ValueError(
                "warm_start_vectors is required when init_strategy='warm_start'"
            )
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init_strategy = init_strategy
        
        if init_strategy == "random":
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        elif init_strategy == "warm_start":
            codebook_dim = warm_start_vectors.shape[-1]
            self._projector = nn.Linear(codebook_dim, embed_dim, bias=False)
            self._embedding: Optional[nn.Embedding] = None
            with torch.no_grad():
                self._projector.weight.data.normal_(mean=0.0, std=0.02)
            for param in self._projector.parameters():
                param.requires_grad = False
            self.register_buffer("_facodec_weights", warm_start_vectors.detach())
    
    @property
    def weight(self) -> torch.Tensor:
        if self.init_strategy == "random":
            return self.embedding.weight
        projected = self._projector(self._facodec_weights)
        num_entries = projected.shape[0]
        target_entries = self.vocab_size
        if num_entries >= target_entries:
            return projected[:target_entries]
        repeats = (target_entries + num_entries - 1) // num_entries
        tiled = projected.repeat(repeats, 1)
        return tiled[:target_entries]
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup per-sample timbre embeddings.
        
        Args:
            indices: Long tensor shape ``(batch,)`` with codebook indices.
        
        Returns:
            Tensor shape ``(batch, embed_dim)`` — global timbre vector per sample.
        """
        if indices.dim() != 1:
            raise ValueError(
                f"TimbreEmbedding expects 1D indices (batch,), got shape {indices.shape}"
            )
        w = self.weight
        return nn.functional.embedding(indices, w)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run python -m pytest tests/models/test_embedding.py -v
```

Expected: All PASS (13 tests)

- [ ] **Step 3: Commit embedding tables**

```bash
git add src/models/embedding.py
git commit -m "feat: add ProsodyEmbedding and TimbreEmbedding with random/warm-start init strategies"
```

---

## Task 2: Temporal Pooling

### Task 2.1: Write failing tests for TemporalPool

**Files:**
- Create: `tests/models/test_pooling.py`

- [ ] **Step 1: Create test file**

```python
"""Tests for TemporalPool — compresses embedded prosody 80 Hz → 12.5 Hz."""
import pytest
import torch
from src.models.pooling import TemporalPool

INPUT_RATE = 80.0
OUTPUT_RATE = 12.5
EMBED_DIM = 2560


def make_prosody_sequence(batch=2, duration_sec=2.0):
    """Create synthetic embedded prosody at 80 Hz."""
    num_frames = int(duration_sec * INPUT_RATE)
    return torch.randn(batch, num_frames, EMBED_DIM)


class TestTemporalPool:
    @pytest.fixture
    def pool(self):
        return TemporalPool(input_rate=INPUT_RATE, output_rate=OUTPUT_RATE)

    def test_output_rate_correct_2sec(self, pool):
        x = make_prosody_sequence(duration_sec=2.0)
        out = pool(x)
        expected_frames = int(2.0 * OUTPUT_RATE)
        assert out.shape[1] == expected_frames, (
            f"Expected {expected_frames} frames at {OUTPUT_RATE} Hz, got {out.shape[1]}"
        )

    def test_output_rate_correct_5sec(self, pool):
        x = make_prosody_sequence(batch=1, duration_sec=5.0)
        out = pool(x)
        expected_frames = int(5.0 * OUTPUT_RATE)
        assert out.shape[1] == expected_frames

    def test_output_rate_correct_0_64sec(self, pool):
        """Short input: 0.64 sec = 51 frames at 80 Hz → 8 frames at 12.5 Hz."""
        x = make_prosody_sequence(batch=1, duration_sec=0.64)
        out = pool(x)
        expected_frames = int(0.64 * OUTPUT_RATE)
        assert out.shape[1] == expected_frames

    def test_embed_dim_preserved(self, pool):
        x = make_prosody_sequence()
        out = pool(x)
        assert out.shape[2] == EMBED_DIM

    def test_batch_dim_preserved(self, pool):
        x = make_prosody_sequence(batch=4, duration_sec=1.5)
        out = pool(x)
        assert out.shape[0] == 4

    def test_handles_non_divisible_frame_count(self, pool):
        """Frame count where 80/12.5 doesn't divide cleanly: 127 frames at 80 Hz."""
        x = torch.randn(2, 127, EMBED_DIM)
        out = pool(x)
        expected = round(127 * OUTPUT_RATE / INPUT_RATE)
        assert out.shape[1] == expected

    def test_pooled_values_bounded(self, pool):
        """Mean-pooled values should stay within input range."""
        x = make_prosody_sequence(duration_sec=3.0)
        out = pool(x)
        assert out.min() >= x.min() - 1e-6
        assert out.max() <= x.max() + 1e-6

    def test_gradient_flows_through_pool(self, pool):
        x = make_prosody_sequence(duration_sec=1.0)
        x.requires_grad = True
        out = pool(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_identity_for_single_frame(self, pool):
        """Single frame input should produce single frame output (trivial pool)."""
        x = torch.randn(2, 1, EMBED_DIM)
        out = pool(x)
        assert out.shape == (2, 1, EMBED_DIM)

    def test_deterministic_output(self, pool):
        torch.manual_seed(42)
        x1 = torch.randn(2, 80, EMBED_DIM)
        torch.manual_seed(42)
        x2 = torch.randn(2, 80, EMBED_DIM)
        out1 = pool(x1)
        out2 = pool(x2)
        assert torch.allclose(out1, out2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/models/test_pooling.py -v 2>&1 | tail -15
```

Expected: All FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/models/test_pooling.py
git commit -m "test: add failing tests for TemporalPool"
```

---

### Task 2.2: Implement `TemporalPool`

**Files:**
- Create: `src/models/pooling.py`

- [ ] **Step 1: Implement TemporalPool**

```python
"""Temporal pooling for downsampling embedded prosody streams."""
import torch.nn as nn
import torch.nn.functional as F

class TemporalPool(nn.Module):
    """Compress embedded prosody using adaptive average pooling.
    
    Takes embedded prosody at ``input_rate`` Hz (typically 80 Hz from FACodec)
    and outputs at ``output_rate`` Hz (typically 12.5 Hz for Amy-LM).
    
    Uses ``nn.AdaptiveAvgPool1d`` internally, which handles non-integer
    downsampling ratios (e.g., 80/12.5 = 6.4) by varying window sizes:
    most output frames average 6 input frames, with occasional 7-frame
    windows to absorb the 0.4 remainder.
    
    Args:
        input_rate: Input sequence frame rate in Hz (default 80.0).
        output_rate: Target output frame rate in Hz (default 12.5).
    
    Example:
        >>> pool = TemporalPool(input_rate=80.0, output_rate=12.5)
        >>> x = torch.randn(2, 160, 2560)   # 2 sec at 80 Hz
        >>> out = pool(x)
        >>> out.shape  # (2, 25, 2560) — 2 sec at 12.5 Hz
    """
    
    def __init__(self, input_rate: float = 80.0, output_rate: float = 12.5):
        super().__init__()
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.ratio = input_rate / output_rate  # e.g., 6.4
    
    def forward(self, x):
        """Pool embedded prosody along the time dimension.
        
        Args:
            x: Tensor shape ``(batch, time, embed_dim)`` at input_rate Hz.
        
        Returns:
            Tensor shape ``(batch, target_time, embed_dim)`` at output_rate Hz.
        """
        batch, length, embed_dim = x.shape
        target_len = max(1, round(length / self.ratio))
        x_t = x.transpose(1, 2)  # (batch, embed_dim, time)
        pooled = F.adaptive_avg_pool1d(x_t, target_len)  # (batch, embed_dim, target_len)
        return pooled.transpose(1, 2)  # (batch, target_len, embed_dim)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run python -m pytest tests/models/test_pooling.py -v
```

Expected: All PASS (10 tests)

- [ ] **Step 3: Commit TemporalPool**

```bash
git add src/models/pooling.py
git commit -m "feat: add TemporalPool using AdaptiveAvgPool1d for 80→12.5 Hz downsampling"
```

---

## Task 3: Residual Fusion

### Task 3.1: Write failing tests for ResidualFusion

**Files:**
- Create: `tests/models/test_fusion.py`

- [ ] **Step 1: Create test file**

```python
"""Tests for ResidualFusion — LayerNorm(semantic + λ·(prosody + timbre))."""
import pytest
import torch
from src.models.fusion import ResidualFusion

HIDDEN_DIM = 2560


def make_semantic(batch=2, seq=25):
    return torch.randn(batch, seq, HIDDEN_DIM)

def make_prosody(batch=2, seq=25):
    return torch.randn(batch, seq, HIDDEN_DIM)

def make_timbre(batch=2):
    return torch.randn(batch, HIDDEN_DIM)


class TestResidualFusion:
    @pytest.fixture
    def fusion(self):
        return ResidualFusion(hidden_dim=HIDDEN_DIM)

    def test_output_shape_matches_semantic(self, fusion):
        semantic = make_semantic(batch=2, seq=25)
        prosody = make_prosody(batch=2, seq=25)
        timbre = make_timbre(batch=2)
        out = fusion(semantic, prosody, timbre)
        assert out.shape == semantic.shape

    def test_output_within_reasonable_range(self, fusion):
        semantic = torch.randn(4, 10, HIDDEN_DIM)
        prosody = torch.randn(4, 10, HIDDEN_DIM)
        timbre = torch.randn(4, HIDDEN_DIM)
        out = fusion(semantic, prosody, timbre)
        assert out.std() < 3.0, f"Output std {out.std()} too high — LayerNorm should normalize"

    def test_identity_when_residual_is_zero(self, fusion):
        """When prosody+timbre are zero, output should equal LayerNorm(semantic)."""
        semantic = torch.randn(3, 15, HIDDEN_DIM)
        prosody = torch.zeros(3, 15, HIDDEN_DIM)
        timbre = torch.zeros(3, HIDDEN_DIM)
        out = fusion(semantic, prosody, timbre)
        expected = torch.nn.functional.layer_norm(
            semantic, (HIDDEN_DIM,), weight=fusion.norm.weight, bias=fusion.norm.bias, eps=1e-5
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_identity_when_lambda_is_zero(self, fusion):
        """At init (λ=0), output == LayerNorm(semantic) regardless of prosody/timbre."""
        semantic = make_semantic(batch=1, seq=10)
        prosody = torch.randn(1, 10, HIDDEN_DIM) * 100
        timbre = torch.randn(1, HIDDEN_DIM) * 100
        out = fusion(semantic, prosody, timbre)
        expected = torch.nn.functional.layer_norm(
            semantic, (HIDDEN_DIM,), weight=fusion.norm.weight, bias=fusion.norm.bias, eps=1e-5
        )
        assert torch.allclose(out, expected, atol=1e-5), "λ=0 should ignore residuals"

    def test_lambda_is_learnable(self, fusion):
        semantic = make_semantic()
        prosody = make_prosody()
        timbre = make_timbre()
        out = fusion(semantic, prosody, timbre)
        loss = out.sum()
        loss.backward()
        assert fusion._lambda.grad is not None, "λ should receive gradient"
        assert fusion._lambda.grad.item() != 0

    def test_gradient_flows_through_all_inputs(self, fusion):
        semantic = make_semantic()
        prosody = make_prosody()
        timbre = make_timbre()
        
        semantic = semantic.clone().requires_grad_(True)
        prosody = prosody.clone().requires_grad_(True)
        timbre = timbre.clone().requires_grad_(True)
        
        out = fusion(semantic, prosody, timbre)
        loss = out.sum()
        loss.backward()
        
        assert semantic.grad is not None
        assert prosody.grad is not None
        assert timbre.grad is not None

    def test_timbre_broadcast_handled_correctly(self, fusion):
        """Timbre is per-sample (B, D); must broadcast to (B, T, D) internally."""
        semantic = make_semantic(batch=4, seq=12)
        prosody = make_prosody(batch=4, seq=12)
        timbre = make_timbre(batch=4)
        out = fusion(semantic, prosody, timbre)
        assert out.shape == (4, 12, HIDDEN_DIM)

    def test_lambda_initialized_at_zero(self, fusion):
        assert fusion._lambda.item() == 0.0

    def test_lambda_can_be_set(self):
        fusion = ResidualFusion(hidden_dim=HIDDEN_DIM)
        with torch.no_grad():
            fusion._lambda.data.fill_(0.5)
        assert fusion._lambda.item() == 0.5
        
        semantic = make_semantic()
        prosody = make_prosody()
        timbre = make_timbre()
        out = fusion(semantic, prosody, timbre)
        
        zero_fusion = ResidualFusion(hidden_dim=HIDDEN_DIM)
        zero_out = zero_fusion(semantic, prosody, timbre)
        assert not torch.allclose(out, zero_out), "λ=0.5 should differ from λ=0"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/models/test_fusion.py -v 2>&1 | tail -12
```

Expected: All FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/models/test_fusion.py
git commit -m "test: add failing tests for ResidualFusion"
```

---

### Task 3.2: Implement `ResidualFusion`

**Files:**
- Create: `src/models/fusion.py`

- [ ] **Step 1: Implement ResidualFusion**

```python
"""Residual fusion module: combines semantic, prosody, and timbre streams."""
import torch.nn as nn


class ResidualFusion(nn.Module):
    """Fuse semantic, prosody, and timbre via residual additive combination.
    
    Computes::
    
        out = LayerNorm(semantic + λ · (prosody + timbre))
    
    where λ is a learnable scalar initialized at zero. At initialization,
    the residual has no effect — gradients slowly grow λ from zero.
    Timbre is per-sample ``(batch, dim)`` and is broadcast to the sequence
    dimension internally.
    
    Args:
        hidden_dim: Dimension of all input/output tensors (2560).
    
    Example:
        >>> fusion = ResidualFusion(hidden_dim=2560)
        >>> semantic = torch.randn(2, 25, 2560)   # (B, T, D)
        >>> prosody = torch.randn(2, 25, 2560)    # (B, T, D)
        >>> timbre = torch.randn(2, 2560)         # (B, D)
        >>> out = fusion(semantic, prosody, timbre)
        >>> out.shape  # (2, 25, 2560)
    """
    
    def __init__(self, hidden_dim: int = 2560):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self._lambda = nn.Parameter(torch.zeros(1))
    
    def forward(self, semantic, prosody, timbre):
        """Fuse three streams via learnable residual addition.
        
        Args:
            semantic: ``(B, T, D)`` — the main semantic stream.
            prosody: ``(B, T, D)`` — pooled prosody embeddings.
            timbre: ``(B, D)`` — per-sample timbre vector 
                (expanded to ``(B, 1, D)``, then broadcast to T).
        
        Returns:
            ``(B, T, D)`` — fused representation.
        """
        residual = (prosody + timbre.unsqueeze(1)) * self._lambda
        fused = semantic + residual
        return self.norm(fused)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run python -m pytest tests/models/test_fusion.py -v
```

Expected: All PASS (9 tests)

- [ ] **Step 3: Commit ResidualFusion**

```bash
git add src/models/fusion.py
git commit -m "feat: add ResidualFusion — learnable λ, LayerNorm, timbre broadcast"
```

---

## Task 4: Module Exports and Integration Verification

### Task 4.1: Create `src/models/__init__.py` with public exports

**Files:**
- Create: `src/models/__init__.py`

- [ ] **Step 1: Create init file**

```python
"""Amy-LM model modules."""
from .embedding import ProsodyEmbedding, TimbreEmbedding
from .pooling import TemporalPool
from .fusion import ResidualFusion

__all__ = ["ProsodyEmbedding", "TimbreEmbedding", "TemporalPool", "ResidualFusion"]
```

- [ ] **Step 2: Verify imports work**

```bash
uv run python -c "from src.models import ProsodyEmbedding, TimbreEmbedding, TemporalPool, ResidualFusion; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 3: Run full test suite**

```bash
uv run python -m pytest tests/models/ -v --tb=short
```

Expected: All 32 tests PASS (13 embedding + 10 pooling + 9 fusion)

- [ ] **Step 4: Commit exports**

```bash
git add src/models/__init__.py
git commit -m "chore: add public exports for deep modules in src/models/__init__.py"
```

---

## Acceptance Criteria Checklist

| # | Criterion | Covered By |
|---|-----------|------------|
| 1 | ProsodyEmbedding shape (B, T, 2560) | `test_random_init_output_shape` |
| 2 | TimbreEmbedding shape (B, 2560) | `test_random_init_output_shape` |
| 3 | Random init ~ N(0, 0.02) | `test_random_init_weight_stats` |
| 4 | Warm-start init from FACodec vectors | `test_warm_start_produces_correct_shape`, `test_warm_start_shape` |
| 5 | Warm-start projector gradients flow | `test_warm_start_gradients_flow` |
| 6 | 80 Hz input → 12.5 Hz output | `test_output_rate_correct_2sec`, `test_output_rate_correct_5sec` |
| 7 | Handles non-divisible frame counts | `test_handles_non_divisible_frame_count` |
| 8 | λ initialized at zero, identity behavior | `test_lambda_initialized_at_zero`, `test_identity_when_lambda_is_zero` |
| 9 | λ learnable, gradient flows | `test_lambda_is_learnable` |
| 10 | Invalid init strategy raises | `test_invalid_init_strategy_raises` |
| 11 | Warm-start without vectors raises | `test_warm_start_without_vectors_raises` |
| 12 | Timbre broadcast (B, D) → (B, T, D) | `test_broadcastable_to_sequence_dim`, `test_timbre_broadcast_handled_correctly` |
| 13 | All three modules importable | Task 4.1, Step 2 |

---

## Execution Order Summary

```
Task 1.1  →  Task 1.2  (Embedding: tests first, then impl)
Task 2.1  →  Task 2.2  (Pooling: tests first, then impl)
Task 3.1  →  Task 3.2  (Fusion: tests first, then impl)
Task 4.1                 (Exports + integration verify)
```

Each component is independent — embedding, pooling, and fusion can be built in parallel if desired. Task 4.1 requires all prior tasks.

**Total commits:** 7
**Total new files:** 8
**Total test cases:** 32