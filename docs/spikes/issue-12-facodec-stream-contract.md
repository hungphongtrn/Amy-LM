# Issue #12: FACodec Stream Contract Spike

## Summary

This spike investigates the FACodec tensor contract to clarify the distinction between **Acoustic Stream** (discrete per-frame residual indices) and **Timbre Vector** (continuous utterance-level embedding from `spk_embs`). The current codebase conflates these, storing averaged acoustic VQ indices under the misleading name `timbre_codebooks_idx`.

## Evidence Sources

### CONTEXT.md (post-grill 2026-05-06)

- Already distinguishes Acoustic Stream (`vq_id[3:]` indices) from Timbre Vector (`spk_embs` continuous vector)
- Documents the naming error: `timbre_codebooks_idx` stored residual acoustic VQ indices
- Confirms corrected contract per grill session

#### Key Extracts

- **Line 17-19**: FACodec definition
  > "**FACodec**: A third-party factorized neural audio codec (Microsoft, arXiv:2403.03100). Produces 1 prosody codebook (vocab=1024, 80 Hz), 2 content codebooks, 3 acoustic detail codebooks, and 1 global timbre vector. Used as a substitute for Amy Codec during pilot validation."

- **Line 35-38**: Acoustic Stream definition
  > "**Acoustic Stream (a_t)**: Discrete residual acoustic detail indices from FACodec's three acoustic codebooks at 80 Hz. Captures speaker/environment/reconstruction artifacts beyond prosody. Optional for the pilot; not to be conflated with Timbre Vector."

- **Line 43-45**: Timbre Vector definition
  > "**Timbre Vector**: A single global utterance-level embedding representing speaker identity. Sourced from FACodec's timbre encoder, not per-frame."

- **Line 119-121**: Flagged ambiguity in dialogue
  > "**Domain expert**: No. `timbre_codebooks_idx` was a mistake — it stored averaged residual acoustic VQ indices under the wrong name. Timbre Vector is a separate continuous embedding from FACodec's `spk_embs`, utterance-level, float32."

- **Line 126**: Flagged ambiguities resolution
  > "`timbre_codebooks_idx` stored averaged residual acoustic VQ indices under the wrong name — resolved: renamed to **Acoustic Stream** (a_t). The true **Timbre Vector** is a separate continuous utterance-level embedding from FACodec `spk_embs`."

---

### src/preprocessing/facodec_encoder.py

**Issue**: The encoder completely ignores FACodec's `spk_embs` output, and incorrectly labels averaged residual VQ indices as "timbre".

#### Critical Code: Decoder call ignores timbre vector (lines 246-247)
```python
with torch.no_grad():
    enc_out = self._encoder(batch)
    _, vq_id, _, _, _ = self._decoder(enc_out, eval_vq=False, vq=True)
```
**Problem**: The decoder returns 5 values. Per Amphion FACodec source, the 4th return value is `spk_embs` (the true timbre vector). It's being discarded with `_`.

#### Misleading Documentation (lines 319-326)
```python
"""FACodec produces 6 codebooks:
- vq_id[0:1] = prosody (1 codebook)
- vq_id[1:3] = content (2 codebooks) 
- vq_id[3:6] = residual/timbre (3 codebooks)
"""
```
**Issue**: The comment calls `vq_id[3:6]` "residual/timbre" — this naming ambiguity contributed to the confusion.

#### Incorrect Index Averaging (lines 367-369)
```python
# Timbre: average of residual codebooks [3], [4], [5]
timbre_val = int((vq_id[3, frame_idx].item() + vq_id[4, frame_idx].item() + vq_id[5, frame_idx].item()) // 3)
timbre_indices.append(timbre_val)
```
**Issue**: This produces discrete integer indices by averaging VQ indices 3, 4, 5. This is **Acoustic Stream** data, not Timbre Vector. The true Timbre Vector would come from `spk_embs`.

---

### src/preprocessing/dataset_processor.py

**Issue**: The HF Features schema stores the wrong data under the wrong name.

#### Schema Definition (lines 178-185)
```python
features = Features({
    "dataset": Value("string"),
    "id": Value("string"),
    "audio": HFAudio(sampling_rate=16000),
    "content_codebooks_idx": Sequence(Value("int64")),
    "prosody_codebooks_idx": Sequence(Value("int64")),
    "timbre_codebooks_idx": Sequence(Value("int64")),  # <-- MISLEADING NAME
})
```
**Issue**: `timbre_codebooks_idx` is defined as a sequence of int64 — this stores the averaged residual VQ indices (Acoustic Stream), not the continuous Timbre Vector from `spk_embs`.

#### Entry Building (lines 252-259)
```python
return {
    "dataset": dataset_name,
    "id": sample_id,
    "audio": {"array": audio_array, "sampling_rate": sampling_rate},
    "content_codebooks_idx": content_indices,
    "prosody_codebooks_idx": prosody_indices,
    "timbre_codebooks_idx": timbre_indices,  # <-- Stores averaged acoustic indices
}
```

---

### Amphion FACodec README

- **Source URL**: https://github.com/open-mmlab/Amphion/blob/main/models/codec/ns3_codec/README.md
- **Mirror**: https://huggingface.co/amphion/naturalspeech3_facodec
- `vq_id[:1]` → prosody (1 codebook)
- `vq_id[1:3]` → content (2 codebooks)  
- `vq_id[3:]` → residual/acoustic detail (3 codebooks in current checkpoint)
- `spk_embs` → global speaker/timbre embedding, used in `fa_decoder.inference()`
- **Conclusion**: `spk_embs` is the utterance-level timbre vector, not frame-level VQ indices

#### Key API Snippet (from Amphion README)
```python
# quantize
vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)

# codes
print("vq id shape:", vq_id.shape)

# get prosody code
prosody_code = vq_id[:1]
print("prosody code shape:", prosody_code.shape)

# get content code
cotent_code = vq_id[1:3]
print("content code shape:", cotent_code.shape)

# get residual code (acoustic detail codes)
residual_code = vq_id[3:]
print("residual code shape:", residual_code.shape)

# speaker embedding
print("speaker embedding shape:", spk_embs.shape)

# decode (recommand)
recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
```

---

### src/models/embedding.py

**Issue**: `TimbreEmbedding` expects discrete indices and performs embedding table lookup, but the true Timbre Vector is a continuous float tensor that requires projection.

#### Current Implementation: Discrete Embedding (lines 60-114)
```python
class TimbreEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,  # Discrete vocabulary
        embed_dim: int = 2560,
        ...
    ):
    ...
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() != 1:
            raise ValueError(
                f"TimbreEmbedding expects 1D indices (batch,), got shape {indices.shape}"
            )
        w = self.weight
        return nn.functional.embedding(indices, w)  # Discrete lookup!
```

**Current behavior**: 
- Takes integer `indices` (batch,)
- Performs `nn.functional.embedding()` — a discrete codebook lookup
- Expects `vocab_size` parameter (256)

**Required behavior**:
- Should accept `spk_embs` — a continuous float tensor (typically shape `[batch, timbre_dim]` where timbre_dim ≈ 256-512)
- Should project continuous vector into MOSS-Audio embedding space
- Should be utterance-level (broadcast to all frames, skip TemporalPool)

---

## Verified FACodec Tensor Contract

| Output | Variable | Shape | Type | Purpose |
|--------|----------|-------|------|---------|
| `vq_id[0]` | prosody_indices | `[T80]` | int64 | Prosody Stream (p_t) |
| `vq_id[1:3]` | content_indices | `[2, T80]` | int64 | FACodec Content (c_t) — disabled in pilot |
| `vq_id[3:6]` | acoustic_indices | `[3, T80]` | int64 | **Acoustic Stream (a_t)** — was misnamed "timbre" |
| `spk_embs` | timbre_vector | `[timbre_dim]` | float32 | **Timbre Vector** — utterance-level speaker embedding |

**Key distinction**: `vq_id[3:6]` are discrete per-frame indices (Acoustic Stream). `spk_embs` is a continuous utterance-level vector (Timbre Vector). They are fundamentally different data types.

## Canonical Amy LM Terms

- **Acoustic Stream (a_t)**: Per-frame discrete indices from FACodec's 3 residual acoustic codebooks. Undergoes TemporalPool (80 Hz → 12.5 Hz) after embedding. Currently misnamed as `timbre_codebooks_idx` in preprocessing output.
- **Timbre Vector**: Global utterance-level continuous embedding from FACodec's `spk_embs`. Broadcast to all frames, skips TemporalPool. Currently not extracted or stored.

## Corrected Training Sample Schema

### Field Migration (current → corrected)

| Old Field | Old Type | → | New Field | New Type | Change |
|-----------|---------|---|-----------|----------|--------|
| `timbre_codebooks_idx` | `Sequence(int64)` | → | `acoustic_codebooks_idx` | `Sequence(Sequence(int64))` | Rename + flatten scalar → `[3, T80]` nested |
| *(missing)* | — | → | `timbre_vector` | `Sequence(float32)` | New field, D=256 from `spk_embs` |
| `content_codebooks_idx` | `Sequence(int64)` | → | `content_codebooks_idx` | `Sequence(Sequence(int64))` | Flatten scalar → `[2, T80]` nested |
| `prosody_codebooks_idx` | `Sequence(int64)` | → | `prosody_codebooks_idx` | `Sequence(int64)` | No change (single codebook) |
| `audio` | `HFAudio(16000)` | → | `audio` | `HFAudio(16000)` | No change |
| `label` | *(dataset-native)* | → | `label` | *(dataset-native)* | No change |
| `dataset` | `string` | → | `dataset` | `string` | No change |
| `id` | `string` | → | `id` | `string` | No change |

### FACodecEncoder corrected return contract

The current encoder returns `(content_indices, prosody_indices, timbre_indices)` as lists of scalar ints.
The corrected encoder must return:

```python
@dataclass
class FACodecStreams:
    prosody_codebooks_idx: torch.Tensor   # [1, T80], int64
    content_codebooks_idx: torch.Tensor   # [2, T80], int64
    acoustic_codebooks_idx: torch.Tensor  # [3, T80], int64
    timbre_vector: torch.Tensor           # [256], float32

FACodecEncoder.encode(audio) -> FACodecStreams
```

Per-frame processing (line 363-381 region of facodec_encoder.py) must change from:
- Averaging `vq_id[3:6]` into scalar int → Emitting `vq_id[3:6]` as-is
- Ignoring `spk_embs` → Capturing and returning `spk_embs.squeeze()`

### Corrected HF Features schema

```python
features = Features({
    "dataset": Value("string"),
    "id": Value("string"),
    "audio": HFAudio(sampling_rate=16000),
    "content_codebooks_idx": Sequence(Sequence(Value("int64"))),  # [2, T80] nested
    "prosody_codebooks_idx": Sequence(Value("int64")),  # [T80]
    "acoustic_codebooks_idx": Sequence(Sequence(Value("int64"))),  # [3, T80] — RENAMED from timbre
    "timbre_vector": Sequence(Value("float32")),  # [256] — NEW, from spk_embs
})
```

## Test Audit

### Test corrections needed (#6)

#### test_facodec_encoder.py

**Return value count assertions (3-tuple → 4-tuple):**
- Line 28: `content_indices, prosody_indices, timbre_indices = encoder.encode(audio)` → Must unpack 4 values (`prosody_codebooks_idx, content_codebooks_idx, acoustic_codebooks_idx, timbre_vector`)
- Lines 40, 56, 71, 86: Same tuple unpacking issue throughout test methods
- Lines 98, 103: `result_1s = encoder.encode(audio_1s)` and frame count assertions on 3 items
- Lines 136-139: Batch result unpacking expects 3 items per result → needs 4
- Lines 150-155: Batch comparison assertions on 3 items per result
- Lines 163, 165: Frame count assertions on `results[0]` (3 indices)
- Lines 183-184, 203-211: Real batch test assertions on 3 items
- Lines 259-264: `test_real_encode_returns_three_index_lists` - expects 3 lists, needs 4 (with timbre_vector)
- Lines 272-278: Range validation loops over 3 index lists
- Lines 283-285: Length equality check for 3 lists
- Lines 292-301: Different audio output test checks 3 streams
- Lines 309-317: Frame count scaling test on 3 items
- Line 323: 2D audio handling test unpacks 3 items

**Type assertions to update:**
- Line 43: `for indices in [content_indices, prosody_indices, timbre_indices]:` → Replace `timbre_indices` with `acoustic_codebooks_idx`
- Lines 76-78: Stream difference assertions reference `timbre_indices` → should reference `acoustic_codebooks_idx`

**Mock mode corrections needed:**
- Mock encoder must generate `timbre_vector` of shape `(256,)` float32 in addition to current 3 streams
- All mock test methods that verify "3 index lists" need to be updated to "4 items (3 index lists + 1 vector)"

#### test_dataset_processor.py

**Required columns assertions:**
- Lines 116-121: `required_columns` list includes `"timbre_codebooks_idx"` → must rename to `"acoustic_codebooks_idx"` and add `"timbre_vector"`
- Line 301: Same `required_columns` list in save/load test

**Field name references:**
- Line 158: `for col in ["content_codebooks_idx", "prosody_codebooks_idx", "timbre_codebooks_idx"]:` → rename `timbre_codebooks_idx` → `acoustic_codebooks_idx`
- Lines 185-188: Frame count tolerance assertions iterate over 3 columns including `timbre_codebooks_idx` → rename

**Type assertions to add:**
- New assertions needed: `timbre_vector` should be list of float32, length 256 (not frame-dependent)
- Lines 163-165: Integer range checks should exclude `timbre_vector` (it's float, not int indices)

## Preprocessing to MOSS-Audio Data Flow

```
Preprocessing (facodec_encoder.py)
├── Audio → FACodec Encoder → enc_out
├── enc_out → FACodec Decoder → vq_id[6, B, T80], spk_embs[B, timbre_dim]
├── vq_id[0] → prosody_codebooks_idx
├── vq_id[1:2] averaged → content_codebooks_idx
├── vq_id[3:5] averaged → acoustic_codebooks_idx (WAS: timbre_codebooks_idx)
└── spk_embs → timbre_embedding (CURRENTLY DISCARDED — needs extraction)

MOSS-Audio Training (Forward Pass)
├── prosody_codebooks_idx → ProsodyEmbedding → TemporalPool(80→12.5) → p_t
├── content_codebooks_idx → ContentEmbedding → TemporalPool(80→12.5) → c_t [DISABLED]
├── acoustic_codebooks_idx → AcousticEmbedding → TemporalPool(80→12.5) → a_t
└── timbre_embedding → TimbreProjection → broadcast → t_t (no pooling!)
```

## Current Implementation Audit

| Component | Current State | Issue | Required Change |
|-----------|--------------|-------|-----------------|
| `facodec_encoder.py:246-247` | Discards 4th decoder return (`spk_embs`) | Timbre Vector never extracted | Capture and return `spk_embs` |
| `facodec_encoder.py:367-369` | Averages VQ[3:6] into "timbre_indices" | Mislabels Acoustic Stream as Timbre | Rename to `acoustic_indices` |
| `dataset_processor.py:184` | `timbre_codebooks_idx: Sequence(int64)` | Schema stores wrong data type | Rename to `acoustic_codebooks_idx` |
| `dataset_processor.py:258` | Stores averaged VQ indices | No field for true Timbre Vector | Add `timbre_embedding: Sequence(float32)` |
| `embedding.py:60-114` | `TimbreEmbedding` does discrete lookup | Expects indices, not continuous vector | Create `TimbreProjection` for float vectors |

### Module Migration (current → corrected)

| Current Module | Current Input | → | New Module | New Input | Change |
|---------------|--------------|---|------------|-----------|--------|
| `TimbreEmbedding` | `[B]` int64 | → | `TimbreProjection` | `[B, 256]` float32 | Discrete lookup → continuous linear projection |
| *(missing)* | — | → | `AcousticEmbedding` | `[B, 3, T80]` int64 | New module for 3-codebook residual stream |
| *(missing)* | — | → | `ContentEmbedding` | `[B, 2, T80]` int64 | New module for 2-codebook content stream |
| `ProsodyEmbedding` | `[B, T80]` int64 | → | `ProsodyEmbedding` | `[B, 1, T80]` int64 | Add codebook axis (was flattened) |
| `ResidualFusion` | `(sem, prosody, timbre)` | → | `ResidualFusion` | per-stream lambdas | Single λ → λ_p, λ_c, λ_a, λ_t |
| `TemporalPool` | `[B, T80, D]` | → | `TemporalPool` | `[B, T80, D]` | No change |

### __init__.py export changes

Current:
```python
from .embedding import ProsodyEmbedding, TimbreEmbedding
from .pooling import TemporalPool
from .fusion import ResidualFusion
```

Corrected:
```python
from .embedding import ProsodyEmbedding, TimbreProjection, AcousticEmbedding, ContentEmbedding
from .pooling import TemporalPool
from .fusion import ResidualFusion
```

## Test Audit (Models)

### Test corrections needed (#7)

#### test_embedding.py
- TimbreEmbedding tests (lines 87-140): Currently tests discrete embedding table lookup with 1D indices `[B]`. Will need to remain for backward compatibility but should be deprecated in favor of `TimbreProjection`.
- Missing test class for `TimbreProjection`: Needs tests for `[B, 256]` float32 input, Linear projection, broadcast compatibility
- Missing test class for `AcousticEmbedding`: Needs tests for 3-codebook embedding `[B, 3, T80]`, sum aggregation, shape validation
- Missing test class for `ContentEmbedding`: Needs tests for 2-codebook embedding `[B, 2, T80]`, disabled-by-default behavior
- Missing test updates for `ProsodyEmbedding`: Input shape should change from `[B, T80]` to `[B, 1, T80]` (add codebook axis)

#### test_fusion.py
- ResidualFusion tests (lines 1-113): Currently tests single lambda `_lambda` gate for combined (prosody + timbre) residual.
- Line 68: `assert fusion._lambda.grad is not None` — needs update to per-stream lambdas `lambda_p`, `lambda_c`, `lambda_a`, `lambda_t`
- Line 98: `assert fusion._lambda.item() == 0.0` — single lambda initialization test needs expansion
- Lines 40-60: Identity tests assume 2 residual inputs (prosody, timbre) — needs expansion to 4 streams
- Missing tests for stream_config gating: disabled streams should be excluded entirely, not just gated at zero
- Missing tests for multi-stream gradient flow (content, acoustic gradients)

## Required Follow-Up Changes

1. **Preprocessing pipeline** (`facodec_encoder.py`, `dataset_processor.py`):
   - Extract `spk_embs` from decoder output (4th return value)
   - Rename `timbre_codebooks_idx` → `acoustic_codebooks_idx`
   - Add new field `timbre_embedding` (float32 vector)
   - Update HF Features schema

2. **Model architecture** (`embedding.py`, fusion code):
   - Create `TimbreProjection` module (Linear from timbre_dim → moss_dim)
   - Keep `TimbreEmbedding` for discrete case (backward compatibility)
   - Ensure Timbre Vector skips TemporalPool (broadcast instead)

3. **Data migration**:
   - Existing preprocessed datasets have wrong field names
   - Need reprocessing or backward compatibility layer

### TimbreProjection specification (replaces TimbreEmbedding)

```
class TimbreProjection(nn.Module):
    """Project continuous timbre vector into MOSS-Audio embedding space.

    Replaces TimbreEmbedding (discrete lookup over integer indices).
    The timbre vector is an utterance-level float32 tensor from FACodec spk_embs.

    Args:
        timbre_dim: Dimensionality of input timbre vector (default=256)
        output_dim: Target embedding dimension (default=2560, MOSS-Audio hidden dim)
        init_strategy: "random" | "warm_start" (default: "random")
    """

    def __init__(
        self,
        timbre_dim: int = 256,
        output_dim: int = 2560,
        init_strategy: str = "random",
    ): ...

    def forward(self, timbre_vector: torch.Tensor) -> torch.Tensor:
        """Project utterance-level timbre.

        Args:
            timbre_vector: [B, timbre_dim] float32 from FACodec spk_embs

        Returns:
            [B, output_dim] float32 — broadcast to frames during fusion
        """
```

### AcousticEmbedding specification (new)

```
class AcousticEmbedding(nn.Module):
    """Embed 3 acoustic residual codebooks into MOSS-Audio embedding space.

    Each codebook has its own independent embedding table (vocab → D).
    Per-frame output is the sum of all three codebook embeddings at that frame.

    Args:
        vocab_size: Codebook vocabulary size (default=1024)
        num_codebooks: Number of acoustic codebooks (default=3)
        embed_dim: Target embedding dimension (default=2560)
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        num_codebooks: int = 3,
        embed_dim: int = 2560,
    ): ...

    def forward(self, codebook_indices: torch.Tensor) -> torch.Tensor:
        """Embed multi-codebook acoustic stream.

        Args:
            codebook_indices: [B, 3, T80] int64 from FACodec vq_id[3:]

        Returns:
            [B, T80, embed_dim] float32 — sum of per-codebook embeddings
        """
```

### ContentEmbedding specification (new)

```
class ContentEmbedding(nn.Module):
    """Embed 2 content codebooks into MOSS-Audio embedding space.

    Same architecture as AcousticEmbedding but with 2 codebooks.
    Disabled in the initial experiment (Stream Activation Config).

    Args:
        vocab_size: Codebook vocabulary size (default=1024)
        num_codebooks: Number of content codebooks (default=2)
        embed_dim: Target embedding dimension (default=2560)
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        num_codebooks: int = 2,
        embed_dim: int = 2560,
    ): ...

    def forward(self, codebook_indices: torch.Tensor) -> torch.Tensor:
        """Embed multi-codebook content stream.

        Args:
            codebook_indices: [B, 2, T80] int64 from FACodec vq_id[1:3]

        Returns:
            [B, T80, embed_dim] float32 — sum of per-codebook embeddings
        """
```

### ProsodyEmbedding corrected interface

```
class ProsodyEmbedding(nn.Module):
    """Embed FACodec prosody codebook into MOSS-Audio embedding space.

    Args:
        vocab_size: Codebook vocabulary size (default=1024)
        embed_dim: Target embedding dimension (default=2560)
        init_strategy: "random" | "warm_start" (default: "random")
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        embed_dim: int = 2560,
        init_strategy: str = "random",
    ): ...

    def forward(self, codebook_indices: torch.Tensor) -> torch.Tensor:
        """Embed single-codebook prosody stream.

        Args:
            codebook_indices: [B, 1, T80] int64 from FACodec vq_id[:1]

        Returns:
            [B, T80, embed_dim] float32
        """
```

### ResidualFusion corrected interface

```
class ResidualFusion(nn.Module):
    """Gated residual fusion: H = LayerNorm(S + Σ λ_i * stream_i)

    Each stream (prosody, content, acoustic, timbre) has its own
    learnable zero-initialized lambda gate. Disabled streams are
    excluded from the computation entirely (not just gated at zero).

    Args:
        hidden_dim: Dimensionality of all streams (default=2560)
        stream_config: Dict[str, bool] controlling which streams participate
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        stream_config: dict[str, bool] | None = None,
    ): ...

    def forward(
        self,
        semantic: torch.Tensor,                   # [B, T12, D]
        prosody: torch.Tensor | None = None,      # [B, T12, D]
        content: torch.Tensor | None = None,      # [B, T12, D]
        acoustic: torch.Tensor | None = None,     # [B, T12, D]
        timbre: torch.Tensor | None = None,       # [B, T12, D] (pre-broadcast)
    ) -> torch.Tensor:
        """Fuse streams via gated residual summation.

        Each non-None stream must be already aligned to [B, T12, D].
        Timbre must be broadcast to frames before this call.
        None streams are not included in the sum.
        """
```

## Issue #8 Blocker Recommendation

**Status**: BLOCKED on Issue #12 completion.

Issue #8 (Prosody Stream Integration) assumes `timbre_codebooks_idx` contains timbre data. This spike reveals it actually contains averaged acoustic residual indices. Continuing with Issue #8 would:
1. Build the wrong embedding architecture
2. Wire Acoustic Stream into a module expecting Timbre Vector
3. Create technical debt requiring rework

**Recommendation**: Complete Issue #12 preprocessing fixes first, then resume Issue #8.

## Executable Verification

- Status: Complete
- Command: `PYTHONPATH=/home/hungphongtrn/Workspace/Amy-LM uv run python scripts/inspect_facodec_contract.py --device cpu --seconds 1`

### Actual Shapes

```
enc_out:           (1, 256, 80) torch.float32
vq_post_emb:       (1, 256, 80) torch.float32
vq_id:             (6, 1, 80) torch.int64
unknown_return_3:  Tensor
quantized:         list (len=3)
quantized[0]:      (1, 256, 80) torch.float32
spk_embs:          (1, 256) torch.float32
timbre_vector:     (256,) torch.float32
prosody vq_id[:1]: (1, 1, 80)
content vq_id[1:3]:(2, 1, 80)
acoustic vq_id[3:]:(3, 1, 80)
```

### Interpretation

| Variable | Actual Shape | Confirmed Contract |
|----------|--------------|-------------------|
| `enc_out` | (batch=1, 256, frames=80) | ✓ Encoder output at 80 Hz frame rate |
| `vq_id` | (6, batch=1, 80) | ✓ 6 codebooks × batch × 80 frames |
| `vq_id[:1]` (prosody) | (1, 1, 80) | ✓ 1 codebook for prosody |
| `vq_id[1:3]` (content) | (2, 1, 80) | ✓ 2 codebooks for content |
| `vq_id[3:]` (acoustic) | (3, 1, 80) | ✓ 3 codebooks for acoustic/residual |
| `spk_embs` | (batch=1, 256) | ✓ **Timbre Vector** is (256,) continuous float32 |
| `spk_embs.squeeze()` (timbre_vector) | (256,) | ✓ Utterance-level embedding |

**Key Findings:**
1. **Timbre Vector dimensionality confirmed**: `spk_embs` is shape `(256,)` float32 — this resolves Open Question #1
2. **Quantized is a list** of 3 tensors — matches the 3 residual codebook structure
3. All VQ slicing matches the documented contract exactly

## Open Questions

1. ~~**Timbre Vector dimensionality**: What is the exact shape of `spk_embs` from FACodec?~~ **RESOLVED**: Confirmed `spk_embs` is `(256,)` float32 per utterance (batch=1 gives `(1, 256)` → squeeze to `(256,)`)
2. **Acoustic Stream storage**: Should `acoustic_codebooks_idx` store `[3, T80]` as nested sequence or keep averaging? (Averaging loses information)
3. **Backward compatibility**: Do we reprocess all datasets or provide migration script?
4. **Embedding strategy**: Should Timbre Vector use learned projection (random init) or FACodec warm-start vectors?

---

*Spike completed: Evidence recorded from CONTEXT.md, codebase audit, and external Amphion FACodec documentation.*

---

## Confidence

- **High**: VQ stream slicing matches Amphion README documentation exactly (`vq_id[:1]` prosody, `vq_id[1:3]` content, `vq_id[3:]` residual)
- **Medium**: `spk_embs` as utterance-level timbre vector (consistent with README showing it as separate from VQ indices, but confirmation of exact shape requires local checkpoint run)
- **High**: The Amphion README explicitly demonstrates that `spk_embs` is passed to `fa_decoder.inference(vq_post_emb, spk_embs)` as the speaker/timbre parameter, confirming it is the Timbre Vector
