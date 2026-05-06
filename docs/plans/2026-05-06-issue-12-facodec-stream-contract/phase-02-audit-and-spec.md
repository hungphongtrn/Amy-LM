# Phase 2: Audit & Corrected Spec

## Phase Goal
Consolidate the Phase 1 evidence into corrected specifications for issues #6 (preprocessing) and #7 (deep modules). The spike report already contains raw audit findings; this phase synthesizes them into actionable, precise spec documents that downstream implementers can follow directly.

## Phase 1 Learnings Applied
- `D_timbre = 256` (confirmed via local checkpoint run)
- `vq_id = (6, B, 80)` — 6 codebooks × 80 Hz
- `spk_embs = (B, 256)` float32 — utterance-level Timbre Vector
- All VQ slicing matches contract: prosody=1, content=2, acoustic=3 codebooks

## Files to Touch

| File | Action | Responsibility |
|------|--------|----------------|
| `docs/spikes/issue-12-facodec-stream-contract.md` | Populate | Finalize verified contract, corrected schema, current audit table, required changes list, blocker recommendation |
| `src/preprocessing/facodec_encoder.py` | Read-only | Audit for line-specific corrections needed by #6 |
| `src/preprocessing/dataset_processor.py` | Read-only | Audit for schema migration plan needed by #6 |
| `tests/preprocessing/test_facodec_encoder.py` | Read-only | Audit for test assertions encoding wrong contract |
| `tests/preprocessing/test_dataset_processor.py` | Read-only | Audit for schema-related test corrections |
| `src/models/embedding.py` | Read-only | Audit for `TimbreEmbedding` → `TimbreProjection` migration |
| `src/models/__init__.py` | Read-only | Audit exports to plan new module exports |
| `tests/models/test_embedding.py` | Read-only | Audit for test corrections |
| `src/models/fusion.py` | Read-only | Audit for per-stream lambda gates |
| `tests/models/test_fusion.py` | Read-only | Audit for corrected interface tests |

No production code changes in this phase.

---

## Task 1: Finalize Corrected #6 Preprocessing Spec

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md` (fill `Corrected Training Sample Schema` section)
- Audit: `src/preprocessing/facodec_encoder.py`, `src/preprocessing/dataset_processor.py`
- Audit: `tests/preprocessing/test_facodec_encoder.py`, `tests/preprocessing/test_dataset_processor.py`

- [ ] **Step 1: Write precise field migration table**

Replace the current schema placeholder with the exact migration each field needs:

```markdown
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
```

- [ ] **Step 2: Document facodec_encoder.py return contract**

```markdown
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

Per-frame processing (line 363-381 region) must change from:
- Averaging `vq_id[3:6]` into scalar int → Emitting `vq_id[3:6]` as-is
- Ignoring `spk_embs` → Capturing and returning `spk_embs.squeeze()`
```

- [ ] **Step 3: Audit test files for wrong assumptions**

Read `tests/preprocessing/test_facodec_encoder.py` and `tests/preprocessing/test_dataset_processor.py`. Record which assertions need updating. The spike report is the deliverable — do NOT modify the test files.

```markdown
### Test corrections needed (#6)

#### test_facodec_encoder.py
- Any assertion checking output tuple length of 3 → must become 4 (add timbre_vector)
- Any assertion checking `timbre_indices` type → must check `acoustic_codebooks_idx` type
- Mock mode must generate timbre_vector of shape (256,) float32

#### test_dataset_processor.py
- `Features` schema assertion must include `timbre_vector` field
- `timbre_codebooks_idx` references must be renamed
- Nested Sequence type for acoustic codebooks must be tested
```

- [ ] **Step 4: Commit preprocessing spec**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: finalize corrected #6 preprocessing spec"
```

---

## Task 2: Finalize Corrected #7 Module Spec

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md` (fill `Current Implementation Audit` and `Required Follow-Up Changes` sections)
- Audit: `src/models/embedding.py`, `src/models/__init__.py`, `src/models/fusion.py`
- Audit: `tests/models/test_embedding.py`, `tests/models/test_fusion.py`

- [ ] **Step 1: Write precise module migration table**

```markdown
### Module Migration (current → corrected)

| Current Module | Current Input | → | New Module | New Input | Change |
|---------------|--------------|---|------------|-----------|--------|
| `TimbreEmbedding` | `[B]` int64 | → | `TimbreProjection` | `[B, 256]` float32 | Discrete lookup → continuous linear projection |
| *(missing)* | — | → | `AcousticEmbedding` | `[B, 3, T80]` int64 | New module for 3-codebook residual stream |
| *(missing)* | — | → | `ContentEmbedding` | `[B, 2, T80]` int64 | New module for 2-codebook content stream |
| `ProsodyEmbedding` | `[B, T80]` int64 | → | `ProsodyEmbedding` | `[B, 1, T80]` int64 | Add codebook axis (was flattened) |
| `ResidualFusion` | `(sem, prosody, timbre)` | → | `ResidualFusion` | per-stream lambdas | Single λ → λ_p, λ_c, λ_a, λ_t |
| `TemporalPool` | `[B, T80, D]` | → | `TemporalPool` | `[B, T80, D]` | No change |
```

- [ ] **Step 2: Write `TimbreProjection` interface spec**

```markdown
### TimbreProjection specification

```python
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

- [ ] **Step 3: Write `AcousticEmbedding` interface spec**

```markdown
### AcousticEmbedding specification

```python
class AcousticEmbedding(nn.Module):
    """Embed 3 acoustic residual codebooks into MOSS-Audio embedding space.

    Each codebook has its own independent embedding table (vocab → D).
    Per-frame output is the sum of all three codebook embeddings at that frame.

    Args:
        vocab_size: Codebook vocabulary size (default=1024, FACodec codebook vocab)
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

- [ ] **Step 4: Write `ContentEmbedding` interface spec**

```markdown
### ContentEmbedding specification

```python
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

- [ ] **Step 5: Write `ProsodyEmbedding` corrected interface spec**

Note the key change: ProsodyEmbedding keeps its single codebook, but the input now preserves the codebook axis `[B, 1, T80]` instead of being flattened to `[B, T80]`.

```markdown
### ProsodyEmbedding corrected interface

```python
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

- [ ] **Step 6: Write `ResidualFusion` corrected interface spec**

The current fusion has a single lambda gate for prosody+timbre combined. The corrected version needs per-stream lambda gates.

```markdown
### ResidualFusion corrected interface

```python
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

- [ ] **Step 7: Audit model test files**

Read `tests/models/test_embedding.py` and `tests/models/test_fusion.py`. Record which tests need correction:

```markdown
### Test corrections needed (#7)

#### test_embedding.py
- `TimbreEmbedding` tests (lines 87-140): replace with `TimbreProjection` tests
  - Input should be float `[B, 256]` not int `[B]`
  - No vocabulary size parameter — timbre_dim instead
  - Output should be `[B, 2560]` not `[B, T, 2560]`
- Add `AcousticEmbedding` tests: input `[B, 3, T80]` int64 → output `[B, T80, D]`
- Add `ContentEmbedding` tests: input `[B, 2, T80]` int64 → output `[B, T80, D]`
- `ProsodyEmbedding` tests: update input shape from `[B, T80]` to `[B, 1, T80]`

#### test_fusion.py
- Single lambda tests: replace with per-stream lambda tests
- Add test that disabled streams (None) are excluded
- Add test that zero-initialized lambdas yield identity (output = LayerNorm(S))
```

- [ ] **Step 8: Audit `src/models/__init__.py` exports**

Read `src/models/__init__.py` and document the needed export changes:

```markdown
### __init__.py export changes

```python
# Current:
from .embedding import ProsodyEmbedding, TimbreEmbedding

# → Corrected:
from .embedding import ProsodyEmbedding, TimbreProjection, AcousticEmbedding, ContentEmbedding
from .fusion import ResidualFusion
from .pooling import TemporalPool

__all__ = [
    "ProsodyEmbedding",
    "TimbreProjection",   # renamed from TimbreEmbedding
    "AcousticEmbedding",  # new
    "ContentEmbedding",   # new
    "TemporalPool",
    "ResidualFusion",
]
```
```

- [ ] **Step 9: Commit module spec**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: finalize corrected #7 module spec"
```

---

## Task 3: Document Data Flow Diagram

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md` (fill `Preprocessing to MOSS-Audio Data Flow` section)

- [ ] **Step 1: Add stage-by-stage ASCII diagram with confirmed shapes**

```text
Preprocessing (facodec_encoder.py)
  Raw audio waveform, 16 kHz
        │
        ▼
  FACodec encoder/decoder
        │
        ├──→ vq_id[:1]   → prosody_codebooks_idx    (1, T80) int64
        ├──→ vq_id[1:3]  → content_codebooks_idx    (2, T80) int64
        ├──→ vq_id[3:]   → acoustic_codebooks_idx   (3, T80) int64
        └──→ spk_embs    → timbre_vector            (256,) float32

Issue #8 training
  audio waveform ──────────→ MOSS-Audio encoder ─→ Semantic Stream S_t   [B, T12, D]
  prosody_codebooks_idx ──→ ProsodyEmbedding ─────→ TemporalPool ──→ P_t [B, T12, D]
  content_codebooks_idx ──→ ContentEmbedding ─────→ TemporalPool ──→ C_t [B, T12, D]  [DISABLED]
  acoustic_codebooks_idx ─→ AcousticEmbedding ────→ TemporalPool ──→ A_t [B, T12, D]  [OPTIONAL]
  timbre_vector ──────────→ TimbreProjection ─────→ broadcast ────→ T_t [B, T12, D]

  ResidualFusion:
      H_t = LayerNorm(S_t + λ_p·P_t + λ_c·C_t + λ_a·A_t + λ_t·T_t)
```

- [ ] **Step 2: Commit data flow diagram**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: add confirmed data flow diagram with shapes"
```

---

## Phase Completion Criteria
- [ ] Corrected field migration table written (old → new names, types, shapes)
- [ ] Corrected module migration table written (TimbreEmbedding → TimbreProjection, new modules defined)
- [ ] Complete interface specs for all 6 modules (ProsodyEmbedding, ContentEmbedding, AcousticEmbedding, TimbreProjection, TemporalPool, ResidualFusion)
- [ ] Stage-by-stage data flow diagram with confirmed shapes
- [ ] Test file audit complete with specific line references for needed corrections
- [ ] `src/models/__init__.py` export changes documented
- [ ] Each task committed separately

## Handoff Notes
- All interface specs in this phase are the **canonical reference** for issues #6 and #7
- The `D_timbre = 256` constant is confirmed — hardcode it in `TimbreProjection` defaults
- `AcousticEmbedding` and `ContentEmbedding` share the same multi-codebook embedding pattern (N independent embedding tables, sum outputs) — consider a shared `MultiCodebookEmbedding` base class
- The spike report is **not** the implementation — it's the spec. Actual code changes happen in #6 and #7.
