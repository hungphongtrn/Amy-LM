# Issue #12: FACodec Stream Contract Spike — Strategy

## Goal
Verify FACodec's real output contract and publish the corrected data stream spec that issues #6 (preprocessing) and #7 (deep modules) must implement — before issue #8 touches MOSS-Audio internals.

## Architecture
This is a **documentation-first spike**. The deliverable is an evidence-backed contract that maps Amphion FACodec tensors to Amy LM canonical terms, then converts that contract into specific corrective tasks for preprocessing, model modules, and the MOSS-Audio Internal Residual Extension. No production code changes in this spike — only docs and an optional inspection helper script. Renames, schema migration, and module changes are follow-up work in issues #6 and #7.

## Tech Stack
Python 3.11, PyTorch, Amphion FACodec, Hugging Face Hub, pytest, Markdown

## Constraints & Assumptions
- Amphion FACodec checkpoint **confirmed available locally**. Executable verification succeeded.
- `spk_embs` = `(256,)` float32 — D_timbre = 256.
- `vq_id` = `(6, B, 80)` — 6 codebooks × 80 Hz frame rate.
- Issue #8 is blocked until this spike confirms the correct tensor contract.
- `CONTEXT.md` has already been corrected during the 2026-05-06 grill session.

## Resolved Contract (verified against local checkpoint)

### FACodec Tensor Mapping

| FACodec output | Stream | Shape (batch=1) | Codebooks | Rate | Type |
|---|:---:|---|:---:|---:|---|
| `vq_id[:1]` | Prosody Stream (p_t) | `(1, 1, 80)` | 1 | 80 Hz | int64 |
| `vq_id[1:3]` | FACodec Content Stream (c_t) | `(2, 1, 80)` | 2 | 80 Hz | int64 |
| `vq_id[3:]` | Acoustic Stream (a_t) | `(3, 1, 80)` | 3 | 80 Hz | int64 |
| `spk_embs` | Timbre Vector (h_t) | `(1, 256)` | n/a | utterance-level | float32 |

### Stream Dimensionality Contract

Let D = MOSS-Audio hidden dim. Let D_t = 256 (timbre dim). Every FACodec stream module must output D.

```
ProsodyEmbedding:  [B, 1, T80] -> [B, T80, D]
ContentEmbedding:  [B, 2, T80] -> [B, T80, D]
AcousticEmbedding: [B, 3, T80] -> [B, T80, D]
TimbreProjection:  [B, 256] -> [B, D]
```

Multi-codebook streams sum per-codebook embeddings:

```
ContentStream  = EmbC0(content_0) + EmbC1(content_1)
AcousticStream = EmbA0(acoustic_0) + EmbA1(acoustic_1) + EmbA2(acoustic_2)
```

### Stage-by-stage shape flow

```
# Stage 1: Embedding (FACodec VQ streams, 80 Hz)
P80 = ProsodyEmbedding(prosody_codebooks_idx)    # [B, T80, D]
C80 = ContentEmbedding(content_codebooks_idx)    # [B, T80, D]
A80 = AcousticEmbedding(acoustic_codebooks_idx)  # [B, T80, D]
T   = TimbreProjection(timbre_vector)            # [B, D]

# Stage 2: Temporal alignment
P12 = TemporalPool(P80)  # [B, T12, D]
C12 = TemporalPool(C80)  # [B, T12, D]
A12 = TemporalPool(A80)  # [B, T12, D]
T12 = broadcast(T, T12)  # [B, T12, D]

# Stage 3: Semantic encoding
S12 = MOSS-Audio encoder(audio)  # [B, T12, D]

# Stage 4: Gated residual fusion
H = LayerNorm(
  S12
  + lambda_p * P12
  + lambda_c * C12
  + lambda_a * A12
  + lambda_t * T12
)
```

### Corrected #6 Schema

Output: HF Dataset/parquet.

| Field | Shape | Type | Notes |
|---|:---:|---:|---|
| `audio` | waveform | HF Audio, 16 kHz | for MOSS-Audio online encoding |
| `prosody_codebooks_idx` | `[1, T80]` | Sequence(int64) | preserve codebook axis |
| `content_codebooks_idx` | `[2, T80]` | Sequence(int64) | preserve codebook axis |
| `acoustic_codebooks_idx` | `[3, T80]` | Sequence(int64) | renamed from `timbre_codebooks_idx` |
| `timbre_vector` | `[256]` | Sequence(float32) | new field from `spk_embs`, D_t=256 |
| `label` | scalar | dataset-native | bool for MUStARD, string for CREMA-D/SAVEE |

### Corrected #7 Module Set

| Module | Input | Output |
|---|---|---|
| `ProsodyEmbedding` | `[B, 1, T80]` int64 | `[B, T80, D_moss]` |
| `ContentEmbedding` | `[B, 2, T80]` int64 | `[B, T80, D_moss]` (sum 2 tables) |
| `AcousticEmbedding` | `[B, 3, T80]` int64 | `[B, T80, D_moss]` (sum 3 tables) |
| `TimbreProjection` | `[B, 256]` float32 | `[B, D_moss]` |
| `TemporalPool` | `[B, T80, D]` | `[B, T12, D]` |
| `ResidualFusion` | aligned `[B, T12, D]` streams | `[B, T12, D]` |

`TimbreEmbedding` (discrete lookup) is replaced by `TimbreProjection` (continuous linear projection). No discrete timbre codebook exists.

## Phases (High-Level)

### Phase 1: Contract Verification — COMPLETE
**Outcome:** FACodec tensor shapes confirmed: `vq_id=(6,B,80)`, `spk_embs=(B,256)`. All evidence recorded in spike report.
**Completed:** 4 commits — skeleton, internal evidence, external evidence, executable verification.

### Phase 2: Audit & Corrected Spec — IN PROGRESS
**Outcome:** Current #6 preprocessing and #7 model code audited against verified contract. Corrected schema and module specifications written and ready for downstream implementation.
**Depends on:** Phase 1

### Phase 3: Handoff
**Outcome:** All domain docs updated, blocker recommendation for #8 finalized, follow-up implementation issues defined and linked.
**Depends on:** Phase 2

## Open Questions
- ~~Is the FACodec checkpoint available locally?~~ Resolved: Yes, executable verification ran successfully.
- ~~What is the exact shape of `spk_embs`?~~ Resolved: `(256,)` float32 per utterance.
- ~~Does `vq_id` shape confirm 6 codebooks × 80 Hz?~~ Resolved: Confirmed at `(6, B, 80)`.
- Acoustic Stream storage format: nested `[3, T80]` vs. averaged scalar? (Recommendation: preserve codebook axis)
- Backward compatibility: reprocess all datasets or provide migration script?
