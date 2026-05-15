# FACodec Stream Contract Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use @subagent-driven-development (recommended) or @executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify FACodec's real stream semantics and publish the corrected Amy LM data contract before issue #8 changes MOSS-Audio internals.

**Architecture:** This is a documentation-first spike with one optional inspection helper. The deliverable is an evidence-backed contract that maps Amphion FACodec tensors to Amy LM canonical terms, then converts that contract into specific corrective tasks for preprocessing, model modules, and the MOSS-Audio Internal Residual Extension.

**Tech Stack:** Python 3.11, PyTorch, Amphion FACodec, Hugging Face Hub, pytest, Markdown

---

## File Structure

| File | Responsibility |
|------|----------------|
| `docs/spikes/issue-12-facodec-stream-contract.md` | Primary spike report: evidence, verified tensor contract, dataset schema, ASCII data-flow diagram, and blocker recommendation for issue #8 |
| `CONTEXT.md` | Canonical project language for FACodec, Prosody Stream, Acoustic Stream, Timbre Vector, and Residual Summation |
| `scripts/inspect_facodec_contract.py` | Optional executable helper to print actual FACodec output shapes when Amphion/checkpoints are available |
| `src/preprocessing/facodec_encoder.py` | Audit only in this issue; identify required follow-up changes to stop treating residual VQ codes as timbre |
| `src/preprocessing/dataset_processor.py` | Audit only in this issue; identify required follow-up schema changes for HF dataset rows |
| `scripts/preprocess.py` | Audit only in this issue; identify CLI/reporting references to old `timbre_codebooks_idx` naming |
| `scripts/fix_audio_column.py` | Audit only in this issue; identify feature-schema references to old `timbre_codebooks_idx` naming |
| `src/models/embedding.py` | Audit only in this issue; decide whether `TimbreEmbedding` should be replaced by a continuous projection |
| `tests/preprocessing/test_facodec_encoder.py` | Audit only in this issue; identify tests that currently encode the wrong stream assumptions |
| `tests/preprocessing/test_dataset_processor.py` | Audit only in this issue; identify dataset schema tests that need correction after the spike |
| `tests/models/test_embedding.py` | Audit only in this issue; identify tests that assume timbre is a discrete codebook index |

No production behavior should change in issue #12 except docs and the optional inspection helper. Renames, schema migration, and module changes should become follow-up issue updates or new implementation issues.

---

## Known Starting Evidence

Issue #12 exists because the current code likely conflates FACodec residual/acoustic VQ codes with the FACodec utterance-level timbre vector.

Current repo evidence:
- `CONTEXT.md:17-18` already states FACodec produces 1 prosody codebook, 2 content codebooks, 3 acoustic detail codebooks, and 1 global timbre vector.
- `CONTEXT.md:35-37` states Timbre Vector is utterance-level and not per-frame.
- `src/preprocessing/facodec_encoder.py:246-247` calls `fa_decoder(enc_out, eval_vq=False, vq=True)` and ignores `spk_embs` except as an unused return value.
- `src/preprocessing/facodec_encoder.py:318-327` documents `vq_id[3:6]` as `residual/timbre`, then maps the average of those frame-level codebooks into `timbre`.
- `src/preprocessing/dataset_processor.py:178-185` persists `timbre_codebooks_idx` as a sequence of int64 values.
- `src/models/embedding.py:60-114` implements `TimbreEmbedding` as a discrete embedding lookup over 1D integer indices.

External FACodec README evidence to verify during execution:
- Amphion example returns `vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)`.
- Amphion example slices `vq_id[:1]` as prosody, `vq_id[1:3]` as content, and `vq_id[3:]` as residual/acoustic detail codes.
- Amphion example passes `spk_embs` to `fa_decoder.inference(vq_post_emb, spk_embs)`, making it the likely utterance-level timbre/speaker embedding.

Working definition for this spike: `h_t` means the utterance-level Timbre Vector consumed by Amy LM. The expected FACodec source is `spk_embs` after removing batch-only dimensions; `TimbreProjection` should convert that vector into Amy LM embedding space before broadcasting across MOSS-Audio frames.

---

## Task 1: Create Spike Report Skeleton

**Files:**
- Create: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Create the report with required sections**

```markdown
# Issue #12: FACodec Stream Contract Spike

## Summary

## Evidence Sources

## Verified FACodec Tensor Contract

## Canonical Amy LM Terms

## Corrected Training Sample Schema

## Preprocessing To MOSS-Audio Data Flow

## Current Implementation Audit

## Required Follow-Up Changes

## Issue #8 Blocker Recommendation

## Open Questions
```

- [ ] **Step 2: Commit the skeleton**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: scaffold FACodec stream contract spike"
```

---

## Task 2: Verify FACodec API From Source And README

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Record repo-local evidence**

Inspect these exact files and capture line references in the report:

```bash
uv run python - <<'PY'
from pathlib import Path
for path in [
    "CONTEXT.md",
    "src/preprocessing/facodec_encoder.py",
    "src/preprocessing/dataset_processor.py",
    "src/models/embedding.py",
]:
    print(path, Path(path).exists())
PY
```

Expected: all four paths print `True`.

- [ ] **Step 2: Record external FACodec README evidence**

Use GitHub or Hugging Face source for Amphion FACodec and record the exact URL and quoted snippets for:

```python
vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
prosody_code = vq_id[:1]
cotent_code = vq_id[1:3]
residual_code = vq_id[3:]
recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
```

Expected conclusion: `vq_id` has six frame-level codebooks, while `spk_embs` is a separate continuous speaker/timbre tensor.

Add this definition to the report if confirmed:

```markdown
`h_t` is Amy LM's utterance-level Timbre Vector. In Amphion FACodec, the source tensor is `spk_embs` from `fa_decoder(enc_out, eval_vq=False, vq=True)`, after squeezing batch-only dimensions. It is not derived from `vq_id[3:]`.
```

- [ ] **Step 3: Record confidence level and unknowns**

Add a short confidence statement:

```markdown
Confidence: High for VQ stream slicing because Amphion's public README demonstrates the exact slices. Medium for the semantic name `Timbre Vector` until an executable run confirms `spk_embs` shape with the project checkpoint.
```

- [ ] **Step 4: Commit evidence update**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: record FACodec stream evidence"
```

---

## Task 3: Add Optional Executable Contract Inspector

**Files:**
- Create: `scripts/inspect_facodec_contract.py`
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Write the helper script**

```python
"""Print FACodec tensor shapes for issue #12 contract verification.

This script is intentionally read-only. It does not preprocess datasets or save
model outputs; it only loads FACodec, runs one synthetic waveform, and prints the
returned tensor shapes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.preprocessing.facodec_encoder import FACodecEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    args = parser.parse_args()

    encoder = FACodecEncoder(
        device=args.device,
        checkpoint_path=str(args.checkpoint_path) if args.checkpoint_path else None,
    )
    if encoder._mock:
        raise SystemExit("FACodec real checkpoint unavailable; inspector requires real Amphion FACodec.")

    samples = int(encoder.SAMPLE_RATE * args.seconds)
    audio = torch.zeros(samples, device=args.device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        enc_out = encoder._encoder(audio)
        vq_post_emb, vq_id, unknown, quantized, spk_embs = encoder._decoder(
            enc_out, eval_vq=False, vq=True
        )

    print(f"enc_out: {tuple(enc_out.shape)} {enc_out.dtype}")
    print(f"vq_post_emb: {tuple(vq_post_emb.shape)} {vq_post_emb.dtype}")
    print(f"vq_id: {tuple(vq_id.shape)} {vq_id.dtype}")
    print(f"unknown_return_3: {type(unknown).__name__}")
    print(f"quantized: {tuple(quantized.shape)} {quantized.dtype}")
    print(f"spk_embs: {tuple(spk_embs.shape)} {spk_embs.dtype}")
    print(f"timbre_vector_candidate: {tuple(spk_embs.squeeze().shape)} {spk_embs.dtype}")
    print(f"prosody vq_id[:1]: {tuple(vq_id[:1].shape)}")
    print(f"content vq_id[1:3]: {tuple(vq_id[1:3].shape)}")
    print(f"residual_acoustic vq_id[3:]: {tuple(vq_id[3:].shape)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the helper in the current environment**

```bash
uv run python scripts/inspect_facodec_contract.py --device cpu --seconds 1
```

Expected if checkpoints are available: output includes `vq_id: (6, 1, ~80)` and a `spk_embs` shape. Expected if checkpoints are unavailable: script exits with `FACodec real checkpoint unavailable; inspector requires real Amphion FACodec.`

- [ ] **Step 3: Add script output or unavailable status to the report**

If real FACodec runs, paste the exact printed shapes into `Verified FACodec Tensor Contract`. If it does not run, document the reason and keep the source/README evidence as the authoritative basis.

If shapes differ from the expected six-codebook VQ stream or `spk_embs` appears frame-level rather than utterance-level, stop and ask for human review before finalizing the schema. In that case, document the actual shapes under `Open Questions` and mark the blocker recommendation as provisional.

- [ ] **Step 4: Commit inspector and report update**

```bash
git add scripts/inspect_facodec_contract.py docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: add FACodec contract inspector"
```

---

## Task 4: Define Corrected Training Sample Schema

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Add canonical schema table**

Use this schema unless Task 2 or Task 3 disproves it:

| Field | Shape Per Sample | Dtype | Frame Rate | Source | Notes |
|-------|------------------|-------|------------|--------|-------|
| `dataset` | scalar | string | n/a | source dataset | Source HF dataset name |
| `id` | scalar | string | n/a | source dataset | Stable sample identifier |
| `audio` | variable samples | HF `Audio(sampling_rate=16000)` | 16 kHz | source dataset | Needed for Online Semantic Encoding |
| `prosody_codebooks_idx` | `[1, T_facodec]` or flattened `[T_facodec]` with documented single-codebook convention | int64 | 80 Hz | `vq_id[:1]` | Prosody Stream |
| `content_codebooks_idx` | `[2, T_facodec]` | int64 | 80 Hz | `vq_id[1:3]` | Store for audit/reconstruction, not Semantic Stream for issue #8 |
| `acoustic_codebooks_idx` | `[3, T_facodec]` | int64 | 80 Hz | `vq_id[3:]` | Residual/acoustic detail codebooks; do not call timbre |
| `timbre_vector` | capture from inspector line `timbre_vector_candidate`, likely `[D_timbre]` after squeezing batch-only dims | float32 | utterance-level | `spk_embs` | Broadcast to MOSS-Audio frame count during Residual Summation; this is Amy LM `h_t` |
| `label` | scalar or task-specific | int64/string | n/a | downstream benchmark | Required by issue #8 training dataset |

Schema note: `content_codebooks_idx` is a canonical FACodec stream, but issue #8 should not consume it as Amy LM's Semantic Stream. For issue #8, `S_t` comes from Online Semantic Encoding through MOSS-Audio using the `audio` field.

- [ ] **Step 2: Add conversion notes for current implementation**

Document these exact corrections:

```markdown
- `timbre_codebooks_idx` should be renamed to `acoustic_codebooks_idx` if it continues to store `vq_id[3:]`.
- `timbre_vector` should be added as a float field sourced from `spk_embs`.
- Averaging multiple VQ codebooks into one integer stream should stop unless an implementation issue explicitly justifies that compression.
- Issue #8 should not consume FACodec `content_codebooks_idx` as the Semantic Stream; MOSS-Audio computes Semantic Stream online from `audio`.
```

- [ ] **Step 3: Commit schema update**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: define corrected FACodec training schema"
```

---

## Task 5: Draw Corrected Data Flow

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Add ASCII diagram**

```text
Preprocessing
  Raw audio waveform, 16 kHz
        |
        v
  FACodec encoder/decoder
        |
        +--> vq_id[:1]  -> prosody_codebooks_idx, 80 Hz, int64
        +--> vq_id[1:3] -> content_codebooks_idx, 80 Hz, int64, audit/reconstruction only
        +--> vq_id[3:]  -> acoustic_codebooks_idx, 80 Hz, int64
        +--> spk_embs   -> timbre_vector, utterance-level, float32

Issue #8 training
  audio waveform -------------------------> MOSS-Audio encoder -> Semantic Stream S_t, 12.5 Hz
  prosody_codebooks_idx -> ProsodyEmbedding -> TemporalPool 80 Hz to 12.5 Hz -> P_t
  timbre_vector --------> TimbreProjection -> broadcast to 12.5 Hz ----------> H_timbre
  acoustic_codebooks_idx -> optional AcousticEmbedding -> TemporalPool -------> A_t, if accepted

  Residual Summation inside MOSS-Audio:
      H_t = LayerNorm(S_t + lambda * (P_t + H_timbre [+ A_t]))
```

- [ ] **Step 2: Add decision note on Acoustic Stream inclusion**

Document the recommended default for issue #8:

```markdown
Recommendation: issue #8 should include Prosody Stream and Timbre Vector first. Acoustic Stream should remain optional unless the experiment explicitly needs residual acoustic detail, because it risks injecting speaker/environment artifacts beyond the social-prosody hypothesis.
```

- [ ] **Step 3: Commit diagram update**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: diagram corrected residual extension data flow"
```

---

## Task 6: Audit Issue #6 Preprocessing Implementation

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Audit `src/preprocessing/facodec_encoder.py`**

Record required follow-up changes with line references:

```markdown
- Change public return contract from `(content_indices, prosody_indices, timbre_indices)` to a named structure such as `FACodecStreams`.
- Preserve codebook axis for `content_codebooks_idx` and `acoustic_codebooks_idx`; do not average codebooks into one integer per frame.
- Store `spk_embs` as `timbre_vector`.
- Update mock mode to generate tensors matching real contract shape and dtype.
- Correct docstrings that claim FACodec frame rate is nominally 12.5 Hz for preprocessing outputs.
```

- [ ] **Step 2: Audit `src/preprocessing/dataset_processor.py`, `scripts/preprocess.py`, and `scripts/fix_audio_column.py`**

Record required follow-up changes:

```markdown
- Replace `timbre_codebooks_idx: Sequence(Value("int64"))` with `acoustic_codebooks_idx` for residual VQ codes.
- Add `timbre_vector` as a float sequence feature once exact `spk_embs` shape is known.
- Update validation/report summaries to report prosody/acoustic frame counts and timbre vector dimension separately.
- Keep HF Dataset/parquet output unless issue #8 introduces a concrete need for `.pt` files.
```

- [ ] **Step 3: Audit preprocessing tests**

Record required test changes:

```markdown
- Tests should assert 80 Hz FACodec frame counts for VQ streams.
- Tests should assert content has 2 codebooks and acoustic has 3 codebooks.
- Tests should assert timbre is a float vector, not a frame-level integer sequence.
```

- [ ] **Step 4: Commit preprocessing audit**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: audit preprocessing FACodec contract changes"
```

---

## Task 7: Audit Issue #7 Model Modules

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Audit `src/models/embedding.py`**

Record required follow-up changes:

```markdown
- Keep `ProsodyEmbedding` for `prosody_codebooks_idx`.
- Replace or supplement `TimbreEmbedding` with `TimbreProjection`, an `nn.Linear(timbre_dim, embed_dim)` or small projection module accepting float tensors of shape `[B, D_timbre]`.
- Do not keep the name `TimbreEmbedding` for residual/acoustic VQ codes; if acoustic codes are used, create `AcousticEmbedding` with explicit multi-codebook handling.
```

- [ ] **Step 2: Audit `src/models/fusion.py` and `src/models/pooling.py` if present**

Record whether current `ResidualFusion` can accept a projected, broadcast timbre tensor:

```markdown
- `TemporalPool` remains appropriate for Prosody Stream and optional Acoustic Stream.
- `ResidualFusion` should accept tensors already aligned to `[B, T_moss, embed_dim]`.
- Timbre Vector projection should broadcast before fusion, not go through temporal pooling.
```

- [ ] **Step 3: Audit model tests**

Record required test changes:

```markdown
- Replace `TimbreEmbedding` integer-index tests with `TimbreProjection` float-vector tests.
- Add tests for broadcasting projected timbre from `[B, D_timbre]` to `[B, T_moss, embed_dim]` in the issue #8 integration layer.
- Add acoustic embedding tests only if Acoustic Stream is accepted for issue #8.
```

- [ ] **Step 4: Commit model audit**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: audit timbre and acoustic model modules"
```

---

## Task 8: Update Canonical Domain Language

**Files:**
- Modify: `CONTEXT.md`
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Verify domain context file exists**

```bash
uv run python - <<'PY'
from pathlib import Path
path = Path("CONTEXT.md")
if not path.is_file():
    raise SystemExit("CONTEXT.md not found")
print("CONTEXT.md found")
PY
```

Expected: `CONTEXT.md found`.

- [ ] **Step 2: Update FACodec language in `CONTEXT.md` only if confirmed**

Expected text changes:

```markdown
**FACodec**:
A third-party factorized neural audio codec (Microsoft, arXiv:2403.03100). Produces 1 prosody VQ codebook, 2 content VQ codebooks, 3 residual acoustic VQ codebooks, and 1 global continuous timbre vector (`spk_embs` in Amphion examples). Used as a substitute for Amy Codec during pilot validation.

**Acoustic Stream (a_t)**:
Discrete residual acoustic detail indices from FACodec `vq_id[3:]` at 80 Hz. Optional for the Amy LM pilot; do not call this the Timbre Vector.

**Timbre Vector**:
A single global utterance-level continuous embedding representing speaker identity. Sourced from FACodec `spk_embs`, not from frame-level VQ indices.
```

- [ ] **Step 3: Add a short report note describing the language update**

```markdown
`CONTEXT.md` now distinguishes Acoustic Stream (`vq_id[3:]`) from Timbre Vector (`spk_embs`). This resolves the naming ambiguity that caused `timbre_codebooks_idx` to refer to residual acoustic codebooks.
```

- [ ] **Step 4: Commit domain language update**

```bash
git add CONTEXT.md docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: clarify FACodec acoustic and timbre terms"
```

---

## Task 9: Final Recommendation And Issue Handoff

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Add blocker recommendation**

Use this recommendation unless the verified evidence contradicts it:

```markdown
Recommendation: Block issue #8 until issue #6 preprocessing and issue #7 modules are corrected or explicitly patched in the issue #8 branch. The current contract would feed residual acoustic VQ indices through a discrete `TimbreEmbedding`, which does not match FACodec's timbre path and would invalidate the intended Prosody Stream + Timbre Vector experiment.
```

- [ ] **Step 2: Add follow-up checklist**

```markdown
## Follow-Up Implementation Issues

- Update issue #6: emit `acoustic_codebooks_idx` and `timbre_vector`; preserve multi-codebook axes; update HF features and tests.
- Update issue #7: replace `TimbreEmbedding` with `TimbreProjection`; add optional `AcousticEmbedding` only if Acoustic Stream is accepted.
- Update issue #8: consume `audio`, `prosody_codebooks_idx`, and `timbre_vector` by default; treat `acoustic_codebooks_idx` as optional.
```

- [ ] **Step 3: Run documentation checks**

```bash
uv run python - <<'PY'
from pathlib import Path
for path in ["docs/spikes/issue-12-facodec-stream-contract.md", "CONTEXT.md"]:
    text = Path(path).read_text()
    assert "timbre_codebooks_idx" in text
    assert "acoustic_codebooks_idx" in text
    assert "Timbre Vector" in text
print("docs contract terms present")
PY
```

Expected: `docs contract terms present`.

- [ ] **Step 4: Commit final spike report**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: complete FACodec stream contract spike"
```

---

## Final Acceptance Checklist

- [ ] FACodec output contract is verified against Amphion source/README or an executable local run.
- [ ] Report identifies Prosody Stream, Content VQ codes, Acoustic Stream, and Timbre Vector with source tensors.
- [ ] Report identifies the exact Amphion tensor for utterance-level timbre (`spk_embs`, unless disproven by execution).
- [ ] Corrected training sample schema includes field names, shapes, dtypes, and frame rates.
- [ ] ASCII diagram shows preprocessing through MOSS-Audio Internal Residual Extension.
- [ ] Current issue #6 implementation audit lists required renames and field additions.
- [ ] Current issue #7 module audit states whether `TimbreEmbedding` should become `TimbreProjection`.
- [ ] `CONTEXT.md` uses canonical terms for Acoustic Stream and Timbre Vector.
- [ ] Report explicitly recommends whether issue #8 remains blocked.

---

## Implementation Notes

- Keep issue #12 scoped to evidence and documentation. Do not rename dataset fields or model classes inside this spike unless the user explicitly expands scope.
- Prefer HF Dataset/parquet for preprocessed output because issue #6 already implemented that direction and issue #12 has no evidence that `.pt` files are required.
- If real FACodec checkpoints are unavailable locally, do not block the spike; cite the Amphion README/source and mark executable shape verification as pending.
- Use small commits after each task so issue #6, #7, and #8 owners can cherry-pick docs or helper code independently.
