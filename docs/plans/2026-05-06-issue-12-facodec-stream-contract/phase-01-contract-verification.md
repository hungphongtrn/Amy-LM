# Phase 1: Contract Verification

## Phase Goal
Establish an evidence-backed FACodec tensor contract confirmed against Amphion README, source code, and (if available) a local checkpoint run. The output is a spike report with verified tensor mapping, confidence assessment, and identified unknowns.

## Files to Touch

| File | Action | Responsibility |
|------|--------|----------------|
| `docs/spikes/issue-12-facodec-stream-contract.md` | Create | Primary spike report |
| `scripts/inspect_facodec_contract.py` | Create | Optional executable inspector helper |
| `CONTEXT.md` | Read-only audit | Already updated during grill session; verify correctness |
| `src/preprocessing/facodec_encoder.py` | Read-only audit | Identify lines that conflate `vq_id[3:]` with timbre |
| `src/preprocessing/dataset_processor.py` | Read-only audit | Identify `timbre_codebooks_idx` in HF schema |
| `src/models/embedding.py` | Read-only audit | Identify `TimbreEmbedding` (discrete lookup) vs needed `TimbreProjection` |

No production code changes in this phase.

## Tasks

### Task 1: Create Spike Report Skeleton

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
## Preprocessing to MOSS-Audio Data Flow
## Current Implementation Audit
## Required Follow-Up Changes
## Issue #8 Blocker Recommendation
## Open Questions
```

- [ ] **Step 2: Commit the skeleton**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: scaffold FACodec stream contract spike report"
```

---

### Task 2: Record Internal Evidence

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`
- Audit: `CONTEXT.md`, `src/preprocessing/facodec_encoder.py`, `src/preprocessing/dataset_processor.py`, `src/models/embedding.py`

- [ ] **Step 1: Verify internal files exist**

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

- [ ] **Step 2: Record CONTEXT.md evidence**

Capture these lines from `CONTEXT.md`:
- Line 17-19: FACodec produces 1 prosody + 2 content + 3 acoustic + 1 timbre vector
- Line 35-37: Acoustic Stream defined as `vq_id[3:]`
- Line 43-45: Timbre Vector sourced from FACodec `spk_embs`
- Line 120-121: `timbre_codebooks_idx` was a mistake
- Line 126: Flagged ambiguity resolved

Add to `Evidence Sources` section:

```markdown
### CONTEXT.md (post-grill 2026-05-06)
- Already distinguishes Acoustic Stream (`vq_id[3:]`) from Timbre Vector (`spk_embs`)
- Documents the naming error: `timbre_codebooks_idx` stored residual acoustic VQ indices
- Confirms corrected contract per grill session
```

- [ ] **Step 3: Record facodec_encoder.py audit findings**

Inspect `src/preprocessing/facodec_encoder.py:246-247` and `src/preprocessing/facodec_encoder.py:318-369`:

```markdown
### src/preprocessing/facodec_encoder.py
- Line 246-247: Calls `fa_decoder(enc_out, eval_vq=False, vq=True)` and ignores `spk_embs`
- Line 318-327: Documents `vq_id[3:6]` as "residual/timbre" — the key conflation
- Line 367-369: Averages vq_id[3], vq_id[4], vq_id[5] into one integer — destroys codebook separation
```

- [ ] **Step 4: Record dataset_processor.py audit findings**

Inspect `src/preprocessing/dataset_processor.py:178-185`:

```markdown
### src/preprocessing/dataset_processor.py
- Line 182: `timbre_codebooks_idx: Sequence(Value("int64"))` — wrong name and type
- No `timbre_vector` field exists — must be added
```

- [ ] **Step 5: Record embedding.py audit findings**

Inspect `src/models/embedding.py:60-114`:

```markdown
### src/models/embedding.py
- Lines 60-114: `TimbreEmbedding` is a discrete embedding lookup over 1D integer indices
- Assumes timbre is a codebook index, not a continuous float vector
```

- [ ] **Step 6: Commit internal evidence**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: record internal FACodec evidence"
```

---

### Task 3: Record External Amphion Evidence

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Fetch Amphion FACodec README**

Read the Amphion FACodec repository README on GitHub or Hugging Face. Record the exact URL and quoted snippets:

```python
# Expected Amphion FACodec API:
vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
prosody_code = vq_id[:1]    # 1 codebook
cotent_code = vq_id[1:3]    # 2 codebooks
residual_code = vq_id[3:]   # 3+ codebooks
recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
```

- [ ] **Step 2: Document in the spike report**

```markdown
### Amphion FACodec README
- Source URL: [Amphion FACodec on GitHub](https://github.com/Audio-AGI/Amphion)
- `vq_id[:1]` → prosody (1 codebook)
- `vq_id[1:3]` → content (2 codebooks)
- `vq_id[3:]` → residual/acoustic detail (3 codebooks in current checkpoint)
- `spk_embs` → global speaker/timbre embedding, used in `fa_decoder.inference()`
- Conclusion: `spk_embs` is the utterance-level timbre vector, not frame-level VQ indices
```

- [ ] **Step 3: Record confidence assessment**

```markdown
## Confidence
- High: VQ stream slicing matches Amphion README exactly
- Medium-High: `spk_embs` as utterance-level timbre vector (consistent with README, but needs local checkpoint run for shape confirmation)
```

- [ ] **Step 4: Commit external evidence**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: record external Amphion FACodec evidence"
```

---

### Task 4: Add Optional Executable Contract Inspector

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
        raise SystemExit(
            "FACodec real checkpoint unavailable; inspector requires real Amphion FACodec."
        )

    samples = int(encoder.SAMPLE_RATE * args.seconds)
    audio = torch.zeros(samples, device=args.device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        enc_out = encoder._encoder(audio)
        vq_post_emb, vq_id, unknown, quantized, spk_embs = encoder._decoder(
            enc_out, eval_vq=False, vq=True
        )

    print(f"enc_out:           {tuple(enc_out.shape)} {enc_out.dtype}")
    print(f"vq_post_emb:       {tuple(vq_post_emb.shape)} {vq_post_emb.dtype}")
    print(f"vq_id:             {tuple(vq_id.shape)} {vq_id.dtype}")
    print(f"unknown_return_3:  {type(unknown).__name__}")
    print(f"quantized:         {tuple(quantized.shape)} {quantized.dtype}")
    print(f"spk_embs:          {tuple(spk_embs.shape)} {spk_embs.dtype}")
    print(f"timbre_vector:     {tuple(spk_embs.squeeze().shape)} {spk_embs.dtype}")
    print(f"prosody vq_id[:1]: {tuple(vq_id[:1].shape)}")
    print(f"content vq_id[1:3]:{tuple(vq_id[1:3].shape)}")
    print(f"acoustic vq_id[3:]:{tuple(vq_id[3:].shape)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the inspector**

```bash
uv run python scripts/inspect_facodec_contract.py --device cpu --seconds 1
```

**Expected if checkpoints available:** Output shows `vq_id: (6, ~80)` and `spk_embs` shape.
**Expected if checkpoints unavailable:** Script exits with `FACodec real checkpoint unavailable`.

- [ ] **Step 3: Record results or unavailability**

If shapes match the expected contract, paste them into the spike report's `Verified FACodec Tensor Contract` section.
If shapes differ (e.g., `spk_embs` is frame-level rather than utterance-level), stop and flag under `Open Questions`.
If executor unavailable, document and mark as pending:

```markdown
## Executable Verification Status
- Status: Pending (FACodec checkpoint not available locally)
- Fallback: Amphion README/source evidence used as authoritative
```

- [ ] **Step 4: Commit inspector and results**

```bash
git add scripts/inspect_facodec_contract.py docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: add FACodec contract inspector"
```

---

## Phase Completion Criteria
- [ ] Spike report skeleton created at `docs/spikes/issue-12-facodec-stream-contract.md`
- [ ] Internal evidence (CONTEXT.md, facodec_encoder.py, dataset_processor.py, embedding.py) recorded with line references
- [ ] External Amphion FACodec README evidence recorded with source URLs
- [ ] Optional inspector script created; output recorded as verification or documented as pending
- [ ] Confidence assessment for each tensor mapping included
- [ ] Open questions documented

## Handoff Notes
- If FACodec checkpoint was available and shapes confirmed: Phase 2 proceeds with high confidence.
- If checkpoint unavailable: Phase 2 proceeds with README-based evidence; shapes marked as provisional until executable run.
- The `TimbreEmbedding` → `TimbreProjection` replacement decision is confirmed — this is not a question, it's a correction.
- All VQ streams (Prosody, Content, Acoustic) preserve their codebook axes. No averaging.
