# Issue #6: Preprocessing Pipeline

> Implementation plan for running FACodec offline on speech datasets and pushing the results to HuggingFace Hub.

## Status

:white_check_mark: **Implementation Complete** - 74 tests passing
:warning: **Blocked** - No suitable pilot dataset. `hungphongtrn/mustard_plus_plus` doesn't exist.
  The source MUStARD dataset is video-based (large, needs audio extraction).
  Need a small audio-only HF dataset to pilot with.

## Quick Summary

This implements **Issue #6**: A preprocessing pipeline that loads any HuggingFace speech dataset, runs FACodec to extract content, prosody, and timbre codebook indices in a single forward pass, and pushes the result as a reusable HuggingFace dataset.

**Output per utterance (HF Dataset row):**
```python
{
    'dataset': str,                    # source HF repo name
    'id': str,                         # unique ID from source dataset
    'audio': {                         # HF Audio feature
        'array': array,
        'sampling_rate': int,
    },
    'content_codebooks_idx': list[int],  # FACodec content head indices
    'prosody_codebooks_idx': list[int],  # FACodec prosody head indices
    'timbre_codebooks_idx': list[int],   # FACodec timbre head indices
}
```

## Start Here

Read **[phase-01-preprocessing.md](./phase-01-preprocessing.md)** for the 7-task implementation plan.

## What This Implements

- [ ] FACodec encoder wrapper: content + prosody + timbre indices in single pass
- [ ] Generic HF dataset loader (not tied to MUStARD++)
- [ ] Batch processor with progress tracking and failure handling
- [ ] HF Dataset assembly and push to Hub
- [ ] Indices-only storage (vectors reconstructed later from codebook lookup tables)
- [ ] Summary report with statistics and failures

## Key Decisions

See [decisions.md](./decisions.md) for:
- FACodec as single encoder for all 3 codebooks (no MOSS-Audio)
- Indices-only storage (RAM-safe for large datasets)
- Generic dataset loader (not MUStARD-specific)
- HF Dataset as output format (not `.pt` files)

## Execution

Implement the 7 tasks sequentially (test-first):

```bash
# Task order: 1 → 2 → 3 → 4 → 5 → 6 → 7
# Each task: write failing test → implement → verify pass → commit
```

Validation:
```bash
python scripts/preprocess.py --dataset hungphongtrn/mustard_plus_plus --split test --max-samples 5
```

---

*This plan implements GitHub Issue #6: Preprocessing Pipeline*
