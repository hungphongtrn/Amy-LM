# Issue #6: Preprocessing Pipeline

Implementation plan for running FACodec offline on speech datasets and pushing the results to HuggingFace Hub as a reusable dataset.

## Acceptance Criteria

- [ ] FACodec encoder loads from Amphion checkpoint, extracts content, prosody, and timbre codebook indices per utterance in a single forward pass
- [ ] Pipeline processes any HuggingFace audio dataset (not tied to MUStARD++ specifically)
- [ ] Output is a HuggingFace dataset with the required columns (see below)
- [ ] Dataset pushed to HuggingFace Hub for reuse in training
- [ ] Only codebook indices stored (not full vectors) — vectors are reconstructed later from codebook lookup tables

## Output Format

Each row in the HuggingFace dataset:

```python
{
    'dataset': str,       # source HF repo name (e.g., "hungphongtrn/mustard_plus_plus")
    'id': str,            # unique ID from source dataset
    'audio': {            # HF Audio feature
        'array': array,   # waveform
        'sampling_rate': int,
    },
    'content_codebooks_idx': list[int],   # FACodec content head indices
    'prosody_codebooks_idx': list[int],   # FACodec prosody head indices
    'timbre_codebooks_idx': list[int],    # FACodec timbre head indices
}
```

Columns no longer needed (was in old plan):
- `content_codebooks` / `prosody_codebooks` / `timbre_codebooks` — vectors are reconstructed from indices + codebook lookup tables at training time

## Files to Create

| File | Responsibility |
|------|----------------|
| `src/preprocessing/__init__.py` | Module marker |
| `src/preprocessing/facodec_encoder.py` | FACodec wrapper: content + prosody + timbre indices |
| `src/preprocessing/dataset_processor.py` | Iterates HF dataset, encodes each sample, builds output HF dataset |
| `src/preprocessing/reporting.py` | Summary report generation |
| `scripts/preprocess.py` | CLI entry point |
| `tests/preprocessing/test_facodec_encoder.py` | Unit tests for FACodec wrapper |
| `tests/preprocessing/test_dataset_processor.py` | Unit tests for batch processor |
| `tests/preprocessing/test_integration.py` | End-to-end pipeline test |

Removed from old plan:
- ~~`src/preprocessing/mustard_downloader.py`~~ — use generic HF `datasets` library instead
- ~~`src/preprocessing/moss_encoder.py`~~ — FACodec handles all 3 codebooks
- ~~`src/preprocessing/alignment.py`~~ — no cross-encoder alignment needed
- ~~`src/preprocessing/batch_processor.py`~~ — consolidated into `dataset_processor.py`

## Implementation Tasks

### Task 1: Project Setup

**Files:** `pyproject.toml`, `src/preprocessing/__init__.py`

1. Add preprocessing dependencies:
```toml
[project.optional-dependencies]
preprocessing = [
    "datasets>=2.15.0",
    "huggingface-hub>=0.20.0",
    "soundfile>=0.13.1",
    "torchaudio>=2.9.1",
    "tqdm>=4.66.0",
]
```

2. Create module structure:
```bash
mkdir -p src/preprocessing tests/preprocessing scripts
touch src/preprocessing/__init__.py
touch tests/preprocessing/__init__.py
```

3. Install: `uv pip install -e ".[preprocessing]"`

---

### Task 2: FACodec Encoder Wrapper

**Files:** `src/preprocessing/facodec_encoder.py`, `tests/preprocessing/test_facodec_encoder.py`

Interface:
- `FACodecEncoder(device: str, checkpoint_path: Optional[str])`
- `encode(audio: Tensor) -> Tuple[content_indices, prosody_indices, timbre_indices]`
  - Single forward pass through FACodec
  - Returns only indices (list[int] or Tensor), not vectors
  - Content: discrete indices from content head (vocab size ~2048)
  - Prosody: discrete indices from prosody head (vocab size ~2048)
  - Timbre: discrete indices from timbre head (vocab size ~2048)
- Mock fallback if Amphion not installed (for tests)
- Frame rate: ~12.5 Hz (native FACodec output)

---

### Task 3: Dataset Processor

**Files:** `src/preprocessing/dataset_processor.py`, `tests/preprocessing/test_dataset_processor.py`

Interface:
- `DatasetProcessor(facodec: FACodecEncoder, output_dir: Path, device: str)`
- `process_dataset(dataset_name: str, split: str, max_samples: Optional[int] = None) -> Dataset`
  - Loads dataset from HF Hub using `datasets.load_dataset(dataset_name, split=split)`
  - Iterates samples, runs FACodec on each audio
  - Builds HF Dataset with all required columns
  - Reports failures without stopping
- `save(dataset: Dataset, repo_id: str)` — saves to disk as parquet
- `push_to_hub(dataset: Dataset, repo_id: str)` — pushes to HF Hub

---

### Task 4: Summary Reporting

**Files:** `src/preprocessing/reporting.py`, `tests/preprocessing/test_reporting.py`

Interface:
- `ProcessingSummary` dataclass — tracks:
  - Total utterances processed / failed
  - Average content / prosody / timbre frame counts
  - Duration histogram
  - Failed utterance IDs + error messages
- `generate_report(summary, output_path)` — outputs JSON report

---

### Task 5: CLI Entry Point

**File:** `scripts/preprocess.py`

CLI:
```bash
python scripts/preprocess.py \
    --dataset hungphongtrn/mustard_plus_plus \
    --split train \
    --output-repo hungphongtrn/mustard_facodec \
    [--max-samples N] \
    [--device cuda/cpu] \
    [--output-dir data/processed]
```

Supports processing multiple datasets by running the script multiple times with different `--dataset` values.

---

### Task 6: Integration Test

**File:** `tests/preprocessing/test_integration.py`

Validates:
- Mock audio (2 sec) processes end-to-end through FACodec mock
- Output HF dataset has all required columns with correct dtypes
- Indices are integers within valid codebook range
- Dataset can be round-tripped through save/load

---

### Task 7: Run Full Test Suite

```bash
uv run pytest tests/preprocessing/ -v --tb=short
```

---

## Completion Checklist

Issue #6 acceptance criteria:

- [ ] FACodec encoder loads, extracts content + prosody + timbre indices in single pass
- [ ] Pipeline works with any HF audio dataset
- [ ] Output HF dataset has columns: `dataset`, `id`, `audio`, `content_codebooks_idx`, `prosody_codebooks_idx`, `timbre_codebooks_idx`
- [ ] Only indices stored (no vectors) — RAM-safe for large datasets
- [ ] Dataset pushed to HuggingFace Hub
- [ ] Summary report generated with totals, averages, failures

Run validation:
```bash
python scripts/preprocess.py --dataset hungphongtrn/mustard_plus_plus --split test --max-samples 5
```
