# Issue #6: MUStARD++ Preprocessing Pipeline

Implementation plan for downloading MUStARD++ dataset and extracting aligned features using FACodec and MOSS-Audio encoders.

## Acceptance Criteria

Per Issue #6, the implementation must satisfy:

- [ ] MUStARD++ dataset downloaded and accessible
- [ ] FACodec encoder loads from Amphion checkpoint, extracts prosody indices (80 Hz) and timbre vector per utterance
- [ ] MOSS-Audio encoder loads, extracts semantic frames (12.5 Hz) per utterance
- [ ] Temporal alignment logged: prosody frame count, semantic frame count, pooling ratio per file
- [ ] `.pt` files saved per utterance with all four fields
- [ ] A summary report: total utterances processed, avg frame counts, any failures

## Output Format

Each `.pt` file contains:

```python
{
    'utterance_id': str,
    'semantic_frames': (N_sem, 2560),       # MOSS-Audio at 12.5 Hz
    'prosody_indices': (N_sem, 1),          # FACodec prosody pooled to match semantic
    'prosody_indices_raw': (N_pros,),         # Original FACodec at 80 Hz
    'timbre_vector': (256,),                # Global timbre from FACodec
    'label': int,                           # 0 or 1 (sarcasm)
    'duration_sec': float,
    'alignment_info': {
        'prosody_frames': int,      # Original count at 80 Hz
        'semantic_frames': int,     # Count at 12.5 Hz
        'pooling_ratio': float,     # N_pros / N_sem
    },
    'metadata': dict
}
```

## Files to Create

| File | Responsibility |
|------|----------------|
| `src/preprocessing/__init__.py` | Module marker |
| `src/preprocessing/mustard_downloader.py` | Download/cache MUStARD++ from HuggingFace |
| `src/preprocessing/facodec_encoder.py` | FACodec wrapper (prosody + timbre extraction) |
| `src/preprocessing/moss_encoder.py` | MOSS-Audio wrapper (semantic frames) |
| `src/preprocessing/alignment.py` | Temporal pooling: 80 Hz → 12.5 Hz |
| `src/preprocessing/batch_processor.py` | Orchestration + `.pt` file generation |
| `src/preprocessing/reporting.py` | Summary report generation |
| `scripts/preprocess_mustard.py` | CLI entry point |
| `tests/preprocessing/test_*.py` | Unit tests per module |
| `tests/preprocessing/test_integration.py` | End-to-end pipeline test |

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

### Task 2: MUStARD++ Downloader

**Files:** `src/preprocessing/mustard_downloader.py`, `tests/preprocessing/test_mustard_downloader.py`

Interface:
- `MustardDownloader(cache_dir: Path)` — dataset: `hungphongtrn/mustard_plus_plus`
- `download(split: str) -> Path` — returns audio directory path
- `iter_utterances(split: str) -> Iterator[Tuple[utt_id, audio_array, sr, label, metadata]]`

---

### Task 3: FACodec Encoder

**Files:** `src/preprocessing/facodec_encoder.py`, `tests/preprocessing/test_facodec_encoder.py`

Interface:
- `FACodecEncoder(device: str, checkpoint_path: Optional[str])`
- `encode(audio: Tensor) -> Tuple[prosody_indices, timbre_vector]`
- Prosody: ~80 Hz, discrete indices (vocab size 1024)
- Timbre: 256-dim global vector per utterance
- Mock fallback if Amphion not installed

---

### Task 4: MOSS-Audio Encoder

**Files:** `src/preprocessing/moss_encoder.py`, `tests/preprocessing/test_moss_encoder.py`

Interface:
- `MOSSAudioEncoder(device: str, model_size: str = "4B")`
- `encode(audio: Tensor) -> semantic_frames: (batch, N_frames, 2560)`
- Frame rate: 12.5 Hz
- Mock fallback if transformers not available

---

### Task 5: Temporal Alignment

**Files:** `src/preprocessing/alignment.py`, `tests/preprocessing/test_alignment.py`

Interface:
- `TemporalAligner(prosody_rate=80.0, semantic_rate=12.5)`
- `pool_prosody_to_semantic(prosody_indices, target_frames) -> pooled_indices`
- Uses adaptive average pooling over time dimension
- `AlignmentInfo` dataclass for logging

---

### Task 6: Batch Processor

**Files:** `src/preprocessing/batch_processor.py`, `tests/preprocessing/test_batch_processor.py`

Interface:
- `BatchProcessor(facodec, moss, aligner, output_dir, device)`
- `process_utterance(utt_id, audio, sr, label, metadata) -> Optional[Path]`
- `process_dataset(split, max_utterances) -> ProcessingSummary`
- Saves `.pt` files with all required fields per Issue #6

---

### Task 7: Summary Reporting

**Files:** `src/preprocessing/reporting.py`, `tests/preprocessing/test_reporting.py`

Interface:
- `ProcessingSummary` — tracks total utterances, frame counts, failures
- `generate_report(summary, output_path)` — JSON summary with:
  - Total utterances processed
  - Average prosody frames per utterance
  - Average semantic frames per utterance
  - Average pooling ratio
  - Failed utterances list

---

### Task 8: CLI Entry Point

**File:** `scripts/preprocess_mustard.py`

CLI:
```bash
python scripts/preprocess_mustard.py \
    --split {train,validation,test,all} \
    --output-dir data/mustard_pp_processed \
    [--max-utterances N] \
    [--device cuda/cpu] \
    [--cache-dir ~/.cache/mustard]
```

---

### Task 9: Integration Test

**File:** `tests/preprocessing/test_integration.py`

Validates:
- Mock audio (2 sec) processes end-to-end
- Output `.pt` file exists and loads correctly
- All Issue #6 required fields present with correct shapes
- Alignment info logged correctly

---

### Task 10: Run Full Test Suite

```bash
uv run pytest tests/preprocessing/ -v --tb=short
```

---

## Completion Checklist

Issue #6 acceptance criteria:

- [ ] MUStARD++ dataset downloaded and accessible
- [ ] FACodec encoder loads, extracts prosody indices (80 Hz) + timbre vector
- [ ] MOSS-Audio encoder loads, extracts semantic frames (12.5 Hz)
- [ ] Temporal alignment logged with frame counts and pooling ratio
- [ ] `.pt` files saved per utterance with `semantic_frames`, `prosody_indices`, `timbre_vector`, `label`
- [ ] Summary report generated with totals, averages, and failures

Run validation:
```bash
python scripts/preprocess_mustard.py --split test --max-utterances 5
```

---

## Implementation Reference

Full implementation details: `OLD-full-plan.md` (lines 117-1865)
