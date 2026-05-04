# Phase 1: MUStARD++ Preprocessing

## Phase Goal

Preprocessed `.pt` files ready for training, with aligned (semantic frames, prosody indices, timbre vector, label) tuples per utterance.

**Success criteria:**
- ✓ MUStARD++ dataset downloaded from HuggingFace with caching
- ✓ FACodec encoder loads and extracts prosody + timbre
- ✓ MOSS-Audio encoder loads and extracts semantic frames
- ✓ Temporal alignment logged (prosody count, semantic count, pooling ratio)
- ✓ `.pt` files saved per utterance with all four required fields
- ✓ Summary report with statistics and any failures

## Files to Touch

| File | Responsibility |
|------|----------------|
| `src/preprocessing/__init__.py` | Module marker |
| `src/preprocessing/mustard_downloader.py` | Download/cache MUStARD++ |
| `src/preprocessing/facodec_encoder.py` | FACodec wrapper (prosody + timbre) |
| `src/preprocessing/moss_encoder.py` | MOSS-Audio wrapper (semantic) |
| `src/preprocessing/alignment.py` | Temporal pooling 80 Hz → 12.5 Hz |
| `src/preprocessing/batch_processor.py` | Orchestration + .pt file generation |
| `scripts/preprocess_mustard.py` | CLI entry point |
| `tests/preprocessing/test_*.py` | Unit tests per module |
| `tests/preprocessing/test_integration.py` | End-to-end pipeline test |
| `tests/preprocessing/smoke_test.py` | Quick validation |

## Tasks

### Task 1: Project Setup

**Files:**
- Create: `src/preprocessing/__init__.py`
- Modify: `pyproject.toml`

**Steps:**

- [ ] **Step 1: Add preprocessing dependencies to pyproject.toml**

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

- [ ] **Step 2: Create module structure**

```bash
mkdir -p src/preprocessing tests/preprocessing scripts
touch src/preprocessing/__init__.py
touch tests/preprocessing/__init__.py
```

- [ ] **Step 3: Install dependencies**

```bash
uv pip install -e ".[preprocessing]"
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/preprocessing/ tests/preprocessing/
git commit -m "chore: add preprocessing dependencies and module structure"
```

---

### Task 2: MUStARD++ Dataset Downloader

**Files:**
- Create: `src/preprocessing/mustard_downloader.py`
- Create: `tests/preprocessing/test_mustard_downloader.py`

**Steps:**

- [ ] **Step 1: Write the failing test**

```python
# tests/preprocessing/test_mustard_downloader.py
import pytest
import tempfile
from pathlib import Path
from src.preprocessing.mustard_downloader import MustardDownloader

def test_mustard_downloader_initialization():
    """Test that downloader initializes with correct paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        downloader = MustardDownloader(cache_dir=cache_dir)
        assert downloader.cache_dir == cache_dir
        assert downloader.dataset_name == "hungphongtrn/mustard_plus_plus"
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
uv run pytest tests/preprocessing/test_mustard_downloader.py::test_mustard_downloader_initialization -v
```

- [ ] **Step 3: Write minimal implementation**

See OLD-full-plan.md lines 117-246 for full implementation reference.

Key interface:
- `MustardDownloader(cache_dir: Optional[Path])`
- `download(split: str) -> Path`
- `iter_utterances(split: str) -> Iterator[(utt_id, audio, sr, label, metadata)]`

- [ ] **Step 4: Run test — expect PASS**

```bash
uv run pytest tests/preprocessing/test_mustard_downloader.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/mustard_downloader.py tests/preprocessing/test_mustard_downloader.py
git commit -m "feat: add MUStARD++ dataset downloader"
```

---

### Task 3: FACodec Encoder Wrapper

**Files:**
- Create: `src/preprocessing/facodec_encoder.py`
- Create: `tests/preprocessing/test_facodec_encoder.py`

**Steps:**

- [ ] **Step 1: Write the failing test**

```python
# tests/preprocessing/test_facodec_encoder.py
import pytest
import torch
from src.preprocessing.facodec_encoder import FACodecEncoder

def test_facodec_encoder_initialization():
    """Test FACodec encoder loads correctly."""
    encoder = FACodecEncoder(device="cpu")
    assert encoder.device == "cpu"
    assert encoder.sample_rate == 16000
    assert hasattr(encoder, 'encode')

def test_facodec_encode_shape():
    """Test encoding produces expected output shapes."""
    encoder = FACodecEncoder(device="cpu")
    audio = torch.randn(1, 16000)  # 1 second at 16kHz
    prosody_indices, timbre_vector = encoder.encode(audio)
    assert prosody_indices.shape[0] == 1
    assert timbre_vector.shape == (1, 256)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
uv run pytest tests/preprocessing/test_facodec_encoder.py::test_facodec_encoder_initialization -v
```

- [ ] **Step 3: Write minimal implementation**

See OLD-full-plan.md lines 319-482 for full implementation.

Key interface:
- `FACodecEncoder(device: str, checkpoint_path: Optional[str])`
- `encode(audio) -> (prosody_indices: Tensor, timbre_vector: Tensor)`
- Prosody: ~80 Hz, vocab size 1024
- Timbre: 256-dim global vector
- Falls back to mock if Amphion not installed

- [ ] **Step 4: Run test — expect PASS**

```bash
uv run pytest tests/preprocessing/test_facodec_encoder.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/facodec_encoder.py tests/preprocessing/test_facodec_encoder.py
git commit -m "feat: add FACodec encoder wrapper"
```

---

### Task 4: MOSS-Audio Encoder Wrapper

**Files:**
- Create: `src/preprocessing/moss_encoder.py`
- Create: `tests/preprocessing/test_moss_encoder.py`

**Steps:**

- [ ] **Step 1: Write the failing test**

```python
# tests/preprocessing/test_moss_encoder.py
import pytest
import torch
from src.preprocessing.moss_encoder import MOSSAudioEncoder

def test_moss_encoder_initialization():
    """Test MOSS-Audio encoder loads correctly."""
    encoder = MOSSAudioEncoder(device="cpu")
    assert encoder.device == "cpu"
    assert encoder.frame_rate == 12.5
    assert encoder.feature_dim == 2560

def test_moss_encode_shape():
    """Test encoding produces expected output shape."""
    encoder = MOSSAudioEncoder(device="cpu")
    audio = torch.randn(1, 16000)
    features = encoder.encode(audio)
    assert features.dim() == 3  # (batch, N_frames, 2560)
    assert features.shape[2] == 2560
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
uv run pytest tests/preprocessing/test_moss_encoder.py::test_moss_encoder_initialization -v
```

- [ ] **Step 3: Write minimal implementation**

See OLD-full-plan.md lines 553-693 for full implementation.

Key interface:
- `MOSSAudioEncoder(device: str, model_size: str = "4B")`
- `encode(audio) -> semantic_frames: Tensor` (batch, N_frames, 2560)
- Frame rate: 12.5 Hz
- Falls back to mock if transformers not available

- [ ] **Step 4: Run test — expect PASS**

```bash
uv run pytest tests/preprocessing/test_moss_encoder.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/moss_encoder.py tests/preprocessing/test_moss_encoder.py
git commit -m "feat: add MOSS-Audio encoder wrapper"
```

---

### Task 5: Temporal Alignment Module

**Files:**
- Create: `src/preprocessing/alignment.py`
- Create: `tests/preprocessing/test_alignment.py`

**Steps:**

- [ ] **Step 1: Write the failing test**

```python
# tests/preprocessing/test_alignment.py
import pytest
import torch
from src.preprocessing.alignment import TemporalAligner, AlignmentInfo

def test_temporal_aligner_pooling():
    """Test pooling prosody to match semantic frames."""
    aligner = TemporalAligner(prosody_rate=80.0, semantic_rate=12.5)
    prosody = torch.arange(80).unsqueeze(0).unsqueeze(-1).float()
    target_frames = 13
    pooled = aligner.pool_prosody_to_semantic(prosody, target_frames)
    assert pooled.shape == (1, 13, 1)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
uv run pytest tests/preprocessing/test_alignment.py::test_temporal_aligner_pooling -v
```

- [ ] **Step 3: Write minimal implementation**

See OLD-full-plan.md lines 780-916 for full implementation.

Key interface:
- `TemporalAligner(prosody_rate=80.0, semantic_rate=12.5)`
- `pool_prosody_to_semantic(prosody_indices, target_frames) -> pooled_indices`
- Uses adaptive average pooling over time dimension
- `AlignmentInfo` dataclass for logging frame counts and ratio

- [ ] **Step 4: Run test — expect PASS**

```bash
uv run pytest tests/preprocessing/test_alignment.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/alignment.py tests/preprocessing/test_alignment.py
git commit -m "feat: add temporal alignment module"
```

---

### Task 6: Batch Processor

**Files:**
- Create: `src/preprocessing/batch_processor.py`
- Create: `tests/preprocessing/test_batch_processor.py`

**Steps:**

- [ ] **Step 1: Write the failing test**

```python
# tests/preprocessing/test_batch_processor.py
import pytest
import torch
import tempfile
from pathlib import Path
from src.preprocessing.batch_processor import BatchProcessor

def test_save_utterance_data():
    """Test saving individual utterance data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        processor = BatchProcessor(output_dir=output_dir)
        data = {
            'utterance_id': 'test_001',
            'semantic_frames': torch.randn(1, 13, 2560),
            'prosody_indices': torch.randint(0, 1024, (1, 80, 1)).float(),
            'timbre_vector': torch.randn(1, 256),
            'label': 1,
        }
        saved_path = processor._save_utterance_data(data)
        assert saved_path.exists()
        loaded = torch.load(saved_path)
        assert loaded['utterance_id'] == 'test_001'
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
uv run pytest tests/preprocessing/test_batch_processor.py::test_save_utterance_data -v
```

- [ ] **Step 3: Write minimal implementation**

See OLD-full-plan.md lines 1011-1281 for full implementation.

Key interface:
- `BatchProcessor(facodec_encoder, moss_encoder, aligner, output_dir, device)`
- `process_utterance(utt_id, audio, sample_rate, label, metadata) -> Optional[Path]`
- `process_dataset(split, max_utterances) -> ProcessingSummary`
- Output `.pt` format: `{utterance_id, semantic_frames, prosody_indices, prosody_indices_raw, timbre_vector, label, duration_sec, alignment_info, metadata}`

- [ ] **Step 4: Run test — expect PASS**

```bash
uv run pytest tests/preprocessing/test_batch_processor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/batch_processor.py tests/preprocessing/test_batch_processor.py
git commit -m "feat: add batch processor for .pt file generation"
```

---

### Task 7: CLI Entry Point

**Files:**
- Create: `scripts/preprocess_mustard.py`
- Create: `tests/preprocessing/test_cli.py`

**Steps:**

- [ ] **Step 1: Write the failing test**

```python
# tests/preprocessing/test_cli.py
import pytest
import subprocess
import sys
from pathlib import Path

def test_cli_help():
    """Test CLI help command works."""
    result = subprocess.run(
        [sys.executable, "scripts/preprocess_mustard.py", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "MUStARD++" in result.stdout or "preprocess" in result.stdout.lower()
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
uv run pytest tests/preprocessing/test_cli.py::test_cli_help -v
```

- [ ] **Step 3: Write minimal implementation**

See OLD-full-plan.md lines 1337-1465 for full implementation.

Key CLI:
- `python scripts/preprocess_mustard.py --split {train,validation,test,all}`
- Options: `--output-dir`, `--max-utterances`, `--device`, `--cache-dir`

- [ ] **Step 4: Run test — expect PASS**

```bash
uv run pytest tests/preprocessing/test_cli.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/preprocess_mustard.py tests/preprocessing/test_cli.py
git commit -m "feat: add CLI entry point for MUStARD++ preprocessing"
```

---

### Task 8: Integration Test

**Files:**
- Create: `tests/preprocessing/test_integration.py`

**Steps:**

- [ ] **Step 1: Write integration test**

See OLD-full-plan.md lines 1492-1608 for full test.

Key assertions:
- Mock audio (2 sec) processes successfully
- Output file exists and loads
- Required fields present: `utterance_id`, `semantic_frames`, `prosody_indices`, `timbre_vector`, `label`, `alignment_info`
- Tensor shapes correct

- [ ] **Step 2: Run test — expect PASS**

```bash
uv run pytest tests/preprocessing/test_integration.py -v --tb=short
```

- [ ] **Step 3: Commit**

```bash
git add tests/preprocessing/test_integration.py
git commit -m "test: add integration tests"
```

---

### Task 9: Smoke Test

**Files:**
- Create: `tests/preprocessing/smoke_test.py`

**Steps:**

- [ ] **Step 1: Create smoke test**

See OLD-full-plan.md lines 1785-1865 for implementation.

Purpose: Quick validation without HF downloads.

- [ ] **Step 2: Run smoke test**

```bash
uv run python tests/preprocessing/smoke_test.py
```

Expected: `SMOKE TEST PASSED ✓`

- [ ] **Step 3: Commit**

```bash
git add tests/preprocessing/smoke_test.py
git commit -m "test: add smoke test"
```

---

### Task 10: Run All Tests

**Steps:**

- [ ] **Run complete test suite**

```bash
uv run pytest tests/preprocessing/ -v --tb=short
```

- [ ] **Verify all tests pass**

---

## Phase Completion Criteria

- [ ] All 10 tasks complete
- [ ] Unit tests pass for each module
- [ ] Integration test passes
- [ ] Smoke test passes
- [ ] Can run: `python scripts/preprocess_mustard.py --split test --max-utterances 5`

## Handoff Notes for Phase 2

After Phase 1:
1. Preprocessed `.pt` files exist in `data/mustard_pp_processed/`
2. Each file contains: `semantic_frames` (N_sem, 2560), `prosody_indices` (N_sem, 1), `timbre_vector` (256,), `label` (int)
3. No encoder models needed in training — data is preprocessed
4. Frame rate alignment: 80 Hz → 12.5 Hz via pooling (ratio ~6.4)

Phase 2 will build:
- Embedding tables for prosody indices → 2560-dim
- Timbre projector: 256 → 2560
- λ gate (zero-init, learnable)
- Residual fusion with LayerNorm

---

**Full implementation details available in:** `OLD-full-plan.md` (lines 117-1865)
