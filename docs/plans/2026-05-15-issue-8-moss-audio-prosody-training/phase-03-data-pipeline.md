# Phase 3: Data Pipeline

> **Status:** In progress — implementing

## Phase Goal
MUStARD audio is FACodec-preprocessed to parquet, and a PyTorch `Dataset`/`DataLoader` yields correct `(audio, prosody_indices, timbre_vector, label)` batches for training `AmyForProsodyClassification`.

## Reality Check
Both `data/mustard_dataset/` and `data/mustard_audio/` are empty dirs. `checkpoints/facodec/` doesn't exist. Data pipeline requires running `prepare_mustard_dataset.py` + FACodec checkpoint download + `preprocess.py` before any DataLoader can function.

## Pre-Task: Data Prerequisites (run once before any test/training)

### Step 1: Download MUStARD audio
```bash
uv run python scripts/prepare_mustard_dataset.py
```
Output: `data/mustard_dataset/` (Arrow format via `save_to_disk`), `data/mustard_audio/` (raw WAVs), `data/mustard_raw/` (videos).

### Step 2: Download FACodec checkpoint
```bash
mkdir -p checkpoints/facodec
# Download from https://huggingface.co/amphion/naturalspeech3_facodec
# Files needed: ns3_facodec_decoder.bin, config.json
```
Export env var: `export AMPHION_CHECKPOINT_DIR=$PWD/checkpoints/facodec`

### Step 3: FACodec-encode MUStARD
```bash
uv run python scripts/preprocess.py \
    --dataset data/mustard_dataset \
    --local-dir \
    --output-repo mustard-processed \
    --no-push \
    --device cpu
```
Output: `data/processed/mustard-processed/data.parquet`

Expected columns: `dataset`, `id`, `audio`, `prosody_codebooks_idx`, `content_codebooks_idx`, `acoustic_codebooks_idx`, `timbre_vector`, `label`

---

## Files to Touch

| Action | File | Purpose |
|--------|------|---------|
| **Modify** | `src/preprocessing/dataset_processor.py` | Add label column, support local Dataset loading |
| **Modify** | `scripts/preprocess.py` | Add `--local-dir` flag for local datasets |
| **Modify** | `tests/preprocessing/test_dataset_processor.py` | Add label column test |
| **Create** | `src/data/mustard_dataset.py` | PyTorch `MustardDataset` + `collate_fn` |
| **Create** | `tests/data/test_mustard_dataset.py` | Dataset + DataLoader tests |

---

## Task 1: Label Column in Preprocessing Pipeline

**Files:**
- Modify: `src/preprocessing/dataset_processor.py`
- Modify: `tests/preprocessing/test_dataset_processor.py`

**Rationale:** The parquet needs a `label` column (0/1 for sarcasm) for the PyTorch Dataset to use as the classification target. Currently `_build_processed_entry` discards the source `sarcasm` column. Also need to support local disk-loaded datasets (not just HF Hub names).

### Step 1: Add label extraction to `_build_processed_entry`

At `src/preprocessing/dataset_processor.py:282`, add label field:

```python
# Extract label from source sample (sarcasm for MUStARD, label for generic datasets)
label = sample.get("sarcasm", sample.get("label", -1))
```

And include `"label": label` in the returned dict.

### Step 2: Update Features schema

At `src/preprocessing/dataset_processor.py:179`, add:

```python
"label": Value("int64"),
```

### Step 3: Support local Dataset in `process_dataset()`

Modify `process_dataset` to accept `str | Dataset` for `dataset_name`. When a `Dataset` is passed directly, use it as source without calling `load_dataset`.

### Step 4: Add `--local-dir` flag to `preprocess.py`

When set, load dataset from local Arrow directory with `Dataset.load_from_disk()` instead of `load_dataset()`.

### Step 5: Update tests

Add `test_process_dataset_includes_label_column` to verify label is present and correct. Add `test_process_dataset_with_local_dataset` to test Dataset-direct loading.

---

## Task 2: `MustardDataset` — PyTorch Dataset

**Files:**
- Create: `src/data/mustard_dataset.py`

**Contract:** Each `__getitem__` returns `(audio, prosody_indices, timbre_vector, label)` matching `AmyForProsodyClassification.forward()` input shapes.

```
Each item:
  audio:           torch.Tensor [T_audio] float32  — raw 16kHz waveform
  prosody_indices: torch.Tensor [1, T80]   int64    — sliced from flat list
  timbre_vector:   torch.Tensor [256]      float32  — utterance-level
  label:           int                              — 0 or 1
```

**Implementation:**

```python
class MustardDataset(Dataset):
    def __init__(self, parquet_path: str | Path):
        dataset = Dataset.from_parquet(parquet_path)
        self.samples = [(s["audio"], s["prosody_codebooks_idx"],
                         s["timbre_vector"], s["label"]) for s in dataset]
    
    def __len__(self): ...
    
    def __getitem__(self, idx):
        audio_dict, prosody_list, timbre_list, label = self.samples[idx]
        audio = torch.tensor(audio_dict["array"], dtype=torch.float32)
        prosody = torch.tensor(prosody_list, dtype=torch.long).unsqueeze(0)  # [1, T80]
        timbre = torch.tensor(timbre_list, dtype=torch.float32)
        return audio, prosody, timbre, int(label)
```

**Design decisions:**
- Audio loaded as raw float32 tensor from HF Audio feature's `array`
- Prosody reshaped from flat `[T80]` to `[1, T80]` per model contract
- Label is `int` (0 or 1) for `CrossEntropyLoss`

---

## Task 3: `collate_fn` — Batching with Padding

**File:** `src/data/mustard_dataset.py` (same file)

Pads variable-length audio and prosody tensors to batch max.

```python
def collate_mustard(batch: list[tuple]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    audios, prosodies, timbres, labels = zip(*batch)
    
    # Audio: pad to max length (right-pad with zeros)
    audio_max = max(a.shape[0] for a in audios)
    audio_padded = torch.stack([F.pad(a, (0, audio_max - a.shape[0])) for a in audios])
    
    # Prosody: pad to max T80 (right-pad with 0, valid codebook index)
    pro_max = max(p.shape[1] for p in prosodies)
    pro_padded = torch.stack([F.pad(p, (0, pro_max - p.shape[1])) for p in prosodies])
    
    # Timbre: stack directly ([B, 256])
    timbre_stacked = torch.stack(timbres)
    
    # Label: stack to [B]
    label_stacked = torch.tensor(labels, dtype=torch.long)
    
    return audio_padded, pro_padded, timbre_stacked, label_stacked
```

Tensor shapes out of collate:
- `audio`: `[B, max_T_audio]` float32
- `prosody_indices`: `[B, 1, max_T80]` int64
- `timbre_vector`: `[B, 256]` float32
- `label`: `[B]` int64

---

## Task 4: Split Logic

**File:** `src/data/mustard_dataset.py` (same file)

Random 80/10/10 train/val/test split using `torch.utils.data.random_split`.

```python
def create_mustard_splits(parquet_path, seed=42, train_frac=0.8, val_frac=0.1):
    dataset = MustardDataset(parquet_path)
    n = len(dataset)
    train_n = int(n * train_frac)
    val_n = int(n * val_frac)
    test_n = n - train_n - val_n
    
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_n, val_n, test_n], generator=generator)
```

---

## Task 5: Tests

**Files:**
- Create: `tests/data/test_mustard_dataset.py`

### Test matrix:

| Test | What it verifies |
|------|-----------------|
| `test_dataset_loads_from_parquet` | `MustardDataset` loads a parquet file, `len()` matches row count |
| `test_getitem_returns_correct_shapes` | Each item has `audio [T]`, `prosody [1, T80]`, `timbre [256]`, `label` int |
| `test_getitem_dtypes` | audio=float32, prosody=int64, timbre=float32, label=int |
| `test_label_distribution` | Labels are 0/1, both classes present |
| `test_collate_stacks_batch` | collate_fn produces `[B, max_T]` audio, `[B, 1, max_T80]` prosody, etc. |
| `test_collate_pads_variable_length` | Samples of different lengths produce padded batches correctly |
| `test_dataloader_iterates` | DataLoader yields batches of correct structure |
| `test_random_split_produces_non_overlapping_sets` | 80/10/10 split has correct sizes and disjoint indices |

Tests use synthetic parquet files (create in `tmp_path` fixture), no network access required.

---

## Phase Completion Criteria

- [ ] `DatasetProcessor` includes `label` column in processed parquet (from source `sarcasm`)
- [ ] `preprocess.py` supports `--local-dir` for local Arrow-format datasets
- [ ] MUStARD audio downloaded and FACodec-encoded (at least via real or mock encoder)
- [ ] `MustardDataset` loads parquet, returns correct shapes per `__getitem__`
- [ ] `collate_fn` pads variable-length audio + prosody, stacks timbre + labels
- [ ] Train/val/test split logic produces non-overlapping sets
- [ ] All tests pass: `uv run python -m pytest tests/data/ tests/preprocessing/ -v`
- [ ] Existing tests continue passing: `uv run python -m pytest tests/ -v`
