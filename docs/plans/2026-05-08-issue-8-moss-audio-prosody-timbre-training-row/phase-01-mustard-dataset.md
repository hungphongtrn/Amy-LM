# Phase 1: MUStARD Dataset Foundation

## Phase Goal

Create a tested preprocessing path that converts MUStARD source data into a Hugging Face Dataset/parquet with raw 16 kHz audio, `id`, `prosody_codebooks_idx` shaped as one codebook per sample, `timbre_vector` length 256, and binary `label` values.

## Files to Touch

- `scripts/prepare_mustard_dataset.py` - New CLI for cloning/reading MUStARD and writing processed parquet/HF Dataset output.
- `src/preprocessing/mustard.py` - New source-layout parser and row builder for MUStARD.
- `src/preprocessing/dataset_processor.py` - Minimal reusable changes only if label preservation or `[1, T80]` serialization belongs in shared FACodec processing.
- `src/preprocessing/__init__.py` - Export MUStARD helpers if tests or scripts import them from the package.
- `tests/preprocessing/test_mustard.py` - New unit tests for MUStARD metadata parsing, audio resolution, label mapping, and processed schema.
- `tests/preprocessing/test_dataset_processor.py` - Add or adjust tests if shared processor behavior changes.
- `pyproject.toml` - Add only missing dependencies required by the implemented preprocessing path.

## Tasks

### Task 1: Add A MUStARD Source Parser

**Files:**

- Create: `src/preprocessing/mustard.py`
- Test: `tests/preprocessing/test_mustard.py`

- [ ] **Step 1: Write the failing metadata parsing test**

```python
from pathlib import Path

from src.preprocessing.mustard import load_mustard_examples


def test_load_mustard_examples_reads_binary_labels(tmp_path: Path):
    root = tmp_path / "MUStARD"
    root.mkdir()
    (root / "data").mkdir()
    wav_dir = root / "clips"
    wav_dir.mkdir()
    (wav_dir / "utt_001.wav").write_bytes(b"not-a-real-wav")
    (root / "data" / "sarcasm_data.json").write_text(
        '{"utt_001": {"sarcasm": true, "utterance": "Great.", "context": []}}',
        encoding="utf-8",
    )

    examples = load_mustard_examples(root)

    assert len(examples) == 1
    assert examples[0].id == "utt_001"
    assert examples[0].label == 1
    assert examples[0].audio_path == wav_dir / "utt_001.wav"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/preprocessing/test_mustard.py::test_load_mustard_examples_reads_binary_labels -v`

Expected: FAIL because `src.preprocessing.mustard` does not exist.

- [ ] **Step 3: Write the minimal parser implementation**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MustardExample:
    id: str
    audio_path: Path
    label: int


def _find_metadata_file(root: Path) -> Path:
    candidates = [
        root / "data" / "sarcasm_data.json",
        root / "sarcasm_data.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find MUStARD sarcasm_data.json under {root}")


def _find_audio_path(root: Path, sample_id: str) -> Path:
    for pattern in (f"**/{sample_id}.wav", f"**/{sample_id}.mp4"):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find audio/video file for MUStARD sample {sample_id}")


def load_mustard_examples(root: Path) -> list[MustardExample]:
    root = Path(root)
    metadata_path = _find_metadata_file(root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    examples: list[MustardExample] = []
    for sample_id, row in sorted(metadata.items()):
        if "sarcasm" not in row:
            raise ValueError(f"MUStARD sample {sample_id} missing sarcasm label")
        examples.append(
            MustardExample(
                id=sample_id,
                audio_path=_find_audio_path(root, sample_id),
                label=int(bool(row["sarcasm"])),
            )
        )
    return examples
```

- [ ] **Step 4: Run the parser test to verify it passes**

Run: `uv run pytest tests/preprocessing/test_mustard.py::test_load_mustard_examples_reads_binary_labels -v`

Expected: PASS.

- [ ] **Step 5: Add tests for missing metadata, missing labels, and missing audio**

```python
import pytest


def test_load_mustard_examples_requires_metadata(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="sarcasm_data.json"):
        load_mustard_examples(tmp_path)


def test_load_mustard_examples_requires_sarcasm_label(tmp_path: Path):
    root = tmp_path / "MUStARD"
    (root / "data").mkdir(parents=True)
    (root / "clips").mkdir()
    (root / "clips" / "utt_001.wav").write_bytes(b"not-a-real-wav")
    (root / "data" / "sarcasm_data.json").write_text(
        '{"utt_001": {"utterance": "Great."}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing sarcasm label"):
        load_mustard_examples(root)


def test_load_mustard_examples_requires_audio(tmp_path: Path):
    root = tmp_path / "MUStARD"
    (root / "data").mkdir(parents=True)
    (root / "data" / "sarcasm_data.json").write_text(
        '{"utt_001": {"sarcasm": false}}',
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="utt_001"):
        load_mustard_examples(root)
```

- [ ] **Step 6: Run the full MUStARD parser test file**

Run: `uv run pytest tests/preprocessing/test_mustard.py -v`

Expected: PASS.

- [ ] **Step 7: Commit parser work**

```bash
git add src/preprocessing/mustard.py tests/preprocessing/test_mustard.py
git commit -m "feat: parse MUStARD source examples"
```

### Task 2: Build Processed Dataset Rows With FACodec Fields And Labels

**Files:**

- Modify: `src/preprocessing/mustard.py`
- Test: `tests/preprocessing/test_mustard.py`

- [ ] **Step 1: Write the failing processed-row schema test**

```python
import numpy as np
import torch

from src.preprocessing.facodec_encoder import FACodecStreams
from src.preprocessing.mustard import MustardExample, build_processed_mustard_row


def test_build_processed_mustard_row_preserves_issue_8_schema(tmp_path: Path):
    wav_path = tmp_path / "utt_001.wav"
    wav_path.write_bytes(b"placeholder")
    example = MustardExample(id="utt_001", audio_path=wav_path, label=1)
    audio = np.zeros(16000, dtype=np.float32)
    streams = FACodecStreams(
        prosody_codebooks_idx=torch.arange(80, dtype=torch.long).unsqueeze(0),
        content_codebooks_idx=torch.zeros(2, 80, dtype=torch.long),
        acoustic_codebooks_idx=torch.zeros(3, 80, dtype=torch.long),
        timbre_vector=torch.ones(256, dtype=torch.float32),
    )

    row = build_processed_mustard_row(example, audio, 16000, streams)

    assert row["id"] == "utt_001"
    assert row["label"] == 1
    assert row["audio"]["sampling_rate"] == 16000
    assert row["audio"]["array"].dtype == np.float32
    assert row["prosody_codebooks_idx"] == [list(range(80))]
    assert len(row["timbre_vector"]) == 256
    assert "content_codebooks_idx" in row
    assert "acoustic_codebooks_idx" in row
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/preprocessing/test_mustard.py::test_build_processed_mustard_row_preserves_issue_8_schema -v`

Expected: FAIL because `build_processed_mustard_row` does not exist.

- [ ] **Step 3: Implement processed-row construction**

```python
from typing import Any

import numpy as np

from src.preprocessing.facodec_encoder import FACodecStreams


def build_processed_mustard_row(
    example: MustardExample,
    audio: np.ndarray,
    sampling_rate: int,
    streams: FACodecStreams,
) -> dict[str, Any]:
    audio = np.asarray(audio, dtype=np.float32)
    if sampling_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio after loading, got {sampling_rate}")
    return {
        "id": example.id,
        "audio": {"array": audio, "sampling_rate": sampling_rate},
        "prosody_codebooks_idx": streams.prosody_codebooks_idx.long().tolist(),
        "content_codebooks_idx": streams.content_codebooks_idx.long().tolist(),
        "acoustic_codebooks_idx": streams.acoustic_codebooks_idx.long().tolist(),
        "timbre_vector": streams.timbre_vector.float().tolist(),
        "label": int(example.label),
    }
```

- [ ] **Step 4: Run the processed-row test**

Run: `uv run pytest tests/preprocessing/test_mustard.py::test_build_processed_mustard_row_preserves_issue_8_schema -v`

Expected: PASS.

- [ ] **Step 5: Commit processed-row work**

```bash
git add src/preprocessing/mustard.py tests/preprocessing/test_mustard.py
git commit -m "feat: build MUStARD FACodec rows"
```

### Task 3: Add The MUStARD Preparation CLI

**Files:**

- Create: `scripts/prepare_mustard_dataset.py`
- Modify: `tests/preprocessing/test_mustard.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
from scripts.prepare_mustard_dataset import create_dataset_features


def test_create_dataset_features_matches_issue_8_contract():
    features = create_dataset_features()

    assert set(features.keys()) == {
        "id",
        "audio",
        "prosody_codebooks_idx",
        "content_codebooks_idx",
        "acoustic_codebooks_idx",
        "timbre_vector",
        "label",
    }
    assert features["audio"].sampling_rate == 16000
```

- [ ] **Step 2: Run the CLI smoke test to verify it fails**

Run: `uv run pytest tests/preprocessing/test_mustard.py::test_create_dataset_features_matches_issue_8_contract -v`

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement the CLI skeleton and feature contract**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Audio, Dataset, Features, Sequence, Value


def create_dataset_features() -> Features:
    return Features(
        {
            "id": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "prosody_codebooks_idx": Sequence(Sequence(Value("int64"))),
            "content_codebooks_idx": Sequence(Sequence(Value("int64"))),
            "acoustic_codebooks_idx": Sequence(Sequence(Value("int64"))),
            "timbre_vector": Sequence(Value("float32")),
            "label": Value("int64"),
        }
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MUStARD with FACodec streams for Amy LM issue #8")
    parser.add_argument("--mustard-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--facodec-checkpoint-path", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    raise SystemExit("Implementation continues in the next step")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the feature contract test**

Run: `uv run pytest tests/preprocessing/test_mustard.py::test_create_dataset_features_matches_issue_8_contract -v`

Expected: PASS.

- [ ] **Step 5: Add CLI processing test with mocked audio loader and FACodec**

Use a test that invokes a pure function rather than spawning a subprocess:

```python
from unittest.mock import Mock

import numpy as np

from scripts.prepare_mustard_dataset import prepare_mustard_dataset


def test_prepare_mustard_dataset_writes_expected_rows(tmp_path: Path):
    root = tmp_path / "MUStARD"
    (root / "data").mkdir(parents=True)
    clips = root / "clips"
    clips.mkdir()
    (clips / "utt_001.wav").write_bytes(b"placeholder")
    (root / "data" / "sarcasm_data.json").write_text(
        '{"utt_001": {"sarcasm": true}}',
        encoding="utf-8",
    )
    output_path = tmp_path / "mustard_facodec.parquet"
    facodec = Mock()
    facodec.encode.return_value = FACodecStreams(
        prosody_codebooks_idx=torch.zeros(1, 80, dtype=torch.long),
        content_codebooks_idx=torch.zeros(2, 80, dtype=torch.long),
        acoustic_codebooks_idx=torch.zeros(3, 80, dtype=torch.long),
        timbre_vector=torch.zeros(256, dtype=torch.float32),
    )

    dataset = prepare_mustard_dataset(
        mustard_root=root,
        output_path=output_path,
        facodec=facodec,
        audio_loader=lambda path: (np.zeros(16000, dtype=np.float32), 16000),
    )

    assert output_path.exists()
    assert len(dataset) == 1
    assert dataset[0]["id"] == "utt_001"
    assert dataset[0]["label"] == 1
```

- [ ] **Step 6: Implement `prepare_mustard_dataset`**

```python
import numpy as np
import soundfile as sf

from src.preprocessing.facodec_encoder import FACodecEncoder
from src.preprocessing.mustard import build_processed_mustard_row, load_mustard_examples


def _load_audio_16khz(path: Path) -> tuple[np.ndarray, int]:
    audio, sampling_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sampling_rate != 16000:
        raise ValueError(f"Expected MUStARD audio at 16 kHz for {path}, got {sampling_rate}")
    return audio.astype(np.float32), sampling_rate


def prepare_mustard_dataset(
    mustard_root: Path,
    output_path: Path,
    facodec: FACodecEncoder,
    audio_loader=_load_audio_16khz,
    max_samples: int | None = None,
) -> Dataset:
    examples = load_mustard_examples(mustard_root)
    if max_samples is not None:
        examples = examples[:max_samples]
    rows = []
    for example in examples:
        audio, sampling_rate = audio_loader(example.audio_path)
        streams = facodec.encode(torch.from_numpy(audio).float())
        rows.append(build_processed_mustard_row(example, audio, sampling_rate, streams))
    dataset = Dataset.from_list(rows, features=create_dataset_features())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))
    return dataset
```

- [ ] **Step 7: Wire `main()` to instantiate FACodec and run the preparation**

```python
def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    facodec = FACodecEncoder(
        device=args.device,
        checkpoint_path=str(args.facodec_checkpoint_path) if args.facodec_checkpoint_path else None,
    )
    if facodec._mock:
        raise SystemExit("FACodec real checkpoint unavailable; refusing to prepare issue #8 dataset with mock streams")
    dataset = prepare_mustard_dataset(
        mustard_root=args.mustard_root,
        output_path=args.output_path,
        facodec=facodec,
        max_samples=args.max_samples,
    )
    print(f"Wrote {len(dataset)} MUStARD rows to {args.output_path}")
```

- [ ] **Step 8: Run the MUStARD tests**

Run: `uv run pytest tests/preprocessing/test_mustard.py -v`

Expected: PASS.

- [ ] **Step 9: Commit CLI work**

```bash
git add scripts/prepare_mustard_dataset.py tests/preprocessing/test_mustard.py
git commit -m "feat: prepare MUStARD FACodec dataset"
```

### Task 4: Validate Shared FACodec Contract Compatibility

**Files:**

- Modify: `src/preprocessing/dataset_processor.py` only if tests prove shared behavior is incompatible.
- Modify: `tests/preprocessing/test_dataset_processor.py` only if shared behavior changes.
- Modify: `tests/preprocessing/test_mustard.py`

- [ ] **Step 1: Add a schema assertion that Prosody Stream keeps its codebook axis for issue #8**

```python
def test_processed_mustard_row_keeps_prosody_codebook_axis(tmp_path: Path):
    example = MustardExample(id="utt_001", audio_path=tmp_path / "utt_001.wav", label=0)
    streams = FACodecStreams(
        prosody_codebooks_idx=torch.tensor([[1, 2, 3]], dtype=torch.long),
        content_codebooks_idx=torch.zeros(2, 3, dtype=torch.long),
        acoustic_codebooks_idx=torch.zeros(3, 3, dtype=torch.long),
        timbre_vector=torch.zeros(256, dtype=torch.float32),
    )
    row = build_processed_mustard_row(example, np.zeros(800, dtype=np.float32), 16000, streams)

    assert row["prosody_codebooks_idx"] == [[1, 2, 3]]
```

- [ ] **Step 2: Run preprocessing tests**

Run: `uv run pytest tests/preprocessing -v`

Expected: PASS. If existing generic processor tests still require flattened prosody, leave generic behavior unchanged and keep the issue #8 shape in the MUStARD-specific path.

- [ ] **Step 3: Run model primitive tests to catch schema regressions**

Run: `uv run pytest tests/models/test_embedding.py tests/models/test_pooling.py tests/models/test_fusion.py -v`

Expected: PASS.

- [ ] **Step 4: Commit compatibility fixes if any files changed**

```bash
git add src/preprocessing/dataset_processor.py tests/preprocessing/test_dataset_processor.py tests/preprocessing/test_mustard.py
git commit -m "test: validate issue 8 FACodec schema"
```

If only tests ran and no files changed, skip this commit.

### Task 5: Produce A Small Local Smoke Dataset

**Files:**

- No required source changes.
- Optional output: `data/processed/mustard_issue8_smoke.parquet` must stay uncommitted unless explicitly requested.

- [ ] **Step 1: Run the CLI with a tiny sample cap and real FACodec**

Run:

```bash
uv run python scripts/prepare_mustard_dataset.py \
  --mustard-root data/raw/MUStARD \
  --output-path data/processed/mustard_issue8_smoke.parquet \
  --device cuda \
  --max-samples 2
```

Expected: either the parquet is written with 2 rows, or the command fails clearly because MUStARD/FACodec checkpoints are not available locally.

- [ ] **Step 2: If the command fails for missing local data/checkpoints, record the exact blocker in the handoff notes**

Example blocker text:

```markdown
Local smoke dataset not produced because `data/raw/MUStARD` is absent. The CLI and tests are ready; rerun the Phase 1 smoke command after cloning https://github.com/soujanyaporia/MUStARD into `data/raw/MUStARD` and installing FACodec checkpoints.
```

- [ ] **Step 3: Verify git status before phase completion**

Run: `git status --short`

Expected: source/test changes are committed; generated parquet files are untracked or ignored and not committed.

## Phase Completion Criteria

- [ ] `tests/preprocessing/test_mustard.py` passes.
- [ ] Existing FACodec preprocessing tests pass or have deliberate, reviewed updates.
- [ ] Existing model primitive tests for embedding, pooling, and fusion pass.
- [ ] The MUStARD CLI refuses to use mock FACodec for issue #8 dataset generation.
- [ ] The output dataset schema includes `audio`, `id`, `prosody_codebooks_idx`, `timbre_vector`, and `label`; optional Content/Acoustic streams may be present but are not used in the Phase 2 stream activation config.
- [ ] Local smoke generation either succeeds or records an explicit environmental blocker.

## Handoff Notes

Phase 2 should not assume MOSS-Audio frame count from FACodec duration. It must align embedded Prosody Stream to the actual `S_t.shape[1]` returned by the MOSS-Audio semantic path.

If Phase 1 keeps generic `DatasetProcessor` prosody flattened for backward compatibility, the Phase 2 data collator must support the issue #8 MUStARD-specific `[1, T80]` field and not rely on the generic processor's old flattened convention.
