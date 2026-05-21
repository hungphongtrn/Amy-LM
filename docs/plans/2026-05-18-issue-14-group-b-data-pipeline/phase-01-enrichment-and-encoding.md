# Phase 1: Enrichment + Processor

## Phase Goal

NVTTS enriched dataset saved to `data/nvtts_enriched/` with all provenance columns. `PreferenceDatasetProcessor` class built, tested, ready to process Script 3's intermediate parquet.

**Issues:** #20 (B1), #21 (B4)

## Files to Touch

- `src/data/nv_enrich_mapping.py` — NV emoji-to-text mapping (#17, read by Script 1)
- `data/speaker_lookup.json` — Speaker context cache (#18, read by Script 1)
- `scripts/enrich_nvtts.py` — Script 1: load NVTTS, apply mappings, save enriched dataset
- `src/preprocessing/preference_dataset_processor.py` — PreferenceDatasetProcessor class
- `tests/preprocessing/test_preference_dataset_processor.py` — Processor tests
- `tests/data/test_enrich_nvtts.py` — Script 1 tests

---

## Tasks

### Task 1: Script 1 — NVTTS Enrichment

**Files:**
- Create: `scripts/enrich_nvtts.py`
- Create: `tests/data/test_enrich_nvtts.py`

- [ ] **Step 1: Write integration test**

```python
# tests/data/test_enrich_nvtts.py
import os
from datasets import Dataset
from scripts.enrich_nvtts import load_nvtts, enrich_sample, apply_nv_mapping, resolve_speaker

def test_apply_nv_mapping_converts_emojis():
    """NV emoji tags in Result column become [Tag] text labels."""
    from src.data.nv_enrich_mapping import emojis_to_tags
    result = emojis_to_tags("I'm 🌬️ fine 🤣 thanks")
    assert "[Breathing]" in result
    assert "[Laughter]" in result

def test_resolve_speaker_expresso():
    """Expresso speaker IDs resolve correctly."""
    from scripts.enrich_nvtts import resolve_speaker
    ctx = resolve_speaker("ex01")
    assert ctx["name"] == "Jack"
    assert ctx["gender"] == "male"

def test_resolve_speaker_voxceleb():
    """VoxCeleb speaker IDs resolve via lookup cache."""
    ctx = resolve_speaker("id03621")
    assert ctx["name"] is not None  # resolved or "unknown"

def test_enrich_sample_produces_all_columns():
    """Enriched sample has all 11 output columns."""
    from scripts.enrich_nvtts import enrich_sample
    raw = {
        "audio": {"path": "dummy.wav", "array": b"", "sampling_rate": 16000},
        "Emotion": "happy",
        "Initial text": "hello world",
        "Result": "hello 🌬️ world",
        "speaker_id": "ex01",
        "data_name": "Expresso",
        "gender": "m",
    }
    enriched = enrich_sample(raw)
    expected_cols = {"id", "audio", "emotion_label", "speaker_name", "speaker_gender",
                     "speaker_age_context", "speaker_nationality", "transcript_with_tags",
                     "bare_transcript", "source"}
    assert expected_cols.issubset(set(enriched.keys()))
    assert enriched["transcript_with_tags"] == "hello [Breathing] world"
    assert enriched["bare_transcript"] == "hello world"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/data/test_enrich_nvtts.py -v
```
Expected: FAIL — `enrich_nvtts.py` not found or functions not defined.

- [ ] **Step 3: Write the script**

```python
# scripts/enrich_nvtts.py
"""Script 1: Enrich NVTTS with speaker context and NV tag mapping.

Loads NVTTS (deepvk/NonverbalTTS), applies:
  - NV emoji→text mapping from #17 (src/data/nv_enrich_mapping.py)
  - Speaker context lookup from #18 (data/speaker_lookup.json)

Output: HF Dataset saved to data/nvtts_enriched/
"""
import json
import os
import sys
from typing import Any, Dict

from datasets import Dataset, load_dataset


NVTTS_DATASET = "deepvk/NonverbalTTS"
SPLITS = ["train", "dev", "test"]
OUTPUT_DIR = "data/nvtts_enriched"
LOOKUP_PATH = "data/speaker_lookup.json"

EXPRESSO_SPEAKERS = {
    "ex01": {"name": "Jack", "gender": "male", "age_context": "unknown", "nationality": "North American"},
    "ex02": {"name": "Lisa", "gender": "female", "age_context": "unknown", "nationality": "North American"},
    "ex03": {"name": "Bert", "gender": "male", "age_context": "unknown", "nationality": "North American"},
    "ex04": {"name": "Emma", "gender": "female", "age_context": "unknown", "nationality": "North American"},
}


def load_speaker_lookup(lookup_path: str = LOOKUP_PATH) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(lookup_path):
        raise FileNotFoundError(
            f"Speaker lookup not found at {lookup_path}. "
            f"Run issue #18 first to build the cache."
        )
    with open(lookup_path, "r") as f:
        return json.load(f)


def resolve_speaker(
    speaker_id: str, lookup: Dict[str, Dict[str, str]]
) -> Dict[str, str]:
    default = {"name": "unknown", "gender": "unknown", "age_context": "unknown", "nationality": "unknown"}
    # Expresso hardcoded fallback
    if speaker_id.lower() in EXPRESSO_SPEAKERS:
        return EXPRESSO_SPEAKERS[speaker_id.lower()]
    # Cascading from lookup cache
    return lookup.get(speaker_id, default)


def enrich_sample(
    sample: Dict[str, Any], lookup: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    from src.data.nv_enrich_mapping import emojis_to_tags

    speaker = resolve_speaker(sample["speaker_id"], lookup)
    return {
        "id": str(sample.get("index", "")),
        "audio": sample["audio"],
        "emotion_label": sample.get("Emotion", "unknown"),
        "speaker_name": speaker["name"],
        "speaker_gender": speaker["gender"],
        "speaker_age_context": speaker["age_context"],
        "speaker_nationality": speaker["nationality"],
        "transcript_with_tags": emojis_to_tags(sample.get("Result", "")),
        "bare_transcript": sample.get("Initial text", ""),
        "source": sample.get("data_name", "unknown"),
    }


def load_nvtts() -> Dataset:
    datasets = []
    for split in SPLITS:
        ds = load_dataset(NVTTS_DATASET, split=split, trust_remote_code=True)
        datasets.append(ds)
    return concatenate_datasets(datasets)


def main():
    print(f"Loading NVTTS ({NVTTS_DATASET})...")
    ds = load_nvtts()
    print(f"Loaded {len(ds)} samples across splits: {SPLITS}")

    print(f"Loading speaker lookup from {LOOKUP_PATH}...")
    lookup = load_speaker_lookup()

    print("Enriching samples...")
    enriched = []
    failed = 0
    for i, sample in enumerate(ds):
        try:
            enriched.append(enrich_sample(sample, lookup))
        except Exception as e:
            print(f"  WARNING: Failed sample {i}: {e}")
            failed += 1
    print(f"Enriched {len(enriched)} samples ({failed} failed)")

    result = Dataset.from_list(enriched)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result.to_parquet(os.path.join(OUTPUT_DIR, "nvtts_enriched.parquet"))
    print(f"Saved to {OUTPUT_DIR}/nvtts_enriched.parquet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python -m pytest tests/data/test_enrich_nvtts.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/enrich_nvtts.py tests/data/test_enrich_nvtts.py
git commit -m "feat: add NVTTS enrichment script (Issue #20)"
```

---

### Task 2: PreferenceDatasetProcessor

**Files:**
- Create: `src/preprocessing/preference_dataset_processor.py`
- Create: `tests/preprocessing/test_preference_dataset_processor.py`

- [ ] **Step 1: Write integration test**

```python
# tests/preprocessing/test_preference_dataset_processor.py
import numpy as np
import pytest
from datasets import Dataset

from src.preprocessing.facodec_encoder import FACodecEncoder
from src.preprocessing.preference_dataset_processor import PreferenceDatasetProcessor


@pytest.fixture
def mock_preference_dataset():
    """Synthetic preference pair dataset matching Script 3 output schema."""
    rng = np.random.RandomState(42)
    samples = []
    for i in range(3):
        audio_arr = rng.randn(16000).astype(np.float32)  # 1s @ 16kHz
        samples.append({
            "id": f"sample_{i}",
            "audio": {
                "path": None,
                "array": audio_arr.tobytes(),
                "sampling_rate": 16000,
            },
            "chosen": f"chosen response {i}",
            "rejected": f"rejected response {i}",
            "rationale_chosen": f"rationale chosen {i}",
            "rationale_rejected": f"rationale rejected {i}",
            "emotion_label": "happy",
            "speaker_name": "Test Speaker",
            "speaker_gender": "female",
            "speaker_age_context": "30, born 1996",
            "speaker_nationality": "American",
            "transcript_with_tags": "hello [Breathing] world",
            "bare_transcript": "hello world",
            "cosine_similarity": float(i) / 10.0,
        })
    return Dataset.from_list(samples)


def test_processor_output_columns(mock_preference_dataset):
    """Output has all expected columns, no content/acoustic streams."""
    encoder = FACodecEncoder(force_mock=True, device="cpu")
    processor = PreferenceDatasetProcessor(encoder)
    result = processor.process_dataset(mock_preference_dataset)

    expected_columns = {
        "dataset", "id", "audio",
        "prosody_codebooks_idx", "timbre_vector",
        "chosen", "rejected", "rationale_chosen", "rationale_rejected",
        "emotion_label", "speaker_name", "speaker_gender",
        "speaker_age_context", "speaker_nationality",
        "transcript_with_tags", "bare_transcript",
        "cosine_similarity", "label",
    }
    actual = set(result.column_names)
    assert expected_columns == actual, f"Missing: {expected_columns - actual}, Extra: {actual - expected_columns}"


def test_processor_skips_content_acoustic(mock_preference_dataset):
    """Content and acoustic codebooks are NOT in output."""
    encoder = FACodecEncoder(force_mock=True, device="cpu")
    processor = PreferenceDatasetProcessor(encoder)
    result = processor.process_dataset(mock_preference_dataset)
    assert "content_codebooks_idx" not in result.column_names
    assert "acoustic_codebooks_idx" not in result.column_names


def test_processor_has_prosody_and_timbre(mock_preference_dataset):
    """Prosody and timbre streams are present with correct shapes."""
    encoder = FACodecEncoder(force_mock=True, device="cpu")
    processor = PreferenceDatasetProcessor(encoder)
    result = processor.process_dataset(mock_preference_dataset)

    sample = result[0]
    assert len(sample["prosody_codebooks_idx"]) > 0  # flat list [T80]
    assert len(sample["timbre_vector"]) == 256


def test_processor_label_is_neg_one(mock_preference_dataset):
    """All labels are -1 (no classification label)."""
    encoder = FACodecEncoder(force_mock=True, device="cpu")
    processor = PreferenceDatasetProcessor(encoder)
    result = processor.process_dataset(mock_preference_dataset)
    for sample in result:
        assert sample["label"] == -1


def test_processor_save_roundtrip(mock_preference_dataset, tmp_path):
    """Save to parquet and reload preserves all columns."""
    encoder = FACodecEncoder(force_mock=True, device="cpu")
    processor = PreferenceDatasetProcessor(encoder)
    result = processor.process_dataset(mock_preference_dataset)

    save_path = tmp_path / "test_processor" / "test-repo" / "train.parquet"
    save_path.parent.mkdir(parents=True)
    result.to_parquet(str(save_path))

    reloaded = Dataset.from_parquet(str(save_path))
    assert set(reloaded.column_names) == set(result.column_names)
    assert len(reloaded) == len(result)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/preprocessing/test_preference_dataset_processor.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Write the processor**

```python
# src/preprocessing/preference_dataset_processor.py
"""PreferenceDatasetProcessor — FACodec encoding for preference pair datasets.

Extends the DatasetProcessor pattern for DPO training data. Encodes only
prosody and timbre streams (skips content + acoustic to save compute/storage).
"""
from __future__ import annotations

from typing import List, Optional

import torch
from datasets import Dataset, Features, Sequence, Value, Audio as HFAudio

from src.preprocessing.facodec_encoder import FACodecEncoder, FACodecStreams


_PREFERENCE_SCHEMA = Features({
    "dataset": Value("string"),
    "id": Value("string"),
    "audio": HFAudio(sampling_rate=16000),
    "prosody_codebooks_idx": Sequence(Value("int64")),
    "timbre_vector": Sequence(Value("float32")),
    "chosen": Value("string"),
    "rejected": Value("string"),
    "rationale_chosen": Value("string"),
    "rationale_rejected": Value("string"),
    "emotion_label": Value("string"),
    "speaker_name": Value("string"),
    "speaker_gender": Value("string"),
    "speaker_age_context": Value("string"),
    "speaker_nationality": Value("string"),
    "transcript_with_tags": Value("string"),
    "bare_transcript": Value("string"),
    "cosine_similarity": Value("float32"),
    "label": Value("int64"),
})


class PreferenceDatasetProcessor:
    """Process preference pair datasets through FACodec (prosody + timbre only)."""

    def __init__(self, encoder: FACodecEncoder, batch_size: int = 8) -> None:
        self.encoder = encoder
        self.batch_size = batch_size

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Encode audio through FACodec, appending prosody + timbre columns."""
        processed = []
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i : i + self.batch_size]
            audio_list = []
            for sample in batch:
                arr = sample["audio"]
                if isinstance(arr, dict):
                    arr = arr["array"]
                audio_list.append(torch.tensor(arr, dtype=torch.float32))

            streams: FACodecStreams = self.encoder.encode_batch(audio_list)

            for j in range(len(audio_list)):
                entry = dict(batch[j])
                entry.pop("prosody_codebooks_idx", None)
                entry.pop("timbre_vector", None)

                # Prosody: single codebook, squeeze to flat list [T80]
                if streams.prosody_codebooks_idx is not None:
                    p_idx = streams.prosody_codebooks_idx[j]
                    if p_idx.dim() == 2:
                        p_idx = p_idx.squeeze(0)
                    entry["prosody_codebooks_idx"] = p_idx.tolist()
                else:
                    entry["prosody_codebooks_idx"] = []

                # Timbre: [256] float32
                if streams.timbre_vector is not None:
                    entry["timbre_vector"] = streams.timbre_vector[j].tolist()
                else:
                    entry["timbre_vector"] = [0.0] * 256

                entry["label"] = -1
                entry.setdefault("dataset", "nvtts-preference")
                processed.append(entry)

        return Dataset.from_list(processed, features=_PREFERENCE_SCHEMA)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python -m pytest tests/preprocessing/test_preference_dataset_processor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/preference_dataset_processor.py tests/preprocessing/test_preference_dataset_processor.py
git commit -m "feat: add PreferenceDatasetProcessor for DPO data (Issue #21)"
```

---

## Phase Completion Criteria
- [ ] Script 1 (`scripts/enrich_nvtts.py`) loads NVTTS, applies NV mapping + speaker lookup, saves enriched dataset
- [ ] `PreferenceDatasetProcessor` class passes all tests with mock FACodec
- [ ] Tests passing: `uv run python -m pytest tests/data/test_enrich_nvtts.py tests/preprocessing/test_preference_dataset_processor.py -v`

## Handoff Notes
- Script 1 requires #17 (NV mapping module) and #18 (speaker lookup JSON) to exist
- PreferenceDatasetProcessor requires #19 (AmyLM — no direct dependency, but schema is informed by it)
- Phase 2 (DeepSeek) needs the enriched dataset output from Script 1 at `data/nvtts_enriched/nvtts_enriched.parquet`
- Phase 3 feeds into PreferenceDatasetProcessor via `processor.process_dataset(intermediate_dataset)`
