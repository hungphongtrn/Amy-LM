# Phase 1: Enrichment + Processor

## Phase Goal

NVTTS enriched dataset saved to `data/nvtts_enriched/` with all provenance columns. `PreferenceDatasetProcessor` class built, tested, ready to process Script 3's intermediate parquet.

**Issues:** #20 (B1), #21 (B4)

## Files to Create

| File | Purpose |
|---|---|
| `scripts/enrich_nvtts.py` | Script 1: load NVTTS, apply NV mapping + speaker lookup, save enriched dataset |
| `tests/data/test_enrich_nvtts.py` | Tests for enrichment functions and script-level integration |
| `src/preprocessing/preference_dataset_processor.py` | PreferenceDatasetProcessor class — FACodec prosody+timbre only |
| `tests/preprocessing/test_preference_dataset_processor.py` | Processor tests with mock FACodec |

## Files to Read (reference only)

| File | Purpose |
|---|---|
| `src/data/nv_tag_mapping.py` | `emojis_to_tags(text) -> str` |
| `src/data/speaker_cache.py` | `SpeakerCache` class, `resolve_age_context()` |
| `src/preprocessing/facodec_encoder.py` | `FACodecEncoder`, `FACodecStreams` |
| `src/preprocessing/dataset_processor.py` | Pattern reference for processor class |

## Group A API Cheat Sheet

```python
# NV tag mapping (#17)
from src.data.nv_tag_mapping import emojis_to_tags
emojis_to_tags("hello 🌬️ world")  # → "hello [Breathing] world"

# Speaker cache (#18)
from src.data.speaker_cache import SpeakerCache, resolve_age_context
cache = SpeakerCache.from_mock()          # Offline mock (no network)
speaker = cache.lookup("ex01")            # → {"name": "Jack", "gender": "Male", "nationality": "American", "age": None, "birth_year": None}
resolve_age_context(speaker)              # → "unknown" (age=None, birth_year=None)

# FACodec encoder
from src.preprocessing.facodec_encoder import FACodecEncoder
encoder = FACodecEncoder(device="cpu", force_mock=True)
streams_list = encoder.encode_batch(audio_tensors)  # → List[FACodecStreams]
# streams_list[0].prosody_codebooks_idx  → torch.Tensor [1, T] int64
# streams_list[0].timbre_vector          → torch.Tensor [256] float32
```

---

## Tasks

### Task 1: Script 1 — NVTTS Enrichment

**Files:**
- Create: `scripts/enrich_nvtts.py`
- Create: `tests/data/test_enrich_nvtts.py`

#### Step 1: Write the test

```python
# tests/data/test_enrich_nvtts.py
import pytest
from src.data.nv_tag_mapping import emojis_to_tags
from src.data.speaker_cache import SpeakerCache, resolve_age_context


class TestEnrichNVTTS:
    """Tests for NVTTS enrichment script functions and integration."""

    @pytest.fixture
    def speaker_cache(self):
        return SpeakerCache.from_mock()

    def test_emojis_to_tags_converts_emojis(self):
        result = emojis_to_tags("I'm 🌬️ fine 🤣 thanks")
        assert "[Breathing]" in result
        assert "[Laughter]" in result

    def test_emojis_to_tags_no_emojis_unchanged(self):
        result = emojis_to_tags("hello world")
        assert result == "hello world"

    def test_speaker_cache_expresso_lookup(self, speaker_cache):
        speaker = speaker_cache.lookup("ex01")
        assert speaker["name"] == "Jack"
        assert speaker["gender"] == "Male"
        assert speaker["nationality"] == "American"

    def test_speaker_cache_unknown_speaker(self, speaker_cache):
        speaker = speaker_cache.lookup("nonexistent_id")
        assert speaker["name"] == "unknown"
        assert speaker["gender"] == "unknown"

    def test_resolve_age_context_both(self):
        result = resolve_age_context({"age": 45, "birth_year": 1979})
        assert result == "45, born 1979"

    def test_resolve_age_context_age_only(self):
        result = resolve_age_context({"age": 30})
        assert result == "30"

    def test_resolve_age_context_birth_only(self):
        result = resolve_age_context({"birth_year": 1996})
        assert result == "born 1996"

    def test_resolve_age_context_neither(self):
        result = resolve_age_context({})
        assert result == "unknown"

    def test_enrich_sample_produces_all_columns(self, speaker_cache):
        """Enriched sample has all 10 output columns with correct values."""
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
        enriched = enrich_sample(raw, speaker_cache)
        expected_cols = {
            "id", "audio", "emotion_label", "speaker_name", "speaker_gender",
            "speaker_age_context", "speaker_nationality", "transcript_with_tags",
            "bare_transcript", "source",
        }
        assert expected_cols.issubset(set(enriched.keys()))
        assert enriched["transcript_with_tags"] == "hello [Breathing] world"
        assert enriched["bare_transcript"] == "hello world"
        assert enriched["speaker_name"] == "Jack"
        assert enriched["source"] == "Expresso"
```

#### Step 2: Run test to verify it fails

```bash
uv run python -m pytest tests/data/test_enrich_nvtts.py -v
```

Expected: FAIL — `scripts.enrich_nvtts` module not found.

#### Step 3: Write the script

```python
# scripts/enrich_nvtts.py
"""Script 1: Enrich NVTTS with speaker context and NV tag mapping.

Loads NVTTS (deepvk/NonverbalTTS), applies:
  - NV emoji-to-text mapping from #17 (src/data/nv_tag_mapping.py)
  - Speaker context lookup from #18 (src/data/speaker_cache.py)

Output: HF Dataset saved to data/nvtts_enriched/
"""
import os
from typing import Any, Dict

from datasets import Dataset, concatenate_datasets, load_dataset

from src.data.nv_tag_mapping import emojis_to_tags
from src.data.speaker_cache import SpeakerCache, resolve_age_context

NVTTS_DATASET = "deepvk/NonverbalTTS"
SPLITS = ["train", "dev", "test"]
OUTPUT_DIR = "data/nvtts_enriched"


def load_nvtts() -> Dataset:
    datasets = []
    for split in SPLITS:
        ds = load_dataset(NVTTS_DATASET, split=split, trust_remote_code=True)
        datasets.append(ds)
    return concatenate_datasets(datasets)


def enrich_sample(sample: Dict[str, Any], cache: SpeakerCache) -> Dict[str, Any]:
    speaker = cache.lookup(sample["speaker_id"])
    return {
        "id": str(sample.get("index", "")),
        "audio": sample["audio"],
        "emotion_label": sample.get("Emotion", "unknown"),
        "speaker_name": speaker["name"],
        "speaker_gender": speaker["gender"],
        "speaker_age_context": resolve_age_context(speaker),
        "speaker_nationality": speaker["nationality"],
        "transcript_with_tags": emojis_to_tags(sample.get("Result", "")),
        "bare_transcript": sample.get("Initial text", ""),
        "source": sample.get("data_name", "unknown"),
    }


def main():
    print(f"Loading NVTTS ({NVTTS_DATASET})...")
    ds = load_nvtts()
    print(f"Loaded {len(ds)} samples across splits: {SPLITS}")

    print("Building speaker cache (mock, no network)...")
    cache = SpeakerCache.from_mock()

    print("Enriching samples...")
    enriched = []
    failed = 0
    for i, sample in enumerate(ds):
        try:
            enriched.append(enrich_sample(sample, cache))
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

#### Step 4: Run test to verify it passes

```bash
uv run python -m pytest tests/data/test_enrich_nvtts.py -v
```

#### Step 5: Commit

```bash
git add scripts/enrich_nvtts.py tests/data/test_enrich_nvtts.py
git commit -m "feat: add NVTTS enrichment script (Issue #20)"
```

---

### Task 2: PreferenceDatasetProcessor

**Files:**
- Create: `src/preprocessing/preference_dataset_processor.py`
- Create: `tests/preprocessing/test_preference_dataset_processor.py`

**Key difference from old plan:** `FACodecEncoder.encode_batch()` returns `List[FACodecStreams]`, not a single `FACodecStreams`. Iterate over the list, pairing each entry with its dataset row.

#### Step 1: Write the test

```python
# tests/preprocessing/test_preference_dataset_processor.py
import numpy as np
import torch
from datasets import Dataset

from src.preprocessing.facodec_encoder import FACodecEncoder


def _make_mock_preference_dataset(num_samples: int = 3) -> Dataset:
    """Synthetic preference pair dataset matching Script 3 output schema."""
    rng = np.random.RandomState(42)
    samples = []
    for i in range(num_samples):
        audio_arr = rng.randn(16000).astype(np.float32)
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


def _get_processor():
    from src.preprocessing.preference_dataset_processor import PreferenceDatasetProcessor
    encoder = FACodecEncoder(device="cpu", force_mock=True)
    return PreferenceDatasetProcessor(encoder)


class TestPreferenceDatasetProcessor:
    def test_output_columns(self):
        """Output has all expected columns; no content/acoustic streams."""
        dataset = _make_mock_preference_dataset()
        processor = _get_processor()
        result = processor.process_dataset(dataset)

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
        assert expected_columns == actual, \
            f"Missing: {expected_columns - actual}, Extra: {actual - expected_columns}"

    def test_skips_content_acoustic(self):
        """Content and acoustic codebooks are NOT in output."""
        dataset = _make_mock_preference_dataset()
        processor = _get_processor()
        result = processor.process_dataset(dataset)
        assert "content_codebooks_idx" not in result.column_names
        assert "acoustic_codebooks_idx" not in result.column_names

    def test_has_prosody_and_timbre(self):
        """Prosody and timbre streams are present with correct shapes."""
        dataset = _make_mock_preference_dataset()
        processor = _get_processor()
        result = processor.process_dataset(dataset)

        sample = result[0]
        assert len(sample["prosody_codebooks_idx"]) > 0
        assert len(sample["timbre_vector"]) == 256

    def test_label_is_neg_one(self):
        """All labels are -1 (no classification label)."""
        dataset = _make_mock_preference_dataset()
        processor = _get_processor()
        result = processor.process_dataset(dataset)
        for sample in result:
            assert sample["label"] == -1

    def test_save_roundtrip(self, tmp_path):
        """Save to parquet and reload preserves all columns."""
        dataset = _make_mock_preference_dataset()
        processor = _get_processor()
        result = processor.process_dataset(dataset)

        save_dir = tmp_path / "test_processor" / "test-repo"
        save_dir.mkdir(parents=True)
        save_path = save_dir / "train.parquet"
        result.to_parquet(str(save_path))

        reloaded = Dataset.from_parquet(str(save_path))
        assert set(reloaded.column_names) == set(result.column_names)
        assert len(reloaded) == len(result)

    def test_preserves_text_columns(self):
        """Text columns (chosen, rejected, rationale) survive unchanged."""
        dataset = _make_mock_preference_dataset(5)
        processor = _get_processor()
        result = processor.process_dataset(dataset)
        for i, sample in enumerate(result):
            assert sample["chosen"] == f"chosen response {i}"
            assert sample["rejected"] == f"rejected response {i}"
            assert sample["rationale_chosen"] == f"rationale chosen {i}"
            assert sample["rationale_rejected"] == f"rationale rejected {i}"
```

#### Step 2: Run test to verify it fails

```bash
uv run python -m pytest tests/preprocessing/test_preference_dataset_processor.py -v
```

Expected: FAIL — module not found.

#### Step 3: Write the processor

```python
# src/preprocessing/preference_dataset_processor.py
"""PreferenceDatasetProcessor — FACodec encoding for preference pair datasets.

Extends the DatasetProcessor pattern for DPO training data. Encodes only
prosody and timbre streams (skips content + acoustic to save compute/storage).

Key: FACodecEncoder.encode_batch() returns List[FACodecStreams] — one per sample.
"""
from __future__ import annotations

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
        """Encode audio through FACodec, appending prosody + timbre columns.

        encode_batch() returns List[FACodecStreams] — one per audio sample.
        We iterate over both the batch rows and the stream results in lockstep.
        """
        processed = []
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i : i + self.batch_size]
            audio_list = []
            for sample in batch:
                arr = sample["audio"]
                if isinstance(arr, dict):
                    arr = arr["array"]
                audio_list.append(torch.tensor(arr, dtype=torch.float32))

            # encode_batch() returns List[FACodecStreams] — one per sample
            streams_list = self.encoder.encode_batch(audio_list)

            for j, streams in enumerate(streams_list):
                entry = dict(batch[j])
                entry.pop("prosody_codebooks_idx", None)
                entry.pop("timbre_vector", None)

                # Prosody: single codebook, squeeze to flat list [T]
                if streams.prosody_codebooks_idx is not None:
                    p_idx = streams.prosody_codebooks_idx[j] if streams.prosody_codebooks_idx.dim() > 2 else streams.prosody_codebooks_idx
                    if p_idx.dim() == 2:
                        p_idx = p_idx.squeeze(0)
                    entry["prosody_codebooks_idx"] = p_idx.tolist()
                else:
                    entry["prosody_codebooks_idx"] = []

                # Timbre: [256] float32
                if streams.timbre_vector is not None:
                    entry["timbre_vector"] = streams.timbre_vector.tolist()
                else:
                    entry["timbre_vector"] = [0.0] * 256

                entry["label"] = -1
                entry.setdefault("dataset", "nvtts-preference")
                processed.append(entry)

        return Dataset.from_list(processed, features=_PREFERENCE_SCHEMA)
```

#### Step 4: Run test to verify it passes

```bash
uv run python -m pytest tests/preprocessing/test_preference_dataset_processor.py -v
```

#### Step 5: Commit

```bash
git add src/preprocessing/preference_dataset_processor.py tests/preprocessing/test_preference_dataset_processor.py
git commit -m "feat: add PreferenceDatasetProcessor for DPO data (Issue #21)"
```

---

## Phase Completion Criteria
- [ ] Script 1 (`scripts/enrich_nvtts.py`) loads NVTTS, applies NV mapping + speaker lookup, saves enriched dataset
- [ ] `PreferenceDatasetProcessor` class passes all tests with mock FACodec
- [ ] All tests passing:
  ```bash
  uv run python -m pytest tests/data/test_enrich_nvtts.py tests/preprocessing/test_preference_dataset_processor.py -v
  ```

## Handoff Notes
- Uses `SpeakerCache.from_mock()` — no network needed for testing
- `FACodecEncoder.encode_batch()` returns `List[FACodecStreams]`, NOT a single `FACodecStreams`
- Phase 2 (DeepSeek) needs the enriched dataset output from Script 1 at `data/nvtts_enriched/nvtts_enriched.parquet`
- Phase 3 feeds into PreferenceDatasetProcessor via `processor.process_dataset(intermediate_dataset)`
