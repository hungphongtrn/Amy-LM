"""Script 1: Enrich NVTTS with speaker context and NV tag mapping.

Loads NVTTS (deepvk/NonverbalTTS), applies:
  - NV emoji-to-text mapping from #17 (src/data/nv_tag_mapping.py)
  - Speaker context lookup from #18 (src/data/speaker_cache.py)

Output: HF Dataset saved to data/nvtts_enriched/
"""

from __future__ import annotations

import os
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from src.data.nv_tag_mapping import emojis_to_tags
from src.data.speaker_cache import SpeakerCache, resolve_age_context

NVTTS_DATASET = "deepvk/NonverbalTTS"
SPLITS = ["train", "dev", "test"]
OUTPUT_DIR = "data/nvtts_enriched"


def load_nvtts() -> Dataset:
    split_datasets = []
    for split in SPLITS:
        ds = load_dataset(NVTTS_DATASET, split=split, trust_remote_code=True)
        split_datasets.append(ds)
    return concatenate_datasets(split_datasets)


def enrich_sample(sample: dict[str, Any], cache: SpeakerCache) -> dict[str, Any]:
    speaker = cache.lookup(sample.get("speaker_id", ""))
    sample_id = sample.get("id")
    if sample_id is None:
        sample_id = sample.get("index", "")

    result_text = sample.get("Result", "").replace("\ufe0f", "")

    return {
        "id": str(sample_id),
        "audio": sample["audio"],
        "emotion_label": sample.get("Emotion", "unknown"),
        "speaker_name": speaker["name"],
        "speaker_gender": speaker["gender"],
        "speaker_age_context": resolve_age_context(speaker),
        "speaker_nationality": speaker["nationality"],
        "transcript_with_tags": emojis_to_tags(result_text),
        "bare_transcript": sample.get("Initial text", ""),
        "source": sample.get("data_name", "unknown"),
    }


def main() -> None:
    print(f"Loading NVTTS ({NVTTS_DATASET})...")
    ds = load_nvtts()
    print(f"Loaded {len(ds)} samples across splits: {SPLITS}")

    # Phase 1: use mock cache (no network). For production, replace with:
    #   SpeakerCache.build("data/speaker_lookup.json")
    cache = SpeakerCache.from_mock()

    print("Enriching samples...")
    enriched_rows = []
    failed = 0
    for i, sample in enumerate(ds):
        try:
            enriched_rows.append(enrich_sample(sample, cache))
        except Exception as e:
            print(f"  WARNING: Failed sample {i}: {e}")
            failed += 1
    print(f"Enriched {len(enriched_rows)} samples ({failed} failed)")
    result = Dataset.from_list(enriched_rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "nvtts_enriched.parquet")
    result.to_parquet(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
