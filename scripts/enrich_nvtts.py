"""Script 1: Enrich NVTTS with speaker context and NV tag mapping.

Loads NVTTS (deepvk/NonverbalTTS), applies:
  - NV emoji-to-text mapping from #17 (src/data/nv_tag_mapping.py)
  - Speaker context lookup from #18 (src/data/speaker_cache.py)

Output: text-only parquet saved to data/nvtts_enriched/.
Audio is NOT included — the enrichment only touches text metadata.
Downstream FACodec encoding loads audio directly from NVTTS by sample id.
Uses streaming=True to avoid OOM from loading ~4 GB of 48 kHz audio.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset, concatenate_datasets, load_dataset

from src.data.nv_tag_mapping import emojis_to_tags, strip_nv_emojis
from src.data.speaker_cache import SpeakerCache, resolve_age_context

NVTTS_DATASET = "deepvk/NonverbalTTS"
SPLITS = ["train", "dev", "test"]
BATCH_SIZE = 100
OUTPUT_DIR = "data/nvtts_enriched"

# Text-only — no audio. FACodec phase loads audio separately by id.
OUTPUT_COLUMNS = [
    "id", "emotion_label", "speaker_name", "speaker_gender",
    "speaker_age_context", "speaker_nationality", "transcript_with_tags",
    "bare_transcript", "source",
]


def enrich_sample(sample: dict[str, Any], cache: SpeakerCache) -> dict[str, Any]:
    speaker = cache.lookup(sample.get("speaker_id") or "")
    sample_id = sample.get("id") or sample.get("index") or ""

    result_text = (sample.get("Result") or "").replace("\ufe0f", "")

    return {
        "id": str(sample_id),
        "emotion_label": sample.get("Emotion") or "unknown",
        "speaker_name": speaker["name"],
        "speaker_gender": speaker["gender"],
        "speaker_age_context": resolve_age_context(speaker),
        "speaker_nationality": speaker["nationality"],
        "transcript_with_tags": emojis_to_tags(result_text),
        "bare_transcript": strip_nv_emojis(result_text),
        "source": sample.get("data_name") or "unknown",
    }


def _enrich_batch(batch: dict[str, list], cache: SpeakerCache) -> dict[str, list]:
    result = {key: [] for key in OUTPUT_COLUMNS}
    n = len(next(iter(batch.values())))
    for i in range(n):
        raw = {k: v[i] for k, v in batch.items()}
        enriched = enrich_sample(raw, cache)
        for key in result:
            result[key].append(enriched.get(key, ""))
    return result


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache = SpeakerCache.build("data/speaker_lookup.json")

    enriched_paths: list[str] = []
    total_samples = 0

    for split in SPLITS:
        print(f"\n--- Processing split: {split} ---")
        ds = load_dataset(NVTTS_DATASET, split=split, streaming=True)
        enriched = ds.map(
            _enrich_batch,
            fn_kwargs={"cache": cache},
            batched=True,
            batch_size=BATCH_SIZE,
            remove_columns=ds.column_names,
        )
        split_path = os.path.join(OUTPUT_DIR, f"nvtts_{split}.parquet")
        enriched.to_parquet(split_path)
        enriched_paths.append(split_path)
        count = sum(1 for _ in Dataset.from_parquet(split_path))
        total_samples += count
        print(f"  Saved {count} samples to {split_path}")

    merged = concatenate_datasets(
        [Dataset.from_parquet(p) for p in enriched_paths]
    )
    out_path = os.path.join(OUTPUT_DIR, "nvtts_enriched.parquet")
    merged.to_parquet(out_path)
    print(f"\nMerged {total_samples} samples \u2192 {out_path}")


if __name__ == "__main__":
    main()
