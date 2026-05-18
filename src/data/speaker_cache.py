"""Speaker Context Lookup Cache — Issue #18.

Merges 3 VoxCeleb metadata sources plus Expresso speaker data into a
cascading-fallback JSON lookup table for speaker context enrichment.

Sources:
  1. hechmik/voxceleb_enrichment_age_gender (HF) — richest: name, gender, age, birth_year
  2. ProgramComputer/voxceleb — vox1_meta.csv (ID -> Name, Gender, Nationality)
  3. johbac/voxceleb-language-metadata (HF) — VoxCeleb2 names
  4. Expresso: 4 known speakers (Jack/Lisa/Bert/Emma), North American English

Cascade: enrichment > vox1_meta > language-metadata > Expresso > "unknown"
"""

from __future__ import annotations

import json
import os
from typing import Any

EXPRESSO_SPEAKERS: dict[str, dict[str, Any]] = {
    "ex01": {"name": "Jack", "gender": "Male", "nationality": "American", "age": None, "birth_year": None},
    "ex02": {"name": "Lisa", "gender": "Female", "nationality": "American", "age": None, "birth_year": None},
    "ex03": {"name": "Bert", "gender": "Male", "nationality": "American", "age": None, "birth_year": None},
    "ex04": {"name": "Emma", "gender": "Female", "nationality": "American", "age": None, "birth_year": None},
}

_UNKNOWN_SPEAKER: dict[str, Any] = {
    "name": "unknown",
    "gender": "unknown",
    "nationality": "unknown",
    "age": None,
    "birth_year": None,
}

_MOCK_VOX1_META: dict[str, dict[str, Any]] = {
    "id00012": {"name": "John Doe", "gender": "Male", "nationality": "American"},
    "id00013": {"name": "Jane Smith", "gender": "Female", "nationality": "British"},
    "id03621": {"name": "Alice Johnson", "gender": "Female", "nationality": "Canadian"},
}

_MOCK_ENRICHMENT: dict[str, dict[str, Any]] = {
    "id00012": {"name": "Johnathan Doe", "gender": "Male", "age": 45, "birth_year": 1979, "nationality": "American"},
    "id03621": {"name": None, "gender": None, "age": 32, "birth_year": 1992, "nationality": None},
}


def resolve_age_context(speaker: dict[str, Any]) -> str:
    """Format age context string from speaker metadata.

    Args:
        speaker: Dict with optional 'age' (int|None) and 'birth_year' (int|None).

    Returns:
        Formatted string like "45, born 1979", "45", "born 1979", or "unknown".
    """
    age = speaker.get("age")
    birth_year = speaker.get("birth_year")

    has_age = age is not None
    has_birth = birth_year is not None

    if has_age and has_birth:
        return f"{age}, born {birth_year}"
    elif has_age:
        return str(age)
    elif has_birth:
        return f"born {birth_year}"
    else:
        return "unknown"


class SpeakerCache:
    """Cascading-fallback speaker context lookup.

    Merges multiple metadata sources with priority:
    enrichment > vox1_meta > language-metadata > Expresso > "unknown".

    Each field (name, gender, nationality, age, birth_year) is resolved
    independently through the cascade.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _merge_fields(*sources: dict[str, Any]) -> dict[str, Any]:
        """Merge fields from multiple sources with cascading fallback.

        First non-None, non-missing value for each field wins.
        Final fallback is 'unknown' for string fields, None for numeric.
        """
        FIELD_DEFAULTS: dict[str, Any] = {
            "name": "unknown",
            "gender": "unknown",
            "nationality": "unknown",
            "age": None,
            "birth_year": None,
        }
        result: dict[str, Any] = {}
        for field, default in FIELD_DEFAULTS.items():
            value = default
            for source in sources:
                src_val = source.get(field)
                if src_val is not None:
                    value = src_val
                    break
            result[field] = value
        return result

    def lookup(self, speaker_id: str) -> dict[str, Any]:
        """Look up speaker context by ID.

        Args:
            speaker_id: Speaker ID (e.g. "ex03" for Expresso, "id03621" for VoxCeleb).

        Returns:
            Dict with fields: name, gender, nationality, age, birth_year.
            Returns _UNKNOWN_SPEAKER if speaker not found.
        """
        if speaker_id in self._data:
            return dict(self._data[speaker_id])
        return dict(_UNKNOWN_SPEAKER)

    def save(self, path: str) -> None:
        """Save cache to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> SpeakerCache:
        """Load cache from JSON file."""
        cache = cls()
        with open(path) as f:
            cache._data = json.load(f)
        return cache

    @classmethod
    def from_mock(cls) -> SpeakerCache:
        """Build cache from mock data (no network required)."""
        cache = cls()
        # Start with Expresso
        for sid, spk in EXPRESSO_SPEAKERS.items():
            cache._data[sid] = dict(spk)

        # Layer vox1_meta (lower priority than enrichment)
        for sid, meta in _MOCK_VOX1_META.items():
            if sid in cache._data:
                cache._data[sid] = cls._merge_fields(
                    cache._data[sid], meta
                )
            else:
                cache._data[sid] = cls._merge_fields(meta)

        # Layer enrichment (highest priority after Expresso base is set)
        for sid, enrich in _MOCK_ENRICHMENT.items():
            if sid in cache._data:
                cache._data[sid] = cls._merge_fields(
                    enrich, cache._data[sid]
                )
            else:
                cache._data[sid] = cls._merge_fields(enrich)

        return cache

    @classmethod
    def build(
        cls,
        output_path: str | None = None,
        force: bool = False,
    ) -> SpeakerCache:
        """Download and merge all speaker metadata sources.

        Fetches from HF datasets (requires network). Caches result to JSON.

        Args:
            output_path: Path to save JSON cache (default: data/speaker_lookup.json).
            force: If True, rebuild even if cache exists.

        Returns:
            Built SpeakerCache.
        """
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data",
                "speaker_lookup.json",
            )

        if not force and os.path.exists(output_path):
            return cls.load(output_path)

        cache = cls()

        # 1. Expresso (base layer)
        for sid, spk in EXPRESSO_SPEAKERS.items():
            cache._data[sid] = dict(spk)

        # 2. VoxCeleb enrichment (highest priority VoxCeleb source)
        try:
            from datasets import load_dataset

            enrich_ds = load_dataset(
                "hechmik/voxceleb_enrichment_age_gender", split="train"
            )
            for row in enrich_ds:
                sid = row.get("VoxCeleb1 ID") or row.get("voxceleb_id") or row.get("speaker_id")
                if not sid:
                    continue
                speaker = {
                    "name": row.get("Name") or row.get("name"),
                    "gender": row.get("Gender") or row.get("gender"),
                    "nationality": row.get("Nationality") or row.get("nationality"),
                    "age": row.get("Age") or row.get("age"),
                    "birth_year": row.get("Birth year") or row.get("birth_year"),
                }
                cache._data[sid] = cls._merge_fields(speaker, cache._data.get(sid, {}))
        except Exception:
            pass  # Source unavailable, skip

        # 3. VoxCeleb1 meta CSV
        try:
            from datasets import load_dataset

            vox1_ds = load_dataset("ProgramComputer/voxceleb", "vox1_meta", split="train")
            for row in vox1_ds:
                sid = row.get("VoxCeleb1 ID") or row.get("speaker_id")
                if not sid:
                    continue
                speaker = {
                    "name": row.get("Name") or row.get("name"),
                    "gender": row.get("Gender") or row.get("gender"),
                    "nationality": row.get("Nationality") or row.get("nationality"),
                }
                cache._data[sid] = cls._merge_fields(
                    cache._data.get(sid, {}), speaker
                )
        except Exception:
            pass

        # 4. VoxCeleb language metadata (names only)
        try:
            from datasets import load_dataset

            lang_ds = load_dataset(
                "johbac/voxceleb-language-metadata", split="train"
            )
            for row in lang_ds:
                sid = row.get("speaker_id") or row.get("VoxCeleb ID")
                if not sid:
                    continue
                speaker = {
                    "name": row.get("name") or row.get("Name"),
                }
                cache._data[sid] = cls._merge_fields(
                    cache._data.get(sid, {}), speaker
                )
        except Exception:
            pass

        if output_path:
            cache.save(output_path)

        return cache
