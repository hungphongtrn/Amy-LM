"""Speaker Context Lookup Cache — Issue #18.

Merges VoxCeleb metadata sources plus Expresso speaker data into a
cascading-fallback JSON lookup table for speaker context enrichment.

Sources:
  1. VoxCeleb enrichment CSV (data/voxceleb_enrichment.csv) — name, gender, nationality, birth_year
     Downloaded from: https://github.com/hechmik/voxceleb_enrichment_age_gender
  2. johbac/voxceleb-language-metadata (HF) — VoxCeleb2 names + gender
  3. Expresso: 4 known speakers (Jack/Lisa/Bert/Emma), North American English

Cascade: enrichment CSV > language-metadata > Expresso > "unknown"
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

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
    enrichment CSV > language-metadata > Expresso > "unknown".

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
            cached = cls.load(output_path)
            # Detect stale cache: only Expresso + mock speakers, no real VoxCeleb data.
            if len(cached._data) <= len(EXPRESSO_SPEAKERS) + len(_MOCK_VOX1_META):
                logger.warning(
                    "Loaded speaker cache has only %d speakers — appears stale/incomplete. "
                    "Rebuilding to fetch full VoxCeleb metadata.",
                    len(cached._data),
                )
            else:
                return cached
            # Fall through to rebuild if stale.

        cache = cls()

        # 1. Expresso (base layer)
        for sid, spk in EXPRESSO_SPEAKERS.items():
            cache._data[sid] = dict(spk)

        sources_loaded = 0

        # 2. VoxCeleb enrichment CSV (highest priority — name, gender, nationality, age, birth_year)
        try:
            cache._ingest_voxceleb_enrichment_csv()
            sources_loaded += 1
        except Exception:
            logger.warning(
                "VoxCeleb enrichment CSV (data/voxceleb_enrichment.csv) "
                "unavailable — download it from "
                "https://github.com/hechmik/voxceleb_enrichment_age_gender"
            )

        # 3. VoxCeleb language metadata (names + gender — loaded from raw CSV)
        try:
            cache._ingest_vox2_language_metadata()
            sources_loaded += 1
        except Exception:
            logger.warning(
                "VoxCeleb language metadata (johbac/voxceleb-language-metadata) unavailable."
            )

        if sources_loaded == 0:
            logger.warning(
                "No VoxCeleb metadata sources were loaded — speaker cache will only "
                "contain %d Expresso speakers. Speaker metadata will be 'unknown' for "
                "all VoxCeleb IDs.",
                len(EXPRESSO_SPEAKERS),
            )

        if output_path:
            cache.save(output_path)

        return cache

    def _ingest_voxceleb_enrichment_csv(self) -> None:
        """Ingest VoxCeleb enrichment from local CSV (hechmik's dataset).

        The CSV (data/voxceleb_enrichment.csv) has ~149K rows covering
        6,112 unique speakers with name, gender, nationality, birth_year.
        Multiple rows per speaker (one per video) — we aggregate to take
        the first non-empty value for each field.
        """
        import csv
        from collections import Counter
        from pathlib import Path

        csv_path = Path("data/voxceleb_enrichment.csv")
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Download from "
                "https://github.com/hechmik/voxceleb_enrichment_age_gender/"
                "blob/main/dataset/final_dataframe_extended.csv"
            )

        # First pass: collect raw values per speaker.
        raw: dict[str, dict[str, list[str]]] = {}
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("VoxCeleb_ID", "").strip()
                if not sid:
                    continue
                if sid not in raw:
                    raw[sid] = {"names": [], "genders": [], "nationalities": [], "birth_years": []}
                r = raw[sid]
                name = row.get("Name", "").strip()
                if name:
                    r["names"].append(name)
                gender = row.get("gender", "").strip().lower()
                if gender:
                    r["genders"].append(gender)
                nat = (row.get("nationality_wiki") or row.get("nationality_dbpedia") or row.get("nationality_gkg") or "").strip()
                if nat:
                    r["nationalities"].append(nat)
                by_str = row.get("birth_year", "").strip()
                if by_str:
                    r["birth_years"].append(by_str)

        # Merge into cache: first non-empty value wins per field.
        for sid, fields in raw.items():
            if sid in _MOCK_VOX1_META:
                continue
            speaker: dict[str, Any] = {}
            # Name: take the most common name
            if fields["names"]:
                speaker["name"] = Counter(fields["names"]).most_common(1)[0][0]
            # Gender: take majority vote
            if fields["genders"]:
                g = Counter(fields["genders"]).most_common(1)[0][0]
                speaker["gender"] = g.capitalize() if g in ("male", "female") else g
            # Nationality: take first (most entries agree)
            if fields["nationalities"]:
                speaker["nationality"] = fields["nationalities"][0]
            # Birth year: take first valid integer
            for by_str in fields["birth_years"]:
                try:
                    by = int(float(by_str))
                    speaker["birth_year"] = by
                    break
                except (ValueError, TypeError):
                    pass

            if sid in self._data:
                self._data[sid] = self._merge_fields(speaker, self._data[sid])
            else:
                self._data[sid] = self._merge_fields(speaker)

    def _ingest_vox2_language_metadata(self) -> None:
        """Ingest VoxCeleb2 metadata from johbac/voxceleb-language-metadata CSV.

        The dataset CSV is tab-separated but has a malformed header (all column
        names fused into one). We load it directly via HF cache to handle this.
        """
        import csv
        from pathlib import Path

        from datasets import load_dataset

        # Download the dataset to populate the cache directory.
        try:
            load_dataset("johbac/voxceleb-language-metadata", split="train")
        except Exception:
            pass

        # Locate the cached CSV file.
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        csv_files = list(cache_root.glob(
            "datasets--johbac--voxceleb-language-metadata/**/vox2_meta.csv"
        ))
        if not csv_files:
            raise FileNotFoundError("vox2_meta.csv not found in HuggingFace cache")

        csv_path = csv_files[0]

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            header = [h.strip() for h in header]

            col_idx = {h: i for i, h in enumerate(header)}
            name_idx = col_idx.get("Name")
            gender_idx = col_idx.get("Gender")
            vc_id_idx = col_idx.get("VoxCeleb2 ID")

            if name_idx is None or vc_id_idx is None:
                raise ValueError(
                    f"CSV columns not as expected: {header}. "
                    f"Expected 'Name' and 'VoxCeleb2 ID' columns."
                )

            for row in reader:
                if not row or len(row) < max(name_idx, vc_id_idx, (gender_idx or 0)) + 1:
                    continue
                sid = row[vc_id_idx].strip()
                if not sid or sid in _MOCK_VOX1_META:
                    continue
                raw_name = row[name_idx].strip().replace("_", " ")
                speaker: dict[str, Any] = {
                    "name": raw_name,
                }
                if gender_idx is not None and gender_idx < len(row):
                    g = row[gender_idx].strip().lower()
                    if g in ("m", "male"):
                        speaker["gender"] = "Male"
                    elif g in ("f", "female"):
                        speaker["gender"] = "Female"
                if sid in self._data:
                    self._data[sid] = self._merge_fields(
                        self._data.get(sid, {}), speaker
                    )
                else:
                    self._data[sid] = self._merge_fields(speaker)
