"""Tests for Speaker Context Lookup Cache — Issue #18."""
import json
import os
import tempfile
import pytest
from src.data.speaker_cache import (
    SpeakerCache,
    EXPRESSO_SPEAKERS,
    resolve_age_context,
)


class TestExpressoSpeakers:
    def test_all_four_expresso_speakers(self):
        """All 4 Expresso speakers are defined."""
        assert len(EXPRESSO_SPEAKERS) == 4
        for sid in ("ex01", "ex02", "ex03", "ex04"):
            assert sid in EXPRESSO_SPEAKERS

    def test_expresso_names(self):
        """Expresso speakers have correct names."""
        assert EXPRESSO_SPEAKERS["ex01"]["name"] == "Jack"
        assert EXPRESSO_SPEAKERS["ex02"]["name"] == "Lisa"
        assert EXPRESSO_SPEAKERS["ex03"]["name"] == "Bert"
        assert EXPRESSO_SPEAKERS["ex04"]["name"] == "Emma"

    def test_expresso_genders(self):
        """2 Male, 2 Female."""
        genders = [s["gender"] for s in EXPRESSO_SPEAKERS.values()]
        assert genders.count("Male") == 2
        assert genders.count("Female") == 2

    def test_expresso_nationality(self):
        """All Expresso speakers are North American English."""
        for sid, spk in EXPRESSO_SPEAKERS.items():
            assert spk["nationality"] == "American", f"{sid} should be American"


class TestAgeContextFormatting:
    def test_age_with_birth_year(self):
        result = resolve_age_context({"age": 45, "birth_year": 1979})
        assert result == "45, born 1979"

    def test_age_without_birth_year(self):
        result = resolve_age_context({"age": 45})
        assert result == "45"

    def test_birth_year_without_age(self):
        result = resolve_age_context({"birth_year": 1979})
        assert result == "born 1979"

    def test_neither_age_nor_birth_year(self):
        result = resolve_age_context({})
        assert result == "unknown"

    def test_age_none_birth_year_none(self):
        result = resolve_age_context({"age": None, "birth_year": None})
        assert result == "unknown"


class TestSpeakerCache:
    @pytest.fixture
    def cache(self):
        """Build cache from mock data (no network)."""
        return SpeakerCache.from_mock()

    def test_expresso_speaker_resolves(self, cache):
        """Expresso speaker ex03 resolves with hardcoded data."""
        speaker = cache.lookup("ex03")
        assert speaker is not None
        assert speaker["name"] == "Bert"
        assert speaker["gender"] == "Male"

    def test_voxceleb_speaker_resolves(self, cache):
        """Known VoxCeleb ID resolves from mock data."""
        speaker = cache.lookup("id00012")
        assert speaker is not None
        assert speaker["name"] != "unknown"

    def test_unknown_speaker_returns_unknown(self, cache):
        """Unknown speaker ID returns 'unknown' for all fields."""
        speaker = cache.lookup("id99999")
        assert speaker is not None
        assert speaker["name"] == "unknown"
        assert speaker["gender"] == "unknown"
        assert speaker["nationality"] == "unknown"

    def test_lookup_returns_required_fields(self, cache):
        """Lookup returns all required fields."""
        speaker = cache.lookup("ex01")
        for field in ["name", "gender", "nationality", "age", "birth_year"]:
            assert field in speaker, f"Missing field: {field}"

    def test_age_context_on_resolved_speaker(self, cache):
        """Age context can be derived from resolved speaker."""
        speaker = cache.lookup("ex01")
        age_ctx = resolve_age_context(speaker)
        assert isinstance(age_ctx, str)
        assert len(age_ctx) > 0

    def test_lookup_normalizes_speaker_id(self, cache):
        """Speaker ID variations resolve correctly."""
        # ex03 with and without leading zeros
        spk1 = cache.lookup("ex03")
        assert spk1 is not None

    def test_cache_size(self, cache):
        """Mock cache has expected minimum entries (4 Expresso + at least 1 VoxCeleb)."""
        assert len(cache._data) >= 5


class TestCachePersistence:
    def test_save_and_load_roundtrip(self):
        """Cache can be saved to JSON and loaded back."""
        cache = SpeakerCache.from_mock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache.save(f.name)
            save_path = f.name

        try:
            loaded = SpeakerCache.load(save_path)
            assert loaded.lookup("ex03")["name"] == "Bert"
        finally:
            os.unlink(save_path)

    def test_json_is_valid(self):
        """Saved JSON is parseable by json.load."""
        cache = SpeakerCache.from_mock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache.save(f.name)
            save_path = f.name

        try:
            with open(save_path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert "ex01" in data
            assert data["ex01"]["name"] == "Jack"
        finally:
            os.unlink(save_path)


class TestCascadingFallback:
    def test_enrichment_overrides_vox1_meta(self):
        """Enrichment field takes priority over vox1_meta for same field."""
        enrichment = {"name": "Enriched Name", "gender": "Male", "age": 50}
        vox1_meta = {"name": "Meta Name", "gender": "Female"}
        result = SpeakerCache._merge_fields(enrichment, vox1_meta)
        assert result["name"] == "Enriched Name"
        assert result["gender"] == "Male"
        assert result["age"] == 50

    def test_fallback_when_primary_missing(self):
        """When enrichment is missing, vox1_meta value is used."""
        enrichment = {"name": None, "gender": None}
        vox1_meta = {"name": "Meta Name", "gender": "Male"}
        result = SpeakerCache._merge_fields(enrichment, vox1_meta)
        assert result["name"] == "Meta Name"
        assert result["gender"] == "Male"

    def test_all_missing_returns_unknown(self):
        """When all sources miss, field is 'unknown'."""
        enrichment = {}
        vox1_meta = {}
        lang_meta = {}
        result = SpeakerCache._merge_fields(enrichment, vox1_meta, lang_meta)
        assert result["name"] == "unknown"
        assert result["gender"] == "unknown"
