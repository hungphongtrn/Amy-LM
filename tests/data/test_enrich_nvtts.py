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
