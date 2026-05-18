import os
from unittest.mock import patch

import pytest
from datasets import Dataset

from src.data.nv_tag_mapping import emojis_to_tags
from src.data.speaker_cache import SpeakerCache, resolve_age_context


class TestEnrichNVTTS:
    """Tests for NVTTS enrichment script functions and integration."""

    @pytest.fixture
    def speaker_cache(self):
        return SpeakerCache.from_mock()

    @pytest.fixture
    def raw_sample_expresso(self):
        return {
            "audio": {"path": "dummy.wav", "array": b"", "sampling_rate": 16000},
            "Emotion": "happy",
            "Initial text": "hello world",
            "Result": "hello 🌬️ world",
            "speaker_id": "ex01",
            "data_name": "Expresso",
            "gender": "m",
        }

    @pytest.fixture
    def raw_sample_voxceleb(self):
        return {
            "audio": {"path": "dummy.wav", "array": b"", "sampling_rate": 16000},
            "Emotion": "neutral",
            "Initial text": "good morning",
            "Result": "good 😷 morning",
            "speaker_id": "id00012",
            "data_name": "VoxCeleb",
            "gender": "m",
        }

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

    def test_enrich_sample_produces_all_columns(self, speaker_cache, raw_sample_expresso):
        """Enriched sample has all 10 output columns with correct values."""
        from scripts.enrich_nvtts import enrich_sample

        enriched = enrich_sample(raw_sample_expresso, speaker_cache)
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

    def test_enrich_sample_voxceleb_speaker(self, speaker_cache, raw_sample_voxceleb):
        """VoxCeleb speaker resolves correctly."""
        from scripts.enrich_nvtts import enrich_sample

        enriched = enrich_sample(raw_sample_voxceleb, speaker_cache)
        assert enriched["speaker_name"] == "Johnathan Doe"
        assert enriched["source"] == "VoxCeleb"
        assert enriched["transcript_with_tags"] == "good [Cough] morning"

    def test_enrich_sample_variation_selector_normalized(self, speaker_cache):
        """Emoji with U+FE0F variation selector is normalized before tag conversion."""
        from scripts.enrich_nvtts import enrich_sample

        raw = {
            "audio": {"path": "dummy.wav", "array": b"", "sampling_rate": 16000},
            "Emotion": "sad",
            "Initial text": "I am tired",
            "Result": "I am \U0001f634\ufe0f tired",
            "speaker_id": "ex01",
            "data_name": "Expresso",
            "gender": "m",
        }
        enriched = enrich_sample(raw, speaker_cache)
        assert "[Snore]" in enriched["transcript_with_tags"]

    def test_enrich_sample_missing_fields(self, speaker_cache):
        """Missing optional fields get sensible defaults."""
        from scripts.enrich_nvtts import enrich_sample

        raw = {
            "audio": {"path": "dummy.wav", "array": b"", "sampling_rate": 16000},
            "speaker_id": "ex01",
        }
        enriched = enrich_sample(raw, speaker_cache)
        assert enriched["emotion_label"] == "unknown"
        assert enriched["bare_transcript"] == ""
        assert enriched["transcript_with_tags"] == ""

    def test_load_nvtts_uses_all_splits(self):
        """load_nvtts loads train+dev+test and concatenates."""
        from scripts.enrich_nvtts import load_nvtts

        def make_split(split_name):
            return Dataset.from_dict({"index": [split_name]})

        with patch("scripts.enrich_nvtts.load_dataset") as mock_load:
            mock_load.side_effect = lambda name, split, **kw: make_split(split)
            result = load_nvtts()

        assert len(result) == 3  # one row per split
        assert mock_load.call_count == 3
        splits_called = [c.kwargs["split"] for c in mock_load.call_args_list]
        assert splits_called == ["train", "dev", "test"]

    def test_main_handles_bad_samples(self, tmp_path):
        """main() suppresses exceptions per sample, reports failures, and exits cleanly."""
        from scripts.enrich_nvtts import main

        bad_ds = Dataset.from_list([{
            "audio": {"path": "a.wav", "array": b"\x00", "sampling_rate": 16000},
            "Emotion": "happy",
            "Initial text": "hello",
            "Result": "hello",
            "speaker_id": "ex01",
            "data_name": "Expresso",
            "gender": "f",
        }])

        call_count = [0]

        def failing_enrich(sample, cache):
            call_count[0] += 1
            raise ValueError("simulated failure")

        with patch("scripts.enrich_nvtts.OUTPUT_DIR", str(tmp_path / "nvtts_enriched")):
            with patch("scripts.enrich_nvtts.enrich_sample", side_effect=failing_enrich):
                with patch("scripts.enrich_nvtts.load_nvtts", return_value=bad_ds):
                    main()

        assert call_count[0] == 1  # enrich was attempted
        # main() exited without raising — the exception was caught internally

    def test_main_saves_parquet(self, speaker_cache, tmp_path):
        """main() saves parquet to the expected output path."""
        from scripts.enrich_nvtts import main

        fake_ds = Dataset.from_dict({
            "index": [0, 1],
            "audio": [
                {"path": "a.wav", "array": b"\x00", "sampling_rate": 16000},
                {"path": "b.wav", "array": b"\x01", "sampling_rate": 16000},
            ],
            "Emotion": ["happy", "sad"],
            "Initial text": ["hello", "goodbye"],
            "Result": ["hello 🌬️", "goodbye 😷"],
            "speaker_id": ["ex01", "id00012"],
            "data_name": ["Expresso", "VoxCeleb"],
            "gender": ["f", "m"],
        })

        with patch("scripts.enrich_nvtts.OUTPUT_DIR", str(tmp_path / "nvtts_enriched")):
            with patch("scripts.enrich_nvtts.load_nvtts", return_value=fake_ds):
                main()

        out_path = tmp_path / "nvtts_enriched" / "nvtts_enriched.parquet"
        assert out_path.exists()

        loaded = Dataset.from_parquet(str(out_path))
        assert len(loaded) == 2
        assert set(loaded.column_names) == {
            "id", "audio", "emotion_label", "speaker_name", "speaker_gender",
            "speaker_age_context", "speaker_nationality", "transcript_with_tags",
            "bare_transcript", "source",
        }
