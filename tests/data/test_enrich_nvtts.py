import os
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from src.data.nv_tag_mapping import emojis_to_tags, NV_EMOJI_TO_TAG
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
        """Enriched sample has all 9 text output columns (no audio)."""
        from scripts.enrich_nvtts import enrich_sample

        enriched = enrich_sample(raw_sample_expresso, speaker_cache)
        expected_cols = {
            "id", "emotion_label", "speaker_name", "speaker_gender",
            "speaker_age_context", "speaker_nationality", "transcript_with_tags",
            "bare_transcript", "source",
        }
        assert expected_cols.issubset(set(enriched.keys()))
        assert "audio" not in enriched
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

    def test_enrich_sample_strips_nv_emojis_from_bare_transcript(self, speaker_cache):
        """bare_transcript has all NV emoji markers stripped."""
        from scripts.enrich_nvtts import enrich_sample

        # Pick two representative emojis from the mapping
        emojis = list(NV_EMOJI_TO_TAG.keys())
        e1, e2 = emojis[0], emojis[1]

        raw = {
            "audio": {"path": "dummy.wav", "array": b"", "sampling_rate": 16000},
            "Emotion": "happy",
            "Initial text": f"hello {e1} world {e2} today",
            "Result": f"hello {e1} world {e2} today",
            "speaker_id": "ex01",
            "data_name": "Expresso",
            "gender": "m",
        }
        enriched = enrich_sample(raw, speaker_cache)
        # bare_transcript should have no emojis
        for e in (e1, e2):
            assert e not in enriched["bare_transcript"], f"Emoji {repr(e)} not stripped from bare_transcript"
        assert enriched["bare_transcript"] == "hello world today"
        # transcript_with_tags should still have the [Tag] labels
        tag1 = NV_EMOJI_TO_TAG[e1]
        tag2 = NV_EMOJI_TO_TAG[e2]
        assert tag1 in enriched["transcript_with_tags"]
        assert tag2 in enriched["transcript_with_tags"]

    def test_main_calls_load_dataset_for_all_splits(self):
        """main() loads train+dev+test via streaming load_dataset."""
        from scripts.enrich_nvtts import main

        call_args = []

        def fake_load_dataset(name, split, streaming, **kw):
            call_args.append(split)
            return Dataset.from_dict({"index": [split]}).to_iterable_dataset()

        with patch("scripts.enrich_nvtts.SpeakerCache.build", return_value=SpeakerCache.from_mock()):
            with patch("scripts.enrich_nvtts.load_dataset", side_effect=fake_load_dataset):
                with patch("scripts.enrich_nvtts.OUTPUT_DIR", "/tmp/nvtts_dummy"):
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        with patch("scripts.enrich_nvtts.OUTPUT_DIR", tmpdir):
                            main()

        assert call_args == ["train", "dev", "test"]

    def test_main_streaming_pipeline_works(self, speaker_cache, tmp_path):
        """main() processes a streaming dataset with minimal data."""
        import tempfile
        from scripts.enrich_nvtts import main

        streaming = Dataset.from_dict({
            "index": ["sample_0"],
            "audio": [{"path": "a.wav", "array": b"\x00", "sampling_rate": 16000}],
            "Emotion": ["neutral"],
            "Initial text": ["test"],
            "Result": ["test \U0001f32c"],
            "speaker_id": ["ex01"],
            "data_name": ["Expresso"],
            "gender": ["f"],
        }).to_iterable_dataset()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scripts.enrich_nvtts.SpeakerCache.build", return_value=speaker_cache):
                with patch("scripts.enrich_nvtts.OUTPUT_DIR", tmpdir):
                    with patch("scripts.enrich_nvtts.load_dataset", return_value=streaming):
                        main()

            out_path = Path(tmpdir) / "nvtts_enriched.parquet"
            assert out_path.exists()
            loaded = Dataset.from_parquet(str(out_path))
            assert len(loaded) == 3  # one row per split (train/dev/test all get same mock)
            assert "audio" not in loaded.column_names

    def test_main_saves_text_parquet(self, speaker_cache, tmp_path):
        """main() saves text-only parquet to the expected output path."""
        import tempfile
        from scripts.enrich_nvtts import main

        fake_streaming_ds = Dataset.from_dict({
            "index": ["0", "1"],
            "audio": [
                {"path": "a.wav", "array": b"\x00", "sampling_rate": 16000},
                {"path": "b.wav", "array": b"\x01", "sampling_rate": 16000},
            ],
            "Emotion": ["happy", "sad"],
            "Initial text": ["hello", "goodbye"],
            "Result": ["hello \U0001f32c", "goodbye \U0001f637"],
            "speaker_id": ["ex01", "id00012"],
            "data_name": ["Expresso", "VoxCeleb"],
            "gender": ["f", "m"],
        }).to_iterable_dataset()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scripts.enrich_nvtts.SpeakerCache.build", return_value=speaker_cache):
                with patch("scripts.enrich_nvtts.OUTPUT_DIR", tmpdir):
                    with patch("scripts.enrich_nvtts.load_dataset", return_value=fake_streaming_ds):
                        main()

            out_path = Path(tmpdir) / "nvtts_enriched.parquet"
            assert out_path.exists()

            loaded = Dataset.from_parquet(str(out_path))
            assert len(loaded) == 6  # 2 rows × 3 splits (mock returns same ds for all)
            assert set(loaded.column_names) == {
                "id", "emotion_label", "speaker_name", "speaker_gender",
                "speaker_age_context", "speaker_nationality", "transcript_with_tags",
                "bare_transcript", "source",
            }
            assert "audio" not in loaded.column_names
            assert loaded[0]["emotion_label"] == "happy"
