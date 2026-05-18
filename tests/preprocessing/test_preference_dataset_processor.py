import numpy as np
from datasets import Dataset

from src.preprocessing.facodec_encoder import FACodecEncoder


def _make_mock_preference_dataset(num_samples: int = 3) -> Dataset:
    """Synthetic preference pair dataset matching Script 3 output schema."""
    rng = np.random.RandomState(42)
    samples = []
    for i in range(num_samples):
        audio_arr = rng.randn(16000).astype(np.float32)
        samples.append(
            {
                "id": f"sample_{i}",
                "audio": {
                    "path": None,
                    "array": audio_arr,
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
            }
        )
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
            "dataset",
            "id",
            "audio",
            "prosody_codebooks_idx",
            "timbre_vector",
            "chosen",
            "rejected",
            "rationale_chosen",
            "rationale_rejected",
            "emotion_label",
            "speaker_name",
            "speaker_gender",
            "speaker_age_context",
            "speaker_nationality",
            "transcript_with_tags",
            "bare_transcript",
            "cosine_similarity",
            "label",
        }
        actual = set(result.column_names)
        assert expected_columns == actual, (
            f"Missing: {expected_columns - actual}, Extra: {actual - expected_columns}"
        )

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

    def test_dataset_tag_applied(self):
        """dataset column defaults to 'nvtts-preference' if not present."""
        dataset = _make_mock_preference_dataset()
        processor = _get_processor()
        result = processor.process_dataset(dataset)
        for sample in result:
            assert sample["dataset"] == "nvtts-preference"
