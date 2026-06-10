import json
import os
import unittest.mock

import numpy as np
import pytest
from datasets import Dataset

from src.preprocessing.facodec_encoder import FACodecEncoder, FACodecStreams


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
        """Save to parquet and reload preserves columns, row count, and key values."""
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

        # Value-level integrity
        for i, (orig, rel) in enumerate(zip(result, reloaded)):
            assert rel["id"] == orig["id"]
            assert rel["chosen"] == orig["chosen"]
            assert rel["rejected"] == orig["rejected"]
            assert rel["label"] == -1
            assert rel["dataset"] == "nvtts-preference"
            assert abs(rel["cosine_similarity"] - orig["cosine_similarity"]) < 1e-6
            assert len(rel["prosody_codebooks_idx"]) == len(orig["prosody_codebooks_idx"])
            assert len(rel["timbre_vector"]) == 256

    def test_cardinality_mismatch_raises(self):
        """ValueError if encode_batch returns different count than inputs."""
        import torch
        from src.preprocessing.preference_dataset_processor import PreferenceDatasetProcessor

        dataset = _make_mock_preference_dataset(2)
        encoder = FACodecEncoder(device="cpu", force_mock=True)
        processor = PreferenceDatasetProcessor(encoder, batch_size=2)

        real_encode_batch = encoder.encode_batch

        def short_encode_batch(audios):
            result = real_encode_batch(audios)
            return result[:1]  # return 1 stream for 2 inputs

        with unittest.mock.patch.object(encoder, "encode_batch", side_effect=short_encode_batch):
            with pytest.raises(ValueError, match="encode_batch returned"):
                processor.process_dataset(dataset)

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


def _make_adversarial_jsonl(tmp_path, pairs_data: list[dict]) -> str:
    jsonl_path = tmp_path / "pairs.jsonl"
    with open(jsonl_path, "w") as f:
        for p in pairs_data:
            f.write(json.dumps(p) + "\n")
    return str(jsonl_path)


class TestAdversarialDatasetProcessor:
    def test_output_columns(self, tmp_path):
        """Adversarial output has all standard + adversarial columns."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "sample_0",
                "chosen": "adv chosen 0",
                "rejected": "adv rejected 0",
                "strategy": "test strategy",
                "emotion_label": "happy",
                "inverse_emotion": "sad",
                "judge_fidelity_chosen": 5,
                "judge_fidelity_rejected": 4,
                "judge_ambiguity_chosen": 2,
                "judge_ambiguity_rejected": 1,
                "generation_attempts": 1,
            },
        ])
        nvtts = _make_mock_preference_dataset(1)
        processor = _get_processor()
        result = processor.process_adversarial_dataset(jsonl_path, nvtts)

        assert "inverse_emotion" in result.column_names
        assert "strategy" in result.column_names
        assert "judge_fidelity_chosen" in result.column_names
        assert "judge_ambiguity_chosen" in result.column_names
        assert "generation_attempts" in result.column_names
        assert "chosen" in result.column_names
        assert "rejected" in result.column_names

    def test_chosen_rejected_from_adversarial(self, tmp_path):
        """Chosen/rejected come from adversarial pairs, not original NVTTS."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "sample_0",
                "chosen": "ADVERSARIAL CHOSEN",
                "rejected": "ADVERSARIAL REJECTED",
                "strategy": "s",
                "emotion_label": "happy",
                "inverse_emotion": "sad",
                "judge_fidelity_chosen": 5,
                "judge_fidelity_rejected": 5,
                "judge_ambiguity_chosen": 1,
                "judge_ambiguity_rejected": 1,
                "generation_attempts": 1,
            },
        ])
        nvtts = _make_mock_preference_dataset(1)
        processor = _get_processor()
        result = processor.process_adversarial_dataset(jsonl_path, nvtts)

        assert result[0]["chosen"] == "ADVERSARIAL CHOSEN"
        assert result[0]["rejected"] == "ADVERSARIAL REJECTED"

    def test_adversarial_metadata_preserved(self, tmp_path):
        """Judge scores, inverse emotion, strategy, generation_attempts survive."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "sample_0",
                "chosen": "c",
                "rejected": "r",
                "strategy": "my strategy",
                "emotion_label": "angry",
                "inverse_emotion": "neutral",
                "judge_fidelity_chosen": 4,
                "judge_fidelity_rejected": 5,
                "judge_ambiguity_chosen": 2,
                "judge_ambiguity_rejected": 1,
                "generation_attempts": 3,
            },
        ])
        nvtts = _make_mock_preference_dataset(1)
        processor = _get_processor()
        result = processor.process_adversarial_dataset(jsonl_path, nvtts)

        sample = result[0]
        assert sample["strategy"] == "my strategy"
        assert sample["emotion_label"] == "angry"
        assert sample["inverse_emotion"] == "neutral"
        assert sample["judge_fidelity_chosen"] == 4
        assert sample["judge_fidelity_rejected"] == 5
        assert sample["judge_ambiguity_chosen"] == 2
        assert sample["judge_ambiguity_rejected"] == 1
        assert sample["generation_attempts"] == 3

    def test_no_matching_pairs_raises(self, tmp_path):
        """ValueError when no JSONL ids match NVTTS ids."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "no_such_id",
                "chosen": "c",
                "rejected": "r",
                "strategy": "",
                "emotion_label": "",
                "inverse_emotion": "",
                "judge_fidelity_chosen": 0,
                "judge_fidelity_rejected": 0,
                "judge_ambiguity_chosen": 0,
                "judge_ambiguity_rejected": 0,
                "generation_attempts": 0,
            },
        ])
        nvtts = _make_mock_preference_dataset(1)
        processor = _get_processor()
        with pytest.raises(ValueError, match="No matching adversarial pairs"):
            processor.process_adversarial_dataset(jsonl_path, nvtts)

    def test_cosine_similarity_is_zero(self, tmp_path):
        """Adversarial pairs have cosine_similarity=0.0 (always passes filter)."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "sample_0",
                "chosen": "c",
                "rejected": "r",
                "strategy": "",
                "emotion_label": "",
                "inverse_emotion": "",
                "judge_fidelity_chosen": 5,
                "judge_fidelity_rejected": 5,
                "judge_ambiguity_chosen": 1,
                "judge_ambiguity_rejected": 1,
                "generation_attempts": 1,
            },
        ])
        nvtts = _make_mock_preference_dataset(1)
        processor = _get_processor()
        result = processor.process_adversarial_dataset(jsonl_path, nvtts)
        assert result[0]["cosine_similarity"] == 0.0

    def test_label_is_neg_one(self, tmp_path):
        """Label is -1 for all adversarial samples."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "sample_0",
                "chosen": "c",
                "rejected": "r",
                "strategy": "",
                "emotion_label": "",
                "inverse_emotion": "",
                "judge_fidelity_chosen": 0,
                "judge_fidelity_rejected": 0,
                "judge_ambiguity_chosen": 0,
                "judge_ambiguity_rejected": 0,
                "generation_attempts": 0,
            },
        ])
        nvtts = _make_mock_preference_dataset(1)
        processor = _get_processor()
        result = processor.process_adversarial_dataset(jsonl_path, nvtts)
        assert result[0]["label"] == -1

    def test_only_matching_ids_processed(self, tmp_path):
        """Only NVTTS samples with matching adversarial pair ids are included."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "sample_1",
                "chosen": "c1",
                "rejected": "r1",
                "strategy": "",
                "emotion_label": "",
                "inverse_emotion": "",
                "judge_fidelity_chosen": 5,
                "judge_fidelity_rejected": 5,
                "judge_ambiguity_chosen": 0,
                "judge_ambiguity_rejected": 0,
                "generation_attempts": 1,
            },
        ])
        nvtts = _make_mock_preference_dataset(3)
        processor = _get_processor()
        result = processor.process_adversarial_dataset(jsonl_path, nvtts)
        assert len(result) == 1
        assert result[0]["id"] == "sample_1"

    def test_has_prosody_and_timbre(self, tmp_path):
        """Prosody and timbre streams are present with correct shapes."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "sample_0",
                "chosen": "c",
                "rejected": "r",
                "strategy": "",
                "emotion_label": "",
                "inverse_emotion": "",
                "judge_fidelity_chosen": 5,
                "judge_fidelity_rejected": 5,
                "judge_ambiguity_chosen": 0,
                "judge_ambiguity_rejected": 0,
                "generation_attempts": 1,
            },
        ])
        nvtts = _make_mock_preference_dataset(1)
        processor = _get_processor()
        result = processor.process_adversarial_dataset(jsonl_path, nvtts)
        assert len(result[0]["prosody_codebooks_idx"]) > 0
        assert len(result[0]["timbre_vector"]) == 256

    def test_rationale_fields_are_empty(self, tmp_path):
        """Rationale fields are empty for adversarial pairs (not applicable)."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "sample_0",
                "chosen": "c",
                "rejected": "r",
                "strategy": "",
                "emotion_label": "",
                "inverse_emotion": "",
                "judge_fidelity_chosen": 5,
                "judge_fidelity_rejected": 5,
                "judge_ambiguity_chosen": 0,
                "judge_ambiguity_rejected": 0,
                "generation_attempts": 1,
            },
        ])
        nvtts = _make_mock_preference_dataset(1)
        processor = _get_processor()
        result = processor.process_adversarial_dataset(jsonl_path, nvtts)
        assert result[0]["rationale_chosen"] == ""
        assert result[0]["rationale_rejected"] == ""

    def test_dataset_tag_default(self, tmp_path):
        """Default dataset tag for adversarial is 'nvtts-adversarial'."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "sample_0",
                "chosen": "c",
                "rejected": "r",
                "strategy": "",
                "emotion_label": "",
                "inverse_emotion": "",
                "judge_fidelity_chosen": 5,
                "judge_fidelity_rejected": 5,
                "judge_ambiguity_chosen": 0,
                "judge_ambiguity_rejected": 0,
                "generation_attempts": 1,
            },
        ])
        nvtts = _make_mock_preference_dataset(1)
        processor = _get_processor()
        result = processor.process_adversarial_dataset(jsonl_path, nvtts)
        assert result[0]["dataset"] == "nvtts-adversarial"

    def test_skips_content_acoustic(self, tmp_path):
        """Content and acoustic codebooks are NOT in adversarial output."""
        jsonl_path = _make_adversarial_jsonl(tmp_path, [
            {
                "id": "sample_0",
                "chosen": "c",
                "rejected": "r",
                "strategy": "",
                "emotion_label": "",
                "inverse_emotion": "",
                "judge_fidelity_chosen": 5,
                "judge_fidelity_rejected": 5,
                "judge_ambiguity_chosen": 0,
                "judge_ambiguity_rejected": 0,
                "generation_attempts": 1,
            },
        ])
        nvtts = _make_mock_preference_dataset(1)
        processor = _get_processor()
        result = processor.process_adversarial_dataset(jsonl_path, nvtts)
        assert "content_codebooks_idx" not in result.column_names
        assert "acoustic_codebooks_idx" not in result.column_names
