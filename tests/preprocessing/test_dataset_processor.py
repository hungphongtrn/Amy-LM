"""Tests for DatasetProcessor - preprocessing pipeline for Amy-LM.

This module tests the DatasetProcessor which orchestrates loading HF datasets
and running FACodec on each sample to produce codebook indices.

Tests MUST work without internet access - we mock datasets.load_dataset().
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Generator

from preprocessing.facodec_encoder import FACodecEncoder
from preprocessing.dataset_processor import DatasetProcessor


class TestDatasetProcessor:
    """Test DatasetProcessor with synthetic data and mocked dependencies."""

    @pytest.fixture
    def facodec_encoder(self):
        """Create a FACodecEncoder in mock mode for testing."""
        return FACodecEncoder(device="cpu")

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create a temporary output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def processor(self, facodec_encoder, temp_output_dir):
        """Create a DatasetProcessor with mocked FACodec."""
        return DatasetProcessor(
            facodec=facodec_encoder,
            output_dir=temp_output_dir,
            device="cpu"
        )

    def _create_synthetic_dataset(self, num_samples: int = 5) -> dict:
        """Create a synthetic dataset dict for testing.
        
        Returns a dict that can be used with Dataset.from_dict().
        """
        # Generate synthetic audio at 16kHz
        samples = []
        ids = []
        
        for i in range(num_samples):
            # 2 seconds of audio at 16kHz = 32000 samples
            # Use a simple sine wave pattern for variety
            duration_sec = 2.0
            sample_rate = 16000
            num_samples_audio = int(duration_sec * sample_rate)
            
            # Generate sine wave at 440 Hz
            t = np.linspace(0, duration_sec, num_samples_audio)
            audio_array = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
            
            samples.append({
                "array": audio_array,
                "sampling_rate": sample_rate
            })
            ids.append(f"sample_{i:03d}")
        
        return {
            "id": ids,
            "audio": samples
        }

    def _create_mock_dataset(self, num_samples: int = 5):
        """Create a mock HF Dataset object."""
        from datasets import Dataset
        data = self._create_synthetic_dataset(num_samples)
        return Dataset.from_dict(data)

    def test_processor_initializes_correctly(self, facodec_encoder, temp_output_dir):
        """DatasetProcessor should initialize with required parameters."""
        processor = DatasetProcessor(
            facodec=facodec_encoder,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        assert processor.facodec == facodec_encoder
        assert processor.output_dir == temp_output_dir
        assert processor.device == "cpu"

    def test_process_dataset_returns_dataset_with_required_columns(self, processor):
        """process_dataset() should return HF Dataset with all required columns."""
        # Create synthetic data
        mock_data = self._create_synthetic_dataset(num_samples=3)
        
        # Mock datasets.load_dataset to return our synthetic data
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            # Process the dataset
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train",
                max_samples=3
            )
            
            # Verify result is a Dataset
            from datasets import Dataset
            assert isinstance(result, Dataset)
            
            # Verify all required columns are present
            required_columns = [
                "dataset", "id", "audio",
                "content_codebooks_idx", "prosody_codebooks_idx", "timbre_codebooks_idx"
            ]
            for col in required_columns:
                assert col in result.column_names, f"Missing column: {col}"

    def test_process_dataset_has_correct_dataset_field(self, processor):
        """The 'dataset' field should match the source dataset name."""
        mock_data = self._create_synthetic_dataset(num_samples=2)
        dataset_name = "my/test-dataset"
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name=dataset_name,
                split="train"
            )
            
            # All rows should have the correct dataset field
            for row in result:
                assert row["dataset"] == dataset_name

    def test_process_dataset_has_valid_indices(self, processor):
        """Indices should be valid integers in codebook range."""
        mock_data = self._create_synthetic_dataset(num_samples=2)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            for row in result:
                # Verify all indices are lists of integers
                for col in ["content_codebooks_idx", "prosody_codebooks_idx", "timbre_codebooks_idx"]:
                    indices = row[col]
                    assert isinstance(indices, list), f"{col} should be a list"
                    assert len(indices) > 0, f"{col} should not be empty"
                    
                    for idx in indices:
                        assert isinstance(idx, int), f"{col} should contain integers"
                        assert 0 <= idx < 2048, f"{col} index out of range"

    def test_process_dataset_has_correct_frame_counts(self, processor):
        """Frame counts should match expected from audio duration."""
        mock_data = self._create_synthetic_dataset(num_samples=2)
        # 2 seconds at 16kHz should produce ~25 frames at 12.5 Hz
        expected_frames = 25
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            for row in result:
                # Allow some tolerance (20-30 frames)
                for col in ["content_codebooks_idx", "prosody_codebooks_idx", "timbre_codebooks_idx"]:
                    frame_count = len(row[col])
                    assert 20 <= frame_count <= 30, \
                        f"Expected ~25 frames, got {frame_count} for {col}"

    def test_process_dataset_preserves_audio_field(self, processor):
        """Audio field should be preserved with array and sampling_rate."""
        mock_data = self._create_synthetic_dataset(num_samples=1)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            row = result[0]
            assert "audio" in row
            assert "array" in row["audio"]
            assert "sampling_rate" in row["audio"]
            assert row["audio"]["sampling_rate"] == 16000
            # HF datasets may convert arrays to lists, either is valid
            audio_array = row["audio"]["array"]
            assert isinstance(audio_array, (np.ndarray, list))
            assert len(audio_array) > 0

    def test_process_dataset_respects_max_samples(self, processor):
        """max_samples parameter should limit the number of samples processed."""
        mock_data = self._create_synthetic_dataset(num_samples=10)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train",
                max_samples=3
            )
            
            assert len(result) == 3

    def test_process_dataset_generates_ids_if_missing(self, processor):
        """If source dataset has no 'id' column, use row index."""
        # Create data without 'id' column
        mock_data = {
            "audio": self._create_synthetic_dataset(3)["audio"]
        }
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Should have generated IDs
            for i, row in enumerate(result):
                assert row["id"] is not None
                assert isinstance(row["id"], str)

    def test_process_dataset_continues_on_error(self, processor):
        """Processing should continue even if some samples fail."""
        # Create data that will cause an error on one sample
        mock_data = self._create_synthetic_dataset(num_samples=3)
        # Add an empty audio sample that should cause error
        mock_data["audio"][1] = {"array": np.array([], dtype=np.float32), "sampling_rate": 16000}
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            # Should not raise, should process the other samples
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Should have 2 samples (one failed)
            assert len(result) == 2

    def test_save_creates_parquet_file(self, processor, temp_output_dir):
        """save() should create a parquet file that can be loaded."""
        mock_data = self._create_synthetic_dataset(num_samples=2)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Save to disk
            repo_id = "my-org/processed-dataset"
            processor.save(result, repo_id, split="train")
            
            # Check file was created
            expected_path = temp_output_dir / repo_id / "train.parquet"
            assert expected_path.exists(), f"Expected file at {expected_path}"
            
            # Load it back and verify
            from datasets import Dataset
            loaded = Dataset.from_parquet(str(expected_path))
            
            # Verify all columns are present
            required_columns = [
                "dataset", "id", "audio",
                "content_codebooks_idx", "prosody_codebooks_idx", "timbre_codebooks_idx"
            ]
            for col in required_columns:
                assert col in loaded.column_names, f"Missing column after load: {col}"
            
            # Verify row count matches
            assert len(loaded) == len(result)

    def test_save_creates_directory_structure(self, processor, temp_output_dir):
        """save() should create directory structure for repo_id."""
        mock_data = self._create_synthetic_dataset(num_samples=1)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            repo_id = "org-name/dataset-name"
            processor.save(result, repo_id, split="train")
            
            # Should create nested directory
            expected_dir = temp_output_dir / repo_id
            assert expected_dir.exists()
            assert expected_dir.is_dir()

    def test_push_to_hub_calls_upload(self, processor):
        """push_to_hub() should call dataset.push_to_hub()."""
        mock_data = self._create_synthetic_dataset(num_samples=1)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Mock the push_to_hub method
            with patch.object(result, 'push_to_hub') as mock_push:
                repo_id = "my-org/my-dataset"
                processor.push_to_hub(result, repo_id)
                
                # Verify push_to_hub was called with correct repo_id
                mock_push.assert_called_once_with(repo_id)

    def test_process_dataset_without_mock(self, processor):
        """Test that process_dataset works without internet by using from_dict."""
        # This test verifies our mocking works - should not require internet
        mock_data = self._create_synthetic_dataset(num_samples=2)
        
        # Ensure no actual network calls happen
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Verify we got results
            assert len(result) == 2
            assert "content_codebooks_idx" in result.column_names

    def test_process_dataset_with_resampling(self, processor):
        """Should handle audio that needs resampling to 16kHz."""
        # Create audio at 22kHz that needs resampling
        duration_sec = 2.0
        sample_rate = 22050  # Not 16kHz
        num_samples_audio = int(duration_sec * sample_rate)
        
        t = np.linspace(0, duration_sec, num_samples_audio)
        audio_array = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        
        mock_data = {
            "id": ["sample_001"],
            "audio": [{
                "array": audio_array,
                "sampling_rate": sample_rate
            }]
        }
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Should still produce valid results
            assert len(result) == 1
            row = result[0]
            assert len(row["content_codebooks_idx"]) > 0
            # Output should have 16kHz sampling rate
            assert row["audio"]["sampling_rate"] == 16000