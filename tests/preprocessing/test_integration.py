"""Integration tests for Amy-LM preprocessing pipeline.

This module validates the end-to-end preprocessing workflow:
- FACodecEncoder + DatasetProcessor integration
- Dataset round-trip (save → load)
- Full pipeline with reporting

All tests work offline using mock data and mocked datasets.load_dataset().
"""

import json
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch

from preprocessing.facodec_encoder import FACodecEncoder
from preprocessing.dataset_processor import DatasetProcessor
from preprocessing.reporting import ProcessingSummary, generate_report


class TestEndToEndPipeline:
    """Test 1: End-to-end pipeline with DatasetProcessor."""

    @pytest.fixture
    def facodec_encoder(self):
        """Create FACodecEncoder in mock mode."""
        encoder = FACodecEncoder(device="cpu", force_mock=True)
        # Verify we're in mock mode for offline testing
        assert encoder._mock is True
        return encoder

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create a temporary output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def dataset_processor(self, facodec_encoder, temp_output_dir):
        """Create DatasetProcessor with mocked FACodec."""
        return DatasetProcessor(
            facodec=facodec_encoder,
            output_dir=temp_output_dir,
            device="cpu"
        )

    def _create_mock_audio_dataset(self, num_samples: int = 3, duration_sec: float = 2.0) -> dict:
        """Create synthetic audio dataset for mocking.
        
        Args:
            num_samples: Number of audio samples to generate
            duration_sec: Duration of each audio sample in seconds
            
        Returns:
            Dictionary suitable for Dataset.from_dict()
        """
        samples = []
        ids = []
        
        sample_rate = 16000  # 16kHz as expected by FACodec
        num_samples_audio = int(duration_sec * sample_rate)
        
        for i in range(num_samples):
            # Generate sine wave at 440 Hz with slight variation per sample
            t = np.linspace(0, duration_sec, num_samples_audio)
            frequency = 440 + (i * 10)  # Slight frequency variation
            audio_array = (np.sin(2 * np.pi * frequency * t) * 0.5).astype(np.float32)
            
            samples.append({
                "array": audio_array,
                "sampling_rate": sample_rate
            })
            ids.append(f"sample_{i:03d}")
        
        return {
            "id": ids,
            "audio": samples
        }

    def test_pipeline_outputs_all_required_columns(self, dataset_processor):
        """End-to-end: Output dataset has all 6 required columns with correct dtypes."""
        # Create synthetic audio data (2 seconds each)
        mock_data = self._create_mock_audio_dataset(num_samples=3, duration_sec=2.0)
        dataset_name = "test/integration-dataset"
        
        # Mock datasets.load_dataset to return synthetic data
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            # Run the pipeline
            result = dataset_processor.process_dataset(
                dataset_name=dataset_name,
                split="train",
                max_samples=3
            )
            
            # Verify all 7 required columns are present
            required_columns = [
                "dataset",      # Source dataset name
                "id",           # Unique sample ID
                "audio",        # Audio dict with array and sampling_rate
                "prosody_codebooks_idx",   # Prosody codebook indices [T80]
                "content_codebooks_idx",   # Content codebook indices [2, T80]
                "acoustic_codebooks_idx",  # Acoustic codebook indices [3, T80]
                "timbre_vector"            # Timbre vector [256] float32
            ]
            
            for col in required_columns:
                assert col in result.column_names, f"Missing required column: {col}"

    def test_pipeline_indices_are_integers_in_valid_range(self, dataset_processor):
        """End-to-end: Indices are integers within valid codebook range (0-1023)."""
        mock_data = self._create_mock_audio_dataset(num_samples=2, duration_sec=2.0)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = dataset_processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Verify all indices are valid integers in range [0, 1023]
            for row in result:
                # Prosody: flat list [T80]
                prosody = row["prosody_codebooks_idx"]
                assert isinstance(prosody, list), "prosody should be a list"
                for idx in prosody:
                    assert isinstance(idx, int), "prosody should contain integers"
                    assert 0 <= idx < 1024, f"prosody index {idx} out of range"
                
                # Content: nested list [2, T80]
                content = row["content_codebooks_idx"]
                assert isinstance(content, list) and len(content) == 2
                for cb_idx in range(2):
                    for idx in content[cb_idx]:
                        assert isinstance(idx, int)
                        assert 0 <= idx < 1024, f"content[{cb_idx}] index out of range"
                
                # Acoustic: nested list [3, T80]
                acoustic = row["acoustic_codebooks_idx"]
                assert isinstance(acoustic, list) and len(acoustic) == 3
                for cb_idx in range(3):
                    for idx in acoustic[cb_idx]:
                        assert isinstance(idx, int)
                        assert 0 <= idx < 1024, f"acoustic[{cb_idx}] index out of range"
                
                # Timbre vector: list of 256 floats
                timbre = row["timbre_vector"]
                assert isinstance(timbre, list) and len(timbre) == 256
                for val in timbre:
                    assert isinstance(val, float), "timbre_vector should contain floats"

    def test_pipeline_dataset_field_matches_source(self, dataset_processor):
        """End-to-end: The 'dataset' field matches the source dataset name."""
        mock_data = self._create_mock_audio_dataset(num_samples=2)
        dataset_name = "my-org/my-test-dataset"
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = dataset_processor.process_dataset(
                dataset_name=dataset_name,
                split="train"
            )
            
            # All rows should have the correct dataset field
            for row in result:
                assert row["dataset"] == dataset_name, \
                    f"Expected dataset='{dataset_name}', got '{row['dataset']}'"

    def test_pipeline_frame_counts_are_correct(self, dataset_processor):
        """End-to-end: Frame counts match expected from 2-second audio at 80 Hz."""
        duration_sec = 2.0
        mock_data = self._create_mock_audio_dataset(num_samples=2, duration_sec=duration_sec)
        
        # Expected: 2 seconds at 16kHz with hop_size=200 -> 32000/200 = 160 frames at 80 Hz
        expected_frames = 160
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            from datasets import Dataset
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = dataset_processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            for row in result:
                # Prosody: flat list
                prosody_frames = len(row["prosody_codebooks_idx"])
                assert prosody_frames == expected_frames, \
                    f"Expected {expected_frames} prosody frames, got {prosody_frames}"
                
                # Content: nested list [2, T80], check first codebook
                content = row["content_codebooks_idx"]
                content_frames = len(content[0])
                assert content_frames == expected_frames, \
                    f"Expected {expected_frames} content frames, got {content_frames}"
                
                # Acoustic: nested list [3, T80], check first codebook
                acoustic = row["acoustic_codebooks_idx"]
                acoustic_frames = len(acoustic[0])
                assert acoustic_frames == expected_frames, \
                    f"Expected {expected_frames} acoustic frames, got {acoustic_frames}"
                
                # Timbre vector: always 256 elements (utterance-level)
                timbre_len = len(row["timbre_vector"])
                assert timbre_len == 256, f"Expected 256 timbre values, got {timbre_len}"


class TestDatasetRoundTrip:
    """Test 2: Dataset round-trip (save → load)."""

    @pytest.fixture
    def facodec_encoder(self):
        """Create FACodecEncoder in mock mode."""
        return FACodecEncoder(device="cpu")

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create a temporary output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def dataset_processor(self, facodec_encoder, temp_output_dir):
        """Create DatasetProcessor."""
        return DatasetProcessor(
            facodec=facodec_encoder,
            output_dir=temp_output_dir,
            device="cpu"
        )

    def _create_mock_audio_dataset(self, num_samples: int = 3) -> dict:
        """Create synthetic audio dataset."""
        samples = []
        ids = []
        
        sample_rate = 16000
        duration_sec = 2.0
        num_samples_audio = int(duration_sec * sample_rate)
        
        for i in range(num_samples):
            t = np.linspace(0, duration_sec, num_samples_audio)
            audio_array = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
            
            samples.append({
                "array": audio_array,
                "sampling_rate": sample_rate
            })
            ids.append(f"sample_{i:03d}")
        
        return {"id": ids, "audio": samples}

    def test_dataset_save_and_load_preserves_columns(self, dataset_processor, temp_output_dir):
        """Round-trip: Dataset can be saved and loaded with same columns."""
        from datasets import Dataset
        
        # Create and process dataset
        mock_data = self._create_mock_audio_dataset(num_samples=3)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            processed = dataset_processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Save to disk
            repo_id = "test-org/processed-dataset"
            saved_path = dataset_processor.save(processed, repo_id, split="train")
            
            # Verify file was created
            assert saved_path.exists(), f"Saved file should exist at {saved_path}"
            
            # Load back
            loaded = Dataset.from_parquet(str(saved_path))
            
            # Verify all columns are preserved
            original_columns = set(processed.column_names)
            loaded_columns = set(loaded.column_names)
            
            assert original_columns == loaded_columns, \
                f"Column mismatch: original={original_columns}, loaded={loaded_columns}"

    def test_dataset_save_and_load_preserves_row_count(self, dataset_processor, temp_output_dir):
        """Round-trip: Dataset preserves row count after save and load."""
        from datasets import Dataset
        
        mock_data = self._create_mock_audio_dataset(num_samples=5)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            processed = dataset_processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            original_count = len(processed)
            
            # Save and load
            repo_id = "test-org/processed-dataset"
            saved_path = dataset_processor.save(processed, repo_id, split="train")
            loaded = Dataset.from_parquet(str(saved_path))
            
            # Verify row count matches
            assert len(loaded) == original_count, \
                f"Row count mismatch: original={original_count}, loaded={len(loaded)}"

    def test_dataset_save_and_load_preserves_indices(self, dataset_processor, temp_output_dir):
        """Round-trip: Codebook indices are preserved after save and load."""
        from datasets import Dataset
        
        mock_data = self._create_mock_audio_dataset(num_samples=2)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            processed = dataset_processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Get original indices from first sample
            original_content = list(processed[0]["content_codebooks_idx"])
            
            # Save and load
            repo_id = "test-org/processed-dataset"
            saved_path = dataset_processor.save(processed, repo_id, split="train")
            loaded = Dataset.from_parquet(str(saved_path))
            
            # Verify indices match
            loaded_content = list(loaded[0]["content_codebooks_idx"])
            assert loaded_content == original_content, \
                f"Indices mismatch: original={original_content[:5]}..., loaded={loaded_content[:5]}..."


class TestFullPipelineWithReporting:
    """Test 3: Full pipeline with reporting."""

    @pytest.fixture
    def facodec_encoder(self):
        """Create FACodecEncoder in mock mode."""
        return FACodecEncoder(device="cpu")

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create a temporary output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def dataset_processor(self, facodec_encoder, temp_output_dir):
        """Create DatasetProcessor."""
        return DatasetProcessor(
            facodec=facodec_encoder,
            output_dir=temp_output_dir,
            device="cpu"
        )

    def _create_mock_audio_dataset(self, num_samples: int = 3) -> dict:
        """Create synthetic audio dataset."""
        samples = []
        ids = []
        
        sample_rate = 16000
        duration_sec = 2.0
        num_samples_audio = int(duration_sec * sample_rate)
        
        for i in range(num_samples):
            t = np.linspace(0, duration_sec, num_samples_audio)
            audio_array = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
            
            samples.append({
                "array": audio_array,
                "sampling_rate": sample_rate
            })
            ids.append(f"sample_{i:03d}")
        
        return {"id": ids, "audio": samples}

    def test_pipeline_generates_report_with_expected_structure(self, dataset_processor, temp_output_dir):
        """Full pipeline: Report has expected structure with all fields."""
        from datasets import Dataset
        
        mock_data = self._create_mock_audio_dataset(num_samples=3)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            # Process dataset
            result = dataset_processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Create ProcessingSummary from results
            summary = ProcessingSummary()
            
            for row in result:
                # Calculate duration from audio array
                audio_array = row["audio"]["array"]
                sampling_rate = row["audio"]["sampling_rate"]
                duration_sec = len(audio_array) / sampling_rate

                # Get frame counts (prosody is flat, content/acoustic are nested)
                content_frames = len(row["content_codebooks_idx"][0]) if row["content_codebooks_idx"] else 0
                prosody_frames = len(row["prosody_codebooks_idx"])
                acoustic_frames = len(row["acoustic_codebooks_idx"][0]) if row["acoustic_codebooks_idx"] else 0

                summary.add_processed(
                    content_frames=content_frames,
                    prosody_frames=prosody_frames,
                    acoustic_frames=acoustic_frames,
                    duration_sec=duration_sec
                )
            
            # Generate report
            report_path = temp_output_dir / "processing_report.json"
            generate_report(summary, report_path)
            
            # Verify report was created
            assert report_path.exists(), "Report file should exist"
            
            # Load and verify structure
            with open(report_path) as f:
                report = json.load(f)
            
            # Verify all expected fields are present
            expected_fields = [
                "total_processed",
                "total_failed",
                "avg_content_frames",
                "avg_prosody_frames",
                "avg_acoustic_frames",
                "avg_duration_sec",
                "duration_histogram",
                "failed_samples"
            ]
            
            for field in expected_fields:
                assert field in report, f"Missing field in report: {field}"

    def test_pipeline_report_has_correct_values(self, dataset_processor, temp_output_dir):
        """Full pipeline: Report values match processed data."""
        from datasets import Dataset
        
        mock_data = self._create_mock_audio_dataset(num_samples=3)
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = dataset_processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Build summary from processed data
            summary = ProcessingSummary()
            total_duration = 0.0
            total_content_frames = 0
            
            for row in result:
                audio_array = row["audio"]["array"]
                sampling_rate = row["audio"]["sampling_rate"]
                duration_sec = len(audio_array) / sampling_rate
                total_duration += duration_sec

                # Get frame counts (prosody is flat, content/acoustic are nested)
                content_frames = len(row["content_codebooks_idx"][0]) if row["content_codebooks_idx"] else 0
                total_content_frames += content_frames
                prosody_frames = len(row["prosody_codebooks_idx"])
                acoustic_frames = len(row["acoustic_codebooks_idx"][0]) if row["acoustic_codebooks_idx"] else 0

                summary.add_processed(
                    content_frames=content_frames,
                    prosody_frames=prosody_frames,
                    acoustic_frames=acoustic_frames,
                    duration_sec=duration_sec
                )
            
            # Generate report
            report_path = temp_output_dir / "processing_report.json"
            generate_report(summary, report_path)
            
            # Load and verify values
            with open(report_path) as f:
                report = json.load(f)
            
            expected_avg_duration = total_duration / len(result)
            expected_avg_content = total_content_frames / len(result)
            
            assert report["total_processed"] == len(result), \
                f"Expected {len(result)} processed, got {report['total_processed']}"
            assert report["avg_duration_sec"] == pytest.approx(expected_avg_duration, abs=0.01), \
                f"Expected avg_duration ~{expected_avg_duration}, got {report['avg_duration_sec']}"
            assert report["avg_content_frames"] == pytest.approx(expected_avg_content, abs=0.1), \
                f"Expected avg_content_frames ~{expected_avg_content}, got {report['avg_content_frames']}"

    def test_pipeline_report_with_failures(self, dataset_processor, temp_output_dir):
        """Full pipeline: Report correctly captures failed samples."""
        from datasets import Dataset
        
        # Create dataset with one sample that will fail (empty audio)
        mock_data = self._create_mock_audio_dataset(num_samples=3)
        mock_data["audio"][1] = {"array": np.array([], dtype=np.float32), "sampling_rate": 16000}
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = dataset_processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            # Build summary with failures
            summary = ProcessingSummary()
            
            # Add successful samples
            for row in result:
                audio_array = row["audio"]["array"]
                sampling_rate = row["audio"]["sampling_rate"]
                duration_sec = len(audio_array) / sampling_rate

                # Get frame counts (prosody is flat, content/acoustic are nested)
                content_frames = len(row["content_codebooks_idx"][0]) if row["content_codebooks_idx"] else 0
                prosody_frames = len(row["prosody_codebooks_idx"])
                acoustic_frames = len(row["acoustic_codebooks_idx"][0]) if row["acoustic_codebooks_idx"] else 0

                summary.add_processed(
                    content_frames=content_frames,
                    prosody_frames=prosody_frames,
                    acoustic_frames=acoustic_frames,
                    duration_sec=duration_sec
                )

            # Add the failed sample
            summary.add_failed(sample_id="sample_001", error="Empty audio array")
            
            # Generate report
            report_path = temp_output_dir / "processing_report.json"
            generate_report(summary, report_path)
            
            # Load and verify
            with open(report_path) as f:
                report = json.load(f)
            
            assert report["total_processed"] == 2, \
                f"Expected 2 processed, got {report['total_processed']}"
            assert report["total_failed"] == 1, \
                f"Expected 1 failed, got {report['total_failed']}"
            assert len(report["failed_samples"]) == 1, \
                f"Expected 1 failed sample entry, got {len(report['failed_samples'])}"
            assert report["failed_samples"][0]["id"] == "sample_001", \
                f"Expected failed sample ID 'sample_001', got {report['failed_samples'][0]['id']}"


class TestIntegrationEdgeCases:
    """Additional edge case integration tests."""

    @pytest.fixture
    def facodec_encoder(self):
        return FACodecEncoder(device="cpu")

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def dataset_processor(self, facodec_encoder, temp_output_dir):
        return DatasetProcessor(
            facodec=facodec_encoder,
            output_dir=temp_output_dir,
            device="cpu"
        )

    def test_single_sample_pipeline(self, dataset_processor):
        """Pipeline works with single sample."""
        from datasets import Dataset
        
        samples = []
        ids = []
        
        sample_rate = 16000
        duration_sec = 2.0
        num_samples_audio = int(duration_sec * sample_rate)
        
        t = np.linspace(0, duration_sec, num_samples_audio)
        audio_array = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        
        samples.append({"array": audio_array, "sampling_rate": sample_rate})
        ids.append("sample_000")
        
        mock_data = {"id": ids, "audio": samples}
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = dataset_processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            assert len(result) == 1
            assert "content_codebooks_idx" in result.column_names
            # content_codebooks_idx is now nested [2, T80]
            assert len(result[0]["content_codebooks_idx"]) == 2  # 2 codebooks
            assert len(result[0]["content_codebooks_idx"][0]) > 0  # Has frames

    def test_varying_duration_pipeline(self, dataset_processor):
        """Pipeline handles varying audio durations."""
        from datasets import Dataset
        
        samples = []
        ids = []
        
        sample_rate = 16000
        durations = [1.0, 2.0, 3.0]  # Different durations
        
        for i, duration_sec in enumerate(durations):
            num_samples_audio = int(duration_sec * sample_rate)
            t = np.linspace(0, duration_sec, num_samples_audio)
            audio_array = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
            
            samples.append({"array": audio_array, "sampling_rate": sample_rate})
            ids.append(f"sample_{i:03d}")
        
        mock_data = {"id": ids, "audio": samples}
        
        with patch("preprocessing.dataset_processor.load_dataset") as mock_load:
            mock_dataset = Dataset.from_dict(mock_data)
            mock_load.return_value = mock_dataset
            
            result = dataset_processor.process_dataset(
                dataset_name="test/dataset",
                split="train"
            )
            
            assert len(result) == 3

            # Verify frame counts scale with duration
            # content_codebooks_idx is nested [2, T80], so check inner list length
            frame_counts = [len(row["content_codebooks_idx"][0]) for row in result]
            # Longer audio should have more frames (roughly proportional)
            assert frame_counts[0] < frame_counts[1] < frame_counts[2], \
                f"Frame counts should increase with duration: {frame_counts}"
