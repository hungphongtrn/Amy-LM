"""Tests for reporting module - summary reporting for preprocessing pipeline.

This module tests the ProcessingSummary dataclass and generate_report function
which collect statistics during preprocessing and generate JSON reports.
"""

import json
import pytest
from pathlib import Path

from preprocessing.reporting import ProcessingSummary, generate_report


class TestProcessingSummary:
    """Test ProcessingSummary dataclass for tracking preprocessing statistics."""

    def test_summary_initializes_with_defaults(self):
        """ProcessingSummary should initialize with zero/empty defaults."""
        summary = ProcessingSummary()
        
        assert summary.total_processed == 0
        assert summary.total_failed == 0
        assert summary.content_frame_counts == []
        assert summary.prosody_frame_counts == []
        assert summary.acoustic_frame_counts == []
        assert summary.durations_seconds == []
        assert summary.failed_samples == []

    def test_add_processed_increments_total(self):
        """add_processed() should increment total_processed."""
        summary = ProcessingSummary()
        
        summary.add_processed(
            content_frames=25,
            prosody_frames=25,
            acoustic_frames=25,
            duration_sec=2.0
        )
        
        assert summary.total_processed == 1

    def test_add_processed_stores_frame_counts(self):
        """add_processed() should store frame counts in lists."""
        summary = ProcessingSummary()
        
        summary.add_processed(
            content_frames=25,
            prosody_frames=25,
            acoustic_frames=25,
            duration_sec=2.0
        )
        
        assert summary.content_frame_counts == [25]
        assert summary.prosody_frame_counts == [25]
        assert summary.acoustic_frame_counts == [25]
        assert summary.durations_seconds == [2.0]

    def test_add_processed_appends_multiple_samples(self):
        """add_processed() should append multiple samples."""
        summary = ProcessingSummary()
        
        summary.add_processed(content_frames=20, prosody_frames=20, acoustic_frames=20, duration_sec=1.5)
        summary.add_processed(content_frames=30, prosody_frames=30, acoustic_frames=30, duration_sec=2.5)
        
        assert summary.total_processed == 2
        assert summary.content_frame_counts == [20, 30]
        assert summary.durations_seconds == [1.5, 2.5]

    def test_add_failed_increments_total(self):
        """add_failed() should increment total_failed."""
        summary = ProcessingSummary()
        
        summary.add_failed(sample_id="sample_001", error="Encoding failed")
        
        assert summary.total_failed == 1

    def test_add_failed_stores_failure_info(self):
        """add_failed() should store failure details."""
        summary = ProcessingSummary()
        
        summary.add_failed(sample_id="sample_001", error="Encoding failed")
        
        assert len(summary.failed_samples) == 1
        assert summary.failed_samples[0]["id"] == "sample_001"
        assert summary.failed_samples[0]["error"] == "Encoding failed"

    def test_add_failed_appends_multiple_failures(self):
        """add_failed() should append multiple failures."""
        summary = ProcessingSummary()
        
        summary.add_failed(sample_id="sample_001", error="Error 1")
        summary.add_failed(sample_id="sample_002", error="Error 2")
        
        assert summary.total_failed == 2
        assert len(summary.failed_samples) == 2
        assert summary.failed_samples[0]["id"] == "sample_001"
        assert summary.failed_samples[1]["id"] == "sample_002"

    def test_to_dict_returns_dict(self):
        """to_dict() should return a dictionary."""
        summary = ProcessingSummary()
        
        result = summary.to_dict()
        
        assert isinstance(result, dict)

    def test_to_dict_computes_averages(self):
        """to_dict() should compute average frame counts."""
        summary = ProcessingSummary()
        summary.add_processed(content_frames=20, prosody_frames=25, acoustic_frames=30, duration_sec=1.5)
        summary.add_processed(content_frames=30, prosody_frames=35, acoustic_frames=40, duration_sec=2.5)

        result = summary.to_dict()

        assert result["avg_content_frames"] == 25.0
        assert result["avg_prosody_frames"] == 30.0
        assert result["avg_acoustic_frames"] == 35.0
        assert result["avg_duration_sec"] == 2.0

    def test_to_dict_averages_empty_lists(self):
        """to_dict() should handle empty lists with zero averages."""
        summary = ProcessingSummary()
        
        result = summary.to_dict()
        
        assert result["avg_content_frames"] == 0.0
        assert result["avg_prosody_frames"] == 0.0
        assert result["avg_acoustic_frames"] == 0.0
        assert result["avg_duration_sec"] == 0.0

    def test_to_dict_includes_totals(self):
        """to_dict() should include total processed and failed."""
        summary = ProcessingSummary()
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=2.0)
        summary.add_failed(sample_id="sample_001", error="Error")
        
        result = summary.to_dict()
        
        assert result["total_processed"] == 1
        assert result["total_failed"] == 1

    def test_to_dict_includes_duration_histogram(self):
        """to_dict() should include duration histogram with bins."""
        summary = ProcessingSummary()
        # Add durations in different bins
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=0.5)  # 0-1s
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=1.5)  # 1-2s
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=2.5)  # 2-5s
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=7.0)  # 5-10s
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=15.0)  # 10s+
        
        result = summary.to_dict()
        
        assert "duration_histogram" in result
        hist = result["duration_histogram"]
        assert hist["0-1s"] == 1
        assert hist["1-2s"] == 1
        assert hist["2-5s"] == 1
        assert hist["5-10s"] == 1
        assert hist["10s+"] == 1

    def test_to_dict_duration_histogram_empty(self):
        """Duration histogram should handle empty durations list."""
        summary = ProcessingSummary()
        
        result = summary.to_dict()
        
        hist = result["duration_histogram"]
        assert hist["0-1s"] == 0
        assert hist["1-2s"] == 0
        assert hist["2-5s"] == 0
        assert hist["5-10s"] == 0
        assert hist["10s+"] == 0

    def test_to_dict_includes_failed_samples(self):
        """to_dict() should include failed samples list."""
        summary = ProcessingSummary()
        summary.add_failed(sample_id="sample_001", error="Error 1")
        summary.add_failed(sample_id="sample_002", error="Error 2")
        
        result = summary.to_dict()
        
        assert "failed_samples" in result
        assert len(result["failed_samples"]) == 2
        assert result["failed_samples"][0] == {"id": "sample_001", "error": "Error 1"}


class TestGenerateReport:
    """Test generate_report function for writing JSON reports."""

    def test_generate_report_creates_file(self, tmp_path):
        """generate_report() should create a JSON file."""
        summary = ProcessingSummary()
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=2.0)

        output_path = tmp_path / "report.json"
        generate_report(summary, output_path)

        assert output_path.exists()

    def test_generate_report_creates_directories(self, tmp_path):
        """generate_report() should create parent directories if needed."""
        summary = ProcessingSummary()
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=2.0)

        output_path = tmp_path / "reports" / "batch1" / "summary.json"
        generate_report(summary, output_path)

        assert output_path.exists()

    def test_generate_report_outputs_valid_json(self, tmp_path):
        """generate_report() should output valid JSON."""
        summary = ProcessingSummary()
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=2.0)
        summary.add_failed(sample_id="sample_001", error="Test error")

        output_path = tmp_path / "report.json"
        generate_report(summary, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_generate_report_has_expected_structure(self, tmp_path):
        """generate_report() JSON should have all expected fields."""
        summary = ProcessingSummary()
        summary.add_processed(content_frames=20, prosody_frames=22, acoustic_frames=24, duration_sec=1.5)
        summary.add_processed(content_frames=30, prosody_frames=32, acoustic_frames=34, duration_sec=2.5)
        summary.add_failed(sample_id="sample_fail", error="Sample failed")

        output_path = tmp_path / "report.json"
        generate_report(summary, output_path)

        with open(output_path) as f:
            data = json.load(f)

        # Check all expected fields are present
        expected_fields = [
            "total_processed", "total_failed",
            "avg_content_frames", "avg_prosody_frames", "avg_acoustic_frames", "avg_duration_sec",
            "duration_histogram", "failed_samples"
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_generate_report_values_are_correct(self, tmp_path):
        """generate_report() should have correct computed values."""
        summary = ProcessingSummary()
        summary.add_processed(content_frames=20, prosody_frames=25, acoustic_frames=30, duration_sec=2.0)
        summary.add_processed(content_frames=30, prosody_frames=35, acoustic_frames=40, duration_sec=3.0)
        summary.add_failed(sample_id="fail_001", error="Encoding error")

        output_path = tmp_path / "report.json"
        generate_report(summary, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["total_processed"] == 2
        assert data["total_failed"] == 1
        assert data["avg_content_frames"] == 25.0
        assert data["avg_prosody_frames"] == 30.0
        assert data["avg_acoustic_frames"] == 35.0
        assert data["avg_duration_sec"] == 2.5
        assert len(data["failed_samples"]) == 1
        assert data["failed_samples"][0]["id"] == "fail_001"

    def test_generate_report_duration_histogram_correct(self, tmp_path):
        """generate_report() should have correct duration histogram."""
        summary = ProcessingSummary()
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=0.5)
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=1.5)
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=1.8)
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=6.0)

        output_path = tmp_path / "report.json"
        generate_report(summary, output_path)

        with open(output_path) as f:
            data = json.load(f)

        hist = data["duration_histogram"]
        assert hist["0-1s"] == 1
        assert hist["1-2s"] == 2
        assert hist["2-5s"] == 0
        assert hist["5-10s"] == 1
        assert hist["10s+"] == 0

    def test_generate_report_uses_indent(self, tmp_path):
        """generate_report() should output indented JSON for readability."""
        summary = ProcessingSummary()
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=2.0)

        output_path = tmp_path / "report.json"
        generate_report(summary, output_path)

        with open(output_path) as f:
            content = f.read()

        # Check for indentation (lines should have leading spaces)
        lines = content.split("\n")
        indented_lines = [line for line in lines if line.startswith("  ")]
        assert len(indented_lines) > 0, "JSON should be indented"

    def test_generate_report_overwrites_existing(self, tmp_path):
        """generate_report() should overwrite existing files."""
        output_path = tmp_path / "report.json"

        # Create existing file with old data
        with open(output_path, "w") as f:
            json.dump({"old": "data"}, f)

        summary = ProcessingSummary()
        summary.add_processed(content_frames=25, prosody_frames=25, acoustic_frames=25, duration_sec=2.0)
        generate_report(summary, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "total_processed" in data
        assert data["total_processed"] == 1


class TestProcessingSummaryIntegration:
    """Integration tests for ProcessingSummary with realistic usage patterns."""

    def test_typical_batch_processing(self):
        """Test ProcessingSummary with a realistic batch processing scenario."""
        summary = ProcessingSummary()
        
        # Simulate processing 100 samples with some failures
        for i in range(95):
            duration = 1.0 + (i % 10) * 0.5  # Varying durations
            summary.add_processed(
                content_frames=20 + (i % 5),
                prosody_frames=22 + (i % 5),
                acoustic_frames=24 + (i % 5),
                duration_sec=duration
            )
        
        # Simulate 5 failures
        for i in range(5):
            summary.add_failed(
                sample_id=f"sample_{i:03d}",
                error="FACodec encoding failed"
            )
        
        result = summary.to_dict()
        
        assert result["total_processed"] == 95
        assert result["total_failed"] == 5
        assert result["avg_content_frames"] == pytest.approx(22.0, abs=0.1)
        assert len(result["duration_histogram"]) == 5
        assert len(result["failed_samples"]) == 5

    def test_empty_summary_report(self, tmp_path):
        """Test that empty summary generates valid empty report."""
        summary = ProcessingSummary()
        
        output_path = tmp_path / "empty_report.json"
        generate_report(summary, output_path)
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert data["total_processed"] == 0
        assert data["total_failed"] == 0
        assert data["avg_content_frames"] == 0.0
        assert data["failed_samples"] == []
        assert all(v == 0 for v in data["duration_histogram"].values())
