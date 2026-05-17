"""Tests for FACodec codebook vector extraction utilities."""
import os
import tempfile
import torch
import pytest
from src.models.codebook_utils import load_prosody_codebook_vectors


class TestLoadProosityCodebookVectors:
    """Tests for loading prosody codebook vectors from FACodec decoder checkpoint."""

    def test_extracts_correct_shape_from_mock_checkpoint(self):
        """Load vectors from a mock checkpoint and verify shape [1024, 8]."""
        mock_state = {"quantizer.0.layers.0._codebook.weight": torch.randn(1024, 8)}
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            torch.save(mock_state, f.name)
            checkpoint_path = f.name

        try:
            vectors = load_prosody_codebook_vectors(checkpoint_path)
            assert vectors.shape == (1024, 8)
            assert vectors.dtype == torch.float32
            assert torch.equal(vectors, mock_state["quantizer.0.layers.0._codebook.weight"])
        finally:
            os.unlink(checkpoint_path)

    def test_raises_if_checkpoint_not_found(self):
        """Should raise FileNotFoundError for missing checkpoint."""
        with pytest.raises(FileNotFoundError):
            load_prosody_codebook_vectors("/nonexistent/path/checkpoint.bin")

    def test_raises_if_key_missing(self):
        """Should raise KeyError if quantizer key is missing from state_dict."""
        mock_state = {"some.other.key": torch.randn(10, 10)}
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            torch.save(mock_state, f.name)
            checkpoint_path = f.name

        try:
            with pytest.raises(KeyError, match="quantizer.0.layers.0._codebook.weight"):
                load_prosody_codebook_vectors(checkpoint_path)
        finally:
            os.unlink(checkpoint_path)
