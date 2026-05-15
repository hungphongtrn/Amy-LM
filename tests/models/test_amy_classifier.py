"""Tests for AmyForProsodyClassification — end-to-end model assembly."""
import pytest
import torch
from src.models.amy_classifier import AmyForProsodyClassification


def make_prosody_codebook_vectors():
    """Mock FACodec prosody codebook vectors [1024, 8]."""
    return torch.randn(1024, 8)


class TestAmyForwardShape:
    """Verify forward pass produces correct output shapes."""

    @pytest.fixture
    def model(self):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device="cpu",
        )

    def test_forward_output_shape(self, model):
        """Forward pass should produce [B, 2] logits."""
        batch = 2
        audio = torch.randn(batch, 32000)
        prosody_indices = torch.randint(0, 1024, (batch, 1, 160))
        timbre_vector = torch.randn(batch, 256)

        logits = model(audio, prosody_indices, timbre_vector)
        assert logits.shape == (batch, 2)
        assert logits.dtype == torch.float32

    def test_single_sample_batch(self, model):
        """Should handle batch_size=1."""
        audio = torch.randn(1, 16000)
        prosody_indices = torch.randint(0, 1024, (1, 1, 80))
        timbre_vector = torch.randn(1, 256)

        logits = model(audio, prosody_indices, timbre_vector)
        assert logits.shape == (1, 2)
