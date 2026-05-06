"""Tests for TemporalPool — compresses embedded prosody 80 Hz → 12.5 Hz."""
import pytest
import torch
from src.models.pooling import TemporalPool

INPUT_RATE = 80.0
OUTPUT_RATE = 12.5
EMBED_DIM = 2560


def make_prosody_sequence(batch=2, duration_sec=2.0):
    """Create synthetic embedded prosody at 80 Hz."""
    num_frames = int(duration_sec * INPUT_RATE)
    return torch.randn(batch, num_frames, EMBED_DIM)


class TestTemporalPool:
    @pytest.fixture
    def pool(self):
        return TemporalPool(input_rate=INPUT_RATE, output_rate=OUTPUT_RATE)

    def test_output_rate_correct_2sec(self, pool):
        x = make_prosody_sequence(duration_sec=2.0)
        out = pool(x)
        expected_frames = int(2.0 * OUTPUT_RATE)
        assert out.shape[1] == expected_frames

    def test_output_rate_correct_5sec(self, pool):
        x = make_prosody_sequence(batch=1, duration_sec=5.0)
        out = pool(x)
        expected_frames = int(5.0 * OUTPUT_RATE)
        assert out.shape[1] == expected_frames

    def test_output_rate_correct_0_64sec(self, pool):
        """Short input: 0.64 sec = 51 frames at 80 Hz → 8 frames at 12.5 Hz."""
        x = make_prosody_sequence(batch=1, duration_sec=0.64)
        out = pool(x)
        expected_frames = int(0.64 * OUTPUT_RATE)
        assert out.shape[1] == expected_frames

    def test_embed_dim_preserved(self, pool):
        x = make_prosody_sequence()
        out = pool(x)
        assert out.shape[2] == EMBED_DIM

    def test_batch_dim_preserved(self, pool):
        x = make_prosody_sequence(batch=4, duration_sec=1.5)
        out = pool(x)
        assert out.shape[0] == 4

    def test_handles_non_divisible_frame_count(self, pool):
        """Pool 127 frames at 80 Hz → target = int(127 * 12.5 / 80) = 19."""
        x = torch.randn(2, 127, EMBED_DIM)
        out = pool(x)
        expected = int(127 * OUTPUT_RATE / INPUT_RATE)
        assert out.shape[1] == expected

    def test_pooled_values_bounded(self, pool):
        """Mean-pooled values should stay within input range."""
        x = make_prosody_sequence(duration_sec=3.0)
        out = pool(x)
        assert out.min() >= x.min() - 1e-6
        assert out.max() <= x.max() + 1e-6

    def test_gradient_flows_through_pool(self, pool):
        x = make_prosody_sequence(duration_sec=1.0)
        x.requires_grad = True
        out = pool(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_identity_for_single_frame(self, pool):
        """Single frame input should produce single frame output (trivial pool)."""
        x = torch.randn(2, 1, EMBED_DIM)
        out = pool(x)
        assert out.shape == (2, 1, EMBED_DIM)

    def test_deterministic_output(self, pool):
        torch.manual_seed(42)
        x1 = torch.randn(2, 80, EMBED_DIM)
        torch.manual_seed(42)
        x2 = torch.randn(2, 80, EMBED_DIM)
        out1 = pool(x1)
        out2 = pool(x2)
        assert torch.allclose(out1, out2)
