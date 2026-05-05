"""Tests for FACodecEncoder wrapper."""

import pytest
import torch

from src.preprocessing.facodec_encoder import FACodecEncoder


class TestFACodecEncoderMock:
    """Test the mock fallback when Amphion is not installed."""

    def test_mock_encoder_initializes_without_amphion(self):
        """Encoder should initialize without error when Amphion not available."""
        # When Amphion is not installed, should fall back to mock mode
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None)
        
        # In mock mode, _mock flag should be True
        assert encoder._mock is True
        
    def test_mock_encode_returns_three_index_lists(self):
        """Mock encode should return 3 lists of indices."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None)
        
        # 2 seconds of audio at 16kHz = 32000 samples
        audio = torch.zeros(32000)
        
        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)
        
        # Should return exactly 3 items
        assert isinstance(content_indices, list)
        assert isinstance(prosody_indices, list)
        assert isinstance(timbre_indices, list)
        
    def test_mock_encode_indices_are_valid_range(self):
        """Mock indices should be in valid codebook range (0-2047)."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None)
        
        audio = torch.zeros(32000)
        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)
        
        # All indices should be integers in range [0, 2047]
        for indices in [content_indices, prosody_indices, timbre_indices]:
            for idx in indices:
                assert isinstance(idx, int)
                assert 0 <= idx <= 2047
                
    def test_mock_encode_frame_count_is_correct(self):
        """Mock should generate ~12.5 frames per second (25 frames for 2s audio)."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None)
        
        # 2 seconds at 16kHz = 32000 samples
        # At 12.5 Hz, we expect ~25 frames
        audio = torch.zeros(32000)
        
        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)
        
        # Allow some tolerance (e.g., 20-30 frames)
        assert 20 <= len(content_indices) <= 30
        assert 20 <= len(prosody_indices) <= 30
        assert 20 <= len(timbre_indices) <= 30
        
        # All three should have same length
        assert len(content_indices) == len(prosody_indices) == len(timbre_indices)
        
    def test_mock_encode_indices_vary_by_stream(self):
        """Content, prosody, and timbre indices should be different streams."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None)
        
        audio = torch.zeros(32000)
        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)
        
        # At least some frames should have different values across streams
        # This proves we're not just returning the same indices for all
        differences = sum(
            1 for c, p, t in zip(content_indices, prosody_indices, timbre_indices)
            if c != p or p != t or c != t
        )
        assert differences > 0, "Indices should vary across streams"
        
    def test_mock_encode_varies_across_frames(self):
        """Indices should vary across frames (not all same value)."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None)
        
        audio = torch.zeros(32000)
        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)
        
        # Content should have some variation across frames
        unique_content = len(set(content_indices))
        assert unique_content > 1, "Content indices should vary across frames"
        
    def test_mock_encode_different_audio_lengths(self):
        """Mock should handle different audio durations correctly."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None)
        
        # 1 second = ~12-13 frames
        audio_1s = torch.zeros(16000)
        result_1s = encoder.encode(audio_1s)
        frames_1s = len(result_1s[0])
        
        # 4 seconds = ~50 frames
        audio_4s = torch.zeros(64000)
        result_4s = encoder.encode(audio_4s)
        frames_4s = len(result_4s[0])
        
        # 4s should have roughly 4x the frames of 1s
        assert frames_4s > frames_1s * 3, "Longer audio should produce more frames"
        assert frames_4s < frames_1s * 5, "Frame count should scale linearly"
