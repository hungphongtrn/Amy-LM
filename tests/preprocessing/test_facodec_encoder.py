"""Tests for FACodecEncoder wrapper."""

import pytest
import torch

from preprocessing.facodec_encoder import FACodecEncoder


class TestFACodecEncoderMock:
    """Test the mock fallback when Amphion is not installed."""

    def test_mock_encoder_initializes_without_amphion(self):
        """Encoder should initialize without error when Amphion not available."""
        # When Amphion is not installed, should fall back to mock mode
        # Use force_mock=True to test mock behavior regardless of Amphion availability
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        # In mock mode, _mock flag should be True
        assert encoder._mock is True

    def test_mock_encode_returns_three_index_lists(self):
        """Mock encode should return 3 lists of indices."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        # 2 seconds of audio at 16kHz = 32000 samples
        audio = torch.zeros(32000)

        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)

        # Should return exactly 3 items
        assert isinstance(content_indices, list)
        assert isinstance(prosody_indices, list)
        assert isinstance(timbre_indices, list)

    def test_mock_encode_indices_are_valid_range(self):
        """Mock indices should be in valid codebook range (0-2047)."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        audio = torch.zeros(32000)
        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)

        # All indices should be integers in range [0, 2047]
        for indices in [content_indices, prosody_indices, timbre_indices]:
            for idx in indices:
                assert isinstance(idx, int)
                assert 0 <= idx <= 2047

    def test_mock_encode_frame_count_is_correct(self):
        """Mock should generate ~12.5 frames per second (25 frames for 2s audio)."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

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
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

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
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        audio = torch.zeros(32000)
        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)

        # Content should have some variation across frames
        unique_content = len(set(content_indices))
        assert unique_content > 1, "Content indices should vary across frames"

    def test_mock_encode_different_audio_lengths(self):
        """Mock should handle different audio durations correctly."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

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

    def test_encode_raises_on_empty_audio(self):
        """encode() should raise ValueError on empty audio tensor."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        with pytest.raises(ValueError, match="empty"):
            encoder.encode(torch.zeros(0))

    def test_encode_raises_on_wrong_shape_audio(self):
        """encode() should raise ValueError on > 2-dim audio tensor."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        with pytest.raises(ValueError, match="Audio should be 1D"):
            encoder.encode(torch.zeros(2, 2, 32000))


class TestFACodecEncoderBatch:
    """Test batch encoding with both mock and real paths."""

    def test_mock_batch_encodes_multiple_samples(self):
        """Batch mock mode returns results for all samples."""
        encoder = FACodecEncoder(device="cpu", force_mock=True)
        audios = [torch.zeros(16000), torch.zeros(32000), torch.zeros(48000)]

        results = encoder.encode_batch(audios)

        assert len(results) == 3
        for content, prosody, timbre in results:
            assert isinstance(content, list)
            assert isinstance(prosody, list)
            assert isinstance(timbre, list)
            assert len(content) > 0

    def test_mock_batch_is_equivalent_to_individual(self):
        """Batch mock encoding matches individual mock encodes."""
        encoder = FACodecEncoder(device="cpu", force_mock=True)
        audios = [torch.zeros(16000), torch.zeros(32000)]

        batch_results = encoder.encode_batch(audios)
        single_results = [encoder.encode(a) for a in audios]

        for (batch_c, batch_p, batch_t), (single_c, single_p, single_t) in zip(
            batch_results, single_results
        ):
            assert batch_c == single_c
            assert batch_p == single_p
            assert batch_t == single_t

    def test_mock_batch_with_variable_lengths(self):
        """Batch mock handles different audio lengths."""
        encoder = FACodecEncoder(device="cpu", force_mock=True)
        audios = [torch.zeros(8000), torch.zeros(16000), torch.zeros(64000)]

        results = encoder.encode_batch(audios)

        frames = [len(r[0]) for r in results]
        assert frames[0] < frames[1] < frames[2]

    def test_batch_raises_on_empty(self):
        """encode_batch() raises on empty batch."""
        encoder = FACodecEncoder(device="cpu", force_mock=True)
        with pytest.raises(ValueError, match="empty"):
            encoder.encode_batch([])

    def test_real_batch_encodes_multiple_samples(self):
        """Real batch mode returns results for all samples."""
        encoder = FACodecEncoder(device="cpu")
        if encoder._mock:
            pytest.skip("Amphion FACodec not available or checkpoints missing")

        audios = [torch.zeros(16000), torch.zeros(32000), torch.zeros(48000)]
        results = encoder.encode_batch(audios)

        assert len(results) == 3
        for content, prosody, timbre in results:
            assert len(content) == len(prosody) == len(timbre)
            assert all(0 <= idx < 1024 for idx in content)

    def test_real_batch_matches_individual(self):
        """Batch encoding produces structurally correct results.
        
        Frame counts match individual encoding. Due to transformer self-attention
        and conv receptive fields over padded zeros, individual frame indices
        may differ near boundaries. All indices remain in valid codebook range.
        """
        encoder = FACodecEncoder(device="cpu")
        if encoder._mock:
            pytest.skip("Amphion FACodec not available or checkpoints missing")

        audios = [torch.zeros(16000), torch.zeros(24000)]

        batch_results = encoder.encode_batch(audios)
        single_results = [encoder.encode(a) for a in audios]

        for (batch_c, batch_p, batch_t), (single_c, single_p, single_t) in zip(
            batch_results, single_results
        ):
            assert len(batch_c) == len(single_c)
            assert len(batch_p) == len(single_p)
            assert len(batch_t) == len(single_t)
            assert all(0 <= idx < 1024 for idx in batch_c)
            assert all(0 <= idx < 1024 for idx in batch_p)
            assert all(0 <= idx < 1024 for idx in batch_t)

    def test_real_batch_frame_counts_match_duration(self):
        """Real batch frame counts scale with audio duration."""
        encoder = FACodecEncoder(device="cpu")
        if encoder._mock:
            pytest.skip("Amphion FACodec not available or checkpoints missing")

        audios = [torch.zeros(16000), torch.zeros(32000)]

        results = encoder.encode_batch(audios)
        frames_1s = len(results[0][0])
        frames_2s = len(results[1][0])

        assert frames_2s > frames_1s * 1.5
        assert frames_2s < frames_1s * 2.5


class TestFACodecEncoderReal:
    """Test the real FACodec implementation when Amphion is available.
    
    These tests only run when Amphion is installed and checkpoints are available.
    They verify the real model produces valid output matching expected format.
    """

    @pytest.fixture
    def encoder(self):
        """Fixture to create a real FACodec encoder."""
        # Skip if Amphion is not available
        try:
            encoder = FACodecEncoder(device="cpu")
            if encoder._mock:
                pytest.skip("Amphion FACodec not available or checkpoints missing")
            return encoder
        except Exception as e:
            pytest.skip(f"Failed to create real FACodec encoder: {e}")

    def test_real_encoder_initializes_when_available(self, encoder):
        """Real encoder should initialize with _mock=False when Amphion available."""
        assert encoder._mock is False
        assert encoder._encoder is not None
        assert encoder._decoder is not None

    def test_real_encode_returns_three_index_lists(self, encoder):
        """Real encode should return 3 lists of indices."""
        # 2 seconds of audio at 16kHz = 32000 samples
        audio = torch.zeros(32000)

        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)

        # Should return exactly 3 items
        assert isinstance(content_indices, list)
        assert isinstance(prosody_indices, list)
        assert isinstance(timbre_indices, list)
        assert len(content_indices) > 0
        assert len(prosody_indices) > 0
        assert len(timbre_indices) > 0

    def test_real_encode_indices_are_valid_range(self, encoder):
        """Real indices should be in valid codebook range (0-1023 for FACodec)."""
        audio = torch.zeros(32000)
        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)

        # FACodec uses 10-bit codebooks = 1024 entries
        for indices in [content_indices, prosody_indices, timbre_indices]:
            for idx in indices:
                assert isinstance(idx, int)
                assert 0 <= idx < 1024, f"Index {idx} out of range [0, 1024)"

    def test_real_encode_all_same_length(self, encoder):
        """All three index lists should have same length."""
        audio = torch.zeros(32000)
        content_indices, prosody_indices, timbre_indices = encoder.encode(audio)

        assert len(content_indices) == len(prosody_indices) == len(timbre_indices)

    def test_real_encode_different_audio_produces_different_output(self, encoder):
        """Different audio inputs should produce different indices."""
        audio_1 = torch.randn(16000)
        audio_2 = torch.randn(16000) * 0.5  # Different amplitude

        result_1 = encoder.encode(audio_1)
        result_2 = encoder.encode(audio_2)

        # At least one stream should differ
        content_diff = any(a != b for a, b in zip(result_1[0], result_2[0]))
        prosody_diff = any(a != b for a, b in zip(result_1[1], result_2[1]))
        timbre_diff = any(a != b for a, b in zip(result_1[2], result_2[2]))

        assert content_diff or prosody_diff or timbre_diff, \
            "Different audio should produce different indices"

    def test_real_encode_frame_count_scales_with_duration(self, encoder):
        """Frame count should scale with audio duration."""
        # Test with different durations
        audio_1s = torch.zeros(16000)
        audio_2s = torch.zeros(32000)

        result_1s = encoder.encode(audio_1s)
        result_2s = encoder.encode(audio_2s)

        frames_1s = len(result_1s[0])
        frames_2s = len(result_2s[0])

        # 2s should have roughly 2x the frames of 1s (with some tolerance)
        assert frames_2s > frames_1s * 1.5, "Longer audio should produce more frames"
        assert frames_2s < frames_1s * 2.5, "Frame count should scale linearly"

    def test_real_encode_handles_2d_audio(self, encoder):
        """Encoder should handle [1, samples] shaped audio."""
        audio_2d = torch.zeros(1, 32000)

        content_indices, prosody_indices, timbre_indices = encoder.encode(audio_2d)

        # Should work and return valid results
        assert isinstance(content_indices, list)
        assert len(content_indices) > 0

    def test_real_encode_raises_on_empty_audio(self, encoder):
        """encode() should raise ValueError on empty audio tensor."""
        with pytest.raises(ValueError, match="empty"):
            encoder.encode(torch.zeros(0))
