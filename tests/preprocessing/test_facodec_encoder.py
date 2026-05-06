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

    def test_mock_encode_returns_facodec_streams(self):
        """Mock encode should return FACodecStreams dataclass."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        # 2 seconds of audio at 16kHz = 32000 samples
        audio = torch.zeros(32000)

        streams = encoder.encode(audio)

        # Should return FACodecStreams with all 4 fields
        assert hasattr(streams, 'prosody_codebooks_idx')
        assert hasattr(streams, 'content_codebooks_idx')
        assert hasattr(streams, 'acoustic_codebooks_idx')
        assert hasattr(streams, 'timbre_vector')

    def test_mock_encode_indices_are_valid_range(self):
        """Mock indices should be in valid codebook range (0-1023)."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        audio = torch.zeros(32000)
        streams = encoder.encode(audio)

        # Check prosody indices: [1, T] tensor
        prosody = streams.prosody_codebooks_idx
        assert prosody.dtype == torch.int64
        assert prosody.shape[0] == 1  # 1 codebook
        assert (prosody >= 0).all() and (prosody < 1024).all()

        # Check content indices: [2, T] tensor
        content = streams.content_codebooks_idx
        assert content.dtype == torch.int64
        assert content.shape[0] == 2  # 2 codebooks
        assert (content >= 0).all() and (content < 1024).all()

        # Check acoustic indices: [3, T] tensor
        acoustic = streams.acoustic_codebooks_idx
        assert acoustic.dtype == torch.int64
        assert acoustic.shape[0] == 3  # 3 codebooks
        assert (acoustic >= 0).all() and (acoustic < 1024).all()

    def test_mock_encode_frame_count_is_correct(self):
        """Mock should generate correct frames at 80 Hz FACodec rate.
        
        2 seconds at 16kHz = 32000 samples
        hop_size=200 -> 32000/200 = 160 frames at 80 Hz
        """
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        audio = torch.zeros(32000)
        streams = encoder.encode(audio)

        # FACodec operates at 80 Hz with hop_size=200
        # 32000 samples / 200 = 160 frames
        expected_frames = 160
        
        # Prosody: [1, T]
        assert streams.prosody_codebooks_idx.shape == (1, expected_frames)
        
        # Content: [2, T]
        assert streams.content_codebooks_idx.shape == (2, expected_frames)
        
        # Acoustic: [3, T]
        assert streams.acoustic_codebooks_idx.shape == (3, expected_frames)
        
        # Timbre vector: [256]
        assert streams.timbre_vector.shape == (256,)
        assert streams.timbre_vector.dtype == torch.float32

    def test_mock_encode_indices_vary_by_stream(self):
        """Different streams should have different indices."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        audio = torch.zeros(32000)
        streams = encoder.encode(audio)

        # Streams should have different values (not all identical)
        # Prosody [0] should differ from Content [0] and Acoustic [0]
        prosody_0 = streams.prosody_codebooks_idx[0]
        content_0 = streams.content_codebooks_idx[0]
        acoustic_0 = streams.acoustic_codebooks_idx[0]

        # At least some frames should differ between streams
        assert not torch.allclose(prosody_0.float(), content_0.float()), \
            "Prosody and content should differ"
        assert not torch.allclose(content_0.float(), acoustic_0.float()), \
            "Content and acoustic should differ"

    def test_mock_encode_varies_across_frames(self):
        """Indices should vary across frames (not all same value)."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        audio = torch.zeros(32000)
        streams = encoder.encode(audio)

        # Content should have some variation across frames
        content_0 = streams.content_codebooks_idx[0]
        unique_values = len(torch.unique(content_0))
        assert unique_values > 1, "Content indices should vary across frames"

    def test_mock_encode_different_audio_lengths(self):
        """Mock should handle different audio durations correctly."""
        encoder = FACodecEncoder(device="cpu", checkpoint_path=None, force_mock=True)

        # 1 second = 16000 samples -> 16000/200 = 80 frames at 80 Hz
        audio_1s = torch.zeros(16000)
        streams_1s = encoder.encode(audio_1s)
        frames_1s = streams_1s.prosody_codebooks_idx.shape[1]

        # 4 seconds = 64000 samples -> 64000/200 = 320 frames at 80 Hz
        audio_4s = torch.zeros(64000)
        streams_4s = encoder.encode(audio_4s)
        frames_4s = streams_4s.prosody_codebooks_idx.shape[1]

        # 4s should have roughly 4x the frames of 1s
        assert frames_4s == frames_1s * 4, "Frame count should scale linearly with duration"

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
        """Batch mock mode returns FACodecStreams for all samples."""
        encoder = FACodecEncoder(device="cpu", force_mock=True)
        audios = [torch.zeros(16000), torch.zeros(32000), torch.zeros(48000)]

        results = encoder.encode_batch(audios)

        assert len(results) == 3
        for streams in results:
            assert isinstance(streams.prosody_codebooks_idx, torch.Tensor)
            assert isinstance(streams.content_codebooks_idx, torch.Tensor)
            assert isinstance(streams.acoustic_codebooks_idx, torch.Tensor)
            assert isinstance(streams.timbre_vector, torch.Tensor)

    def test_mock_batch_is_equivalent_to_individual(self):
        """Batch mock encoding matches individual mock encodes."""
        encoder = FACodecEncoder(device="cpu", force_mock=True)
        audios = [torch.zeros(16000), torch.zeros(32000)]

        batch_results = encoder.encode_batch(audios)
        single_results = [encoder.encode(a) for a in audios]

        for batch_streams, single_streams in zip(batch_results, single_results):
            assert torch.equal(batch_streams.prosody_codebooks_idx, single_streams.prosody_codebooks_idx)
            assert torch.equal(batch_streams.content_codebooks_idx, single_streams.content_codebooks_idx)
            assert torch.equal(batch_streams.acoustic_codebooks_idx, single_streams.acoustic_codebooks_idx)
            assert torch.equal(batch_streams.timbre_vector, single_streams.timbre_vector)

    def test_mock_batch_with_variable_lengths(self):
        """Batch mock handles different audio lengths."""
        encoder = FACodecEncoder(device="cpu", force_mock=True)
        audios = [torch.zeros(8000), torch.zeros(16000), torch.zeros(64000)]

        results = encoder.encode_batch(audios)

        frames = [r.prosody_codebooks_idx.shape[1] for r in results]
        assert frames[0] < frames[1] < frames[2]

    def test_batch_raises_on_empty(self):
        """encode_batch() raises on empty batch."""
        encoder = FACodecEncoder(device="cpu", force_mock=True)
        with pytest.raises(ValueError, match="empty"):
            encoder.encode_batch([])

    def test_real_batch_encodes_multiple_samples(self):
        """Real batch mode returns FACodecStreams for all samples."""
        encoder = FACodecEncoder(device="cpu")
        if encoder._mock:
            pytest.skip("Amphion FACodec not available or checkpoints missing")

        audios = [torch.zeros(16000), torch.zeros(32000), torch.zeros(48000)]
        results = encoder.encode_batch(audios)

        assert len(results) == 3
        for streams in results:
            # Check shapes
            T = streams.prosody_codebooks_idx.shape[1]
            assert streams.prosody_codebooks_idx.shape == (1, T)
            assert streams.content_codebooks_idx.shape == (2, T)
            assert streams.acoustic_codebooks_idx.shape == (3, T)
            assert streams.timbre_vector.shape == (256,)
            # Check value ranges
            assert (streams.content_codebooks_idx >= 0).all()
            assert (streams.content_codebooks_idx < 1024).all()

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

        for batch_streams, single_streams in zip(batch_results, single_results):
            # Frame counts should match
            assert batch_streams.prosody_codebooks_idx.shape == single_streams.prosody_codebooks_idx.shape
            assert batch_streams.content_codebooks_idx.shape == single_streams.content_codebooks_idx.shape
            assert batch_streams.acoustic_codebooks_idx.shape == single_streams.acoustic_codebooks_idx.shape
            assert batch_streams.timbre_vector.shape == single_streams.timbre_vector.shape
            
            # All indices should be in valid range
            assert (batch_streams.content_codebooks_idx >= 0).all()
            assert (batch_streams.content_codebooks_idx < 1024).all()
            assert (batch_streams.prosody_codebooks_idx >= 0).all()
            assert (batch_streams.prosody_codebooks_idx < 1024).all()
            assert (batch_streams.acoustic_codebooks_idx >= 0).all()
            assert (batch_streams.acoustic_codebooks_idx < 1024).all()

    def test_real_batch_frame_counts_match_duration(self):
        """Real batch frame counts scale with audio duration."""
        encoder = FACodecEncoder(device="cpu")
        if encoder._mock:
            pytest.skip("Amphion FACodec not available or checkpoints missing")

        audios = [torch.zeros(16000), torch.zeros(32000)]

        results = encoder.encode_batch(audios)
        frames_1s = results[0].prosody_codebooks_idx.shape[1]
        frames_2s = results[1].prosody_codebooks_idx.shape[1]

        assert frames_2s == frames_1s * 2


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

    def test_real_encode_returns_facodec_streams(self, encoder):
        """Real encode should return FACodecStreams with all fields."""
        # 2 seconds of audio at 16kHz = 32000 samples
        audio = torch.zeros(32000)

        streams = encoder.encode(audio)

        # Should return FACodecStreams with all 4 fields
        assert hasattr(streams, 'prosody_codebooks_idx')
        assert hasattr(streams, 'content_codebooks_idx')
        assert hasattr(streams, 'acoustic_codebooks_idx')
        assert hasattr(streams, 'timbre_vector')
        
        # Check tensor shapes
        T = 32000 // 200  # 160 frames at 80 Hz
        assert streams.prosody_codebooks_idx.shape == (1, T)
        assert streams.content_codebooks_idx.shape == (2, T)
        assert streams.acoustic_codebooks_idx.shape == (3, T)
        assert streams.timbre_vector.shape == (256,)

    def test_real_encode_indices_are_valid_range(self, encoder):
        """Real indices should be in valid codebook range (0-1023 for FACodec)."""
        audio = torch.zeros(32000)
        streams = encoder.encode(audio)

        # FACodec uses 10-bit codebooks = 1024 entries
        # Check prosody indices
        assert streams.prosody_codebooks_idx.dtype == torch.int64
        assert (streams.prosody_codebooks_idx >= 0).all()
        assert (streams.prosody_codebooks_idx < 1024).all()
        
        # Check content indices
        assert streams.content_codebooks_idx.dtype == torch.int64
        assert (streams.content_codebooks_idx >= 0).all()
        assert (streams.content_codebooks_idx < 1024).all()
        
        # Check acoustic indices
        assert streams.acoustic_codebooks_idx.dtype == torch.int64
        assert (streams.acoustic_codebooks_idx >= 0).all()
        assert (streams.acoustic_codebooks_idx < 1024).all()
        
        # Check timbre vector is float32
        assert streams.timbre_vector.dtype == torch.float32

    def test_real_encode_all_same_length(self, encoder):
        """All codebook streams should have same temporal length."""
        audio = torch.zeros(32000)
        streams = encoder.encode(audio)

        # All streams should have same number of frames
        T = streams.prosody_codebooks_idx.shape[1]
        assert streams.content_codebooks_idx.shape[1] == T
        assert streams.acoustic_codebooks_idx.shape[1] == T

    def test_real_encode_different_audio_produces_different_output(self, encoder):
        """Different audio inputs should produce different indices."""
        audio_1 = torch.randn(16000)
        audio_2 = torch.randn(16000) * 0.5  # Different amplitude

        streams_1 = encoder.encode(audio_1)
        streams_2 = encoder.encode(audio_2)

        # At least one stream should differ
        prosody_diff = not torch.equal(streams_1.prosody_codebooks_idx, streams_2.prosody_codebooks_idx)
        content_diff = not torch.equal(streams_1.content_codebooks_idx, streams_2.content_codebooks_idx)
        acoustic_diff = not torch.equal(streams_1.acoustic_codebooks_idx, streams_2.acoustic_codebooks_idx)
        timbre_diff = not torch.equal(streams_1.timbre_vector, streams_2.timbre_vector)

        assert prosody_diff or content_diff or acoustic_diff or timbre_diff, \
            "Different audio should produce different outputs"

    def test_real_encode_frame_count_scales_with_duration(self, encoder):
        """Frame count should scale with audio duration."""
        # Test with different durations
        audio_1s = torch.zeros(16000)
        audio_2s = torch.zeros(32000)

        streams_1s = encoder.encode(audio_1s)
        streams_2s = encoder.encode(audio_2s)

        frames_1s = streams_1s.prosody_codebooks_idx.shape[1]
        frames_2s = streams_2s.prosody_codebooks_idx.shape[1]

        # 2s should have exactly 2x the frames of 1s at 80 Hz
        assert frames_2s == frames_1s * 2, "Frame count should scale linearly"

    def test_real_encode_handles_2d_audio(self, encoder):
        """Encoder should handle [1, samples] shaped audio."""
        audio_2d = torch.zeros(1, 32000)

        streams = encoder.encode(audio_2d)

        # Should work and return valid FACodecStreams
        assert hasattr(streams, 'prosody_codebooks_idx')
        assert streams.prosody_codebooks_idx.shape[0] == 1  # 1 codebook
        assert streams.prosody_codebooks_idx.shape[1] > 0  # Has frames

    def test_real_encode_raises_on_empty_audio(self, encoder):
        """encode() should raise ValueError on empty audio tensor."""
        with pytest.raises(ValueError, match="empty"):
            encoder.encode(torch.zeros(0))
