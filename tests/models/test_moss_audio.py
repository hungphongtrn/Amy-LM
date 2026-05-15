"""Tests for MOSS-Audio backbone integration."""

import pytest
import torch

from src.models.moss_audio import MossAudioWrapper


class TestMossAudioWrapper:
    """Tests for MossAudioWrapper model loading and sub-module extraction."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_init_loads_model_and_extracts_submodules(self, device):
        """Wrapper should load model and expose encoder, adapter, language_model."""
        wrapper = MossAudioWrapper(device=device)

        assert wrapper.audio_encoder is not None
        assert wrapper.audio_adapter is not None
        assert wrapper.language_model is not None

    def test_semantic_stream_output_shape(self, device):
        """Semantic stream should produce [B, T_frames, 2560] from raw audio."""
        wrapper = MossAudioWrapper(device=device)

        # Simulate ~2 seconds of 16kHz audio
        audio = torch.randn(2, 32000, device=device)  # [B, T_audio]
        semantic = wrapper.encode_semantic(audio)

        # Qwen3 hidden_dim = 2560
        assert semantic.dim() == 3
        assert semantic.shape[0] == 2
        assert semantic.shape[2] == 2560
        # 2s audio -> ~200 mel frames (hop=160), then /8 conv downsample -> ~25 frames
        assert 20 <= semantic.shape[1] <= 30

    def test_submodules_are_frozen_by_default(self, device):
        """Audio encoder, adapter, and language model should have no trainable params."""
        wrapper = MossAudioWrapper(device=device)

        for name, param in wrapper.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_encode_semantic_different_lengths(self, device):
        """Should handle different audio lengths in a batch via padding."""
        wrapper = MossAudioWrapper(device=device)

        audio_1s = torch.randn(1, 16000, device=device)   # 1s
        audio_3s = torch.randn(1, 48000, device=device)   # 3s
        semantic_1s = wrapper.encode_semantic(audio_1s)
        semantic_3s = wrapper.encode_semantic(audio_3s)

        assert semantic_1s.dim() == 3
        assert semantic_3s.dim() == 3
        assert semantic_1s.shape[0] == 1
        assert semantic_3s.shape[0] == 1
        # 3s should have ~3x more frames than 1s
        ratio = semantic_3s.shape[1] / semantic_1s.shape[1]
        assert 2.0 <= ratio <= 4.0, (
            f"Expected ~3x frames for 3s vs 1s, got {ratio:.2f}"
        )

    def test_encode_semantic_empty_batch(self, device):
        """Empty batch should return empty tensor without crashing."""
        wrapper = MossAudioWrapper(device=device)

        audio = torch.empty(0, 16000, device=device)
        semantic = wrapper.encode_semantic(audio)
        assert semantic.shape == (0, 0, 2560)

    def test_audio_encoder_output_dim(self, device):
        """Audio encoder hidden dim should match expected Whisper config."""
        wrapper = MossAudioWrapper(device=device)
        assert hasattr(wrapper.audio_encoder, "config")
        # Encoder hidden dim is the adapter input dim
        adapter_in_features = wrapper.audio_adapter.gate_proj.in_features
        assert adapter_in_features > 0

    def test_language_model_hidden_size(self, device):
        """Qwen3 hidden dim should be 2560 for 4B variant."""
        wrapper = MossAudioWrapper(device=device)
        hidden_size = wrapper.language_model.config.hidden_size
        assert hidden_size == 2560, (
            f"Expected Qwen3 hidden_size=2560, got {hidden_size}"
        )

    def test_adapter_output_dim_matches_llm_input(self, device):
        """Audio adapter output dim must equal Qwen3 hidden dim."""
        wrapper = MossAudioWrapper(device=device)
        adapter_out = wrapper.audio_adapter.down_proj.out_features
        llm_hidden = wrapper.language_model.config.hidden_size
        assert adapter_out == llm_hidden, (
            f"Adapter output {adapter_out} != LLM hidden {llm_hidden}"
        )

    def test_encode_semantic_produces_valid_embeddings(self, device):
        """Output values should be finite and non-zero."""
        wrapper = MossAudioWrapper(device=device)
        audio = torch.randn(1, 16000, device=device)  # 1s audio
        semantic = wrapper.encode_semantic(audio)
        assert torch.isfinite(semantic).all()
        assert not torch.allclose(semantic, torch.zeros_like(semantic), atol=1e-6)

    def test_semantic_frame_rate_is_approximately_12_5_hz(self, device):
        """2 seconds of 16kHz audio should produce ~25 frames (~12.5 Hz)."""
        wrapper = MossAudioWrapper(device=device)

        audio = torch.randn(1, 32000, device=device)
        semantic = wrapper.encode_semantic(audio)

        n_frames = semantic.shape[1]
        # 2s at 16kHz = 32000 samples -> 200 mel frames (hop=160) -> /8 conv -> ~25 frames
        # MOSS-Audio encoder uses conv downsample rate of 8
        assert 22 <= n_frames <= 28, (
            f"Expected ~25 frames for 2s audio (12.5 Hz), got {n_frames}"
        )

    def test_frame_rate_scales_with_duration(self, device):
        """Longer audio should produce proportionally more frames."""
        wrapper = MossAudioWrapper(device=device)

        audio_1s = torch.randn(1, 16000, device=device)
        audio_2s = torch.randn(1, 32000, device=device)

        frames_1s = wrapper.encode_semantic(audio_1s).shape[1]
        frames_2s = wrapper.encode_semantic(audio_2s).shape[1]

        ratio = frames_2s / frames_1s
        assert 1.7 <= ratio <= 2.3, (
            f"Expected ~2x frames for 2s vs 1s, got {ratio:.2f}"
        )
