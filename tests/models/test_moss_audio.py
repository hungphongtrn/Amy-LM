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
