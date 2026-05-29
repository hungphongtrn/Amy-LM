"""Tests for AmyForProsodyClassification — end-to-end model assembly."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.amy_classifier import AmyForProsodyClassification


def make_prosody_codebook_vectors():
    """Mock FACodec prosody codebook vectors [1024, 8]."""
    return torch.randn(1024, 8)


class TestAmyForwardShape:
    """Verify forward pass produces correct output shapes.

    These tests run full forward passes and require GPU.
    """

    @pytest.fixture
    def model(self, require_gpu, device):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device=device,
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

    def test_forward_output_shape_batch_3(self, model):
        """Forward pass with batch_size=3 produces [3, 2] logits (regression)."""
        batch = 3
        audio = torch.randn(batch, 32000)
        prosody_indices = torch.randint(0, 1024, (batch, 1, 160))
        timbre_vector = torch.randn(batch, 256)

        logits = model(audio, prosody_indices, timbre_vector)
        assert logits.shape == (batch, 2)

    def test_forward_output_shape_batch_8(self, model):
        """Forward pass with batch_size=8 produces [8, 2] logits (regression)."""
        batch = 8
        audio = torch.randn(batch, 32000)
        prosody_indices = torch.randint(0, 1024, (batch, 1, 160))
        timbre_vector = torch.randn(batch, 256)

        logits = model(audio, prosody_indices, timbre_vector)
        assert logits.shape == (batch, 2)

    def test_single_sample_batch(self, model):
        """Should handle batch_size=1."""
        audio = torch.randn(1, 16000)
        prosody_indices = torch.randint(0, 1024, (1, 1, 80))
        timbre_vector = torch.randn(1, 256)

        logits = model(audio, prosody_indices, timbre_vector)
        assert logits.shape == (1, 2)


class TestAmyGradientFlow:
    """Verify which parameters receive gradients.

    These tests run backward passes and require GPU.
    """

    @pytest.fixture
    def model(self, require_gpu, device):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device=device,
        )

    @pytest.fixture
    def batch(self):
        return (
            torch.randn(2, 32000),
            torch.randint(0, 1024, (2, 1, 160)),
            torch.randn(2, 256),
        )

    def test_only_trainable_params_get_gradients(self, model, batch):
        """Trainable params in the forward path receive gradients;
        frozen backbone params do not."""
        audio, prosody_idx, timbre = batch
        with torch.no_grad():
            semantic, _ = model.amy_moss.encode_enriched_audio_embeds(audio)
        semantic = semantic.float()
        T_moss = semantic.shape[1]

        p_emb = model.amy_moss.prosody_embedding(prosody_idx)
        P = model.amy_moss.temporal_pool(p_emb)
        if P.shape[1] != T_moss:
            P = P.transpose(1, 2)
            P = F.adaptive_avg_pool1d(P, T_moss)
            P = P.transpose(1, 2)
        t_proj = model.amy_moss.timbre_projection(timbre)
        T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)

        H = model.amy_moss.residual_fusion(semantic, prosody=P, timbre=T, content=None, acoustic=None)
        loss = H.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, f"Frozen param '{name}' should not have grad"
        assert model.amy_moss.timbre_projection.linear.weight.grad is not None
        assert model.amy_moss.residual_fusion.lambda_p.grad is not None
        assert model.amy_moss.residual_fusion.lambda_t.grad is not None

    def test_fusion_lambdas_are_trainable(self, model):
        """lambda_p and lambda_t should require grad."""
        assert model.amy_moss.residual_fusion.lambda_p.requires_grad
        assert model.amy_moss.residual_fusion.lambda_t.requires_grad

    def test_classifier_is_trainable(self, model):
        """Classifier head should require grad."""
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_timbre_projection_is_trainable(self, model):
        """TimbreProjection should require grad."""
        for name, param in model.amy_moss.timbre_projection.named_parameters():
            assert param.requires_grad, f"timbre_projection.{name} should be trainable"

    def test_backbone_fully_frozen(self, model):
        """All MOSS-Audio backbone params should have requires_grad=False."""
        for name, param in model.amy_moss.moss.named_parameters():
            assert not param.requires_grad, (
                f"Backbone param '{name}' should be frozen"
            )


class TestAmyBaselineEquivalence:
    """Verify Amy model equals MOSS-Audio baseline when lambdas are zero.

    These tests run forward passes through the 4B language model and require GPU.
    """

    @pytest.fixture
    def model(self, require_gpu, device):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device=device,
        )

    @pytest.fixture
    def audio(self):
        return torch.randn(1, 32000)

    @pytest.fixture
    def prosody_indices(self):
        return torch.randint(0, 1024, (1, 1, 160))

    @pytest.fixture
    def timbre_vector(self):
        return torch.randn(1, 256)

    def test_lambdas_start_at_zero(self, model):
        """lambda_p and lambda_t must be zero at initialization."""
        assert model.amy_moss.residual_fusion.lambda_p.item() == 0.0
        assert model.amy_moss.residual_fusion.lambda_t.item() == 0.0

    def test_semantic_alone_equals_baseline(self, model, audio, prosody_indices, timbre_vector):
        """With lambdas=0 and prosody/timbre fed, output should equal
        running only semantic through the same path."""
        with torch.no_grad():
            semantic, _ = model.amy_moss.encode_enriched_audio_embeds(audio)
        semantic = semantic.float()
        T_moss = semantic.shape[1]
        lm_dtype = next(model.get_language_model().parameters()).dtype

        with torch.no_grad():
            H_baseline = model.amy_moss.residual_fusion(
                semantic,
                prosody=None, content=None, acoustic=None, timbre=None,
            )
            H_baseline_lm = H_baseline.to(dtype=lm_dtype)
            lm_out_baseline = model.get_language_model()(
                inputs_embeds=H_baseline_lm
            ).last_hidden_state.float()
            pooled_baseline = lm_out_baseline.mean(dim=1)
            logits_baseline = model.classifier(pooled_baseline)

        with torch.no_grad():
            p_emb = model.amy_moss.prosody_embedding(prosody_indices)
            P = model.amy_moss.temporal_pool(p_emb)
            if P.shape[1] != T_moss:
                P = P.transpose(1, 2)
                P = F.adaptive_avg_pool1d(P, T_moss)
                P = P.transpose(1, 2)
            t_proj = model.amy_moss.timbre_projection(timbre_vector)
            T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)
            H_full = model.amy_moss.residual_fusion(
                semantic, prosody=P, timbre=T, content=None, acoustic=None,
            )
            H_full_lm = H_full.to(dtype=lm_dtype)
            lm_out_full = model.get_language_model()(
                inputs_embeds=H_full_lm
            ).last_hidden_state.float()
            pooled_full = lm_out_full.mean(dim=1)
            logits_full = model.classifier(pooled_full)

        assert torch.allclose(logits_baseline, logits_full, atol=1e-4)


class TestAmyTemporalAlignment:
    """Verify FACodec 80Hz stream aligns to MOSS-Audio ~12.5Hz frames.

    These tests run forward passes and require GPU.
    """

    @pytest.fixture
    def model(self, require_gpu, device):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device=device,
        )

    def test_prosody_pool_matches_semantic_frames(self, model):
        """Pooled prosody [B, T_moss, 2560] must have same T_moss as semantic."""
        audio = torch.randn(2, 48000)
        prosody_indices = torch.randint(0, 1024, (2, 1, 240))

        with torch.no_grad():
            semantic, _ = model.amy_moss.encode_enriched_audio_embeds(audio)
        T_moss = semantic.shape[1]

        p_emb = model.amy_moss.prosody_embedding(prosody_indices)
        P = model.amy_moss.temporal_pool(p_emb)

        if P.shape[1] != T_moss:
            P = P.transpose(1, 2)
            P = F.adaptive_avg_pool1d(P, T_moss)
            P = P.transpose(1, 2)

        assert P.shape[1] == T_moss, (
            f"Pooled prosody frames ({P.shape[1]}) must equal "
            f"semantic frames ({T_moss}) after alignment"
        )

    def test_timbre_broadcast_matches_semantic_frames(self, model):
        """Broadcast timbre must have correct T_moss."""
        audio = torch.randn(1, 16000)
        timbre = torch.randn(1, 256)

        with torch.no_grad():
            semantic, _ = model.amy_moss.encode_enriched_audio_embeds(audio)
        T_moss = semantic.shape[1]

        t_proj = model.amy_moss.timbre_projection(timbre)
        T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)

        assert T.shape[1] == T_moss
        assert T.shape[0] == 1
        assert T.shape[2] == 2560


class TestAmyLoopLogic:
    """Verify batching logic in forward() handles batch_size correctly.

    These tests mock the heavy model components and test the batching logic.
    """

    class _MockLM(nn.Module):
        def __init__(self, T_out=25, D_out=2560):
            super().__init__()
            self._T_out = T_out
            self._D_out = D_out
            self._dummy = nn.Parameter(torch.zeros(1))

        def forward(self, inputs_embeds, attention_mask=None):
            B = inputs_embeds.shape[0]
            out = MagicMock()
            out.last_hidden_state = torch.randn(B, inputs_embeds.shape[1], self._D_out)
            return out

    @pytest.fixture
    def mock_modules(self):
        """Create an AmyForProsodyClassification with mock heavy components."""
        model = object.__new__(AmyForProsodyClassification)
        nn.Module.__init__(model)
        model._device = torch.device("cpu")
        model.classifier = nn.Linear(2560, 2)

        def _fake_encode(audio, prosody_indices=None, timbre_vector=None):
            B = audio.shape[0]
            T_out = 25
            return torch.randn(B, T_out, 2560), torch.full((B,), T_out, dtype=torch.long)

        model.amy_moss = MagicMock()
        model.amy_moss.encode_enriched_audio_embeds = MagicMock(
            side_effect=_fake_encode
        )
        model.get_language_model = MagicMock(
            return_value=self._MockLM()
        )
        return model

    def test_single_sample_forward(self, mock_modules):
        B = 1
        audio = torch.randn(B, 16000)
        prosody = torch.randint(0, 1024, (B, 1, 80))
        timbre = torch.randn(B, 256)
        logits = mock_modules(audio, prosody, timbre)
        assert logits.shape == (B, 2)
        assert mock_modules.amy_moss.encode_enriched_audio_embeds.call_count == 1

    def test_batch_3_forward(self, mock_modules):
        B = 3
        audio = torch.randn(B, 16000)
        prosody = torch.randint(0, 1024, (B, 1, 80))
        timbre = torch.randn(B, 256)
        logits = mock_modules(audio, prosody, timbre)
        assert logits.shape == (B, 2)
        assert mock_modules.amy_moss.encode_enriched_audio_embeds.call_count == 1

    def test_batch_8_forward(self, mock_modules):
        B = 8
        audio = torch.randn(B, 32000)
        prosody = torch.randint(0, 1024, (B, 1, 160))
        timbre = torch.randn(B, 256)
        logits = mock_modules(audio, prosody, timbre)
        assert logits.shape == (B, 2)
        assert mock_modules.amy_moss.encode_enriched_audio_embeds.call_count == 1
