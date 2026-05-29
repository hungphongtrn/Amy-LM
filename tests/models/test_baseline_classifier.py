"""Tests for BaselineClassifier — MOSS-Audio frozen backbone + Linear classifier."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.models.baseline_classifier import BaselineClassifier


class TestBaselineLoopLogic:
    """Verify the per-sample loop in forward() handles batch_size > 1.

    These tests mock the heavy model components and test the loop logic.
    """

    class _MockLM(nn.Module):
        def __init__(self, T_out=25, D_out=2560):
            super().__init__()
            self._T_out = T_out
            self._D_out = D_out
            self._dummy = nn.Parameter(torch.zeros(1))

        def forward(self, inputs_embeds):
            B = inputs_embeds.shape[0]
            out = MagicMock()
            out.last_hidden_state = torch.randn(B, self._T_out, self._D_out)
            return out

    @pytest.fixture
    def mock_modules(self):
        """Create a BaselineClassifier with mocked encode_enriched_audio_embeds
        and language_model."""
        baseline = object.__new__(BaselineClassifier)
        nn.Module.__init__(baseline)
        baseline._device = torch.device("cpu")
        baseline.norm = nn.Identity()
        baseline.classifier = nn.Linear(2560, 2)
        baseline.amy_moss = MagicMock()
        baseline.amy_moss.encode_enriched_audio_embeds = MagicMock(
            return_value=torch.randn(1, 25, 2560)
        )
        baseline.get_language_model = MagicMock(
            return_value=self._MockLM()
        )
        return baseline

    def test_single_sample_forward(self, mock_modules):
        B = 1
        audio = torch.randn(B, 16000)
        logits = mock_modules(audio)
        assert logits.shape == (B, 2)
        assert mock_modules.amy_moss.encode_enriched_audio_embeds.call_count == 1

    def test_batch_3_forward(self, mock_modules):
        B = 3
        audio = torch.randn(B, 16000)
        logits = mock_modules(audio)
        assert logits.shape == (B, 2)
        assert mock_modules.amy_moss.encode_enriched_audio_embeds.call_count == B

    def test_batch_8_forward(self, mock_modules):
        B = 8
        audio = torch.randn(B, 32000)
        logits = mock_modules(audio)
        assert logits.shape == (B, 2)
        assert mock_modules.amy_moss.encode_enriched_audio_embeds.call_count == B


class TestBaselineForwardShape:
    """Verify forward pass produces correct output shapes.

    These tests run full forward passes through the 4B model and require GPU.
    """

    @pytest.fixture
    def model(self, require_gpu, device):
        return BaselineClassifier(device=device)

    def test_forward_output_shape(self, model):
        """Forward pass produces [B, 2] logits."""
        audio = torch.randn(2, 32000)
        logits = model(audio)
        assert logits.shape == (2, 2)
        assert logits.dtype == torch.float32

    def test_forward_output_shape_batch_3(self, model):
        """Forward pass with batch_size=3 produces [3, 2] logits (regression)."""
        audio = torch.randn(3, 32000)
        logits = model(audio)
        assert logits.shape == (3, 2)

    def test_forward_output_shape_batch_8(self, model):
        """Forward pass with batch_size=8 produces [8, 2] logits (regression)."""
        audio = torch.randn(8, 32000)
        logits = model(audio)
        assert logits.shape == (8, 2)

    def test_single_sample_batch(self, model):
        """Handles batch_size=1."""
        audio = torch.randn(1, 16000)
        logits = model(audio)
        assert logits.shape == (1, 2)


class TestBaselineFreeze:
    """Verify backbone is frozen and trainable params are correct.

    These tests instantiate the 4B model and require GPU.
    """

    @pytest.fixture
    def model(self, require_gpu, device):
        return BaselineClassifier(device=device)

    def test_backbone_fully_frozen(self, model):
        for name, param in model.amy_moss.moss.named_parameters():
            assert not param.requires_grad, (
                f"Backbone param '{name}' should be frozen"
            )

    def test_classifier_is_trainable(self, model):
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_norm_is_trainable(self, model):
        """LayerNorm is trainable."""
        for param in model.norm.parameters():
            assert param.requires_grad

    def test_only_norm_classifier_params_exist(self, model):
        """No extra trainable modules beyond norm, classifier."""
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert all(
            n.startswith("norm.") or n.startswith("classifier.")
            for n in trainable
        ), f"Unexpected trainable params: {trainable - {'norm', 'classifier'}}"


class TestBaselineAmyEquivalence:
    """BaselineClassifier must produce identical logits to AmyForProsodyClassification
    when lambdas=0 and both have the same classifier weights.

    These tests run forward passes through the 4B model and require GPU.
    """

    @pytest.fixture
    def audio(self):
        return torch.randn(1, 32000)

    @pytest.fixture
    def prosody_indices(self):
        return torch.randint(0, 1024, (1, 1, 160))

    @pytest.fixture
    def timbre_vector(self):
        return torch.randn(1, 256)

    def test_forward_equivalence_at_zero_lambda(
        self, require_gpu, device, audio, prosody_indices, timbre_vector
    ):
        """With identical classifier weights, baseline and Amy (lambda=0) logits must match."""
        from src.models import AmyForProsodyClassification

        vectors = torch.randn(1024, 8)
        baseline = BaselineClassifier(device=device)
        amy = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)

        amy.classifier.load_state_dict(deepcopy(baseline.classifier.state_dict()))
        assert torch.equal(baseline.classifier.weight, amy.classifier.weight)

        baseline.eval()
        amy.eval()
        with torch.no_grad():
            logits_baseline = baseline(audio)
            logits_amy = amy(audio, prosody_indices, timbre_vector)

        assert torch.allclose(logits_baseline, logits_amy, atol=1e-4)
