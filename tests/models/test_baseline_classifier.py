import pytest
import torch

from src.models.baseline_classifier import BaselineClassifier


class TestBaselineForwardShape:
    @pytest.fixture
    def model(self):
        return BaselineClassifier(device="cpu")

    def test_forward_output_shape(self, model):
        """Forward pass produces [B, 2] logits."""
        audio = torch.randn(2, 32000)
        logits = model(audio)
        assert logits.shape == (2, 2)
        assert logits.dtype == torch.float32

    def test_single_sample_batch(self, model):
        """Handles batch_size=1."""
        audio = torch.randn(1, 16000)
        logits = model(audio)
        assert logits.shape == (1, 2)


class TestBaselineFreeze:
    @pytest.fixture
    def model(self):
        return BaselineClassifier(device="cpu")

    def test_backbone_fully_frozen(self, model):
        for name, param in model.wrapper.named_parameters():
            assert not param.requires_grad, (
                f"Backbone param '{name}' should be frozen"
            )

    def test_classifier_is_trainable(self, model):
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_norm_is_trainable(self, model):
        """LayerNorm is trainable (matches ResidualFusion norm in Amy model)."""
        for param in model.norm.parameters():
            assert param.requires_grad

    def test_only_wrapper_norm_classifier_params_exist(self, model):
        """No extra trainable modules beyond wrapper, norm, classifier."""
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert all(
            n.startswith("norm.") or n.startswith("classifier.")
            for n in trainable
        ), f"Unexpected trainable params: {trainable - {'norm', 'classifier'}}"


class TestBaselineAmyEquivalence:
    """BaselineClassifier must produce identical logits to AmyForProsodyClassification
    when lambdas=0 and both have the same classifier weights."""

    @pytest.fixture
    def audio(self):
        return torch.randn(1, 32000)

    @pytest.fixture
    def prosody_indices(self):
        return torch.randint(0, 1024, (1, 1, 160))

    @pytest.fixture
    def timbre_vector(self):
        return torch.randn(1, 256)

    def test_forward_equivalence_at_zero_lambda(self, audio, prosody_indices, timbre_vector):
        """With identical classifier weights, baseline and Amy (lambda=0) logits must match."""
        from copy import deepcopy

        from src.models import AmyForProsodyClassification

        vectors = torch.randn(1024, 8)
        baseline = BaselineClassifier(device="cpu")
        amy = AmyForProsodyClassification(warm_start_vectors=vectors, device="cpu")

        amy.classifier.load_state_dict(deepcopy(baseline.classifier.state_dict()))
        assert torch.equal(baseline.classifier.weight, amy.classifier.weight)

        with torch.no_grad():
            logits_baseline = baseline(audio)
            logits_amy = amy(audio, prosody_indices, timbre_vector)

        assert torch.allclose(logits_baseline, logits_amy, atol=1e-4)
