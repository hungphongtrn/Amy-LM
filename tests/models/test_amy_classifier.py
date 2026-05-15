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


class TestAmyGradientFlow:
    """Verify which parameters receive gradients."""

    @pytest.fixture
    def model(self):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device="cpu",
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
            semantic = model.wrapper.encode_semantic(audio)
        semantic = semantic.float()
        T_moss = semantic.shape[1]

        p_emb = model.prosody_embedding(prosody_idx)
        P = model.temporal_pool(p_emb)
        if P.shape[1] != T_moss:
            P = P.transpose(1, 2)
            P = torch.nn.functional.adaptive_avg_pool1d(P, T_moss)
            P = P.transpose(1, 2)
        t_proj = model.timbre_projection(timbre)
        T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)

        H = model.fusion(semantic, prosody=P, timbre=T, content=None, acoustic=None)
        loss = H.sum()
        loss.backward()

        # Params that must receive gradients (in the fusion forward path)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, f"Frozen param '{name}' should not have grad"
        # Specific trainable params that should have grads
        assert model.timbre_projection.linear.weight.grad is not None
        assert model.fusion.lambda_p.grad is not None
        assert model.fusion.lambda_t.grad is not None

    def test_fusion_lambdas_are_trainable(self, model):
        """lambda_p and lambda_t should require grad."""
        assert model.fusion.lambda_p.requires_grad
        assert model.fusion.lambda_t.requires_grad

    def test_classifier_is_trainable(self, model):
        """Classifier head should require grad."""
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_timbre_projection_is_trainable(self, model):
        """TimbreProjection should require grad."""
        for name, param in model.timbre_projection.named_parameters():
            assert param.requires_grad, f"timbre_projection.{name} should be trainable"

    def test_backbone_fully_frozen(self, model):
        """All MOSS-Audio backbone params should have requires_grad=False."""
        for name, param in model.wrapper.named_parameters():
            assert not param.requires_grad, (
                f"Backbone param '{name}' should be frozen"
            )


class TestAmyBaselineEquivalence:
    """Verify Amy model equals MOSS-Audio baseline when lambdas are zero."""

    @pytest.fixture
    def model(self):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device="cpu",
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
        assert model.fusion.lambda_p.item() == 0.0
        assert model.fusion.lambda_t.item() == 0.0

    def test_semantic_alone_equals_baseline(self, model, audio, prosody_indices, timbre_vector):
        """With lambdas=0 and prosody/timbre fed, output should equal
        running only semantic through the same path."""
        with torch.no_grad():
            semantic = model.wrapper.encode_semantic(audio)
        semantic = semantic.float()
        T_moss = semantic.shape[1]
        lm_dtype = next(model.get_language_model().parameters()).dtype

        with torch.no_grad():
            H_baseline = model.fusion(
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
            p_emb = model.prosody_embedding(prosody_indices)
            P = model.temporal_pool(p_emb)
            if P.shape[1] != T_moss:
                P = P.transpose(1, 2)
                P = torch.nn.functional.adaptive_avg_pool1d(P, T_moss)
                P = P.transpose(1, 2)
            t_proj = model.timbre_projection(timbre_vector)
            T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)
            H_full = model.fusion(
                semantic, prosody=P, timbre=T, content=None, acoustic=None,
            )
            H_full_lm = H_full.to(dtype=lm_dtype)
            lm_out_full = model.get_language_model()(
                inputs_embeds=H_full_lm
            ).last_hidden_state.float()
            pooled_full = lm_out_full.mean(dim=1)
            logits_full = model.classifier(pooled_full)

        assert torch.allclose(logits_baseline, logits_full, atol=1e-4)


class TestAmyStreamConfig:
    """Verify stream activation config controls module construction."""

    def test_disabled_streams_not_instantiated(self):
        """Content and acoustic modules should not exist when disabled."""
        vectors = torch.randn(1024, 8)
        config = {"prosody": True, "content": False, "acoustic": False, "timbre": True}
        model = AmyForProsodyClassification(
            warm_start_vectors=vectors,
            stream_config=config,
            device="cpu",
        )
        assert hasattr(model, "prosody_embedding")
        assert hasattr(model, "timbre_projection")
        assert not hasattr(model, "content_embedding")
        assert not hasattr(model, "acoustic_embedding")

    def test_config_key_missing_for_disabled_streams(self):
        """Missing keys in config default to False (disabled)."""
        vectors = torch.randn(1024, 8)
        config = {"prosody": True, "timbre": True}
        model = AmyForProsodyClassification(
            warm_start_vectors=vectors,
            stream_config=config,
            device="cpu",
        )
        assert hasattr(model, "prosody_embedding")
        assert hasattr(model, "timbre_projection")
        assert not hasattr(model, "content_embedding")
        assert not hasattr(model, "acoustic_embedding")

    def test_config_stored_as_attribute(self):
        """Stream config should be accessible as an attribute."""
        vectors = torch.randn(1024, 8)
        config = {"prosody": True, "content": True, "acoustic": False, "timbre": True}
        model = AmyForProsodyClassification(
            warm_start_vectors=vectors,
            stream_config=config,
            device="cpu",
        )
        assert model.stream_config == config


class TestAmyTemporalAlignment:
    """Verify FACodec 80Hz stream aligns to MOSS-Audio ~12.5Hz frames."""

    @pytest.fixture
    def model(self):
        vectors = make_prosody_codebook_vectors()
        return AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device="cpu",
        )

    def test_prosody_pool_matches_semantic_frames(self, model):
        """Pooled prosody [B, T_moss, 2560] must have same T_moss as semantic."""
        audio = torch.randn(2, 48000)
        prosody_indices = torch.randint(0, 1024, (2, 1, 240))

        with torch.no_grad():
            semantic = model.wrapper.encode_semantic(audio)
        T_moss = semantic.shape[1]

        p_emb = model.prosody_embedding(prosody_indices)
        P = model.temporal_pool(p_emb)

        if P.shape[1] != T_moss:
            P = P.transpose(1, 2)
            P = torch.nn.functional.adaptive_avg_pool1d(P, T_moss)
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
            semantic = model.wrapper.encode_semantic(audio)
        T_moss = semantic.shape[1]

        t_proj = model.timbre_projection(timbre)
        T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)

        assert T.shape[1] == T_moss
        assert T.shape[0] == 1
        assert T.shape[2] == 2560
