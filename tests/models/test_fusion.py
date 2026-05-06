"""Tests for ResidualFusion — LayerNorm(semantic + λ·(prosody + timbre))."""
import pytest
import torch
from src.models.fusion import ResidualFusion

HIDDEN_DIM = 2560


def make_semantic(batch=2, seq=25):
    return torch.randn(batch, seq, HIDDEN_DIM)


def make_prosody(batch=2, seq=25):
    return torch.randn(batch, seq, HIDDEN_DIM)


def make_timbre(batch=2):
    return torch.randn(batch, HIDDEN_DIM)


class TestResidualFusion:
    @pytest.fixture
    def fusion(self):
        return ResidualFusion(hidden_dim=HIDDEN_DIM)

    def test_output_shape_matches_semantic(self, fusion):
        semantic = make_semantic(batch=2, seq=25)
        prosody = make_prosody(batch=2, seq=25)
        timbre = make_timbre(batch=2)
        out = fusion(semantic, prosody, timbre)
        assert out.shape == semantic.shape

    def test_output_within_reasonable_range(self, fusion):
        semantic = torch.randn(4, 10, HIDDEN_DIM)
        prosody = torch.randn(4, 10, HIDDEN_DIM)
        timbre = torch.randn(4, HIDDEN_DIM)
        out = fusion(semantic, prosody, timbre)
        assert out.std() < 3.0

    def test_identity_when_residual_is_zero(self, fusion):
        """When prosody+timbre are zero, output should equal LayerNorm(semantic)."""
        semantic = torch.randn(3, 15, HIDDEN_DIM)
        prosody = torch.zeros(3, 15, HIDDEN_DIM)
        timbre = torch.zeros(3, HIDDEN_DIM)
        out = fusion(semantic, prosody, timbre)
        expected = torch.nn.functional.layer_norm(
            semantic, (HIDDEN_DIM,), weight=fusion.norm.weight, bias=fusion.norm.bias, eps=1e-5
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_identity_when_lambda_is_zero(self, fusion):
        """At init (λ=0), output == LayerNorm(semantic) regardless of prosody/timbre."""
        semantic = make_semantic(batch=1, seq=10)
        prosody = torch.randn(1, 10, HIDDEN_DIM) * 100
        timbre = torch.randn(1, HIDDEN_DIM) * 100
        out = fusion(semantic, prosody, timbre)
        expected = torch.nn.functional.layer_norm(
            semantic, (HIDDEN_DIM,), weight=fusion.norm.weight, bias=fusion.norm.bias, eps=1e-5
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_lambda_is_learnable(self, fusion):
        semantic = make_semantic()
        prosody = make_prosody()
        timbre = make_timbre()
        out = fusion(semantic, prosody, timbre)
        loss = out.sum()
        loss.backward()
        assert fusion._lambda.grad is not None
        assert fusion._lambda.grad.item() != 0

    def test_gradient_flows_through_all_inputs(self, fusion):
        semantic = make_semantic()
        prosody = make_prosody()
        timbre = make_timbre()
        
        semantic = semantic.clone().requires_grad_(True)
        prosody = prosody.clone().requires_grad_(True)
        timbre = timbre.clone().requires_grad_(True)
        
        out = fusion(semantic, prosody, timbre)
        loss = out.sum()
        loss.backward()
        
        assert semantic.grad is not None
        assert prosody.grad is not None
        assert timbre.grad is not None

    def test_timbre_broadcast_handled_correctly(self, fusion):
        """Timbre is per-sample (B, D); must broadcast to (B, T, D) internally."""
        semantic = make_semantic(batch=4, seq=12)
        prosody = make_prosody(batch=4, seq=12)
        timbre = make_timbre(batch=4)
        out = fusion(semantic, prosody, timbre)
        assert out.shape == (4, 12, HIDDEN_DIM)

    def test_lambda_initialized_at_zero(self, fusion):
        assert fusion._lambda.item() == 0.0

    def test_lambda_can_be_set(self):
        fusion = ResidualFusion(hidden_dim=HIDDEN_DIM)
        with torch.no_grad():
            fusion._lambda.data.fill_(0.5)
        assert fusion._lambda.item() == 0.5
        
        semantic = make_semantic()
        prosody = make_prosody()
        timbre = make_timbre()
        out = fusion(semantic, prosody, timbre)
        
        zero_fusion = ResidualFusion(hidden_dim=HIDDEN_DIM)
        zero_out = zero_fusion(semantic, prosody, timbre)
        assert not torch.allclose(out, zero_out)
