"""Tests for ResidualFusion — gated residual fusion with per-stream lambdas."""
import pytest
import torch
from src.models.fusion import ResidualFusion

HIDDEN_DIM = 2560


def make_semantic(batch=2, seq=25):
    return torch.randn(batch, seq, HIDDEN_DIM)


def make_prosody(batch=2, seq=25):
    return torch.randn(batch, seq, HIDDEN_DIM)


def make_content(batch=2, seq=25):
    return torch.randn(batch, seq, HIDDEN_DIM)


def make_acoustic(batch=2, seq=25):
    return torch.randn(batch, seq, HIDDEN_DIM)


def make_timbre(batch=2, seq=25):
    """Timbre must be pre-broadcast to [B, T, D] before fusion."""
    return torch.randn(batch, HIDDEN_DIM).unsqueeze(1).expand(batch, seq, HIDDEN_DIM)


class TestResidualFusion:
    @pytest.fixture
    def fusion(self):
        return ResidualFusion(hidden_dim=HIDDEN_DIM)

    def test_output_shape_matches_semantic(self, fusion):
        """Output shape matches semantic input shape."""
        semantic = make_semantic(batch=2, seq=25)
        prosody = make_prosody(batch=2, seq=25)
        content = make_content(batch=2, seq=25)
        acoustic = make_acoustic(batch=2, seq=25)
        timbre = make_timbre(batch=2, seq=25)
        out = fusion(semantic, prosody, content, acoustic, timbre)
        assert out.shape == semantic.shape

    def test_output_within_reasonable_range(self, fusion):
        semantic = torch.randn(4, 10, HIDDEN_DIM)
        prosody = torch.randn(4, 10, HIDDEN_DIM)
        content = torch.randn(4, 10, HIDDEN_DIM)
        acoustic = torch.randn(4, 10, HIDDEN_DIM)
        timbre = make_timbre(batch=4, seq=10)
        out = fusion(semantic, prosody, content, acoustic, timbre)
        assert out.std() < 3.0

    def test_identity_at_init_zero_lambdas(self, fusion):
        """At init (all λ=0), output == LayerNorm(semantic) regardless of streams."""
        semantic = make_semantic(batch=1, seq=10)
        prosody = torch.randn(1, 10, HIDDEN_DIM) * 100
        content = torch.randn(1, 10, HIDDEN_DIM) * 100
        acoustic = torch.randn(1, 10, HIDDEN_DIM) * 100
        timbre = make_timbre(batch=1, seq=10) * 100
        
        out = fusion(semantic, prosody, content, acoustic, timbre)
        expected = torch.nn.functional.layer_norm(
            semantic, (HIDDEN_DIM,), weight=fusion.norm.weight, bias=fusion.norm.bias, eps=1e-5
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_all_lambdas_initialized_at_zero(self, fusion):
        """All per-stream lambdas start at zero."""
        assert fusion.lambda_p.item() == 0.0
        assert fusion.lambda_c.item() == 0.0
        assert fusion.lambda_a.item() == 0.0
        assert fusion.lambda_t.item() == 0.0

    def test_each_lambda_independently_learnable(self, fusion):
        """Each lambda can be set independently and affects output."""
        semantic = make_semantic(batch=2, seq=10)
        prosody = make_prosody(batch=2, seq=10)
        content = make_content(batch=2, seq=10)
        acoustic = make_acoustic(batch=2, seq=10)
        timbre = make_timbre(batch=2, seq=10)
        
        # Set only prosody lambda
        with torch.no_grad():
            fusion.lambda_p.data.fill_(0.5)
        
        out = fusion(semantic, prosody, content, acoustic, timbre)
        
        # With only prosody enabled, should be: LayerNorm(semantic + 0.5 * prosody)
        expected = torch.nn.functional.layer_norm(
            semantic + 0.5 * prosody, (HIDDEN_DIM,), 
            weight=fusion.norm.weight, bias=fusion.norm.bias, eps=1e-5
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_all_lambdas_can_be_set_independently(self):
        """Each lambda can be set to different values."""
        fusion = ResidualFusion(hidden_dim=HIDDEN_DIM)
        with torch.no_grad():
            fusion.lambda_p.data.fill_(0.1)
            fusion.lambda_c.data.fill_(0.2)
            fusion.lambda_a.data.fill_(0.3)
            fusion.lambda_t.data.fill_(0.4)
        
        assert fusion.lambda_p.item() == pytest.approx(0.1, abs=1e-6)
        assert fusion.lambda_c.item() == pytest.approx(0.2, abs=1e-6)
        assert fusion.lambda_a.item() == pytest.approx(0.3, abs=1e-6)
        assert fusion.lambda_t.item() == pytest.approx(0.4, abs=1e-6)

    def test_gradient_flows_through_all_enabled_streams(self, fusion):
        """Gradients flow through all streams when enabled."""
        semantic = make_semantic(batch=2, seq=10).requires_grad_(True)
        prosody = make_prosody(batch=2, seq=10).requires_grad_(True)
        content = make_content(batch=2, seq=10).requires_grad_(True)
        acoustic = make_acoustic(batch=2, seq=10).requires_grad_(True)
        timbre = make_timbre(batch=2, seq=10).requires_grad_(True)
        
        # Set all lambdas to non-zero so all streams contribute
        with torch.no_grad():
            fusion.lambda_p.data.fill_(0.5)
            fusion.lambda_c.data.fill_(0.5)
            fusion.lambda_a.data.fill_(0.5)
            fusion.lambda_t.data.fill_(0.5)
        
        out = fusion(semantic, prosody, content, acoustic, timbre)
        loss = out.sum()
        loss.backward()
        
        assert semantic.grad is not None
        assert prosody.grad is not None
        assert content.grad is not None
        assert acoustic.grad is not None
        assert timbre.grad is not None

    def test_prosody_only_enabled(self, fusion):
        """When only prosody is provided (others None), only lambda_p applies."""
        semantic = make_semantic(batch=2, seq=10)
        prosody = make_prosody(batch=2, seq=10)
        
        with torch.no_grad():
            fusion.lambda_p.data.fill_(0.5)
        
        out = fusion(semantic, prosody=prosody, content=None, acoustic=None, timbre=None)
        expected = torch.nn.functional.layer_norm(
            semantic + 0.5 * prosody, (HIDDEN_DIM,),
            weight=fusion.norm.weight, bias=fusion.norm.bias, eps=1e-5
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_none_streams_excluded(self, fusion):
        """None streams are completely excluded from computation."""
        semantic = make_semantic(batch=2, seq=10)
        prosody = make_prosody(batch=2, seq=10)
        
        # Even if prosody lambda is zero, setting content/acoustic/timbre to None
        # should yield same result as just prosody
        out_with_nones = fusion(
            semantic, prosody=prosody, content=None, acoustic=None, timbre=None
        )
        
        # Create dummy streams (but with lambda_c/a/t = 0 they shouldn't contribute)
        content = make_content(batch=2, seq=10)
        acoustic = make_acoustic(batch=2, seq=10)
        timbre = make_timbre(batch=2, seq=10)
        
        out_with_zeros = fusion(
            semantic, prosody=prosody, content=content, acoustic=acoustic, timbre=timbre
        )
        
        # Both should be identical since lambdas for c/a/t are 0
        assert torch.allclose(out_with_nones, out_with_zeros, atol=1e-5)

    def test_timbre_pre_broadcast_required(self):
        """Timbre must be pre-broadcast to [B, T, D] — fusion no longer handles broadcasting."""
        fusion = ResidualFusion(hidden_dim=HIDDEN_DIM)
        semantic = make_semantic(batch=2, seq=10)
        timbre_per_sample = torch.randn(2, HIDDEN_DIM)  # [B, D] not broadcasted
        
        # This should fail or produce wrong shape if we try to use it directly
        # Our implementation assumes timbre is already [B, T, D]
        with torch.no_grad():
            fusion.lambda_t.data.fill_(0.5)
        
        # Timpre must be pre-broadcast
        timbre_broadcast = timbre_per_sample.unsqueeze(1).expand(2, 10, HIDDEN_DIM)
        out = fusion(semantic, prosody=None, content=None, acoustic=None, timbre=timbre_broadcast)
        assert out.shape == (2, 10, HIDDEN_DIM)

    def test_gradient_flows_through_lambdas(self, fusion):
        """Gradients flow through all lambda parameters."""
        semantic = make_semantic(batch=2, seq=10)
        prosody = make_prosody(batch=2, seq=10)
        content = make_content(batch=2, seq=10)
        acoustic = make_acoustic(batch=2, seq=10)
        timbre = make_timbre(batch=2, seq=10)
        
        # Set all lambdas to non-zero
        with torch.no_grad():
            fusion.lambda_p.data.fill_(0.5)
            fusion.lambda_c.data.fill_(0.5)
            fusion.lambda_a.data.fill_(0.5)
            fusion.lambda_t.data.fill_(0.5)
        
        out = fusion(semantic, prosody, content, acoustic, timbre)
        loss = out.sum()
        loss.backward()
        
        # All lambdas should have gradients
        assert fusion.lambda_p.grad is not None
        assert fusion.lambda_c.grad is not None
        assert fusion.lambda_a.grad is not None
        assert fusion.lambda_t.grad is not None

    def test_mixed_enabled_disabled_streams(self, fusion):
        """Can have arbitrary mix of enabled and disabled streams."""
        semantic = make_semantic(batch=2, seq=10)
        prosody = make_prosody(batch=2, seq=10)
        acoustic = make_acoustic(batch=2, seq=10)
        
        with torch.no_grad():
            fusion.lambda_p.data.fill_(0.3)
            fusion.lambda_a.data.fill_(0.7)
        
        # Only prosody and acoustic enabled
        out = fusion(
            semantic, 
            prosody=prosody, 
            content=None, 
            acoustic=acoustic, 
            timbre=None
        )
        
        expected = torch.nn.functional.layer_norm(
            semantic + 0.3 * prosody + 0.7 * acoustic, (HIDDEN_DIM,),
            weight=fusion.norm.weight, bias=fusion.norm.bias, eps=1e-5
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_no_streams_enabled_identity(self, fusion):
        """When all streams are None, output is just LayerNorm(semantic)."""
        semantic = make_semantic(batch=2, seq=10)
        
        out = fusion(semantic, prosody=None, content=None, acoustic=None, timbre=None)
        expected = torch.nn.functional.layer_norm(
            semantic, (HIDDEN_DIM,),
            weight=fusion.norm.weight, bias=fusion.norm.bias, eps=1e-5
        )
        assert torch.allclose(out, expected, atol=1e-5)
