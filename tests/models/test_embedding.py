import pytest
import torch
from src.models.embedding import ProsodyEmbedding, TimbreProjection, AcousticEmbedding, ContentEmbedding

PROSODY_VOCAB = 1024
TIMBRE_DIM = 256
EMBED_DIM = 2560


def make_prosody_indices(batch=2, seq=25):
    """Return prosody indices with shape [B, 1, T] (codebook axis preserved)."""
    return torch.randint(0, PROSODY_VOCAB, (batch, 1, seq))


def make_acoustic_indices(batch=2, seq=80):
    """Return acoustic indices with shape [B, 3, T] (3 codebooks)."""
    return torch.randint(0, PROSODY_VOCAB, (batch, 3, seq))


def make_content_indices(batch=2, seq=80):
    """Return content indices with shape [B, 2, T] (2 codebooks)."""
    return torch.randint(0, PROSODY_VOCAB, (batch, 2, seq))


def make_timbre_vectors(batch=2):
    """Return timbre vectors with shape [B, 256] (continuous float32)."""
    return torch.randn(batch, TIMBRE_DIM)


class TestProsodyEmbedding:
    @pytest.fixture
    def random_emb(self):
        return ProsodyEmbedding(
            vocab_size=PROSODY_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="random",
            init_std=0.02,
        )

    def test_random_init_output_shape(self, random_emb):
        """Prosody indices now have shape [B, 1, T] with codebook axis."""
        indices = make_prosody_indices(batch=2, seq=25)
        out = random_emb(indices)
        assert out.shape == (2, 25, EMBED_DIM)

    def test_random_init_weight_stats(self, random_emb):
        weight = random_emb.weight
        mean = weight.mean().item()
        std = weight.std().item()
        assert abs(mean) < 0.1
        assert 0.01 < std < 0.03

    def test_random_init_same_index_same_vector(self, random_emb):
        indices = torch.tensor([[7], [7]])  # Shape: [2, 1]
        out = random_emb(indices)
        assert torch.allclose(out[0], out[1])

    def test_warm_start_produces_correct_shape(self):
        codebook = torch.randn(PROSODY_VOCAB * 2, 32)
        emb = ProsodyEmbedding(
            vocab_size=PROSODY_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="warm_start",
            warm_start_vectors=codebook,
        )
        indices = make_prosody_indices()
        out = emb(indices)
        assert out.shape == (2, 25, EMBED_DIM)

    def test_warm_start_projector_is_frozen(self):
        """Warm-start projector is static — no grad after init."""
        codebook = torch.randn(PROSODY_VOCAB * 2, 32)
        emb = ProsodyEmbedding(
            vocab_size=PROSODY_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="warm_start",
            warm_start_vectors=codebook,
        )
        assert hasattr(emb, '_projector'), "projector should exist in warm_start mode"
        for name, param in emb.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_invalid_init_strategy_raises(self):
        with pytest.raises(ValueError, match="init_strategy"):
            ProsodyEmbedding(
                vocab_size=PROSODY_VOCAB,
                embed_dim=EMBED_DIM,
                init_strategy="nonexistent",
            )

    def test_warm_start_without_vectors_raises(self):
        with pytest.raises(ValueError, match="warm_start_vectors"):
            ProsodyEmbedding(
                vocab_size=PROSODY_VOCAB,
                embed_dim=EMBED_DIM,
                init_strategy="warm_start",
            )


class TestTimbreProjection:
    """Tests for TimbreProjection — linear projection of continuous timbre vectors."""
    
    @pytest.fixture
    def projection(self):
        return TimbreProjection(timbre_dim=TIMBRE_DIM, output_dim=EMBED_DIM)

    def test_output_shape(self, projection):
        """Input: [B, 256] -> Output: [B, 2560]"""
        timbre_vectors = make_timbre_vectors(batch=4)
        out = projection(timbre_vectors)
        assert out.shape == (4, EMBED_DIM)

    def test_linear_transformation(self, projection):
        """Output is a linear transformation of input."""
        timbre_vectors = make_timbre_vectors(batch=2)
        out = projection(timbre_vectors)
        
        # Verify it's computed via linear layer
        expected = torch.nn.functional.linear(timbre_vectors, projection.linear.weight, projection.linear.bias)
        assert torch.allclose(out, expected)

    def test_broadcastable_to_sequence(self):
        """Timbre projection output can be broadcast to [B, T, D] for fusion."""
        projection = TimbreProjection(timbre_dim=TIMBRE_DIM, output_dim=EMBED_DIM)
        timbre_vectors = make_timbre_vectors(batch=4)
        out = projection(timbre_vectors)
        
        # Broadcast to sequence dimension
        out_expanded = out.unsqueeze(1).expand(4, 30, EMBED_DIM)
        assert out_expanded.shape == (4, 30, EMBED_DIM)
        
        # Verify all frames have same timbre vector
        for b in range(4):
            for t in range(30):
                assert torch.equal(out_expanded[b, t], out[b])

    def test_gradient_flows(self):
        """Gradients flow through the linear projection."""
        projection = TimbreProjection(timbre_dim=TIMBRE_DIM, output_dim=EMBED_DIM)
        timbre_vectors = make_timbre_vectors(batch=2).requires_grad_(True)
        
        out = projection(timbre_vectors)
        loss = out.sum()
        loss.backward()
        
        assert timbre_vectors.grad is not None
        assert projection.linear.weight.grad is not None
        assert projection.linear.bias.grad is not None

    def test_no_warm_start_or_init_strategy(self):
        """TimbreProjection is a simple linear layer — no embedding table features."""
        projection = TimbreProjection(timbre_dim=256, output_dim=2560)
        
        # Should not have embedding-specific attributes
        assert not hasattr(projection, 'embedding')
        assert not hasattr(projection, 'init_strategy')
        assert not hasattr(projection, 'weight')


class TestAcousticEmbedding:
    """Tests for AcousticEmbedding — sum of 3 codebook embeddings."""
    
    @pytest.fixture
    def acoustic_emb(self):
        return AcousticEmbedding(vocab_size=1024, num_codebooks=3, embed_dim=EMBED_DIM)

    def test_output_shape(self, acoustic_emb):
        """Input: [B, 3, T] -> Output: [B, T, D]"""
        indices = make_acoustic_indices(batch=2, seq=80)
        out = acoustic_emb(indices)
        assert out.shape == (2, 80, EMBED_DIM)

    def test_has_three_independent_embeddings(self, acoustic_emb):
        """AcousticEmbedding has 3 independent embedding tables."""
        assert len(acoustic_emb.embeddings) == 3
        
        # Each embedding should have its own parameters
        for i, emb in enumerate(acoustic_emb.embeddings):
            assert emb.weight.shape == (1024, EMBED_DIM)
            # Verify they're different (not shared)
            if i > 0:
                assert not torch.allclose(emb.weight, acoustic_emb.embeddings[i-1].weight)

    def test_sums_all_codebook_embeddings(self, acoustic_emb):
        """Output is sum of per-codebook embeddings, not average."""
        indices = make_acoustic_indices(batch=2, seq=10)
        out = acoustic_emb(indices)
        
        # Manual computation: sum each codebook independently
        manual_sum = torch.zeros(2, 10, EMBED_DIM)
        for cb in range(3):
            manual_sum = manual_sum + acoustic_emb.embeddings[cb](indices[:, cb, :])
        
        assert torch.allclose(out, manual_sum)

    def test_per_frame_same_indices_same_embedding(self, acoustic_emb):
        """Same codebook indices at same frame produce same embedding."""
        # Create indices where all codebooks have same value at frame 5
        indices = torch.zeros(2, 3, 10, dtype=torch.long)
        indices[:, :, 5] = 42  # All codebooks get index 42 at frame 5
        
        out = acoustic_emb(indices)
        # Frame 5 should have consistent embedding across batch
        assert torch.allclose(out[0, 5], out[1, 5])

    def test_gradient_flows(self, acoustic_emb):
        """Gradients flow through all embedding tables."""
        indices = make_acoustic_indices(batch=2, seq=10)
        out = acoustic_emb(indices)
        loss = out.sum()
        loss.backward()
        
        # All embeddings should have gradients
        for emb in acoustic_emb.embeddings:
            assert emb.weight.grad is not None


class TestContentEmbedding:
    """Tests for ContentEmbedding — sum of 2 codebook embeddings."""
    
    @pytest.fixture
    def content_emb(self):
        return ContentEmbedding(vocab_size=1024, num_codebooks=2, embed_dim=EMBED_DIM)

    def test_output_shape(self, content_emb):
        """Input: [B, 2, T] -> Output: [B, T, D]"""
        indices = make_content_indices(batch=2, seq=80)
        out = content_emb(indices)
        assert out.shape == (2, 80, EMBED_DIM)

    def test_has_two_independent_embeddings(self, content_emb):
        """ContentEmbedding has 2 independent embedding tables."""
        assert len(content_emb.embeddings) == 2
        
        for i, emb in enumerate(content_emb.embeddings):
            assert emb.weight.shape == (1024, EMBED_DIM)
            if i > 0:
                assert not torch.allclose(emb.weight, content_emb.embeddings[i-1].weight)

    def test_sums_all_codebook_embeddings(self, content_emb):
        """Output is sum of per-codebook embeddings."""
        indices = make_content_indices(batch=2, seq=10)
        out = content_emb(indices)
        
        manual_sum = torch.zeros(2, 10, EMBED_DIM)
        for cb in range(2):
            manual_sum = manual_sum + content_emb.embeddings[cb](indices[:, cb, :])
        
        assert torch.allclose(out, manual_sum)

    def test_disabled_stream_support(self, content_emb):
        """Content stream can be disabled (return None) in the fusion module."""
        # This test verifies the embedding itself works; fusion handles disabled streams
        indices = make_content_indices(batch=2, seq=10)
        out = content_emb(indices)
        assert out is not None
        assert out.shape == (2, 10, EMBED_DIM)

    def test_gradient_flows(self, content_emb):
        """Gradients flow through both embedding tables."""
        indices = make_content_indices(batch=2, seq=10)
        out = content_emb(indices)
        loss = out.sum()
        loss.backward()
        
        for emb in content_emb.embeddings:
            assert emb.weight.grad is not None
