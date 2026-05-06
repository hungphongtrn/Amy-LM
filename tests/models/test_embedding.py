import pytest
import torch
from src.models.embedding import ProsodyEmbedding, TimbreEmbedding

PROSODY_VOCAB = 1024
TIMBRE_VOCAB = 256
EMBED_DIM = 2560


def make_prosody_indices(batch=2, seq=25):
    return torch.randint(0, PROSODY_VOCAB, (batch, seq))


def make_timbre_indices(batch=2):
    return torch.randint(0, TIMBRE_VOCAB, (batch,))


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
        indices = torch.tensor([7, 7])
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


class TestTimbreEmbedding:
    @pytest.fixture
    def random_emb(self):
        return TimbreEmbedding(
            vocab_size=TIMBRE_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="random",
        )

    def test_random_init_output_shape(self, random_emb):
        indices = make_timbre_indices(batch=4)
        out = random_emb(indices)
        assert out.shape == (4, EMBED_DIM)

    def test_per_sample_constant_vector(self, random_emb):
        """Timbre embedding is per-sample: same index repeated should give same vector."""
        indices = torch.tensor([42, 42])
        out = random_emb(indices)
        assert torch.allclose(out[0], out[1])

    def test_warm_start_shape(self):
        codebook = torch.randn(TIMBRE_VOCAB, 320)
        emb = TimbreEmbedding(
            vocab_size=TIMBRE_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="warm_start",
            warm_start_vectors=codebook,
        )
        indices = make_timbre_indices()
        out = emb(indices)
        assert out.shape == (2, EMBED_DIM)

    def test_broadcastable_to_sequence_dim(self, random_emb):
        """Timbre vector should be broadcastable to (B, T, D) for downstream fusion."""
        indices = make_timbre_indices(batch=4)
        out = random_emb(indices)
        out_expanded = out.unsqueeze(1).expand(4, 30, EMBED_DIM)
        assert out_expanded.shape == (4, 30, EMBED_DIM)
        for b in range(4):
            for t in range(30):
                assert torch.equal(out_expanded[b, t], out[b])

    def test_warm_start_projector_is_frozen(self):
        """Warm-start projector is static — no grad after init."""
        codebook = torch.randn(TIMBRE_VOCAB, 320)
        emb = TimbreEmbedding(
            vocab_size=TIMBRE_VOCAB,
            embed_dim=EMBED_DIM,
            init_strategy="warm_start",
            warm_start_vectors=codebook,
        )
        assert hasattr(emb, '_projector'), "projector should exist in warm_start mode"
        for name, param in emb.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"
