"""Embedding tables for FACodec streams: prosody, timbre, content, acoustic."""
from typing import Optional
import torch
import torch.nn as nn


class ProsodyEmbedding(nn.Module):
    """Embed single-codebook prosody stream into MOSS-Audio embedding space.
    
    Args:
        vocab_size: Codebook vocabulary size (default=1024)
        embed_dim: Target embedding dimension (default=2560)
        init_strategy: 'random' or 'warm_start'
        init_std: Standard deviation for random init (default=0.02)
        warm_start_vectors: FACodec codebook vectors for warm_start mode
    """
    
    def __init__(
        self,
        vocab_size: int = 1024,
        embed_dim: int = 2560,
        init_strategy: str = "random",
        init_std: float = 0.02,
        warm_start_vectors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if init_strategy not in ("random", "warm_start"):
            raise ValueError(
                f"init_strategy must be 'random' or 'warm_start', got '{init_strategy}'"
            )
        if init_strategy == "warm_start" and warm_start_vectors is None:
            raise ValueError(
                "warm_start_vectors is required when init_strategy='warm_start'"
            )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init_strategy = init_strategy

        if init_strategy == "random":
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        elif init_strategy == "warm_start":
            codebook_dim = warm_start_vectors.shape[-1]
            self._projector = nn.Linear(codebook_dim, embed_dim, bias=False)
            with torch.no_grad():
                self._projector.weight.data.normal_(mean=0.0, std=0.02)
            for param in self._projector.parameters():
                param.requires_grad = False
            self.register_buffer("_facodec_weights", warm_start_vectors.detach())

    @property
    def weight(self) -> torch.Tensor:
        if self.init_strategy == "random":
            return self.embedding.weight
        projected = self._projector(self._facodec_weights)
        num_entries = projected.shape[0]
        target_entries = self.vocab_size
        if num_entries >= target_entries:
            return projected[:target_entries]
        repeats = (target_entries + num_entries - 1) // num_entries
        tiled = projected.repeat(repeats, 1)
        return tiled[:target_entries]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Embed single-codebook prosody stream.

        Args:
            indices: [B, 1, T80] int64 from FACodec vq_id[:1]

        Returns:
            [B, T80, embed_dim] float32
        """
        # Squeeze the codebook axis: [B, 1, T] -> [B, T]
        if indices.dim() == 3:
            indices = indices.squeeze(1)
        w = self.weight
        return nn.functional.embedding(indices, w)


class TimbreProjection(nn.Module):
    """Project continuous timbre vector into MOSS-Audio embedding space.

    Replaces TimbreEmbedding (discrete lookup over integer indices).
    The timbre vector is an utterance-level float32 tensor from FACodec spk_embs.
    D_timbre = 256 (confirmed via local checkpoint run).

    Args:
        timbre_dim: Dimensionality of input timbre vector (default=256)
        output_dim: Target embedding dimension (default=2560, MOSS-Audio hidden dim)
    """

    def __init__(self, timbre_dim: int = 256, output_dim: int = 2560):
        super().__init__()
        self.timbre_dim = timbre_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(timbre_dim, output_dim)

    def forward(self, timbre_vector: torch.Tensor) -> torch.Tensor:
        """Project utterance-level timbre.

        Args:
            timbre_vector: [B, 256] float32 from FACodec spk_embs

        Returns:
            [B, output_dim] float32 — broadcast to frames during fusion
        """
        return self.linear(timbre_vector)


class AcousticEmbedding(nn.Module):
    """Embed 3 acoustic residual codebooks into MOSS-Audio embedding space.

    Each codebook has its own independent embedding table (vocab → D).
    Per-frame output is the sum of all three codebook embeddings at that frame.

    Args:
        vocab_size: Codebook vocabulary size (default=1024, FACodec codebook vocab)
        num_codebooks: Number of acoustic codebooks (default=3)
        embed_dim: Target embedding dimension (default=2560)
    """

    def __init__(self, vocab_size: int = 1024, num_codebooks: int = 3, embed_dim: int = 2560):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim
        # Each codebook gets its own independent embedding table
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for _ in range(num_codebooks)
        ])

    def forward(self, codebook_indices: torch.Tensor) -> torch.Tensor:
        """Embed multi-codebook acoustic stream.

        Args:
            codebook_indices: [B, 3, T80] int64 from FACodec vq_id[3:]

        Returns:
            [B, T80, embed_dim] float32 — sum of per-codebook embeddings
        """
        # Sum per-codebook embeddings along the codebook dimension
        result = torch.zeros(
            codebook_indices.shape[0], codebook_indices.shape[2], self.embed_dim,
            device=codebook_indices.device, dtype=torch.float32
        )
        for cb in range(self.num_codebooks):
            result = result + self.embeddings[cb](codebook_indices[:, cb, :])
        return result


class ContentEmbedding(nn.Module):
    """Embed 2 content codebooks into MOSS-Audio embedding space.

    Same architecture as AcousticEmbedding but with 2 codebooks.
    Disabled in the initial experiment (Stream Activation Config).

    Args:
        vocab_size: Codebook vocabulary size (default=1024)
        num_codebooks: Number of content codebooks (default=2)
        embed_dim: Target embedding dimension (default=2560)
    """

    def __init__(self, vocab_size: int = 1024, num_codebooks: int = 2, embed_dim: int = 2560):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for _ in range(num_codebooks)
        ])

    def forward(self, codebook_indices: torch.Tensor) -> torch.Tensor:
        """Embed multi-codebook content stream.

        Args:
            codebook_indices: [B, 2, T80] int64 from FACodec vq_id[1:3]

        Returns:
            [B, T80, embed_dim] float32 — sum of per-codebook embeddings
        """
        result = torch.zeros(
            codebook_indices.shape[0], codebook_indices.shape[2], self.embed_dim,
            device=codebook_indices.device, dtype=torch.float32
        )
        for cb in range(self.num_codebooks):
            result = result + self.embeddings[cb](codebook_indices[:, cb, :])
        return result

