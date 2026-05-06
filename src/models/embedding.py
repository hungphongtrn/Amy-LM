"""Embedding tables for prosody and timbre discrete codebook indices."""
from typing import Optional
import torch
import torch.nn as nn


class ProsodyEmbedding(nn.Module):
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
        w = self.weight
        return nn.functional.embedding(indices, w)


class TimbreEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
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
        if indices.dim() != 1:
            raise ValueError(
                f"TimbreEmbedding expects 1D indices (batch,), got shape {indices.shape}"
            )
        w = self.weight
        return nn.functional.embedding(indices, w)
