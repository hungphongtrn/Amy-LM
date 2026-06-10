"""Residual fusion module: combines semantic with optional prosody, content, acoustic, and timbre streams."""
import torch
import torch.nn as nn


class ResidualFusion(nn.Module):
    """Gated residual fusion with per-stream normalization.

    H = LayerNorm(S + Σ λ_i · norm_i(stream_i))

    Each stream is first normalized via its own LayerNorm so contributions
    have unit magnitude regardless of raw embedding scale. Gates (λ) are
    initialized at 1.0 so FACodec modules receive gradient from epoch 1.
    Disabled streams (None) are excluded entirely.

    Args:
        hidden_dim: Dimensionality of all streams (default=2560)
        dropout: Dropout probability on the residual sum before the outer LayerNorm (default=0.1).
            Preserves identity when all λ=0 (dropout(0)=0).
    """

    def __init__(self, hidden_dim: int = 2560, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_p = nn.LayerNorm(hidden_dim)
        self.norm_c = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.lambda_p = nn.Parameter(torch.ones(1))
        self.lambda_c = nn.Parameter(torch.ones(1))
        self.lambda_a = nn.Parameter(torch.ones(1))
        self.lambda_t = nn.Parameter(torch.ones(1))

    def forward(
        self,
        semantic: torch.Tensor,                   # [B, T, D]
        prosody: torch.Tensor | None = None,      # [B, T, D]
        content: torch.Tensor | None = None,      # [B, T, D]
        acoustic: torch.Tensor | None = None,     # [B, T, D]
        timbre: torch.Tensor | None = None,       # [B, T, D] (pre-broadcast)
    ) -> torch.Tensor:
        """Fuse streams via gated residual summation.

        Args:
            semantic: Base semantic embeddings [B, T, D]
            prosody: Optional prosody embeddings [B, T, D]
            content: Optional content embeddings [B, T, D]
            acoustic: Optional acoustic embeddings [B, T, D]
            timbre: Optional timbre embeddings [B, T, D] (must be pre-broadcast)

        Returns:
            Fused embeddings [B, T, D]
        """
        residual = torch.zeros_like(semantic)
        if prosody is not None:
            residual = residual + self.lambda_p * self.norm_p(prosody)
        if content is not None:
            residual = residual + self.lambda_c * self.norm_c(content)
        if acoustic is not None:
            residual = residual + self.lambda_a * self.norm_a(acoustic)
        if timbre is not None:
            residual = residual + self.lambda_t * self.norm_t(timbre)
        return self.norm(semantic + self.dropout(residual))
