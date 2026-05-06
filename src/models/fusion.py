"""Residual fusion module: combines semantic with optional prosody, content, acoustic, and timbre streams."""
import torch
import torch.nn as nn


class ResidualFusion(nn.Module):
    """Gated residual fusion: H = LayerNorm(S + Σ λ_i · stream_i)

    Each stream (prosody, content, acoustic, timbre) has its own
    learnable zero-initialized lambda gate. Disabled streams (None)
    are excluded from the computation entirely.

    Args:
        hidden_dim: Dimensionality of all streams (default=2560)
    """

    def __init__(self, hidden_dim: int = 2560):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        # Per-stream learnable lambdas, zero-initialized
        self.lambda_p = nn.Parameter(torch.zeros(1))  # Prosody gate
        self.lambda_c = nn.Parameter(torch.zeros(1))  # Content gate
        self.lambda_a = nn.Parameter(torch.zeros(1))  # Acoustic gate
        self.lambda_t = nn.Parameter(torch.zeros(1))  # Timbre gate

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
            residual = residual + self.lambda_p * prosody
        if content is not None:
            residual = residual + self.lambda_c * content
        if acoustic is not None:
            residual = residual + self.lambda_a * acoustic
        if timbre is not None:
            residual = residual + self.lambda_t * timbre
        return self.norm(semantic + residual)
