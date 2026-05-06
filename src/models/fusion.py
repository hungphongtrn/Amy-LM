"""Residual fusion module: combines semantic, prosody, and timbre streams."""
import torch
import torch.nn as nn


class ResidualFusion(nn.Module):
    """Fuse semantic, prosody, and timbre via residual additive combination.
    
    Computes::
    
        out = LayerNorm(semantic + λ · (prosody + timbre))
    
    where λ is a learnable scalar initialized at zero.
    Timbre is per-sample ``(batch, dim)`` and is broadcast to the sequence
    dimension internally.
    
    Args:
        hidden_dim: Dimension of all input/output tensors (2560).
    """
    
    def __init__(self, hidden_dim: int = 2560):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self._lambda = nn.Parameter(torch.zeros(1))
    
    def forward(self, semantic, prosody, timbre):
        residual = (prosody + timbre.unsqueeze(1)) * self._lambda
        fused = semantic + residual
        return self.norm(fused)
