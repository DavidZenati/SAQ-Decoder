from __future__ import annotations

import torch
from torch import nn


class BoundaryFiLM(nn.Module):
    def __init__(self, context_dim: int, feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim * 2),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.mlp(context)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma) * x + beta
