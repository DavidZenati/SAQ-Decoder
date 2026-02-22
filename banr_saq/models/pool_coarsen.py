from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .renorm_cell import BoundaryContext, ScaleEdges
from .scatter_ops import scatter_max, scatter_mean


@dataclass
class Assignments:
    assign_s: torch.Tensor
    assign_l: torch.Tensor


class PoolCoarsen(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pool_s = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pool_l = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, s: torch.Tensor, l: torch.Tensor, assign: Assignments, bctx_next: BoundaryContext, edge_s2l_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, ScaleEdges, BoundaryContext]:
        n_s = int(assign.assign_s.max().item()) + 1
        n_l = int(assign.assign_l.max().item()) + 1

        s_m = scatter_mean(s[1:], assign.assign_s, n_s)
        s_x = scatter_max(s[1:], assign.assign_s, n_s)
        s_next = torch.cat([s[:1], self.pool_s(torch.cat([s_m, s_x], dim=-1))], dim=0)

        l_m = scatter_mean(l, assign.assign_l, n_l)
        l_x = scatter_max(l, assign.assign_l, n_l)
        l_next = self.pool_l(torch.cat([l_m, l_x], dim=-1))

        return s_next, l_next, ScaleEdges(edge_s2l=edge_s2l_next), bctx_next
