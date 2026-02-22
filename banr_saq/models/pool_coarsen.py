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

    @staticmethod
    def _align_index(index: torch.Tensor, src_len: int) -> torch.Tensor:
        """Pad/truncate index to match src length for scatter ops."""
        if index.numel() == src_len:
            return index
        if index.numel() < src_len:
            pad = index.new_zeros(src_len - index.numel())
            return torch.cat([index, pad], dim=0)
        return index[:src_len]

    @staticmethod
    def _sanitize_index(index: torch.Tensor, dim_size: int) -> torch.Tensor:
        if dim_size <= 0:
            return index.new_zeros(index.numel())
        return index.clamp(0, dim_size - 1)

    def forward(self, s: torch.Tensor, l: torch.Tensor, assign: Assignments, bctx_next: BoundaryContext, edge_s2l_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, ScaleEdges, BoundaryContext]:
        s_src = s[1:]
        l_src = l

        assign_s = self._align_index(assign.assign_s, s_src.size(0))
        assign_l = self._align_index(assign.assign_l, l_src.size(0))

        n_s = max(1, int(assign_s.max().item()) + 1) if assign_s.numel() else 1
        n_l = max(1, int(assign_l.max().item()) + 1) if assign_l.numel() else 1

        assign_s = self._sanitize_index(assign_s, n_s)
        assign_l = self._sanitize_index(assign_l, n_l)

        s_m = scatter_mean(s_src, assign_s, n_s)
        s_x = scatter_max(s_src, assign_s, n_s)
        s_next = torch.cat([s[:1], self.pool_s(torch.cat([s_m, s_x], dim=-1))], dim=0)

        l_m = scatter_mean(l_src, assign_l, n_l)
        l_x = scatter_max(l_src, assign_l, n_l)
        l_next = self.pool_l(torch.cat([l_m, l_x], dim=-1))

        return s_next, l_next, ScaleEdges(edge_s2l=edge_s2l_next), bctx_next
