from __future__ import annotations

import torch
from torch import nn

from .pool_coarsen import Assignments, PoolCoarsen
from .renorm_cell import BoundaryContext, RenormCell, ScaleEdges


class BANRSAQEncoder(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int):
        super().__init__()
        self.cell = RenormCell(dim, heads, ff_mult)
        self.pool = PoolCoarsen(dim)

    def forward(self, s0: torch.Tensor, l0: torch.Tensor, edges: list[ScaleEdges], bctxs: list[BoundaryContext], assignments: list[Assignments], primary_patch: list[torch.Tensor], edge_s2l_next: list[torch.Tensor]):
        s, l = s0, l0
        skip_s, skip_l = [], []
        multiscale_l = [l0]
        for k in range(len(assignments)):
            s, l = self.cell(s, l, edges[k], bctxs[k], primary_patch[k])
            skip_s.append(s)
            skip_l.append(l)
            s, l, _, _ = self.pool(s, l, assignments[k], bctxs[k + 1], edge_s2l_next[k])
            multiscale_l.append(l)
        s, l = self.cell(s, l, edges[-1], bctxs[-1], primary_patch[-1])
        return s, l, skip_s, skip_l, multiscale_l
