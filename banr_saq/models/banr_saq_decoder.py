from __future__ import annotations

import torch
from torch import nn

from .renorm_cell import BoundaryContext, RenormCell, ScaleEdges


class BANRSAQDecoder(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int):
        super().__init__()
        self.cell = RenormCell(dim, heads, ff_mult)
        self.fuse_s = nn.Linear(2 * dim, dim)
        self.fuse_l = nn.Linear(2 * dim, dim)

    def forward(self, s: torch.Tensor, l: torch.Tensor, skip_s: list[torch.Tensor], skip_l: list[torch.Tensor], assignments_rev: list[torch.Tensor], edges_rev: list[ScaleEdges], bctx_rev: list[BoundaryContext], primary_patch_rev: list[torch.Tensor]):
        for i in range(len(skip_s)):
            assign_s, assign_l = assignments_rev[i]
            s_up = s[1:][assign_s]
            s = torch.cat([s[:1], self.fuse_s(torch.cat([s_up, skip_s[i][1:]], dim=-1))], dim=0)
            l_up = l[assign_l]
            l = self.fuse_l(torch.cat([l_up, skip_l[i]], dim=-1))
            s, l = self.cell(s, l, edges_rev[i], bctx_rev[i], primary_patch_rev[i])
        return s, l
