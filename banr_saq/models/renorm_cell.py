from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .boundary_film import BoundaryFiLM
from .scatter_ops import scatter_softmax, scatter_sum


@dataclass
class ScaleEdges:
    edge_s2l: torch.Tensor  # [2, E]


@dataclass
class BoundaryContext:
    check_type: torch.Tensor
    check_boundary: torch.Tensor
    patch_type: torch.Tensor


class RenormCell(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 4):
        super().__init__()
        self.dim = dim
        self.s_self = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.s_ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim))
        self.l_ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim))

        self.ln_s1 = nn.LayerNorm(dim)
        self.ln_s2 = nn.LayerNorm(dim)
        self.ln_l1 = nn.LayerNorm(dim)
        self.ln_l2 = nn.LayerNorm(dim)

        self.q_l = nn.Linear(dim, dim)
        self.k_s = nn.Linear(dim, dim)
        self.v_s = nn.Linear(dim, dim)
        self.out_l = nn.Linear(dim, dim)

        self.q_s = nn.Linear(dim, dim)
        self.k_l = nn.Linear(dim, dim)
        self.v_l = nn.Linear(dim, dim)
        self.out_s = nn.Linear(dim, dim)
        self.gate = nn.Linear(2 * dim, dim)

        self.check_ctx_emb = nn.Embedding(8, dim)
        self.patch_emb = nn.Embedding(4, dim)
        self.s_film = BoundaryFiLM(2 * dim, dim)
        self.l_film = BoundaryFiLM(dim, dim)

    @staticmethod
    def _align_1d_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad/truncate a 1D tensor to target length, preserving dtype/device."""
        if x.numel() == target_len:
            return x
        if x.numel() < target_len:
            pad = x.new_zeros(target_len - x.numel())
            return torch.cat([x, pad], dim=0)
        return x[:target_len]

    def _syndrome_self_attention(self, s: torch.Tensor) -> torch.Tensor:
        out, _ = self.s_self(
            self.ln_s1(s).unsqueeze(0),
            self.ln_s1(s).unsqueeze(0),
            self.ln_s1(s).unsqueeze(0),
            need_weights=False,
        )
        s = s + out.squeeze(0)
        return s + self.s_ff(self.ln_s2(s))

    def _logical_from_syndrome(self, s_no_global: torch.Tensor, l: torch.Tensor, edge_s2l: torch.Tensor) -> torch.Tensor:
        check_idx, patch_idx = edge_s2l
        valid = (
            (check_idx >= 0)
            & (check_idx < s_no_global.size(0))
            & (patch_idx >= 0)
            & (patch_idx < l.size(0))
        )
        if not torch.any(valid):
            return l + self.l_ff(self.ln_l2(l))

        check_idx = check_idx[valid]
        patch_idx = patch_idx[valid]

        q = self.q_l(self.ln_l1(l))[patch_idx]
        k = self.k_s(self.ln_s1(s_no_global))[check_idx]
        v = self.v_s(self.ln_s1(s_no_global))[check_idx]
        logits = (q * k).sum(-1) / (self.dim ** 0.5)
        alpha = scatter_softmax(logits, patch_idx, l.size(0)).unsqueeze(-1)
        agg = scatter_sum(alpha * v, patch_idx, l.size(0))
        l = l + self.out_l(agg)
        return l + self.l_ff(self.ln_l2(l))

    def _syndrome_from_logical(self, s_no_global: torch.Tensor, l: torch.Tensor, primary_patch: torch.Tensor) -> torch.Tensor:
        primary_patch = self._align_1d_length(primary_patch, s_no_global.size(0)).clamp(0, max(l.size(0) - 1, 0))
        q = self.q_s(self.ln_s1(s_no_global))
        k = self.k_l(self.ln_l1(l))[primary_patch]
        v = self.v_l(self.ln_l1(l))[primary_patch]
        g = torch.sigmoid(self.gate(torch.cat([q, k], dim=-1)))
        return s_no_global + g * self.out_s(v)

    def forward(self, s: torch.Tensor, l: torch.Tensor, edges: ScaleEdges, bctx: BoundaryContext, primary_patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self._syndrome_self_attention(s)

        n_checks = s.size(0) - 1
        check_boundary = self._align_1d_length(bctx.check_boundary, n_checks)
        check_type = self._align_1d_length(bctx.check_type, n_checks)
        patch_type = self._align_1d_length(bctx.patch_type, l.size(0))

        check_ctx = torch.cat(
            [
                self.check_ctx_emb(check_boundary + 4 * check_type),
                self.check_ctx_emb(check_type),
            ],
            dim=-1,
        )
        s_no_global = self.s_film(s[1:], check_ctx)

        l = self._logical_from_syndrome(s_no_global, l, edges.edge_s2l)
        s_no_global = self._syndrome_from_logical(s_no_global, l, primary_patch)
        l = self.l_film(l, self.patch_emb(patch_type))

        s = torch.cat([s[:1], s_no_global], dim=0)
        return s, l
