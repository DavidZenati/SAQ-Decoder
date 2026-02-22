from __future__ import annotations

import torch
from torch import nn

from banr_saq.config import BANRSAQConfig

from .banr_saq_encoder import BANRSAQEncoder
from .readout_heads import ReadoutHeads


class BANRSAQ(nn.Module):
    def __init__(self, cfg: BANRSAQConfig, logical_classes: int):
        super().__init__()
        c = cfg.embed_dim
        self.cfg = cfg

        self.emb_syn = nn.Embedding(2, c)
        self.emb_check_type = nn.Embedding(2, c)
        self.emb_boundary = nn.Embedding(4, c)
        self.emb_dist = nn.Embedding(cfg.dist_clip + 1, c)
        self.patch_emb = nn.Embedding(4, c)
        self.global_token = nn.Parameter(torch.zeros(1, c))

        self.lp_head = nn.Sequential(nn.Linear(1, c), nn.GELU(), nn.Linear(c, logical_classes))
        self.init_patch = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, c))

        self.encoder = BANRSAQEncoder(c, cfg.num_heads, cfg.ff_mult)
        self.readout = ReadoutHeads(c, logical_classes)

    def build_tokens(self, syndrome: torch.Tensor, check_type: torch.Tensor, check_boundary: torch.Tensor, check_dist: torch.Tensor, patch_type: torch.Tensor, patch_index_per_check: torch.Tensor):
        s_tokens = self.emb_syn(syndrome.long()) + self.emb_check_type(check_type.long()) + self.emb_boundary(check_boundary.long()) + self.emb_dist(check_dist.long())
        s0 = torch.cat([self.global_token, s_tokens], dim=0)
        n_patch = int(patch_type.max().item()) + 1
        patch_sum = torch.zeros(n_patch, s_tokens.shape[-1], device=s_tokens.device)
        patch_sum.index_add_(0, patch_index_per_check, s_tokens)
        cnt = torch.zeros(n_patch, device=s_tokens.device)
        cnt.index_add_(0, patch_index_per_check, torch.ones_like(patch_index_per_check, dtype=torch.float))
        l0 = self.init_patch(patch_sum / cnt.clamp_min(1).unsqueeze(-1)) + self.patch_emb(patch_type)
        return s0, l0

    def forward(self, batch: dict):
        s0, l0 = self.build_tokens(
            batch['syndrome'], batch['check_type'], batch['check_boundary'], batch['check_dist'], batch['patch_type_0'], batch['primary_patch_0']
        )
        prior_logits = self.lp_head(batch['syndrome'].float().mean().view(1, 1)).squeeze(0)

        sK, lK, _, _, multiscale_l = self.encoder(
            s0,
            l0,
            batch['edges'],
            batch['bctxs'],
            batch['assignments'],
            batch['primary_patch'],
            batch['edge_s2l_next'],
        )
        logits, scale_logits, qubit_logits = self.readout(sK, lK, multiscale_l)
        return {
            'prior_logits': prior_logits,
            'logical_logits': logits,
            'scale_logits': scale_logits,
            'qubit_logits': qubit_logits,
        }
