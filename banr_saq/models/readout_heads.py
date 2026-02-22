from __future__ import annotations

import torch
from torch import nn


class ReadoutHeads(nn.Module):
    def __init__(self, dim: int, logical_classes: int):
        super().__init__()
        self.log_head = nn.Linear(2 * dim, logical_classes)
        self.shared_probe = nn.Linear(dim, logical_classes)
        self.qubit_head = nn.Linear(dim, 4)

    def forward(self, sK: torch.Tensor, lK: torch.Tensor, multiscale_l: list[torch.Tensor], q0: torch.Tensor | None = None):
        h_glob = lK.mean(dim=0)
        logits = self.log_head(torch.cat([h_glob, sK[0]], dim=-1))
        scale_logits = [self.shared_probe(l.mean(dim=0)) for l in multiscale_l]
        qubit_logits = self.qubit_head(q0) if q0 is not None else None
        return logits, scale_logits, qubit_logits
