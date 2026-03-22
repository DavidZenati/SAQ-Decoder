from __future__ import annotations

import torch
import torch.nn.functional as F


def rg_consistency_loss(scale_logits: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    if not scale_logits:
        return torch.tensor(0.0, device=target.device)
    losses = [F.cross_entropy(logit.unsqueeze(0), target.unsqueeze(0)) for logit in scale_logits]
    return torch.stack(losses).mean()
