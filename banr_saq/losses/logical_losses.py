from __future__ import annotations

import torch
import torch.nn.functional as F


def logical_prior_loss(prior_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(prior_logits.unsqueeze(0), target.unsqueeze(0))


def logical_classification_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
