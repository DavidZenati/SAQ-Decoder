from __future__ import annotations

import torch
import torch.nn.functional as F


def syndrome_consistency_loss(qubit_logits: torch.Tensor, H: torch.Tensor, syndrome: torch.Tensor) -> torch.Tensor:
    if qubit_logits is None:
        return torch.tensor(0.0, device=syndrome.device)
    p = torch.sigmoid(qubit_logits[:, 1])
    losses = []
    for j in range(H.shape[0]):
        support = H[j].bool()
        pred = 0.5 * (1 - torch.prod(1 - 2 * p[support])) if support.any() else torch.tensor(0.0, device=p.device)
        losses.append(F.binary_cross_entropy(pred, syndrome[j].float()))
    return torch.stack(losses).mean()
