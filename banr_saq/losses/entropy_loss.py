from __future__ import annotations

import torch


def logical_minimum_entropy_loss(qubit_logits: torch.Tensor, e_true: torch.Tensor, logical_ops: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if qubit_logits is None:
        return torch.tensor(0.0, device=e_true.device)
    q = torch.sigmoid((1 - 2 * e_true.float()) * qubit_logits[:, 1])
    vals = []
    for row in logical_ops:
        support = row.bool()
        prod = torch.prod(1 - 2 * q[support]) if support.any() else torch.tensor(1.0, device=q.device)
        vals.append(-torch.log((1 + prod).clamp_min(eps)))
    return torch.stack(vals).mean()
