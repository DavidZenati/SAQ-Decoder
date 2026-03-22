from __future__ import annotations

import torch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = scatter_sum(src, index, dim_size)
    counts = src.new_zeros(dim_size)
    counts.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    counts = counts.clamp_min(1.0).view(-1, *([1] * (src.dim() - 1)))
    return out / counts


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Autograd-safe scatter max without in-place read/write aliasing."""
    groups = []
    for g in range(dim_size):
        mask = index == g
        if torch.any(mask):
            groups.append(src[mask].max(dim=0).values)
        else:
            groups.append(src.new_zeros(src.shape[1:]))
    return torch.stack(groups, dim=0)


def scatter_softmax(logits: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Group softmax avoiding in-place writes on tensors participating in autograd."""
    max_vals = []
    for g in range(dim_size):
        mask = index == g
        if torch.any(mask):
            max_vals.append(logits[mask].max())
        else:
            max_vals.append(logits.new_tensor(float('-inf')))
    max_per = torch.stack(max_vals, dim=0)

    stable = logits - max_per[index]
    expv = torch.exp(stable)
    denom = logits.new_zeros(dim_size)
    denom.index_add_(0, index, expv)
    return expv / denom[index].clamp_min(1e-12)
