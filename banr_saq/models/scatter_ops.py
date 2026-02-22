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
    out = src.new_full((dim_size,) + src.shape[1:], float('-inf'))
    for i in range(src.shape[0]):
        out[index[i]] = torch.maximum(out[index[i]], src[i])
    out[out == float('-inf')] = 0.0
    return out


def scatter_softmax(logits: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    max_per = logits.new_full((dim_size,), float('-inf'))
    for i, idx in enumerate(index):
        max_per[idx] = max(max_per[idx], logits[i])
    stable = logits - max_per[index]
    expv = torch.exp(stable)
    denom = logits.new_zeros(dim_size)
    denom.index_add_(0, index, expv)
    return expv / denom[index].clamp_min(1e-12)
