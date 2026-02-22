from __future__ import annotations

import numpy as np
import torch


def build_patch_assignments(num_checks: int, d: int) -> np.ndarray:
    side = max(1, int(np.ceil(np.sqrt(num_checks))))
    patch_side = max(1, int(np.ceil(side / 2)))
    assign = np.zeros(num_checks, dtype=np.int64)
    for j in range(num_checks):
        r, c = divmod(j, side)
        assign[j] = (r // 2) * patch_side + (c // 2)
    return assign


def build_edge_s2l_with_shared_interface(assign_s: np.ndarray, A_cc) -> torch.Tensor:
    src, dst = [], []
    for j in range(len(assign_s)):
        primary = int(assign_s[j])
        src.append(j)
        dst.append(primary)
        for k in A_cc[j].indices:
            p2 = int(assign_s[k])
            if p2 != primary:
                src.append(j)
                dst.append(p2)
    edges = sorted(set(zip(src, dst)))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def classify_patch_types(assign_s: np.ndarray, check_boundary: np.ndarray) -> np.ndarray:
    n_patch = int(assign_s.max()) + 1
    patch_type = np.zeros(n_patch, dtype=np.int64)
    for p in range(n_patch):
        b = check_boundary[assign_s == p]
        if np.any(b == 3):
            patch_type[p] = 3
        elif np.any(b == 1):
            patch_type[p] = 1
        elif np.any(b == 2):
            patch_type[p] = 2
    return patch_type
