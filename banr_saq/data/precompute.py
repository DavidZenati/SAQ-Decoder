from __future__ import annotations

import math

import numpy as np
import torch

from banr_saq.data.boundary_classifier import classify_boundaries
from banr_saq.data.code_builder import build_css_code
from banr_saq.data.patch_builder import build_edge_s2l_with_shared_interface, build_patch_assignments, classify_patch_types
from banr_saq.models.pool_coarsen import Assignments
from banr_saq.models.renorm_cell import BoundaryContext, ScaleEdges


def precompute_distance(d: int, family: str = 'rotated') -> dict:
    """Precompute graph/boundary/assignment tensors for a distance and code family."""
    code = build_css_code(d, family=family)
    H_X, H_Z, L = code['H_X'], code['H_Z'], code['L']
    H_full = np.vstack([H_X, H_Z]).astype(np.uint8)
    boundary = classify_boundaries(H_X, H_Z)

    K = int(math.ceil(math.log2(d)))
    edges, bctxs, assignments, primary_patch, edge_s2l_next = [], [], [], [], []

    assign_s = build_patch_assignments(H_full.shape[0], d)
    edge_s2l = build_edge_s2l_with_shared_interface(assign_s, boundary['A_cc'])
    patch_type = classify_patch_types(assign_s, boundary['check_boundary'])

    n_patches = int(assign_s.max()) + 1
    coarse_patch_count = max(1, n_patches // 2 + 1)
    for _ in range(K + 1):
        edges.append(ScaleEdges(edge_s2l=edge_s2l))
        bctxs.append(BoundaryContext(
            check_type=torch.tensor(boundary['check_type'], dtype=torch.long),
            check_boundary=torch.tensor(boundary['check_boundary'], dtype=torch.long),
            patch_type=torch.tensor(patch_type, dtype=torch.long),
        ))
        primary_patch.append(torch.tensor(assign_s, dtype=torch.long))

    for _ in range(K):
        assignments.append(
            Assignments(
                assign_s=torch.arange(H_full.shape[0]) % coarse_patch_count,
                assign_l=torch.arange(n_patches) % coarse_patch_count,
            )
        )
        edge_s2l_next.append(edge_s2l)

    return {
        'd': d,
        'family': code['family'],
        'K': K,
        'H_full': torch.tensor(H_full, dtype=torch.float32),
        'L': torch.tensor(L, dtype=torch.float32),
        'boundary': boundary,
        'edges': edges,
        'bctxs': bctxs,
        'assignments': assignments,
        'primary_patch': primary_patch,
        'edge_s2l_next': edge_s2l_next,
        'patch_type_0': torch.tensor(patch_type, dtype=torch.long),
    }
