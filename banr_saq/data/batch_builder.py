from __future__ import annotations

import torch

from .noise_sampler import sample_depolarizing, syndrome_from_error


def make_single_sample(cache: dict, p: float = 0.1) -> dict:
    """Generate one on-the-fly training sample from cached code tensors.

    Data generation happens here:
      1) sample physical error e
      2) compute syndrome s = H e mod 2
      3) compute logical class y from L e mod 2
    """
    n = cache['H_full'].shape[1]
    e = sample_depolarizing(n, p)
    s = syndrome_from_error(cache['H_full'].numpy(), e)
    y_bits = (cache['L'].numpy() @ e) % 2
    y_class = int(sum(int(bit) << i for i, bit in enumerate(y_bits.tolist())))

    return {
        'syndrome': torch.tensor(s, dtype=torch.long),
        'check_type': torch.tensor(cache['boundary']['check_type'], dtype=torch.long),
        'check_boundary': torch.tensor(cache['boundary']['check_boundary'], dtype=torch.long),
        'check_dist': torch.tensor(cache['boundary']['check_dist'], dtype=torch.long),
        'patch_type_0': cache['patch_type_0'],
        'primary_patch_0': cache['primary_patch'][0],
        'edges': cache['edges'],
        'bctxs': cache['bctxs'],
        'assignments': cache['assignments'],
        'primary_patch': cache['primary_patch'],
        'edge_s2l_next': cache['edge_s2l_next'],
        'error': torch.tensor(e, dtype=torch.float32),
        'L': cache['L'],
        'H_full': cache['H_full'],
        'y_class': torch.tensor(y_class, dtype=torch.long),
    }
