from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path


def _mode(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr.astype(int), return_counts=True)
    return int(vals[counts.argmax()]) if len(vals) else 0


def classify_boundaries(H_X: np.ndarray, H_Z: np.ndarray, d_max: int = 5):
    m_X, n = H_X.shape
    m_Z, _ = H_Z.shape

    deg_X = H_X.sum(axis=1)
    deg_Z = H_Z.sum(axis=1)
    bulk_deg_X = _mode(deg_X)
    bulk_deg_Z = _mode(deg_Z)

    check_type = np.concatenate([np.zeros(m_X, dtype=int), np.ones(m_Z, dtype=int)])
    check_boundary = np.zeros(m_X + m_Z, dtype=int)
    check_boundary[:m_X][deg_X < bulk_deg_X] = 2
    check_boundary[m_X:][deg_Z < bulk_deg_Z] = 1

    H_full = sp.csr_matrix(np.vstack([H_X, H_Z]))
    A_cc = ((H_full @ H_full.T) > 0).astype(np.uint8)
    A_cc.setdiag(0)

    for j in np.where(check_boundary > 0)[0]:
        neighbors = A_cc[j].indices
        ntypes = set(check_boundary[neighbors])
        if check_boundary[j] == 1 and 2 in ntypes:
            check_boundary[j] = 3
        if check_boundary[j] == 2 and 1 in ntypes:
            check_boundary[j] = 3

    sources = np.where(check_boundary > 0)[0]
    if len(sources) == 0:
        check_dist = np.full(m_X + m_Z, d_max, dtype=int)
        mode = 'toric'
    else:
        dists = shortest_path(A_cc, directed=False, unweighted=True, indices=sources)
        check_dist = np.clip(np.min(dists, axis=0), 0, d_max).astype(int)
        mode = 'rotated'

    deg_Q_X = H_X.sum(axis=0)
    deg_Q_Z = H_Z.sum(axis=0)
    bulk_Q_X = _mode(deg_Q_X)
    bulk_Q_Z = _mode(deg_Q_Z)
    qubit_boundary = np.zeros(n, dtype=int)
    qubit_boundary[(deg_Q_Z < bulk_Q_Z) & (deg_Q_X >= bulk_Q_X)] = 1
    qubit_boundary[(deg_Q_X < bulk_Q_X) & (deg_Q_Z >= bulk_Q_Z)] = 2
    qubit_boundary[(deg_Q_X < bulk_Q_X) & (deg_Q_Z < bulk_Q_Z)] = 3

    qubit_dist = np.full(n, d_max, dtype=int)
    for i in range(n):
        adj = H_full[:, i].nonzero()[0]
        if len(adj):
            qubit_dist[i] = int(np.min(check_dist[adj]))
    qubit_dist = np.clip(qubit_dist, 0, d_max)

    return {
        'check_type': check_type,
        'check_boundary': check_boundary,
        'check_dist': check_dist,
        'qubit_boundary': qubit_boundary,
        'qubit_dist': qubit_dist,
        'A_cc': A_cc,
        'mode': mode,
    }
