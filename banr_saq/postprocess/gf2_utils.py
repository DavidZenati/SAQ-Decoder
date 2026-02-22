from __future__ import annotations

import numpy as np


def gf2_row_echelon(A: np.ndarray) -> tuple[np.ndarray, list[int]]:
    A = (A.copy() % 2).astype(np.uint8)
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        pivot = next((rr for rr in range(r, m) if A[rr, c]), None)
        if pivot is None:
            continue
        A[[r, pivot]] = A[[pivot, r]]
        for rr in range(m):
            if rr != r and A[rr, c]:
                A[rr] ^= A[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return A, pivots


def nullspace_basis(A: np.ndarray) -> np.ndarray:
    R, pivots = gf2_row_echelon(A)
    m, n = A.shape
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for r, p in enumerate(pivots):
            if R[r, f]:
                v[p] = 1
        basis.append(v)
    return np.stack(basis, axis=1) if basis else np.zeros((n, 0), dtype=np.uint8)
