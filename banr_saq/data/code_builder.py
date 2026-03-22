from __future__ import annotations

import numpy as np


# NOTE: These are lightweight synthetic CSS builders intended for wiring/training pipeline tests.
# They keep dimensions consistent with the high-level spec and expose an explicit family switch.

def build_rotated_css_code(d: int) -> dict:
    """Build a simple rotated-style CSS code with n=d^2 data qubits."""
    n = d * d
    checks = []
    for r in range(d - 1):
        for c in range(d - 1):
            q = [r * d + c, r * d + c + 1, (r + 1) * d + c, (r + 1) * d + c + 1]
            checks.append(q)

    # Split plaquettes into X/Z halves (placeholder partition).
    m_half = len(checks) // 2
    H_X = np.zeros((m_half, n), dtype=np.uint8)
    H_Z = np.zeros((len(checks) - m_half, n), dtype=np.uint8)
    for i, q in enumerate(checks[:m_half]):
        H_X[i, q] = 1
    for i, q in enumerate(checks[m_half:]):
        H_Z[i, q] = 1

    # k=1 -> 2 logical operators.
    L = np.zeros((2, n), dtype=np.uint8)
    L[0, :d] = 1
    L[1, ::d] = 1
    return {
        'family': 'rotated',
        'H_X': H_X,
        'H_Z': H_Z,
        'L': L,
        'n': n,
        'k': 1,
    }


def build_toric_css_code(d: int) -> dict:
    """Build a simple toric-style CSS code with n=2d^2 data qubits."""
    n = 2 * d * d
    m = d * d

    def idx_h(r: int, c: int) -> int:
        return (r % d) * d + (c % d)

    def idx_v(r: int, c: int) -> int:
        return d * d + (r % d) * d + (c % d)

    # Z-checks (stars) and X-checks (plaquettes), each degree 4 with periodic wrap.
    H_Z = np.zeros((m, n), dtype=np.uint8)
    H_X = np.zeros((m, n), dtype=np.uint8)
    for r in range(d):
        for c in range(d):
            j = r * d + c
            # star around vertex (r,c)
            H_Z[j, idx_h(r, c)] = 1
            H_Z[j, idx_h(r, c - 1)] = 1
            H_Z[j, idx_v(r, c)] = 1
            H_Z[j, idx_v(r - 1, c)] = 1
            # plaquette at face (r,c)
            H_X[j, idx_h(r, c)] = 1
            H_X[j, idx_h(r + 1, c)] = 1
            H_X[j, idx_v(r, c)] = 1
            H_X[j, idx_v(r, c + 1)] = 1

    # k=2 -> 4 logical operators.
    L = np.zeros((4, n), dtype=np.uint8)
    # Horizontal non-contractible loop (on horizontal edges, row 0)
    for c in range(d):
        L[0, idx_h(0, c)] = 1
    # Vertical non-contractible loop (on vertical edges, col 0)
    for r in range(d):
        L[1, idx_v(r, 0)] = 1
    # Complementary dual loops (simple placeholders)
    for c in range(d):
        L[2, idx_v(0, c)] = 1
    for r in range(d):
        L[3, idx_h(r, 0)] = 1

    return {
        'family': 'toric',
        'H_X': H_X,
        'H_Z': H_Z,
        'L': L,
        'n': n,
        'k': 2,
    }


def build_css_code(d: int, family: str = 'rotated') -> dict:
    family = family.lower()
    if family == 'rotated':
        return build_rotated_css_code(d)
    if family == 'toric':
        return build_toric_css_code(d)
    raise ValueError(f"Unknown code family '{family}'. Use 'rotated' or 'toric'.")
