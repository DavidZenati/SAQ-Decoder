"""
SAQ: Stabilizer-Aware Quantum Error Correction Decoder
Auxiliary functions of -
Stage 3:  Constraint-Projected Nullspace Descent (CPND)
"""

import numpy as np
import torch
import galois
GF2 = galois.GF(2)

def parity_dot(a, b):
    return int(np.bitwise_and(a, b).sum() & 1)

def gf2_solve(A_field, b_field):

    A = np.asarray(A_field, dtype=np.uint8).copy()
    b = np.asarray(b_field, dtype=np.uint8).copy()

    r, n = A.shape
    aug  = np.hstack([A, b.reshape(-1, 1)])
    piv  = []
    row  = 0
    for col in range(n):
        idx = np.where(aug[row:, col] == 1)[0]
        if idx.size == 0:
            continue
        i = row + idx[0]
        aug[[row, i]] = aug[[i, row]]
        piv.append(col)
        for j in range(r):
            if j != row and aug[j, col]:
                aug[j] ^= aug[row]
        row += 1
        if row == r:
            break
    if row < r:
        raise ValueError("Matrix rank deficient")

    # back-substitution
    x = np.zeros(n, dtype=np.uint8)
    for i in reversed(range(r)):
        col = piv[i]
        rhs = aug[i, -1] ^ parity_dot(aug[i, col+1:n], x[col+1:n])
        x[col] = rhs
    return GF2(x)

def exact_left_inverse(H_hat):
    r, n = H_hat.shape
    I_r  = GF2.Identity(r)
    cols = [gf2_solve(H_hat, I_r[:, i]) for i in range(r)]
    return GF2(np.column_stack(cols))              # shape (n, r)

def kernel_basis(M_field):
    A = np.asarray(M_field, dtype=np.uint8).copy()
    r, n = A.shape
    piv, row = [], 0
    for col in range(n):
        idx = np.where(A[row:, col])[0]
        if idx.size == 0: continue
        i = row + idx[0];  A[[row,i]] = A[[i,row]];  piv.append(col)
        for j in range(r):
            if j!=row and A[j,col]: A[j] ^= A[row]
        row += 1
        if row == r: break
    free   = [c for c in range(n) if c not in piv]
    basis  = np.zeros((n, len(free)), dtype=np.uint8)
    for k,j in enumerate(free):
        v = np.zeros(n, dtype=np.uint8); v[j] = 1
        for i, p in reversed(list(enumerate(piv))):
            v[p] = parity_dot(A[i, p+1:], v[p+1:])
        basis[:, k] = v
    return GF2(basis)

def _index_to_bits(index: torch.Tensor, k: int):
    B = index.size(0)
    device = index.device
    bits = torch.zeros(B, k, dtype=torch.long, device=device)
    for i in range(k):
        bits[:, i] = (index >> i) & 1
    return bits

def logits_to_logical_bits(logits: torch.Tensor, k: int):
    pred_index = logits.argmax(dim=1)          # [B]
    bits = _index_to_bits(pred_index, k)       # [B, k]
    return bits

def greedy_nullspace_refine(e0_t, N, w):
    device = e0_t.device
    Nd   = torch.from_numpy(N.view(np.ndarray)).to(device).bool()   # (n,g)
    e    = e0_t.clone().bool()                                     # (B,n)
    sign = (1 - 2*e.int()).float()                                 # (B,n)

    if isinstance(w, np.ndarray):
        w_t = torch.from_numpy(w).to(device)
    else:
        w_t = w.to(device)

    if w_t.dim() == 1:                              # shape (n,)
        w_t = w_t.unsqueeze(0).expand(e.size(0), -1)   # (B, n)

    for j in range(Nd.shape[1]):
        v = Nd[:, j]                                               # (n,)
        delta = (sign[:, v] * w_t[:, v]).sum(dim=1)                # (B,)
        mask  = delta < 0
        if mask.any():
            e[mask] ^= v
            sign[mask][:, v] *= -1

    return e.int()