from __future__ import annotations

import numpy as np


def cpnd_project(H_aug: np.ndarray, B_left_inv: np.ndarray, syndrome_logical: np.ndarray, e_pred: np.ndarray) -> np.ndarray:
    y = (syndrome_logical ^ ((H_aug @ e_pred) % 2)).astype(np.uint8)
    return (e_pred ^ ((B_left_inv @ y) % 2)).astype(np.uint8)


def nullspace_descent(e_proj: np.ndarray, null_basis: np.ndarray, probs: np.ndarray) -> np.ndarray:
    e = e_proj.copy().astype(np.uint8)
    sign = 1 - 2 * e
    w = -np.log(np.clip(probs, 1e-6, 1 - 1e-6) / np.clip(1 - probs, 1e-6, 1 - 1e-6))
    for j in range(null_basis.shape[1]):
        supp = np.where(null_basis[:, j])[0]
        delta = np.sum(w[supp] * sign[supp])
        if delta < 0:
            e[supp] ^= 1
            sign[supp] *= -1
    return e
