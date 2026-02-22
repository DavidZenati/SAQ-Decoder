from __future__ import annotations

import numpy as np


def sample_depolarizing(n: int, p: float) -> np.ndarray:
    return (np.random.rand(n) < p).astype(np.uint8)


def syndrome_from_error(H: np.ndarray, e: np.ndarray) -> np.ndarray:
    return (H @ e) % 2
