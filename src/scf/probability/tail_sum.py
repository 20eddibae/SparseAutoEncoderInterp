"""
Tail-sum identity for non-negative integer RVs:
    E[N] = sum_{k>=1} P(N >= k).

Plug in P_hat(N >= k) and integrate; compare to the direct sample mean as a
sanity check.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np


def expectation_via_tail_sum(samples: np.ndarray) -> Tuple[float, float]:
    """Returns (E_hat_tail_sum, E_hat_direct)."""
    x = np.asarray(samples)
    if (x < 0).any() or not np.issubdtype(x.dtype, np.integer):
        x = x.astype(int)
    max_k = int(x.max())
    tail = sum(float((x >= k).mean()) for k in range(1, max_k + 1))
    return tail, float(x.mean())
