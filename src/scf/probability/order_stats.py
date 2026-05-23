"""
Order statistics on per-turn feature magnitudes.

For each turn we have F magnitudes; rank them as X_(1) >= X_(2) >= ... >= X_(F).
Quantities of interest:
  - max gap: X_(1) - X_(2).  Large gap => the turn is monosemantic.
  - top-k ratio: X_(1) / sum(X_(1..k)).  Concentration of activation.
  - distribution of X_(1) across turns.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class OrderStats:
    max_activation: np.ndarray            # (N,)
    top1_minus_top2: np.ndarray           # (N,)
    top1_fraction: np.ndarray             # (N,)
    n_active_count: np.ndarray            # (N,)


def per_turn_order_stats(magnitudes: np.ndarray, k: int = 5) -> OrderStats:
    """magnitudes: (N, F). Returns (N,) arrays per metric."""
    M = np.asarray(magnitudes, dtype=np.float64)
    sorted_desc = -np.sort(-M, axis=1)               # (N, F)
    top = sorted_desc[:, :k]
    max_act = top[:, 0]
    gap = top[:, 0] - (top[:, 1] if k > 1 else 0.0)
    denom = top.sum(axis=1)
    frac = np.where(denom > 0, max_act / denom, 0.0)
    n_active = (M > 0).sum(axis=1)
    return OrderStats(max_act, gap, frac, n_active)
