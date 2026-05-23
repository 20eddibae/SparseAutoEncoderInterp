"""
Plug-in entropy estimator with a Chebyshev-style error bound.

H(p) = -sum_i p_i log p_i. Given iid samples x_1,...,x_N from p, the plug-in
estimator is H_hat = -sum_i p_hat_i log p_hat_i where p_hat_i = n_i / N.

Variance bound (Antos-Kontoyiannis 2001-style, simplified):
    Var(H_hat) <= (log N)^2 / N.
Chebyshev then gives
    P(|H_hat - H| >= eps) <= (log N)^2 / (N eps^2).
We use this to attach a 1-sigma error bar in `entropy_with_chebyshev_bound`.

Jensen's inequality is used implicitly: H(p) <= log(support size); we expose
that comparison so the project can plot the slack.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class EntropyEstimate:
    H_hat: float
    n_samples: int
    support_size: int
    log_support_bound: float        # Jensen upper bound (uniform attains it)
    chebyshev_eps: float            # half-width such that Pr(|H_hat-H|>=eps) <= delta
    delta: float


def plug_in_entropy(counts: Sequence[int] | np.ndarray, base: float = np.e) -> float:
    """H_hat from raw counts. Zero counts contribute 0."""
    c = np.asarray(counts, dtype=np.float64)
    N = c.sum()
    if N <= 0:
        return 0.0
    p = c[c > 0] / N
    H = -np.sum(p * np.log(p))
    if base != np.e:
        H /= np.log(base)
    return float(H)


def entropy_with_chebyshev_bound(
    counts: Sequence[int] | np.ndarray,
    delta: float = 0.05,
    base: float = np.e,
) -> EntropyEstimate:
    """Chebyshev half-width eps so that Pr(|H_hat - H| >= eps) <= delta."""
    c = np.asarray(counts, dtype=np.float64)
    N = int(c.sum())
    H_hat = plug_in_entropy(c, base=base)
    support = int((c > 0).sum())
    if N <= 1:
        eps = float("inf")
    else:
        var_bound = (np.log(N)) ** 2 / N
        eps = float(np.sqrt(var_bound / delta))
        if base != np.e:
            eps /= np.log(base)
    return EntropyEstimate(
        H_hat=H_hat,
        n_samples=N,
        support_size=support,
        log_support_bound=float(np.log(max(support, 1)) / np.log(base)),
        chebyshev_eps=eps,
        delta=delta,
    )


def cross_entropy(p_counts: np.ndarray, q_counts: np.ndarray, base: float = np.e) -> float:
    """H(p, q) = -sum p_i log q_i with smoothing on q."""
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    H = -np.sum(p * np.log(q))
    return float(H if base == np.e else H / np.log(base))


def kl_divergence(p_counts: np.ndarray, q_counts: np.ndarray, base: float = np.e) -> float:
    """KL(p||q). Smoothing on q for numerical safety."""
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    kl = np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))
    return float(kl if base == np.e else kl / np.log(base))
