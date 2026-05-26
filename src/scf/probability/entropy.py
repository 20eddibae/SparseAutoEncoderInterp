"""
Plug-in entropy with a delta-method error bar (Experiment 3).

H(p) = E[-log p(X)] = -sum_i p_i log p_i. Given iid samples x_1,...,x_N from p,
the plug-in estimator is H_hat = -sum_i p_hat_i log p_hat_i, p_hat_i = n_i / N.
Writing it as a sample mean of g(X) = -log p(X) lets the **delta method**
(course item 35: Taylor expansion + Var of a sample mean) give its standard
error directly:

    Var(H_hat) ~= (1/N) Var_p[ -log p(X) ]
                = (1/N) ( sum_i p_i (log p_i)^2 - (sum_i p_i log p_i)^2 ).

This is the headline error bar used in `entropy_with_delta_se`. The older
Chebyshev (log N)^2/N half-width is kept in `entropy_with_chebyshev_bound` for
reference only.

Jensen's inequality (item 28) underlies two facts we use: H(p) <= log(support)
(uniform attains it), and KL(p||q) >= 0 -- the latter is why a nonzero KL between
the two role distributions is meaningful. We expose the log-support slack so the
project can plot it.
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
    delta_se: float                 # delta-method standard error of H_hat
    chebyshev_eps: float            # reference Chebyshev half-width (delta=0.05)
    delta: float


def entropy_with_delta_se(
    counts: Sequence[int] | np.ndarray, base: float = np.e
) -> EntropyEstimate:
    """Plug-in entropy + delta-method standard error (1 sigma).

    se(H_hat) = sqrt( (Var_p[-log p] ) / N ),  Var_p[-log p] computed under p_hat.
    Also fills the reference Chebyshev half-width for comparison.
    """
    c = np.asarray(counts, dtype=np.float64)
    N = int(c.sum())
    H_hat = plug_in_entropy(c, base=base)
    support = int((c > 0).sum())
    if N <= 1:
        se = float("inf")
    else:
        p = c[c > 0] / N
        lg = np.log(p)
        var_g = float(np.sum(p * lg * lg) - np.sum(p * lg) ** 2)   # Var_p[-log p]
        se = float(np.sqrt(max(var_g, 0.0) / N))
        if base != np.e:
            se /= np.log(base)
    ref = entropy_with_chebyshev_bound(c, base=base)
    return EntropyEstimate(
        H_hat=H_hat,
        n_samples=N,
        support_size=support,
        log_support_bound=float(np.log(max(support, 1)) / np.log(base)),
        delta_se=se,
        chebyshev_eps=ref.chebyshev_eps,
        delta=ref.delta,
    )


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
        delta_se=float("nan"),
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
