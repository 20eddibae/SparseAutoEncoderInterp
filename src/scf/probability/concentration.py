"""
Markov's and Chebyshev's inequalities (course items 25/26).

For a non-negative RV X with mean mu,
    Markov:    P(X >= t) <= mu / t.
For any RV with mean mu, variance sigma^2,
    Chebyshev: P(|X - mu| >= t) <= sigma^2 / t^2.

We compare both to the empirical upper tail. Markov uses only the mean;
Chebyshev also uses the variance and is the tighter of the two for large t.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TailComparison:
    t: np.ndarray
    empirical: np.ndarray
    markov: np.ndarray
    chebyshev: np.ndarray
    mu: float
    sigma2: float


def markov_bound(mu: float, t: np.ndarray) -> np.ndarray:
    """Valid only for non-negative X and t > 0."""
    t = np.asarray(t, dtype=np.float64)
    out = np.full_like(t, np.inf)
    np.divide(mu, t, out=out, where=t > 0)
    return np.minimum(out, 1.0)


def chebyshev_bound(mu: float, sigma2: float, t: np.ndarray) -> np.ndarray:
    """P(|X - mu| >= t) <= sigma^2 / t^2."""
    t = np.asarray(t, dtype=np.float64)
    out = np.full_like(t, np.inf)
    np.divide(sigma2, t * t, out=out, where=t > 0)
    return np.minimum(out, 1.0)


def compare_tails(samples: np.ndarray, t_grid: np.ndarray) -> TailComparison:
    """Markov and Chebyshev bounds plus the empirical upper-tail."""
    x = np.asarray(samples, dtype=np.float64)
    mu = float(x.mean())
    sigma2 = float(x.var(ddof=1))
    t = np.asarray(t_grid, dtype=np.float64)
    empirical = np.array([(x - mu >= ti).mean() for ti in t])
    markov = markov_bound(mu, t + mu) if x.min() >= 0 else np.full_like(t, np.nan)
    cheb = chebyshev_bound(mu, sigma2, t)
    return TailComparison(
        t=t, empirical=empirical, markov=markov,
        chebyshev=cheb, mu=mu, sigma2=sigma2,
    )
