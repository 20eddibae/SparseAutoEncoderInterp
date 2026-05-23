"""
Markov's inequality, Chebyshev, and an empirical sub-Gaussian fit.

For a non-negative RV X with mean mu,
    Markov:    P(X >= t) <= mu / t.
For any RV with mean mu, variance sigma^2,
    Chebyshev: P(|X - mu| >= t) <= sigma^2 / t^2.
Sub-Gaussian (mean-zero) with proxy sigma_g^2:
    P(X >= t) <= exp(-t^2 / (2 sigma_g^2)).

We compare all three to the empirical tail. Chebyshev should be tight when
sigma is small relative to t; sub-Gaussian (if it holds) is tighter for large t.
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
    sub_gaussian: np.ndarray
    mu: float
    sigma2: float
    sigma2_g: float            # sub-Gaussian proxy


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


def sub_gaussian_proxy(samples: np.ndarray) -> float:
    """
    Estimate a sub-Gaussian variance proxy by matching the empirical MGF
    M(s) = E[exp(s (X - mu))] to the sub-Gaussian envelope exp(s^2 sigma_g^2 / 2)
    at a small s. We use the smallest s in (0.01, 0.5) that does not overflow.
    """
    x = np.asarray(samples, dtype=np.float64)
    mu = x.mean()
    centred = x - mu
    s_grid = np.linspace(0.01, 0.5, 25)
    proxies = []
    for s in s_grid:
        m = np.mean(np.exp(s * centred))
        if not np.isfinite(m) or m <= 1.0:
            continue
        proxies.append(2 * np.log(m) / (s * s))
    if not proxies:
        return float(x.var(ddof=1))
    return float(np.max(proxies))


def compare_tails(samples: np.ndarray, t_grid: np.ndarray) -> TailComparison:
    """All three bounds plus the empirical upper-tail."""
    x = np.asarray(samples, dtype=np.float64)
    mu = float(x.mean())
    sigma2 = float(x.var(ddof=1))
    sigma2_g = sub_gaussian_proxy(x)
    t = np.asarray(t_grid, dtype=np.float64)
    empirical = np.array([(x - mu >= ti).mean() for ti in t])
    markov = markov_bound(mu, t + mu) if x.min() >= 0 else np.full_like(t, np.nan)
    cheb = chebyshev_bound(mu, sigma2, t)
    sg = np.exp(-t * t / (2 * sigma2_g))
    return TailComparison(
        t=t, empirical=empirical, markov=markov,
        chebyshev=cheb, sub_gaussian=sg,
        mu=mu, sigma2=sigma2, sigma2_g=sigma2_g,
    )
