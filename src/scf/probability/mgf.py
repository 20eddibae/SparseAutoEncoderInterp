"""
Empirical MGF M_hat(s) = (1/N) sum exp(s X_i) and a normal comparison.

If X is approximately N(mu, sigma^2) then log M(s) = mu s + sigma^2 s^2 / 2.
We fit (mu, sigma^2) by linear regression on log M_hat against (s, s^2) and
report the residual as a Gaussianity diagnostic.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MGFFit:
    s_grid: np.ndarray
    log_M_hat: np.ndarray
    mu_fit: float
    sigma2_fit: float
    residual_max: float       # sup_s |log M_hat - (mu s + sigma^2 s^2 / 2)|


def empirical_mgf(samples: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    s = np.asarray(s_grid, dtype=np.float64)
    # broadcasting: (S, N)
    return np.mean(np.exp(s[:, None] * x[None, :]), axis=1)


def fit_normal_via_mgf(samples: np.ndarray, s_max: float = 0.5, n_s: int = 25) -> MGFFit:
    s = np.linspace(-s_max, s_max, n_s)
    s = s[s != 0]
    M = empirical_mgf(samples, s)
    log_M = np.log(M)
    A = np.column_stack([s, 0.5 * s * s])
    coef, *_ = np.linalg.lstsq(A, log_M, rcond=None)
    mu_fit, sigma2_fit = float(coef[0]), float(coef[1])
    pred = mu_fit * s + 0.5 * sigma2_fit * s * s
    return MGFFit(
        s_grid=s, log_M_hat=log_M,
        mu_fit=mu_fit, sigma2_fit=sigma2_fit,
        residual_max=float(np.max(np.abs(log_M - pred))),
    )
