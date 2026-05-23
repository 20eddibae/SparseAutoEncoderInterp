"""
WLLN and CLT diagnostics.

Given iid samples, the sample mean converges to the population mean (WLLN)
and sqrt(N)(mean - mu) is approximately N(0, sigma^2) (CLT). We expose two
helpers:
  - running_mean_convergence: plot E[X] estimate vs N for visual WLLN.
  - clt_diagnostic: standardise the sample mean of B bootstrap blocks and
    run a one-sample KS test against N(0,1).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import kstest, norm


@dataclass
class CLTDiagnostic:
    block_size: int
    n_blocks: int
    standardised_means: np.ndarray
    ks_stat: float
    ks_pvalue: float
    mu: float
    sigma2: float


def running_mean(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    return np.cumsum(x) / (np.arange(1, x.size + 1))


def clt_diagnostic(samples: np.ndarray, block_size: int = 64) -> CLTDiagnostic:
    x = np.asarray(samples, dtype=np.float64)
    n_blocks = x.size // block_size
    if n_blocks < 5:
        raise ValueError("need at least 5 blocks for a CLT diagnostic")
    x = x[: n_blocks * block_size].reshape(n_blocks, block_size)
    mu = float(x.mean())
    sigma2 = float(x.var(ddof=1))
    block_means = x.mean(axis=1)
    if sigma2 == 0:
        raise ValueError("zero variance; CLT undefined")
    z = (block_means - mu) * np.sqrt(block_size / sigma2)
    ks_stat, ks_p = kstest(z, "norm")
    return CLTDiagnostic(
        block_size=block_size, n_blocks=n_blocks,
        standardised_means=z,
        ks_stat=float(ks_stat), ks_pvalue=float(ks_p),
        mu=mu, sigma2=sigma2,
    )
