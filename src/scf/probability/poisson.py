"""
Poisson limit + thinning as a null hypothesis for feature firings.

Null model: feature i fires independently across turns with probability p_i,
where p_i is estimated as the empirical firing rate. Under this model the
number of active features per turn is the sum of independent Bernoullis,
which is approximately Poisson(lambda) for small p_i with lambda = sum_i p_i
(Le Cam's theorem; total variation distance <= sum p_i^2).

We compare empirical P(N = k) to the Poisson PMF and report a chi-squared
discrepancy. The hypothesis is that real data deviates *above* the null
because features are correlated — thats the finding to report.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import poisson, chi2


@dataclass
class PoissonTest:
    lambda_hat: float
    le_cam_bound: float        # TV bound: sum p_i^2
    chi2_stat: float
    chi2_dof: int
    chi2_pvalue: float
    empirical_pmf: np.ndarray
    poisson_pmf: np.ndarray
    k_grid: np.ndarray


def poisson_thinning_test(
    firing_indicator: np.ndarray,
    max_k: int | None = None,
    min_expected: float = 5.0,
) -> PoissonTest:
    """
    firing_indicator: shape (N_turns, F), 0/1 (or bool).
    Returns the chi-squared statistic and bin-merged PMFs.
    """
    A = np.asarray(firing_indicator, dtype=np.int8)
    p = A.mean(axis=0)                  # per-feature firing rate
    lam = float(p.sum())
    le_cam = float(np.sum(p ** 2))
    n_active = A.sum(axis=1)            # per-turn count

    if max_k is None:
        max_k = int(n_active.max())
    ks = np.arange(0, max_k + 1)
    emp = np.array([(n_active == k).sum() for k in ks], dtype=np.float64)
    expected = poisson.pmf(ks, mu=lam) * n_active.size

    # merge tail bins until each expected count >= min_expected
    merged_obs, merged_exp = _merge_tail(emp, expected, min_expected)
    dof = max(len(merged_obs) - 2, 1)   # estimated 1 parameter (lambda)
    chi2_stat = float(np.sum((merged_obs - merged_exp) ** 2 / merged_exp))
    pval = float(1 - chi2.cdf(chi2_stat, df=dof))

    return PoissonTest(
        lambda_hat=lam,
        le_cam_bound=le_cam,
        chi2_stat=chi2_stat,
        chi2_dof=dof,
        chi2_pvalue=pval,
        empirical_pmf=emp / emp.sum(),
        poisson_pmf=poisson.pmf(ks, mu=lam),
        k_grid=ks,
    )


def _merge_tail(obs: np.ndarray, exp: np.ndarray, min_exp: float) -> tuple[np.ndarray, np.ndarray]:
    o, e = list(obs), list(exp)
    while len(e) > 2 and e[-1] < min_exp:
        e[-2] += e[-1]; o[-2] += o[-1]
        e.pop(); o.pop()
    return np.array(o), np.array(e)
