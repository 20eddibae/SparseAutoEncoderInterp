"""
Experiment 1 -- Is L0 Poisson?

The random variable is L0 = the number of SAE features firing on a turn
(= sum of F Bernoulli indicators). Course topics (<=2 families):

  * Tail-sum identity (items 5/8):  E[L0] = sum_{k>=1} P(L0 >= k).  One-line
    numerical check against the direct sample mean.
  * Poisson-as-limit-of-Binomial (items 13/14): under independent firing,
    L0 is a sum of independent Bernoullis -> Poisson(lambda = sum_i p_i). We fit
    lambda, compare the empirical PMF to Poisson via chi-squared, and report the
    Fano factor var/mean (=1 for Poisson). The correction term sum_i p_i^2 reports
    how far the small-rate Poisson limit is from applying.
  * Markov & Chebyshev tail bounds (items 25/26): bound P(L0 - mu >= t) and
    compare to the empirical upper tail.

No MGF Gaussian fit and no CLT here -- those belong to Experiment 2.
"""
from __future__ import annotations
from typing import Any

import numpy as np

from ..config import Config
from ..probability.concentration import markov_bound, chebyshev_bound
from ..probability.poisson import poisson_thinning_test
from ..probability.tail_sum import expectation_via_tail_sum
from ._io import FeatureBundle


def run(npz_path: str, cfg: Config) -> dict[str, Any]:
    bundle = FeatureBundle.load(npz_path)
    n_active = bundle.support.sum(axis=1).astype(np.float64)

    # --- tail-sum identity (sanity check) ---
    e_tail, e_direct = expectation_via_tail_sum(n_active.astype(int))

    mu = float(n_active.mean())
    var = float(n_active.var(ddof=1))

    # --- Poisson null: limit of independent Bernoulli firing ---
    poisson = poisson_thinning_test(bundle.support.astype(np.int8))

    # --- Markov + Chebyshev tail bounds vs the empirical upper tail ---
    t_grid = np.linspace(0.0, max(1.0, 2 * n_active.std()), 25)
    empirical = np.array([(n_active - mu >= ti).mean() for ti in t_grid])
    markov = markov_bound(mu, t_grid + mu)        # P(L0 >= mu+t) <= mu/(mu+t)
    cheb = chebyshev_bound(mu, var, t_grid)       # P(|L0-mu| >= t) <= var/t^2

    return {
        "n_turns": int(n_active.size),
        "mu": mu,
        "var": var,
        "fano_factor": var / mu,
        "expectation_tail_sum": e_tail,
        "expectation_direct": e_direct,
        "lambda_hat": poisson.lambda_hat,
        "poisson_approx_error": poisson.poisson_approx_error,
        "poisson_chi2_stat": poisson.chi2_stat,
        "poisson_chi2_pvalue": poisson.chi2_pvalue,
        "k_grid": poisson.k_grid.tolist(),
        "empirical_pmf": poisson.empirical_pmf.tolist(),
        "poisson_pmf": poisson.poisson_pmf.tolist(),
        "tail_grid": t_grid.tolist(),
        "empirical_tail": empirical.tolist(),
        "markov_tail": markov.tolist(),
        "chebyshev_tail": cheb.tolist(),
    }
