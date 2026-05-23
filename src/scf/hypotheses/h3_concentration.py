"""
H3: N_active = number of features firing per turn is sub-Gaussian. Compare
Markov, Chebyshev, and sub-Gaussian (fit via MGF) tails to the empirical tail.

Also runs a Poisson-thinning null test and a CLT diagnostic.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any

import numpy as np

from ..config import Config
from ..probability.concentration import compare_tails
from ..probability.mgf import fit_normal_via_mgf
from ..probability.poisson import poisson_thinning_test
from ..probability.clt import clt_diagnostic
from ..probability.tail_sum import expectation_via_tail_sum
from ._io import FeatureBundle


def run(npz_path: str, cfg: Config) -> dict[str, Any]:
    bundle = FeatureBundle.load(npz_path)
    n_active = bundle.support.sum(axis=1).astype(np.float64)

    e_tail, e_direct = expectation_via_tail_sum(n_active.astype(int))

    t_grid = np.linspace(0.0, max(1.0, 2 * n_active.std()), 25)
    tails = compare_tails(n_active, t_grid)
    mgf_fit = fit_normal_via_mgf(n_active, s_max=0.3)
    poisson = poisson_thinning_test(bundle.support.astype(np.int8))
    clt = clt_diagnostic(n_active, block_size=max(16, n_active.size // 50))

    return {
        "n_turns": int(n_active.size),
        "expectation_tail_sum": e_tail,
        "expectation_direct": e_direct,
        "mu": tails.mu,
        "var": tails.sigma2,
        "sub_gaussian_proxy": tails.sigma2_g,
        "tail_grid": tails.t.tolist(),
        "empirical_tail": tails.empirical.tolist(),
        "markov_tail": tails.markov.tolist(),
        "chebyshev_tail": tails.chebyshev.tolist(),
        "sub_gaussian_tail": tails.sub_gaussian.tolist(),
        "mgf": asdict(mgf_fit) | {"s_grid": mgf_fit.s_grid.tolist(),
                                  "log_M_hat": mgf_fit.log_M_hat.tolist()},
        "poisson": {k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in asdict(poisson).items()},
        "clt": {
            "block_size": clt.block_size, "n_blocks": clt.n_blocks,
            "ks_stat": clt.ks_stat, "ks_pvalue": clt.ks_pvalue,
            "mu": clt.mu, "sigma2": clt.sigma2,
        },
    }
