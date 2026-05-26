"""
Experiment 2 -- Does the sample mean of L0 concentrate?

Same random variable as Experiment 1 (L0 = active features per turn), but now we
ask only the limit-theorem question. Course topics (exactly 2):

  * Weak Law of Large Numbers (item 31): the running sample mean of L0 settles
    onto a single value mu as more turns are included.
  * Central Limit Theorem (item 33): split the turns into B blocks; each block
    mean is itself an average, so sqrt(block_size)*(block_mean - mu)/sigma should
    be approximately N(0,1). We standardise the block means and run a one-sample
    KS test against the standard normal.

Deliverable: the running-mean curve (WLLN) and one histogram of standardised
block means with a Gaussian overlay + KS p-value (CLT). Nothing else.
"""
from __future__ import annotations
from typing import Any

import numpy as np

from ..config import Config
from ..probability.clt import running_mean, clt_diagnostic
from ._io import FeatureBundle


def run(npz_path: str, cfg: Config) -> dict[str, Any]:
    bundle = FeatureBundle.load(npz_path)
    n_active = bundle.support.sum(axis=1).astype(np.float64)
    N = n_active.size

    # --- WLLN: running sample mean (sub-sampled for a compact curve) ---
    rm = running_mean(n_active)
    idx = np.unique(np.linspace(1, N, 200).astype(int)) - 1
    rm_curve_n = (idx + 1).tolist()
    rm_curve = rm[idx].tolist()

    # --- CLT: standardised block means + KS test ---
    block = max(16, N // 50)
    clt = clt_diagnostic(n_active, block_size=block)

    return {
        "n_turns": int(N),
        "mu": float(n_active.mean()),
        "running_mean_n": rm_curve_n,
        "running_mean": rm_curve,
        "block_size": clt.block_size,
        "n_blocks": clt.n_blocks,
        "standardised_block_means": clt.standardised_means.tolist(),
        "ks_stat": clt.ks_stat,
        "ks_pvalue": clt.ks_pvalue,
        "block_mean_mu": clt.mu,
        "block_var": clt.sigma2,
    }
