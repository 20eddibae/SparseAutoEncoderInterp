"""
H1: conversation feature-trajectories are first-order Markov at the coarsened state level.

Test: Chapman-Kolmogorov. Estimate P_hat from 1-step transitions; compare to
the empirical 2-step transition matrix. Report Frobenius residual, max-element
residual, and a chi-squared statistic.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any

from ..config import Config
from ..coarsen import build_topk_space, build_cluster_space
from ..probability.markov import estimate_transition, chapman_kolmogorov_test, mixing_time
from ._io import FeatureBundle


def run(npz_path: str, cfg: Config) -> dict[str, Any]:
    bundle = FeatureBundle.load(npz_path)
    per_conv = bundle.by_conversation()
    flat = [t for conv in per_conv for t in conv]

    if cfg.coarsen.method == "topk":
        space, _ = build_topk_space(flat, cfg.coarsen.k)
    elif cfg.coarsen.method == "cluster":
        space, _ = build_cluster_space(flat, cfg.coarsen.n_clusters)
    else:
        raise ValueError(cfg.coarsen.method)

    trajectories = []
    for conv in per_conv:
        if cfg.coarsen.method == "topk":
            from ..coarsen import topk_state
            traj = [space.label_to_id[topk_state(t, cfg.coarsen.k)] for t in conv]
        else:
            f2c = space.extra["feature_to_cluster"]
            traj = []
            for t in conv:
                if t.max() == 0:
                    traj.append(f2c[0])
                else:
                    traj.append(f2c[int(t.argmax())])
        trajectories.append(traj)

    fit = estimate_transition(trajectories, space.n_states)
    ck = chapman_kolmogorov_test(fit)
    mix = mixing_time(fit.P_hat, fit.pi_hat) if space.n_states <= 200 else None

    return {
        "n_states": space.n_states,
        "n_trajectories": len(trajectories),
        "ck": asdict(ck),
        "stationary_top10": fit.pi_hat.argsort()[::-1][:10].tolist(),
        "mixing_time_eps_0.25": mix,
    }
