"""
H4: Long conversations converge to a stationary feature distribution that
depends on the initial topic. We coarsen and:

  1. Tag each conversation with a topic via its turn-1 state.
  2. Estimate one Markov chain per topic.
  3. Compute the stationary distribution and the per-topic mixing time.
  4. Compare topic-conditional stationaries via KL.

A large KL between two topic-conditional stationaries argues for absorbing-class
structure (i.e., the chain is not ergodic).
"""
from __future__ import annotations
from typing import Any

import numpy as np

from ..config import Config
from ..coarsen import build_topk_space, build_cluster_space, topk_state
from ..probability.markov import estimate_transition, mixing_time
from ..probability.entropy import kl_divergence
from ._io import FeatureBundle


def run(npz_path: str, cfg: Config, n_topics: int = 4) -> dict[str, Any]:
    bundle = FeatureBundle.load(npz_path)
    per_conv = bundle.by_conversation()
    flat = [t for conv in per_conv for t in conv]

    if cfg.coarsen.method == "topk":
        space, _ = build_topk_space(flat, cfg.coarsen.k)
        def state_of(t): return space.label_to_id[topk_state(t, cfg.coarsen.k)]
    else:
        space, _ = build_cluster_space(flat, cfg.coarsen.n_clusters)
        f2c = space.extra["feature_to_cluster"]
        def state_of(t):
            return f2c[0] if t.max() == 0 else f2c[int(t.argmax())]

    trajectories = [[state_of(t) for t in conv] for conv in per_conv]
    initials = np.array([traj[0] for traj in trajectories])
    topic_states, counts = np.unique(initials, return_counts=True)
    top_topics = topic_states[np.argsort(-counts)[:n_topics]].tolist()

    per_topic = {}
    for top in top_topics:
        traj_subset = [tr for tr, init in zip(trajectories, initials) if init == top]
        if len(traj_subset) < 2:
            continue
        fit = estimate_transition(traj_subset, space.n_states)
        mix = mixing_time(fit.P_hat, fit.pi_hat) if space.n_states <= 200 else None
        per_topic[int(top)] = {
            "n_conversations": len(traj_subset),
            "pi": fit.pi_hat.tolist(),
            "mixing_time_eps_0.25": mix,
        }

    kls = {}
    keys = list(per_topic.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pa = np.array(per_topic[a]["pi"])
            pb = np.array(per_topic[b]["pi"])
            kls[f"KL({a}||{b})"] = kl_divergence(pa, pb)
            kls[f"KL({b}||{a})"] = kl_divergence(pb, pa)

    return {
        "n_states": space.n_states,
        "topics": top_topics,
        "per_topic": per_topic,
        "kl_between_topic_stationaries": kls,
    }
