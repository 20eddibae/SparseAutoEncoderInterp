"""
Experiment 3 -- Are the two roles different distributions?

For each role we form the marginal feature-firing distribution p_role over the
F features (normalised firing counts). Course topics:

  * Entropy as an expectation + delta-method error bar (item 35). H(p) =
    E[-log p] estimated by plug-in, with a delta-method 1-sigma standard error.
  * KL divergence both directions, with Jensen's inequality (item 28) as the
    reason KL >= 0 -- so a strictly positive KL is genuine evidence the two
    role distributions differ (not just noise around 0).

The Bayes posterior P(role | feature i fires) (item 3) is reported as a single
supporting line -- the few most role-discriminative features -- not as its own
experiment.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any

import numpy as np

from ..config import Config
from ..probability.entropy import entropy_with_delta_se, kl_divergence
from ..probability.bayes import posterior_role_given_feature
from ._io import FeatureBundle


def run(npz_path: str, cfg: Config) -> dict[str, Any]:
    bundle = FeatureBundle.load(npz_path)
    human = bundle.support[bundle.role == 0]
    ai = bundle.support[bundle.role == 1]

    human_counts = human.sum(axis=0)
    ai_counts = ai.sum(axis=0)

    h_human = entropy_with_delta_se(human_counts)
    h_ai = entropy_with_delta_se(ai_counts)

    # KL both directions (Jensen => both >= 0; positive => roles differ)
    kl_ha = kl_divergence(human_counts, ai_counts)
    kl_ah = kl_divergence(ai_counts, human_counts)

    # Bayes posterior P(role | feature) -- one supporting line of top features
    bayes = posterior_role_given_feature(human, ai, top_n=10)

    return {
        "n_human_turns": int(human.shape[0]),
        "n_ai_turns": int(ai.shape[0]),
        "entropy_human": asdict(h_human),
        "entropy_ai": asdict(h_ai),
        "delta_entropy": h_human.H_hat - h_ai.H_hat,
        "delta_entropy_se": float(np.hypot(h_human.delta_se, h_ai.delta_se)),
        "kl_human_to_ai": kl_ha,
        "kl_ai_to_human": kl_ah,
        "top_features_for_human": bayes.top_human_features.tolist(),
        "top_features_for_ai": bayes.top_ai_features.tolist(),
        "max_posterior_human": float(bayes.p_human_given_feature[bayes.top_human_features[0]]),
        "max_posterior_ai": float(bayes.p_ai_given_feature[bayes.top_ai_features[0]]),
    }
