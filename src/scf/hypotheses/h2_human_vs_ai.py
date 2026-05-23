"""
H2: Human prompts have higher entropy over features than AI completions
(AI is mode-seeking). Test with plug-in entropy + Chebyshev error bars.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any

import numpy as np

from ..config import Config
from ..probability.entropy import entropy_with_chebyshev_bound, kl_divergence
from ..probability.bayes import posterior_role_given_feature
from ._io import FeatureBundle


def run(npz_path: str, cfg: Config) -> dict[str, Any]:
    bundle = FeatureBundle.load(npz_path)
    human = bundle.support[bundle.role == 0]
    ai = bundle.support[bundle.role == 1]

    human_counts = human.sum(axis=0)
    ai_counts = ai.sum(axis=0)

    h_human = entropy_with_chebyshev_bound(human_counts)
    h_ai = entropy_with_chebyshev_bound(ai_counts)

    kl_ha = kl_divergence(human_counts, ai_counts)
    kl_ah = kl_divergence(ai_counts, human_counts)

    bayes = posterior_role_given_feature(human, ai, top_n=20)

    return {
        "n_human_turns": int(human.shape[0]),
        "n_ai_turns": int(ai.shape[0]),
        "entropy_human": asdict(h_human),
        "entropy_ai": asdict(h_ai),
        "delta_entropy": h_human.H_hat - h_ai.H_hat,
        "kl_human_to_ai": kl_ha,
        "kl_ai_to_human": kl_ah,
        "top_features_for_human": bayes.top_human_features.tolist(),
        "top_features_for_ai": bayes.top_ai_features.tolist(),
    }
