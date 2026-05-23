"""
Bayes' theorem on the (role, feature support) join distribution.

P(role | feature i fires) = P(feature i fires | role) P(role) / P(feature i fires).

Inputs are per-turn binary support matrices for the two roles. We return the
posterior over role for each feature, plus the most-discriminative features.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BayesReport:
    p_human_given_feature: np.ndarray     # (F,)
    p_ai_given_feature: np.ndarray        # (F,)
    p_human: float
    p_ai: float
    top_human_features: np.ndarray        # indices, sorted by P(human|i) desc
    top_ai_features: np.ndarray


def posterior_role_given_feature(
    human_support: np.ndarray, ai_support: np.ndarray,
    smoothing: float = 1e-6, top_n: int = 20,
) -> BayesReport:
    """
    human_support, ai_support: (N_role, F), 0/1.
    """
    H = np.asarray(human_support, dtype=np.float64)
    A = np.asarray(ai_support, dtype=np.float64)
    n_h, n_a = H.shape[0], A.shape[0]
    n = n_h + n_a
    p_h = n_h / n
    p_a = n_a / n

    p_feat_h = (H.sum(axis=0) + smoothing) / (n_h + 2 * smoothing)
    p_feat_a = (A.sum(axis=0) + smoothing) / (n_a + 2 * smoothing)
    p_feat = p_feat_h * p_h + p_feat_a * p_a

    post_h = p_feat_h * p_h / p_feat
    post_a = p_feat_a * p_a / p_feat

    top_h = np.argsort(-post_h)[:top_n]
    top_a = np.argsort(-post_a)[:top_n]

    return BayesReport(
        p_human_given_feature=post_h,
        p_ai_given_feature=post_a,
        p_human=p_h, p_ai=p_a,
        top_human_features=top_h, top_ai_features=top_a,
    )
