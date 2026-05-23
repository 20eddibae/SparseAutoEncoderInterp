import numpy as np

from scf.probability.bayes import posterior_role_given_feature
from scf.probability.order_stats import per_turn_order_stats


def test_bayes_posterior_extreme_features():
    """A feature that only fires for one role should give P(role|feat) ~ 1."""
    rng = np.random.default_rng(0)
    H = rng.binomial(1, 0.1, size=(500, 5)).astype(bool)
    A = rng.binomial(1, 0.1, size=(500, 5)).astype(bool)
    # feature 0 fires only for human, feature 1 only for AI
    H[:, 0] = True; A[:, 0] = False
    H[:, 1] = False; A[:, 1] = True
    rep = posterior_role_given_feature(H, A)
    assert rep.p_human_given_feature[0] > 0.95
    assert rep.p_ai_given_feature[1] > 0.95


def test_order_stats_shapes_and_monotonicity():
    rng = np.random.default_rng(1)
    M = rng.random((30, 100))
    stats = per_turn_order_stats(M, k=5)
    assert stats.max_activation.shape == (30,)
    assert np.all(stats.top1_minus_top2 >= 0)
    assert np.all((0 <= stats.top1_fraction) & (stats.top1_fraction <= 1))
    assert np.all(stats.n_active_count == (M > 0).sum(axis=1))
