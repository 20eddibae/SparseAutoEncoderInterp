import numpy as np

from scf.probability.concentration import (
    markov_bound, chebyshev_bound, compare_tails, sub_gaussian_proxy,
)


def test_markov_bound_basic():
    bound = markov_bound(mu=1.0, t=np.array([2.0, 4.0]))
    assert np.allclose(bound, [0.5, 0.25])


def test_chebyshev_bound_basic():
    bound = chebyshev_bound(mu=0.0, sigma2=1.0, t=np.array([1.0, 2.0, 5.0]))
    assert np.allclose(bound, [1.0, 0.25, 0.04])


def test_compare_tails_runs_on_normal():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=1.0, size=5000)
    grid = np.linspace(0.5, 3.0, 10)
    out = compare_tails(x, grid)
    assert out.empirical.shape == grid.shape
    # for a normal RV, chebyshev should be >= empirical at every t
    assert np.all(out.chebyshev + 1e-12 >= out.empirical)


def test_sub_gaussian_proxy_recovers_variance_for_normal():
    rng = np.random.default_rng(1)
    x = rng.normal(scale=2.0, size=20000)
    proxy = sub_gaussian_proxy(x)
    assert 2.0 < proxy < 8.0   # true variance is 4; proxy is conservative
