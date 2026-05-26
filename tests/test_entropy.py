import numpy as np

from scf.probability.entropy import (
    plug_in_entropy, entropy_with_chebyshev_bound, entropy_with_delta_se, kl_divergence,
)


def test_delta_se_decreases_with_N():
    small = entropy_with_delta_se(np.array([20, 30, 50, 40]))
    large = entropy_with_delta_se(np.array([2000, 3000, 5000, 4000]))
    assert large.delta_se < small.delta_se
    assert large.delta_se > 0


def test_delta_se_zero_for_point_mass():
    est = entropy_with_delta_se(np.array([1000, 0, 0]))
    assert np.isclose(est.delta_se, 0.0, atol=1e-9)


def test_uniform_attains_log_support():
    counts = np.array([10, 10, 10, 10])
    H = plug_in_entropy(counts)
    assert np.isclose(H, np.log(4), atol=1e-9)


def test_point_mass_is_zero():
    assert plug_in_entropy(np.array([100, 0, 0])) == 0.0


def test_chebyshev_bound_decreases_with_N():
    small = entropy_with_chebyshev_bound(np.array([20, 20, 20, 20]))
    large = entropy_with_chebyshev_bound(np.array([2000] * 4))
    assert large.chebyshev_eps < small.chebyshev_eps


def test_jensen_upper_bound():
    counts = np.random.default_rng(0).integers(1, 100, size=50)
    est = entropy_with_chebyshev_bound(counts)
    assert est.H_hat <= est.log_support_bound + 1e-9


def test_kl_nonnegative_and_zero_for_same():
    p = np.array([5, 10, 5])
    assert kl_divergence(p, p) >= -1e-9
    assert kl_divergence(p, p) < 1e-6
    q = np.array([1, 1, 50])
    assert kl_divergence(p, q) > 0
