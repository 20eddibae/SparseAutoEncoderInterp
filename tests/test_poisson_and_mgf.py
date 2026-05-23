import numpy as np

from scf.probability.poisson import poisson_thinning_test
from scf.probability.mgf import fit_normal_via_mgf
from scf.probability.tail_sum import expectation_via_tail_sum


def test_poisson_thinning_accepts_independent_bernoullis():
    rng = np.random.default_rng(0)
    p = np.array([0.02, 0.04, 0.01, 0.05])
    A = (rng.random((3000, p.size)) < p).astype(np.int8)
    out = poisson_thinning_test(A)
    assert 0.0 < out.lambda_hat < p.sum() * 1.5
    # le_cam upper-bound TV must be small for small p
    assert out.le_cam_bound < 0.01


def test_mgf_recovers_normal_parameters():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=2.0, scale=1.5, size=5000)
    fit = fit_normal_via_mgf(x, s_max=0.3, n_s=20)
    assert abs(fit.mu_fit - 2.0) < 0.1
    assert abs(fit.sigma2_fit - 1.5 ** 2) < 0.4


def test_tail_sum_matches_direct():
    rng = np.random.default_rng(2)
    x = rng.poisson(lam=3.0, size=10000)
    tail, direct = expectation_via_tail_sum(x)
    assert abs(tail - direct) < 1e-9
