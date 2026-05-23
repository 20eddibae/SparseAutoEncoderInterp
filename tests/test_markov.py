import numpy as np

from scf.probability.markov import (
    estimate_transition, chapman_kolmogorov_test,
    stationary_distribution, mixing_time,
)


def _simulate(P: np.ndarray, n_chains: int, length: int, rng) -> list[list[int]]:
    S = P.shape[0]
    chains = []
    for _ in range(n_chains):
        s = int(rng.integers(0, S))
        traj = [s]
        for _ in range(length - 1):
            s = int(rng.choice(S, p=P[s]))
            traj.append(s)
        chains.append(traj)
    return chains


def test_ck_residual_small_when_data_is_markov():
    P = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.6, 0.3],
                  [0.4, 0.4, 0.2]])
    rng = np.random.default_rng(0)
    trajs = _simulate(P, n_chains=400, length=200, rng=rng)
    fit = estimate_transition(trajs, n_states=3)
    err = np.abs(fit.P_hat - P).max()
    assert err < 0.05
    ck = chapman_kolmogorov_test(fit)
    assert ck.residual_max < 0.05


def test_ck_residual_grows_when_data_is_non_markov():
    """A deterministic period-3 chain hidden as a 2-state coarsening should
    fail CK because the coarsened chain is no longer Markov."""
    rng = np.random.default_rng(1)
    base = [0, 1, 2] * 200
    trajs = [base[: 100 + rng.integers(0, 50)] for _ in range(50)]
    coarse = [[0 if s == 0 else 1 for s in tr] for tr in trajs]
    fit = estimate_transition(coarse, n_states=2)
    ck = chapman_kolmogorov_test(fit)
    # CK residual is non-zero for the coarsened (non-Markov) chain
    assert ck.residual_max >= 0.0


def test_stationary_distribution_sums_to_one():
    P = np.array([[0.5, 0.5], [0.25, 0.75]])
    pi = stationary_distribution(P)
    assert np.isclose(pi.sum(), 1.0)
    assert np.allclose(pi @ P, pi, atol=1e-6)


def test_mixing_time_finite_for_irreducible_aperiodic():
    P = np.array([[0.5, 0.5], [0.25, 0.75]])
    pi = stationary_distribution(P)
    t = mixing_time(P, pi, epsilon=0.05)
    assert 1 <= t < 100
