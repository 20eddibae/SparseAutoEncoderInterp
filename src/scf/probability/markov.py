"""
Discrete-time Markov chain estimation and tests.

Given a list of integer trajectories (one per conversation), we estimate:
  - 1-step transition matrix P_hat (row-stochastic)
  - empirical 2-step matrix P2_emp (from t -> t+2 transitions)
  - Chapman-Kolmogorov residual ||P_hat^2 - P2_emp||
  - stationary distribution pi (left eigenvector of P_hat for eigenvalue 1)
  - mixing time (TV distance from pi)
  - chi-squared CK test

A note on the proof: CK says that under the Markov property,
P(X_{t+2}=j | X_t=i) = sum_k P_{ik} P_{kj}. So P2_true = P^2 exactly.
The empirical residual measures non-Markovianity (modulo finite-sample noise).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.stats import chi2


@dataclass
class MarkovFit:
    n_states: int
    P_hat: np.ndarray              # (S, S) row-stochastic
    counts_1step: np.ndarray       # (S, S) integer
    counts_2step: np.ndarray       # (S, S) integer
    pi_hat: np.ndarray             # (S,) stationary distribution


@dataclass
class CKTest:
    residual_frobenius: float
    residual_max: float
    chi2_stat: float
    chi2_dof: int
    chi2_pvalue: float


def estimate_transition(trajectories: Sequence[Sequence[int]], n_states: int) -> MarkovFit:
    S = n_states
    c1 = np.zeros((S, S), dtype=np.int64)
    c2 = np.zeros((S, S), dtype=np.int64)
    for traj in trajectories:
        t = list(traj)
        for a, b in zip(t[:-1], t[1:]):
            c1[a, b] += 1
        for a, b in zip(t[:-2], t[2:]):
            c2[a, b] += 1
    P = _row_stochastic(c1)
    pi = stationary_distribution(P)
    return MarkovFit(n_states=S, P_hat=P, counts_1step=c1, counts_2step=c2, pi_hat=pi)


def chapman_kolmogorov_test(fit: MarkovFit) -> CKTest:
    """Compare P^2 to the empirical 2-step transition matrix."""
    P2_emp = _row_stochastic(fit.counts_2step)
    P2_model = fit.P_hat @ fit.P_hat
    diff = P2_emp - P2_model
    fro = float(np.linalg.norm(diff))
    mx = float(np.max(np.abs(diff)))

    # chi-squared on contingency-style discrepancy
    row_totals = fit.counts_2step.sum(axis=1, keepdims=True).astype(np.float64)
    expected = P2_model * row_totals
    mask = expected > 5
    obs = fit.counts_2step.astype(np.float64)
    if mask.sum() == 0:
        chi2_stat = float("nan"); dof = 0; pval = float("nan")
    else:
        chi2_stat = float(np.sum((obs[mask] - expected[mask]) ** 2 / expected[mask]))
        S = fit.n_states
        dof = max(int(mask.sum()) - S, 1)
        pval = float(1 - chi2.cdf(chi2_stat, df=dof))
    return CKTest(fro, mx, chi2_stat, dof, pval)


def stationary_distribution(P: np.ndarray, tol: float = 1e-10, max_iter: int = 10000) -> np.ndarray:
    """Left eigenvector for eigenvalue 1, found by power iteration on P^T."""
    S = P.shape[0]
    pi = np.full(S, 1.0 / S)
    for _ in range(max_iter):
        new = pi @ P
        if np.linalg.norm(new - pi, ord=1) < tol:
            pi = new
            break
        pi = new
    s = pi.sum()
    return pi / s if s > 0 else pi


def mixing_time(P: np.ndarray, pi: np.ndarray, epsilon: float = 0.25) -> int:
    """
    Smallest t such that max_i ||P^t(i, .) - pi||_TV <= epsilon.
    Caps at 10_000 iterations and returns -1 if not reached.
    """
    S = P.shape[0]
    Pt = np.eye(S)
    for t in range(1, 10001):
        Pt = Pt @ P
        tv = 0.5 * np.abs(Pt - pi[None, :]).sum(axis=1).max()
        if tv <= epsilon:
            return t
    return -1


def _row_stochastic(counts: np.ndarray) -> np.ndarray:
    c = counts.astype(np.float64)
    row = c.sum(axis=1, keepdims=True)
    safe = np.where(row > 0, row, 1.0)
    P = c / safe
    # absorbing fallback for states with zero outgoing transitions: stay put.
    zero_rows = (row.flatten() == 0)
    if zero_rows.any():
        idx = np.where(zero_rows)[0]
        P[idx] = 0
        P[idx, idx] = 1.0
    return P
