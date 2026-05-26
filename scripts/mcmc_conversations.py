#!/usr/bin/env python3
"""
Generative-MCMC view of the conversation-mode Markov chain (CPU-only, runs on the
already-extracted per-turn features). This extends the turn-level chain in
`markov_chain.py` in two ways:

  1. ROLE-COUPLED kernels (H5/H6 from RESEARCH.md). Role strictly alternates, so
     the pooled 1-step chain is not first-order (CK residual ~1.6). We estimate
     two kernels separately --

         T_{h->a}  : human-turn mode -> next ai-turn mode   ("AI self-surprise")
         T_{a->h}  : ai-turn mode    -> next human-turn mode ("human surprise")

     and show (a) the role-coupled 2-step model lowers the CK residual vs the
     pooled chain, and (b) the two kernels genuinely differ (role-asymmetric
     dynamics) -- the structural reason the human is the less-predictable side.

  2. Two samplers over the fitted chain:

     * METROPOLIS-HASTINGS (genuine MCMC) targeting the stationary distribution
       pi. Symmetric random-walk proposal over the K modes; acceptance uses pi
       only. We watch TV(MH-empirical, pi) -> 0 to confirm the sampler converges
       to the same pi the eigen/power method found -- a textbook MCMC validation
       on real data.

     * FORWARD (ancestral) SIMULATION of whole synthetic conversations from the
       role-coupled kernels, matching the real conversation-length and opening-
       mode distributions. We then run the *same* statistics on synthetic vs real
       turns -- a posterior-predictive check. The 1-step model reproduces the
       headline human>AI surprise gap by construction, but the dwell-time / modes-
       per-conversation comparison exposes exactly the higher-order structure the
       Markov model misses (the source of the CK residual).

Usage:
  python scripts/mcmc_conversations.py --features artifacts/features_stripped.npz \
      --k 32 --out-prefix results
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from scf.probability.markov import (
    estimate_transition, chapman_kolmogorov_test, stationary_distribution, mixing_time,
)


# --------------------------------------------------------------------------- #
#  clustering + trajectory helpers (shared with markov_chain.py)
# --------------------------------------------------------------------------- #
def cluster_states(mag, k, seed=0):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize
    X = normalize(mag.astype(np.float32))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096,
                         n_init=3, max_iter=100)
    return km.fit_predict(X)


def ordered_indices(conv, turn):
    return np.lexsort((turn, conv))


def trajectories(states, conv, turn, role=None):
    """Per-conversation lists of state (and optional role)."""
    order = ordered_indices(conv, turn)
    trajs, cur, cid = [], [], None
    for idx in order:
        if conv[idx] != cid:
            if len(cur) > 1:
                trajs.append(cur)
            cur, cid = [], conv[idx]
        cur.append(int(states[idx]) if role is None
                   else (int(states[idx]), int(role[idx])))
    if len(cur) > 1:
        trajs.append(cur)
    return trajs


def cond_entropy_rows(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0, -P * np.log(P), 0.0)
    return terms.sum(axis=1)


def row_stochastic(counts):
    c = counts.astype(np.float64)
    row = c.sum(axis=1, keepdims=True)
    safe = np.where(row > 0, row, 1.0)
    P = c / safe
    zero = (row.flatten() == 0)
    if zero.any():
        P[zero] = 0.0
        P[zero, np.where(zero)[0]] = 1.0
    return P


def weighted_entropy(P, counts):
    """Frequency-weighted mean conditional entropy of a kernel (nats)."""
    H = cond_entropy_rows(P)
    w = counts.sum(1)
    w = w / w.sum() if w.sum() else w
    return float((w * H).sum())


# --------------------------------------------------------------------------- #
#  role-coupled kernels  (H5/H6)
# --------------------------------------------------------------------------- #
def role_coupled_kernels(role_trajs, k):
    """Counts for human->ai and ai->next-human transitions."""
    C_ha = np.zeros((k, k))   # human-turn -> ai-turn        (AI self-surprise)
    C_ah = np.zeros((k, k))   # ai-turn    -> human-turn     (human surprise)
    for traj in role_trajs:
        for (s_a, r_a), (s_b, _r_b) in zip(traj[:-1], traj[1:]):
            if r_a == 0:      # leaving a human turn -> next is ai
                C_ha[s_a, s_b] += 1
            else:             # leaving an ai turn   -> next is human
                C_ah[s_a, s_b] += 1
    return C_ha, C_ah


def coupled_ck_residual(role_trajs, T_ha, T_ah, k):
    """2-step (human->human) empirical matrix vs T_ha @ T_ah, Frobenius residual.
    This is the role-aware analogue of the pooled CK test: a human turn maps to
    the next human turn through one ai turn, so the model 2-step kernel is the
    product of the two role kernels."""
    C2 = np.zeros((k, k))     # human -> next-human (two steps)
    for traj in role_trajs:
        for (s_a, r_a), _mid, (s_c, r_c) in zip(traj[:-2], traj[1:-1], traj[2:]):
            if r_a == 0 and r_c == 0:
                C2[s_a, s_c] += 1
    P2_emp = row_stochastic(C2)
    P2_model = T_ha @ T_ah
    return float(np.linalg.norm(P2_emp - P2_model)), C2


# --------------------------------------------------------------------------- #
#  Metropolis-Hastings sampler targeting pi
# --------------------------------------------------------------------------- #
def metropolis_hastings(pi, n_steps, rng, record_every=200):
    """Random-walk MH on {0..K-1} with a symmetric uniform proposal, target pi.
    Returns the visit-frequency vector and the TV(empirical, pi) trace."""
    K = pi.size
    x = int(rng.integers(K))
    visits = np.zeros(K)
    tv_trace = []
    accepts = 0
    for t in range(1, n_steps + 1):
        y = int(rng.integers(K))                       # symmetric proposal
        ratio = (pi[y] / pi[x]) if pi[x] > 0 else 1.0
        if rng.random() < min(1.0, ratio):
            x = y
            accepts += 1
        visits[x] += 1
        if t % record_every == 0:
            emp = visits / visits.sum()
            tv_trace.append(0.5 * float(np.abs(emp - pi).sum()))
    return visits / visits.sum(), np.array(tv_trace), accepts / n_steps


# --------------------------------------------------------------------------- #
#  forward simulation of synthetic conversations
# --------------------------------------------------------------------------- #
def simulate_conversations(role_trajs, T_ha, T_ah, k, rng, n_conv):
    """Ancestral sampling: alternate roles starting human, use the empirical
    opening-mode distribution and the empirical conversation-length distribution."""
    starts = np.zeros(k)
    lengths = []
    for traj in role_trajs:
        s0, r0 = traj[0]
        if r0 == 0:
            starts[s0] += 1
        lengths.append(len(traj))
    starts = starts / starts.sum() if starts.sum() else np.full(k, 1.0 / k)
    lengths = np.array(lengths)

    sims = []
    for _ in range(n_conv):
        L = int(rng.choice(lengths))
        s = int(rng.choice(k, p=starts))
        role = 0
        seq = [(s, role)]
        for _step in range(L - 1):
            T = T_ha if role == 0 else T_ah
            row = T[s]
            if row.sum() <= 0:
                break
            s = int(rng.choice(k, p=row / row.sum()))
            role = 1 - role
            seq.append((s, role))
        if len(seq) > 1:
            sims.append(seq)
    return sims


# --------------------------------------------------------------------------- #
#  posterior-predictive statistics
# --------------------------------------------------------------------------- #
def dwell_and_diversity(role_trajs, k):
    """Mode dwell-times (consecutive identical-mode run lengths) and #distinct
    modes per conversation -- higher-order summaries a 1-step chain need not match."""
    dwell = []
    distinct = []
    for traj in role_trajs:
        seq = [s for s, _r in traj]
        distinct.append(len(set(seq)))
        run = 1
        for a, b in zip(seq[:-1], seq[1:]):
            if a == b:
                run += 1
            else:
                dwell.append(run)
                run = 1
        dwell.append(run)
    return np.array(dwell), np.array(distinct)


def role_surprise(role_trajs, k):
    """Frequency-weighted conditional entropy of each role kernel, recomputed
    from a set of trajectories (works for real and synthetic identically)."""
    C_ha, C_ah = role_coupled_kernels(role_trajs, k)
    ai_self = weighted_entropy(row_stochastic(C_ha), C_ha)     # human->ai
    human_s = weighted_entropy(row_stochastic(C_ah), C_ah)     # ai->human
    return human_s, ai_self


# --------------------------------------------------------------------------- #
def analyze(features, k, seed):
    f = np.load(features)
    mag, conv, turn = f["magnitudes"], f["conv_id"], f["turn_idx"]
    role = f["role"].astype(int)
    print(f"[load] {features}: {mag.shape[0]} turns, {mag.shape[1]} feats, k={k}")

    states = cluster_states(mag, k, seed)
    plain = trajectories(states, conv, turn)
    rtrajs = trajectories(states, conv, turn, role=role)
    n_steps = sum(len(t) - 1 for t in plain)
    print(f"[chain] {len(plain)} conversations, {n_steps} turn-transitions")

    # pooled chain (reproduces markov_chain.py)
    fit = estimate_transition(plain, k)
    P, pi = fit.P_hat, fit.pi_hat
    ck_pooled = chapman_kolmogorov_test(fit).residual_frobenius
    tmix = mixing_time(P, pi, 0.25)
    eig = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    gap = float(1.0 - eig[1])
    h_rate = float((pi * cond_entropy_rows(P)).sum())
    pi_entropy = float(-(pi[pi > 0] * np.log(pi[pi > 0])).sum())
    # TV(start->pi) decay from the modal state
    v = np.zeros(k); v[int(np.argmax(pi))] = 1.0
    tv_power = []
    for _ in range(40):
        v = v @ P
        tv_power.append(0.5 * float(np.abs(v - pi).sum()))

    # role-coupled kernels (H5/H6)
    C_ha, C_ah = role_coupled_kernels(rtrajs, k)
    T_ha, T_ah = row_stochastic(C_ha), row_stochastic(C_ah)
    ck_coupled, _ = coupled_ck_residual(rtrajs, T_ha, T_ah, k)
    kernel_diff = float(np.linalg.norm(T_ha - T_ah))            # do the two differ?
    human_s, ai_self = role_surprise(rtrajs, k)

    # --- MCMC: Metropolis-Hastings targeting pi ---
    rng = np.random.default_rng(seed)
    mh_emp, mh_tv, mh_acc = metropolis_hastings(pi, n_steps=200_000, rng=rng)
    mh_tv_final = float(mh_tv[-1])

    # --- forward simulation + posterior-predictive checks ---
    sims = simulate_conversations(rtrajs, T_ha, T_ah, k, rng, n_conv=len(rtrajs))
    dwell_r, distinct_r = dwell_and_diversity(rtrajs, k)
    dwell_s, distinct_s = dwell_and_diversity(sims, k)
    human_s_sim, ai_self_sim = role_surprise(sims, k)
    # empirical mode distribution real vs synthetic
    pmode_r = np.bincount([s for t in rtrajs for s, _ in t], minlength=k).astype(float)
    pmode_s = np.bincount([s for t in sims for s, _ in t], minlength=k).astype(float)
    pmode_r /= pmode_r.sum(); pmode_s /= pmode_s.sum()
    tv_mode = 0.5 * float(np.abs(pmode_r - pmode_s).sum())

    summary = {
        "features": os.path.basename(features), "k": k,
        "n_conversations": len(plain), "n_turn_transitions": int(n_steps),
        "pooled": {
            "ck_residual": ck_pooled, "spectral_gap": gap, "mixing_time": int(tmix),
            "entropy_rate": h_rate, "pi_entropy": pi_entropy, "pi_entropy_max": float(np.log(k)),
        },
        "role_coupled": {
            "ck_residual_coupled": ck_coupled, "ck_residual_pooled": ck_pooled,
            "ck_improvement": ck_pooled - ck_coupled,
            "kernel_l2_difference": kernel_diff,
            "human_surprise": human_s, "ai_self_surprise": ai_self,
            "surprise_gap": human_s - ai_self,
        },
        "mcmc": {
            "mh_steps": 200_000, "mh_acceptance_rate": mh_acc,
            "mh_tv_to_pi_final": mh_tv_final,
        },
        "generative_check": {
            "n_synthetic_conversations": len(sims),
            "surprise_gap_real": human_s - ai_self,
            "surprise_gap_synthetic": human_s_sim - ai_self_sim,
            "tv_mode_distribution_real_vs_synthetic": tv_mode,
            "mean_dwell_real": float(dwell_r.mean()), "mean_dwell_synthetic": float(dwell_s.mean()),
            "max_dwell_real": int(dwell_r.max()), "max_dwell_synthetic": int(dwell_s.max()),
            "mean_distinct_modes_real": float(distinct_r.mean()),
            "mean_distinct_modes_synthetic": float(distinct_s.mean()),
        },
    }

    arrays = {
        "P": P, "pi": pi, "T_ha": T_ha, "T_ah": T_ah,
        "cond_entropy_rows": cond_entropy_rows(P),
        "tv_power": np.array(tv_power),
        "mh_emp": mh_emp, "mh_tv": mh_tv, "mh_record_every": 200,
        "pmode_real": pmode_r, "pmode_synthetic": pmode_s,
        "dwell_real": dwell_r, "dwell_synthetic": dwell_s,
        "distinct_real": distinct_r, "distinct_synthetic": distinct_s,
    }
    return summary, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/features_stripped.npz")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="results")
    args = ap.parse_args()

    corpus = ("wildchat" if "wildchat" in args.features else
              "stripped" if "stripped" in args.features else "other")
    summary, arrays = analyze(args.features, args.k, args.seed)

    os.makedirs(os.path.join(args.out_prefix, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.out_prefix, "mcmc"), exist_ok=True)
    with open(os.path.join(args.out_prefix, "data", f"mcmc_{corpus}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(os.path.join(args.out_prefix, "mcmc", f"arrays_{corpus}.npz"), **arrays)

    rc = summary["role_coupled"]; gc = summary["generative_check"]; mc = summary["mcmc"]
    print("\n=== role-coupled chain (H5/H6) ===")
    print(f"  CK residual: pooled {rc['ck_residual_pooled']:.3f} -> coupled "
          f"{rc['ck_residual_coupled']:.3f}  (improvement {rc['ck_improvement']:+.3f})")
    print(f"  ||T_h->a - T_a->h||_F = {rc['kernel_l2_difference']:.3f}  (kernels differ)")
    print(f"  human surprise {rc['human_surprise']:.3f} vs AI self-surprise "
          f"{rc['ai_self_surprise']:.3f}  gap {rc['surprise_gap']:+.3f}")
    print("\n=== Metropolis-Hastings (target = pi) ===")
    print(f"  acceptance {mc['mh_acceptance_rate']:.3f}; "
          f"TV(MH-empirical, pi) after 200k steps = {mc['mh_tv_to_pi_final']:.4f}")
    print("\n=== generative posterior-predictive check (synthetic vs real) ===")
    print(f"  surprise gap: real {gc['surprise_gap_real']:+.3f} | "
          f"synthetic {gc['surprise_gap_synthetic']:+.3f}  (Markov inherits the asymmetry)")
    print(f"  TV(mode dist real, synthetic) = {gc['tv_mode_distribution_real_vs_synthetic']:.4f}")
    print(f"  mean dwell: real {gc['mean_dwell_real']:.3f} | synthetic {gc['mean_dwell_synthetic']:.3f}")
    print(f"  distinct modes/conv: real {gc['mean_distinct_modes_real']:.2f} | "
          f"synthetic {gc['mean_distinct_modes_synthetic']:.2f}")


if __name__ == "__main__":
    main()
