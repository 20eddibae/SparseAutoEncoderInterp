#!/usr/bin/env python3
"""
Conversation-level (turn) Markov chain over SAE features. CPU-only, runs on the
already-extracted per-turn features (no GPU, no re-extraction).

This is "Chain 2" from the research plan: states are conversation modes (clusters
of per-turn SAE activation patterns), time steps are TURNS within a conversation.
We ask the probability questions from RESEARCH.md:

  - Markov fit + Chapman-Kolmogorov residual  (markov.py)            [Markov]
  - stationary distribution pi and whether the chain CONVERGES to it [Markov]
  - mixing time / spectral gap  (how fast a conversation forgets its start)
  - entropy rate h = -sum_i pi_i sum_j P_ij log P_ij                 [entropy]
  - per-state conditional entropy H(P_i.)  -> deterministic vs open states
  - role asymmetry: "human surprise" vs "AI self-surprise"
        compares the entropy of ai->human turn transitions (how unpredictable
        the human's next mode is) against human->ai transitions (the model).

Usage:
  python scripts/markov_chain.py --features artifacts/features_stripped.npz --k 32
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from scf.probability.markov import (
    estimate_transition, chapman_kolmogorov_test, stationary_distribution, mixing_time,
)


def cluster_states(mag, k, seed=0):
    """Option 3: cluster L2-normalized per-turn activation vectors into k modes."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize
    X = normalize(mag.astype(np.float32))            # cluster by pattern, not scale
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096,
                         n_init=3, max_iter=100)
    return km.fit_predict(X)


def trajectories_by_conv(states, conv, turn):
    """Ordered per-conversation state sequences."""
    order = np.lexsort((turn, conv))
    trajs, cur, cid = [], [], None
    for idx in order:
        if conv[idx] != cid:
            if len(cur) > 1: trajs.append(cur)
            cur, cid = [], conv[idx]
        cur.append(int(states[idx]))
    if len(cur) > 1: trajs.append(cur)
    return trajs


def cond_entropy_rows(P):
    """H(P_i.) for each row, in nats."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0, -P * np.log(P), 0.0)
    return terms.sum(axis=1)


def role_transition_entropy(states, conv, turn, role, k, src_role):
    """Mean conditional entropy of transitions LEAVING a turn whose role==src_role,
    i.e. how unpredictable the *next* turn's mode is given the current mode.
    src_role=1 (ai) -> next is human  => 'human surprise'
    src_role=0 (human) -> next is ai  => 'AI self-surprise'."""
    order = np.lexsort((turn, conv))
    C = np.zeros((k, k))
    for a, b in zip(order[:-1], order[1:]):
        if conv[a] != conv[b]:        # not consecutive within a conversation
            continue
        if role[a] != src_role:
            continue
        C[states[a], states[b]] += 1
    rowsum = C.sum(1, keepdims=True)
    P = np.divide(C, rowsum, out=np.zeros_like(C), where=rowsum > 0)
    H = cond_entropy_rows(P)
    w = rowsum.flatten(); w = w / w.sum() if w.sum() else w   # weight by frequency
    return float((w * H).sum()), int(rowsum.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/features_stripped.npz")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    f = np.load(args.features)
    mag = f["magnitudes"]; conv = f["conv_id"]; turn = f["turn_idx"]; role = f["role"].astype(int)
    print(f"[load] {args.features}: {mag.shape[0]} turns, {mag.shape[1]} feats, k={args.k}")

    states = cluster_states(mag, args.k, args.seed)
    trajs = trajectories_by_conv(states, conv, turn)
    n_steps = sum(len(t) - 1 for t in trajs)
    print(f"[chain] {len(trajs)} conversations, {n_steps} turn-transitions")

    fit = estimate_transition(trajs, args.k)
    P, pi = fit.P_hat, fit.pi_hat

    # --- convergence to stationary distribution ---
    ck = chapman_kolmogorov_test(fit)
    tmix = mixing_time(P, pi, epsilon=0.25)
    eig = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    gap = 1.0 - eig[1]
    # power-iterate from a point mass to watch TV(.,pi) decay
    v = np.zeros(args.k); v[int(np.argmax(pi))] = 1.0
    tv = []
    for _ in range(40):
        v = v @ P
        tv.append(0.5 * np.abs(v - pi).sum())

    # --- entropy rate + per-state conditional entropy ---
    Hrows = cond_entropy_rows(P)
    h_rate = float((pi * Hrows).sum())
    pi_entropy = float(-(pi[pi > 0] * np.log(pi[pi > 0])).sum())
    order = np.argsort(pi)[::-1]

    # --- role asymmetry: human surprise vs AI self-surprise ---
    h_surprise, n_ah = role_transition_entropy(states, conv, turn, role, args.k, src_role=1)
    ai_surprise, n_ha = role_transition_entropy(states, conv, turn, role, args.k, src_role=0)

    print("\n=== Markov fit (turn-level conversation chain) ===")
    print(f"  Chapman-Kolmogorov residual (Frobenius) = {ck.residual_frobenius:.4f}  "
          f"(0 = perfectly 1st-order Markov)")
    print(f"  spectral gap 1-|lambda2| = {gap:.4f}   |lambda2| = {eig[1]:.4f}")
    print(f"  mixing time (TV<=0.25)   = {tmix} turns")
    print(f"  TV(start->pi) after 1/2/5/10 turns = "
          f"{tv[0]:.3f} / {tv[1]:.3f} / {tv[4]:.3f} / {tv[9]:.3f}")
    converged = tv[-1] < 1e-3
    print(f"  -> converges to a unique stationary distribution: {converged}")

    print("\n=== stationary distribution pi (does the conversation settle on a mode?) ===")
    print(f"  entropy of pi = {pi_entropy:.3f} nats  (max = {np.log(args.k):.3f}; "
          f"low => collapses onto few modes)")
    print(f"  top-5 modes (state: pi): " +
          ", ".join(f"{int(s)}:{pi[s]:.3f}" for s in order[:5]))

    print("\n=== entropy rate & per-state predictability ===")
    print(f"  entropy rate h = {h_rate:.3f} nats/turn  (max = {np.log(args.k):.3f})")
    det = np.argsort(Hrows)[:3]; opn = np.argsort(Hrows)[::-1][:3]
    print(f"  most DETERMINISTIC modes (low H): " +
          ", ".join(f"{int(s)}:{Hrows[s]:.2f}" for s in det))
    print(f"  most OPEN-ENDED modes (high H):   " +
          ", ".join(f"{int(s)}:{Hrows[s]:.2f}" for s in opn))

    print("\n=== role asymmetry (where the human enters) ===")
    print(f"  human surprise  H(ai-turn -> next human-turn) = {h_surprise:.3f} nats  (n={n_ah})")
    print(f"  AI self-surprise H(human-turn -> next ai-turn) = {ai_surprise:.3f} nats  (n={n_ha})")
    print(f"  gap (human - AI) = {h_surprise - ai_surprise:+.3f} nats  "
          f"(>0 => the human's next mode is less predictable than the model's)")


if __name__ == "__main__":
    main()
