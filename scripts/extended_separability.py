#!/usr/bin/env python3
"""
Diagnose whether and why per-turn SAE features are separable by role.

The probes measure the actual features in --features; nothing about the corpus
or expected outcome is assumed. The final verdict is computed from the AUCs.

Probes (CPU only):
  A. role, all turns           - linear (mag + support) and nonlinear
  B. role, generated-only      - exclude turn_idx==0 (no first-turn cue)
  C. seed vs generated         - turn_idx==0 vs >0 (does structure exist at all)
  D. group-aware role          - split by conv_id (no conversation leakage)

Reading: if A and B both separate, role is learnable from the features. If B
collapses toward chance while C stays high, the signal lives only in the first
turn and the features still encode distributional structure. Whether that
reflects genuine human/AI language or a corpus artifact (e.g. formatting) is
not something these probes can decide.
"""
from __future__ import annotations
import argparse
import numpy as np


def auc_acc(clf, Xtr, ytr, Xte, yte):
    from sklearn.metrics import roc_auc_score, accuracy_score
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return accuracy_score(yte, clf.predict(Xte)), roc_auc_score(yte, p)


def balanced_idx(mask_pos, mask_neg, n_per, rng):
    pos = np.where(mask_pos)[0]
    neg = np.where(mask_neg)[0]
    n = min(len(pos), len(neg), n_per)
    sel = np.concatenate([rng.choice(pos, n, replace=False),
                          rng.choice(neg, n, replace=False)])
    rng.shuffle(sel)
    return sel, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/features.npz")
    ap.add_argument("--sample", type=int, default=20000, help="max per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier

    print(f"[load] {args.features}")
    f = np.load(args.features)
    mag = f["magnitudes"]; sup = f["support"].astype(np.float32)
    role = f["role"].astype(int); conv = f["conv_id"]; turn = f["turn_idx"]
    N, F = mag.shape
    print(f"turns={N} features={F}  turn_idx range [{turn.min()},{turn.max()}]")
    print(f"role human={int((role==0).sum())} ai={int((role==1).sum())}  "
          f"turn0={(turn==0).sum()} generated(turn>0)={(turn>0).sum()}")

    def lin(): return LogisticRegression(max_iter=2000, C=1.0)

    # ---------- A: role, all turns ----------
    print("\n=== A. role separability (all turns) ===")
    sel, n = balanced_idx(role == 0, role == 1, args.sample, rng)
    X, y = mag[sel], role[sel]
    sc = StandardScaler().fit(X)
    Xtr, Xte, ytr, yte = train_test_split(sc.transform(X), y, test_size=.25,
                                           random_state=0, stratify=y)
    a, u = auc_acc(lin(), Xtr, ytr, Xte, yte)
    uA = u
    print(f"  linear(mag)   acc={a:.4f} AUC={u:.4f}  (n={2*n}, baseline 0.5)")
    Xb = sup[sel]
    Xbtr, Xbte, _, _ = train_test_split(Xb, y, test_size=.25, random_state=0, stratify=y)
    a, u = auc_acc(lin(), Xbtr, ytr, Xbte, yte)
    print(f"  linear(sup)   acc={a:.4f} AUC={u:.4f}")
    a, u = auc_acc(HistGradientBoostingClassifier(max_iter=200, random_state=0),
                   Xtr, ytr, Xte, yte)
    print(f"  nonlinear(GB) acc={a:.4f} AUC={u:.4f}")

    # ---------- B: role among generated-only ----------
    print("\n=== B. role separability, GENERATED turns only (turn_idx>0) ===")
    g = turn > 0
    sel, n = balanced_idx((role == 0) & g, (role == 1) & g, args.sample, rng)
    X, y = mag[sel], role[sel]
    sc = StandardScaler().fit(X)
    Xtr, Xte, ytr, yte = train_test_split(sc.transform(X), y, test_size=.25,
                                           random_state=0, stratify=y)
    a, u = auc_acc(lin(), Xtr, ytr, Xte, yte)
    uB = u
    print(f"  linear(mag)   acc={a:.4f} AUC={u:.4f}  (n={2*n})")

    # ---------- C: seed (turn 0) vs generated ----------
    print("\n=== C. seed(turn0) vs generated(turn>0) ===")
    sel, n = balanced_idx(turn == 0, turn > 0, args.sample, rng)
    X = mag[sel]; y = (turn[sel] > 0).astype(int)  # 1=generated, 0=seed
    sc = StandardScaler().fit(X)
    Xtr, Xte, ytr, yte = train_test_split(sc.transform(X), y, test_size=.25,
                                           random_state=0, stratify=y)
    a, u = auc_acc(lin(), Xtr, ytr, Xte, yte)
    uC = u
    print(f"  linear(mag)   acc={a:.4f} AUC={u:.4f}  (n={2*n})")

    # ---------- D: group-aware role (no conv leakage) ----------
    print("\n=== D. role, group-aware split by conv_id ===")
    sel, n = balanced_idx(role == 0, role == 1, args.sample, rng)
    X, y, g = mag[sel], role[sel], conv[sel]
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    gss = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=0)
    tr, te = next(gss.split(Xs, y, groups=g))
    a, u = auc_acc(lin(), Xs[tr], y[tr], Xs[te], y[te])
    uD = u
    print(f"  linear(mag)   acc={a:.4f} AUC={u:.4f}  (no shared conversations)")

    # ---------- verdict computed from the actual AUCs ----------
    hi, lo = 0.70, 0.60  # separable / near-chance thresholds
    print("\n=== INTERPRETATION (computed from the AUCs above) ===")
    if uA >= hi and uB >= hi:
        print(f"Role is separable from the features (A={uA:.3f}, B={uB:.3f}), and the")
        print("signal is not just the first turn (B stays high). Group-aware split")
        print(f"D={uD:.3f} {'agrees' if abs(uD-uA) < 0.05 else 'differs from A — check for leakage'}.")
        print("These probes cannot tell genuine human/AI language from a corpus")
        print("artifact (e.g. role-specific formatting); inspect top features to decide.")
    elif uA >= lo > uB:
        print(f"Weak role signal (A={uA:.3f}) that collapses on generated-only turns")
        print(f"(B={uB:.3f} ~ chance): the signal lives in the first turn, not the role.")
        print(f"Structure still exists in the features (C={uC:.3f}). Role is effectively")
        print("unlearnable here — typical of a corpus where both roles share one")
        print("distribution. A corpus whose roles differ in distribution would separate.")
    else:
        print(f"Role near chance (A={uA:.3f}, B={uB:.3f}). seed-vs-generated C={uC:.3f}:")
        print("if C is high the features encode structure but not role; if C is also low")
        print("the features may be degenerate — check sanity output from step 06.")


if __name__ == "__main__":
    main()
