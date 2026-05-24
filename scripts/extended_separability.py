#!/usr/bin/env python3
"""
Diagnose WHY per-turn SAE features are / are not separable.

Hypothesis: the `synthetic` corpus generates every turn after the first from
the SAME transformer (same sampling), so human vs ai turns are drawn from one
distribution and role is unseparable by construction. The only real text is
turn 0 (an OpenWebText seed). These probes test that mechanism and check
whether the features encode ANY recoverable structure.

Probes (all on artifacts/features.npz, CPU only):
  A. role, all turns           - linear (mag + support) and nonlinear
  B. role, generated-only      - exclude turn_idx==0; should collapse to chance
  C. seed vs generated         - turn_idx==0 vs >0; separable => features OK
  D. group-aware role          - split by conv_id (no conversation leakage)
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
    print(f"  linear(mag)   acc={a:.4f} AUC={u:.4f}  (n={2*n})")
    print("  -> expect ~chance: confirms generated human/ai are one distribution")

    # ---------- C: seed (turn 0) vs generated ----------
    print("\n=== C. seed(turn0) vs generated(turn>0) ===")
    sel, n = balanced_idx(turn == 0, turn > 0, args.sample, rng)
    X = mag[sel]; y = (turn[sel] > 0).astype(int)  # 1=generated, 0=seed
    sc = StandardScaler().fit(X)
    Xtr, Xte, ytr, yte = train_test_split(sc.transform(X), y, test_size=.25,
                                           random_state=0, stratify=y)
    a, u = auc_acc(lin(), Xtr, ytr, Xte, yte)
    print(f"  linear(mag)   acc={a:.4f} AUC={u:.4f}  (n={2*n})")
    print("  -> high AUC => features DO encode distribution; role label is the issue")

    # ---------- D: group-aware role (no conv leakage) ----------
    print("\n=== D. role, group-aware split by conv_id ===")
    sel, n = balanced_idx(role == 0, role == 1, args.sample, rng)
    X, y, g = mag[sel], role[sel], conv[sel]
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    gss = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=0)
    tr, te = next(gss.split(Xs, y, groups=g))
    a, u = auc_acc(lin(), Xs[tr], y[tr], Xs[te], y[te])
    print(f"  linear(mag)   acc={a:.4f} AUC={u:.4f}  (no shared conversations)")

    print("\n=== INTERPRETATION ===")
    print("If B ~0.50 and C >>0.50: features are fine; the synthetic corpus makes")
    print("role unlearnable (both roles sampled from the same model). Fix = use a")
    print("real human/AI corpus (sharegpt) where the two roles differ in distribution.")


if __name__ == "__main__":
    main()
