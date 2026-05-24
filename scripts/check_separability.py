#!/usr/bin/env python3
"""
Sanity-check the extracted SAE features and test whether per-turn feature
vectors are linearly separable by role (human vs assistant).

Usage:
  python scripts/check_separability.py --features artifacts/features.npz \
      --plot artifacts/separability.png
"""
from __future__ import annotations
import argparse
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/features.npz")
    ap.add_argument("--plot", default="artifacts/separability.png")
    ap.add_argument("--sample", type=int, default=40000,
                    help="max turns to use for the classifier (balanced)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"[load] {args.features}")
    f = np.load(args.features)
    mag = f["magnitudes"]          # (N, F) float32
    sup = f["support"]             # (N, F) uint8
    role = f["role"].astype(int)   # (N,) 0=human 1=assistant
    N, F = mag.shape

    # ---- sanity ----
    print("\n=== SANITY ===")
    print(f"turns={N}  n_features={F}")
    print(f"role: human={int((role==0).sum())}  assistant={int((role==1).sum())}")
    print(f"NaN={bool(np.isnan(mag).any())}  Inf={bool(np.isinf(mag).any())}")
    print(f"mag min/mean/max = {mag.min():.4g} / {mag.mean():.4g} / {mag.max():.4g}")
    act_per_turn = sup.sum(1)
    print(f"active feats/turn: mean={act_per_turn.mean():.1f} "
          f"min={act_per_turn.min()} max={act_per_turn.max()}")
    feat_freq = sup.sum(0)
    print(f"dead features (never active): {int((feat_freq==0).sum())}/{F}")
    print(f"always-active features: {int((feat_freq==N).sum())}/{F}")
    n_empty = int((act_per_turn == 0).sum())
    print(f"all-zero turns: {n_empty}")

    # ---- balanced subsample ----
    idx0 = np.where(role == 0)[0]
    idx1 = np.where(role == 1)[0]
    per = min(len(idx0), len(idx1), args.sample // 2)
    sel = np.concatenate([rng.choice(idx0, per, replace=False),
                          rng.choice(idx1, per, replace=False)])
    rng.shuffle(sel)
    X = mag[sel].astype(np.float32)
    y = role[sel]
    print(f"\n=== SEPARABILITY (balanced n={len(sel)}, {per}/class) ===")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                           random_state=args.seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xte_s)[:, 1]
    acc = accuracy_score(yte, clf.predict(Xte_s))
    auc = roc_auc_score(yte, proba)
    base = max((yte == 0).mean(), (yte == 1).mean())
    print(f"logreg(magnitudes):  test_acc={acc:.4f}  AUC={auc:.4f}  baseline={base:.4f}")

    # binary support classifier
    Xb = sup[sel].astype(np.float32)
    Xbtr, Xbte = Xb[: len(Xtr)], Xb[len(Xtr):]
    # reuse same split indices
    Xbtr, Xbte, _, _ = train_test_split(Xb, y, test_size=0.25,
                                        random_state=args.seed, stratify=y)
    clf_b = LogisticRegression(max_iter=2000, C=1.0)
    clf_b.fit(Xbtr, ytr)
    acc_b = accuracy_score(yte, clf_b.predict(Xbte))
    auc_b = roc_auc_score(yte, clf_b.predict_proba(Xbte)[:, 1])
    print(f"logreg(support):     test_acc={acc_b:.4f}  AUC={auc_b:.4f}")

    # top discriminative features
    coef = clf.coef_[0]
    top = np.argsort(np.abs(coef))[::-1][:10]
    print("top-10 discriminative features (idx: coef):")
    print("  " + ", ".join(f"{i}:{coef[i]:+.2f}" for i in top))

    # ---- LDA 1-D projection for the plot ----
    lda = LinearDiscriminantAnalysis(n_components=1)
    z = lda.fit_transform(Xtr_s, ytr).ravel()
    zte = lda.transform(Xte_s).ravel()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for c, lab, col in [(0, "human", "tab:blue"), (1, "assistant", "tab:orange")]:
            ax[0].hist(z[ytr == c], bins=60, alpha=0.6, label=lab, color=col, density=True)
        ax[0].set_title(f"LDA projection (train)\nlogreg AUC={auc:.3f} acc={acc:.3f}")
        ax[0].set_xlabel("LDA axis"); ax[0].legend()
        # PCA 2D of a small subset for visual
        from sklearn.decomposition import PCA
        sub = rng.choice(len(Xtr_s), min(4000, len(Xtr_s)), replace=False)
        p = PCA(n_components=2).fit_transform(Xtr_s[sub])
        for c, lab, col in [(0, "human", "tab:blue"), (1, "assistant", "tab:orange")]:
            m = ytr[sub] == c
            ax[1].scatter(p[m, 0], p[m, 1], s=4, alpha=0.4, label=lab, color=col)
        ax[1].set_title("PCA(2) of features"); ax[1].legend()
        fig.tight_layout(); fig.savefig(args.plot, dpi=120)
        print(f"\n[plot] wrote {args.plot}")
    except Exception as e:
        print(f"[plot] skipped: {e}")

    # ---- verdict ----
    print("\n=== VERDICT ===")
    if auc >= 0.9:
        print(f"STRONGLY separable by role (AUC={auc:.3f}).")
    elif auc >= 0.75:
        print(f"Moderately separable by role (AUC={auc:.3f}).")
    elif auc >= 0.6:
        print(f"Weakly separable by role (AUC={auc:.3f}).")
    else:
        print(f"NOT linearly separable by role (AUC={auc:.3f}) — near chance.")


if __name__ == "__main__":
    main()
