#!/usr/bin/env python3
"""
Evaluate how much of the ShareGPT role-separability is genuine human/AI
language vs the HTML-formatting tell (assistant turns are <div class="markdown
prose"> wrapped). Uses ONLY the already-extracted artifacts/features.npz — no
GPU / no re-extraction. Aligns each stored feature row back to its raw turn
text by replaying the corpus loader in the same order.

Tests:
  0. Replay-align rows -> raw text -> html flag; verify P(html|role).
  1. Reproduce role AUC (all turns) to confirm baseline.
  2. How concentrated is the signal? best single-feature AUC + top-k.
  3. Does "html present" alone predict role? (the trivial tell)
  4. KEY: role separability with formatting CONTROLLED:
       (a) human(plain) vs AI turns that are NOT html-wrapped
       (b) within html-wrapped AI vs plain human, do features still help
           beyond the html flag?  (role AUC among matched-format is the clean #)
"""
from __future__ import annotations
import os, sys, json, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.chdir(os.path.join(HERE, ".."))

from scf.config import Config
from scf.corpus import load_corpus

HTML_RE = re.compile(r'<div|class="markdown|class=\\"markdown|</div>|<p>|<ol>|<ul>|<code>|<pre>')

def html_flag(text: str) -> int:
    return 1 if HTML_RE.search(text) else 0

def main():
    cfg = Config.from_yaml("configs/discovery.yaml")
    f = np.load("artifacts/features.npz")
    mag = f["magnitudes"]; role = f["role"].astype(int)
    conv = f["conv_id"]; turn = f["turn_idx"]
    N = mag.shape[0]
    print(f"[features] {N} turns, {mag.shape[1]} feats")

    # ---- 0. replay loader in identical order to recover per-row text ----
    print("[replay] re-walking corpus to tag html (same order as extraction)...")
    html = np.zeros(N, dtype=np.int8); ridx = 0
    ok = True
    for cid, c in enumerate(load_corpus(cfg)):
        for ti, t in enumerate(c.turns):
            if ridx >= N: break
            # sanity: row order must match stored conv_id/turn_idx
            if conv[ridx] != cid or turn[ridx] != ti:
                ok = False
            html[ridx] = html_flag(t.text)
            ridx += 1
        if ridx >= N: break
    print(f"[replay] tagged {ridx}/{N} rows; order-match={ok}")
    if ridx < N:
        html = html[:ridx]; mag = mag[:ridx]; role = role[:ridx]
        conv = conv[:ridx]; turn = turn[:ridx]; N = ridx

    h, a = role == 0, role == 1
    print(f"\n=== P(html | role) ===")
    print(f"  human turns: {h.sum():6d}  html={html[h].mean():.3f}")
    print(f"  ai    turns: {a.sum():6d}  html={html[a].mean():.3f}")
    print(f"  AI turns WITHOUT html: {int(((a)&(html==0)).sum())}")
    print(f"  human turns WITH html: {int(((h)&(html==1)).sum())}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    def bal(pos, neg, n=15000):
        p = np.where(pos)[0]; q = np.where(neg)[0]
        m = min(len(p), len(q), n)
        s = np.concatenate([rng.choice(p, m, False), rng.choice(q, m, False)])
        rng.shuffle(s); return s, m
    def fit_auc(X, y, scale=True):
        if scale: X = StandardScaler().fit_transform(X)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.25,random_state=0,stratify=y)
        clf = LogisticRegression(max_iter=2000).fit(Xtr,ytr)
        return roc_auc_score(yte, clf.predict_proba(Xte)[:,1]), clf

    # ---- 1. reproduce baseline role AUC ----
    sel,m = bal(h, a)
    auc,clf = fit_auc(mag[sel], role[sel])
    print(f"\n=== 1. baseline role AUC (all turns, n={2*m}) = {auc:.4f} ===")

    # ---- 2. concentration: single best feature, top-k ----
    print("\n=== 2. how concentrated is the signal? ===")
    coef = np.abs(clf.coef_[0]); order = np.argsort(coef)[::-1]
    # single-feature AUCs on the magnitude directly (no scaling needed for 1 feat AUC)
    Xs, ys = mag[sel], role[sel]
    single = []
    for fi in order[:8]:
        single.append((int(fi), roc_auc_score(ys, Xs[:,fi])))
    print("  top-|coef| features -> single-feature AUC:")
    for fi,u in single:
        # does this feature fire more on html turns?
        fire_h = (mag[(html==1),fi] > 0).mean(); fire_p = (mag[(html==0),fi] > 0).mean()
        print(f"    feat {fi:4d}: AUC={u:.3f}  fires html={fire_h:.2f} plain={fire_p:.2f}")
    for k in (1,5,20,100):
        topk = order[:k]
        u,_ = fit_auc(mag[sel][:,topk], role[sel])
        print(f"  top-{k:3d} feats role AUC = {u:.4f}")

    # ---- 3. html flag alone ----
    print("\n=== 3. does the html flag ALONE predict role? ===")
    print(f"  AUC(html_flag -> role) = {roc_auc_score(role[sel], html[sel]):.4f}")

    # ---- 4. role separability with formatting controlled ----
    print("\n=== 4. role AUC with formatting CONTROLLED ===")
    # 4a: human(plain) vs AI-without-html  -> language only, no format tell
    pos = (a) & (html==0); neg = (h) & (html==0)
    if pos.sum() > 200 and neg.sum() > 200:
        sel2,m2 = bal(neg, pos, n=min(pos.sum(),neg.sum()))
        u,_ = fit_auc(mag[sel2], role[sel2])
        print(f"  4a clean (human-plain vs AI-plain): AUC={u:.4f}  (n={2*m2})")
    else:
        print(f"  4a skipped: AI-plain={int(pos.sum())} human-plain={int(neg.sum())}")
    # 4b: restrict to html turns only (mostly AI) vs ... not meaningful; instead
    #     control by removing html-correlated features: drop top html-discriminative
    print("  4b drop the most html-predictive features, re-test role:")
    # rank features by html-AUC, drop top D, re-fit role on all turns
    htmlsel,_ = bal(html==0, html==1)
    Xh = mag[htmlsel]; yh = html[htmlsel]
    html_auc_per_feat = np.array([roc_auc_score(yh, Xh[:,j]) if Xh[:,j].any() else 0.5
                                   for j in range(mag.shape[1])])
    html_rank = np.argsort(np.abs(html_auc_per_feat-0.5))[::-1]
    for D in (0,1,10,50,200):
        drop = set(html_rank[:D].tolist())
        keep = np.array([j for j in range(mag.shape[1]) if j not in drop])
        u,_ = fit_auc(mag[sel][:,keep], role[sel])
        print(f"    drop top-{D:3d} html feats -> role AUC = {u:.4f}  (kept {len(keep)})")

if __name__ == "__main__":
    main()
