#!/usr/bin/env python3
"""
Render every result in the project to figures under `results/figures/`.

Data sources (all produced on our 1-layer GPT + 4096-feature SAE):
  - results/data/h2_{corpus}.json      role entropy / Bayes      (run_hypothesis --h 2)
  - results/data/h3_{corpus}.json      L0 concentration / Poisson(run_hypothesis --h 3)
  - results/mcmc/arrays_{corpus}.npz   Markov chain + MCMC arrays (mcmc_conversations.py)
  - SEPARABILITY (below)               classifier AUCs            (documented in RESEARCH.md)

Separability AUCs are point estimates from SLURM classifier jobs (8687826 stripped,
8687859 wildchat); they are stable and recorded in RESEARCH.md, so we plot the
recorded table rather than re-running the classifiers.

Usage:  python scripts/make_plots.py
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
CORPORA = ["stripped", "wildchat"]
NICE = {"stripped": "ShareGPT (stripped)", "wildchat": "WildChat"}
COL = {"stripped": "#2c7fb8", "wildchat": "#d95f0e"}

# Recorded role-separability AUCs (RESEARCH.md, jobs 8687826 / 8687859).
SEPARABILITY = {
    "A linear":      {"stripped": 0.918, "wildchat": 0.921},
    "A nonlinear":   {"stripped": 0.978, "wildchat": 0.981},
    "B generated":   {"stripped": 0.931, "wildchat": 0.934},
    "C seed-vs-gen": {"stripped": 0.757, "wildchat": 0.786},
    "D group-aware": {"stripped": 0.914, "wildchat": 0.919},
}


def load_json(corpus, h):
    p = os.path.join(ROOT, "results", "data", f"h{h}_{corpus}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_arrays(corpus):
    p = os.path.join(ROOT, "results", "mcmc", f"arrays_{corpus}.npz")
    return np.load(p) if os.path.exists(p) else None


def save(fig, name):
    out = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, ROOT)}")


# --------------------------------------------------------------------------- #
def plot_separability():
    probes = list(SEPARABILITY)
    x = np.arange(len(probes)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, c in enumerate(CORPORA):
        vals = [SEPARABILITY[p][c] for p in probes]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=NICE[c], color=COL[c])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance (0.50)")
    ax.set_xticks(x); ax.set_xticklabels(probes, rotation=15, ha="right")
    ax.set_ylabel("ROC AUC"); ax.set_ylim(0.45, 1.02)
    ax.set_title("Role separability of SAE turn-features (genuine, HTML-stripped)\n"
                 "human vs AI is recoverable at AUC ~0.92 linear / ~0.98 nonlinear, "
                 "robust across two corpora")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "01_separability.png")


def plot_h3():
    d = load_json("stripped", 3)
    if d is None:
        print("  [skip] h3_stripped.json missing"); return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # (a) L0 PMF: empirical vs Poisson null
    k = np.array(d["poisson"]["k_grid"]); emp = np.array(d["poisson"]["empirical_pmf"])
    poi = np.array(d["poisson"]["poisson_pmf"])
    ax = axes[0]
    ax.plot(k, emp, label="empirical L0", color=COL["stripped"], lw=1.8)
    ax.plot(k, poi, label=f"Poisson(λ={d['poisson']['lambda_hat']:.1f}) null", color="grey", ls="--")
    ax.set_xlim(0, 200); ax.set_xlabel("active features / turn (L0)"); ax.set_ylabel("probability")
    vmr = d["var"] / d["mu"]
    ax.set_title(f"L0 is super-Poissonian\nvar/mean ≈ {vmr:.1f} (Poisson = 1) → features co-fire")
    ax.legend(fontsize=8)

    # (b) tail concentration bounds
    ax = axes[1]
    t = np.array(d["tail_grid"])
    ax.semilogy(t, np.clip(d["empirical_tail"], 1e-6, 1), label="empirical", color="k", lw=2)
    ax.semilogy(t, np.clip(d["markov_tail"], 1e-6, 1), label="Markov", ls="--")
    ax.semilogy(t, np.clip(d["chebyshev_tail"], 1e-6, 1), label="Chebyshev", ls="--")
    ax.semilogy(t, np.clip(d["sub_gaussian_tail"], 1e-6, 1), label="sub-Gaussian", ls="--")
    ax.set_xlabel("t  (deviation above mean)"); ax.set_ylabel("P(L0 − μ ≥ t)")
    ax.set_title("Concentration bounds vs empirical tail\nsub-Gaussian tightest; all valid")
    ax.legend(fontsize=8)

    # (c) variance/mean + Poisson rejection + CLT
    ax = axes[2]; ax.axis("off")
    p = d["poisson"]; clt = d["clt"]
    lines = [
        "L0 summary (ShareGPT stripped)",
        f"  mean μ            = {d['mu']:.2f}",
        f"  variance          = {d['var']:.1f}",
        f"  var/mean (Fano)   = {d['var']/d['mu']:.2f}   (Poisson = 1)",
        "",
        "Poisson null test",
        f"  λ̂ (Le Cam)       = {p['lambda_hat']:.1f}",
        f"  Le Cam TV bound   = {p['le_cam_bound']:.1f}   (>>1 -> void)",
        f"  χ² p-value        = {p['chi2_pvalue']:.1e}   → rejected",
        "",
        "CLT for the sample mean",
        f"  blocks            = {clt['n_blocks']} × {clt['block_size']}",
        f"  KS p-value        = {clt['ks_pvalue']:.2f}   (block means Gaussian ✓)",
        "",
        "tail-sum identity  E[N]=Σ P(N≥k)",
        f"  {d['expectation_tail_sum']:.6f} vs {d['expectation_direct']:.6f} ✓",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.5)
    fig.suptitle("H3 — feature density (L0): a well-behaved but strongly over-dispersed RV", y=1.04)
    save(fig, "02_h3_concentration.png")


def plot_h2():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    roles = ["human", "ai"]; x = np.arange(len(roles)); w = 0.38

    # (a) marginal entropy with Chebyshev error bars
    ax = axes[0]
    for i, c in enumerate(CORPORA):
        d = load_json(c, 2)
        if d is None: continue
        H = [d["entropy_human"]["H_hat"], d["entropy_ai"]["H_hat"]]
        err = [d["entropy_human"]["chebyshev_eps"], d["entropy_ai"]["chebyshev_eps"]]
        b = ax.bar(x + (i - 0.5) * w, H, w, yerr=err, capsize=4, label=NICE[c], color=COL[c])
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=8)
    ax.set_xticks(x); ax.set_xticklabels(roles)
    ax.set_ylabel("plug-in entropy (nats)"); ax.set_ylim(4.4, 5.0)
    ax.set_title("Marginal feature entropy ≈ equal by role\n(±Chebyshev 1σ) → not a 'mode-seeking' gap")
    ax.legend(fontsize=8)

    # (b) active feature vocabulary
    ax = axes[1]
    for i, c in enumerate(CORPORA):
        d = load_json(c, 2)
        if d is None: continue
        V = [d["entropy_human"]["support_size"], d["entropy_ai"]["support_size"]]
        b = ax.bar(x + (i - 0.5) * w, V, w, label=NICE[c], color=COL[c])
        ax.bar_label(b, fmt="%d", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(roles)
    ax.set_ylabel("# distinct active features")
    ax.set_title("AI uses a broader feature vocabulary")
    ax.legend(fontsize=8)

    # (c) KL asymmetry
    ax = axes[2]
    labels = ["KL(human‖ai)", "KL(ai‖human)"]; xx = np.arange(len(labels))
    for i, c in enumerate(CORPORA):
        d = load_json(c, 2)
        if d is None: continue
        v = [d["kl_human_to_ai"], d["kl_ai_to_human"]]
        b = ax.bar(xx + (i - 0.5) * w, v, w, label=NICE[c], color=COL[c])
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("KL divergence (nats)")
    ax.set_title("Small but asymmetric role divergence")
    ax.legend(fontsize=8)
    fig.suptitle("H2 — role entropy: separability is about WHICH features fire, not how many", y=1.04)
    save(fig, "03_h2_entropy.png")


def plot_markov():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) transition heatmap (stripped)
    a = load_arrays("stripped")
    if a is not None:
        P = a["P"]; pi = a["pi"]
        order = np.argsort(pi)[::-1]
        ax = axes[0, 0]
        im = ax.imshow(P[np.ix_(order, order)], cmap="viridis", aspect="auto")
        ax.set_title("Turn→turn transition matrix P (ShareGPT)\nmodes sorted by stationary mass")
        ax.set_xlabel("next mode"); ax.set_ylabel("current mode")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # (b) stationary distribution
    ax = axes[0, 1]
    for c in CORPORA:
        a = load_arrays(c)
        if a is None: continue
        pi = np.sort(a["pi"])[::-1]
        ax.plot(np.arange(pi.size), pi, marker="o", ms=3, label=NICE[c], color=COL[c])
    ax.axhline(1/32, ls=":", c="grey", label="uniform (1/32)")
    ax.set_xlabel("mode (rank)"); ax.set_ylabel("stationary π")
    ax.set_title("Stationary distribution: spread over modes\n(high entropy, a few mild attractors)")
    ax.legend(fontsize=8)

    # (c) mixing: TV(start→π) decay
    ax = axes[1, 0]
    for c in CORPORA:
        a = load_arrays(c)
        if a is None: continue
        tv = a["tv_power"]
        ax.semilogy(np.arange(1, tv.size + 1), np.clip(tv, 1e-6, 1),
                    marker="o", ms=3, label=NICE[c], color=COL[c])
    ax.axhline(0.25, ls="--", c="grey", label="mixing threshold (TV=0.25)")
    ax.set_xlabel("turns"); ax.set_ylabel("TV(Pᵗ(start,·), π)")
    ax.set_xlim(1, 20)
    ax.set_title("Conversations forget their opening mode\n(ergodic; mixing in ~4–10 turns)")
    ax.legend(fontsize=8)

    # (d) per-state conditional entropy (predictability spectrum)
    ax = axes[1, 1]
    for c in CORPORA:
        a = load_arrays(c)
        if a is None: continue
        H = np.sort(a["cond_entropy_rows"])
        ax.plot(np.arange(H.size), H, marker="o", ms=3, label=NICE[c], color=COL[c])
    ax.axhline(np.log(32), ls=":", c="grey", label="max (ln 32)")
    ax.set_xlabel("mode (sorted)"); ax.set_ylabel("H(next | mode) (nats)")
    ax.set_title("Predictability spectrum across modes\nsome near-deterministic, some wide-open")
    ax.legend(fontsize=8)
    fig.suptitle("H1/H4 — conversation-mode Markov chain (K=32): ergodic, mixing, interpretable", y=1.0)
    save(fig, "04_markov_chain.png")


def plot_mcmc():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Metropolis-Hastings convergence to pi
    ax = axes[0, 0]
    for c in CORPORA:
        a = load_arrays(c)
        if a is None: continue
        tv = a["mh_tv"]; every = int(a["mh_record_every"])
        steps = np.arange(1, tv.size + 1) * every
        ax.loglog(steps, np.clip(tv, 1e-5, 1), label=NICE[c], color=COL[c])
    ax.set_xlabel("MH steps"); ax.set_ylabel("TV(MH empirical, π)")
    ax.set_title("Metropolis–Hastings converges to π\n(MCMC reproduces the analytic stationary law)")
    ax.legend(fontsize=8)

    # (b) real vs synthetic mode distribution (stripped)
    ax = axes[0, 1]
    a = load_arrays("stripped")
    if a is not None:
        pr = a["pmode_real"]; ps = a["pmode_synthetic"]
        order = np.argsort(pr)[::-1]
        xx = np.arange(pr.size)
        ax.bar(xx - 0.2, pr[order], 0.4, label="real", color=COL["stripped"])
        ax.bar(xx + 0.2, ps[order], 0.4, label="synthetic (MCMC)", color="#9ecae1")
        ax.set_xlabel("mode (rank)"); ax.set_ylabel("frequency")
        ax.set_title("Generative check: mode usage\nMarkov sim matches the marginal mode mix")
        ax.legend(fontsize=8)

    # (c) distinct-modes-per-conversation real vs synthetic (the adequacy gap)
    ax = axes[1, 0]
    a = load_arrays("stripped")
    if a is not None:
        dr = a["distinct_real"]; ds = a["distinct_synthetic"]
        mx = int(max(dr.max(), ds.max()))
        bins = np.arange(1, min(mx, 14) + 2) - 0.5
        ax.hist(dr, bins=bins, density=True, alpha=0.6,
                label=f"real (mean {dr.mean():.2f})", color=COL["stripped"])
        ax.hist(ds, bins=bins, density=True, alpha=0.6,
                label=f"synthetic (mean {ds.mean():.2f})", color="#9ecae1")
        ax.axvline(dr.mean(), color=COL["stripped"], ls="--", lw=1)
        ax.axvline(ds.mean(), color="#3182bd", ls="--", lw=1)
        ax.set_xlabel("# distinct modes per conversation")
        ax.set_ylabel("density")
        ax.set_title("Where the memoryless model fails\nreal dialogue is stickier (revisits fewer modes)")
        ax.legend(fontsize=8)

    # (d) surprise gap real vs synthetic, both corpora
    ax = axes[1, 1]
    labels, gr, gs = [], [], []
    for c in CORPORA:
        p = os.path.join(ROOT, "results", "data", f"mcmc_{c}.json")
        if not os.path.exists(p): continue
        d = json.load(open(p))["generative_check"]
        labels.append(NICE[c]); gr.append(d["surprise_gap_real"]); gs.append(d["surprise_gap_synthetic"])
    xx = np.arange(len(labels)); w = 0.38
    b1 = ax.bar(xx - w/2, gr, w, label="real", color="#2c7fb8")
    b2 = ax.bar(xx + w/2, gs, w, label="synthetic (MCMC)", color="#9ecae1")
    ax.bar_label(b1, fmt="%+.3f", fontsize=8); ax.bar_label(b2, fmt="%+.3f", fontsize=8)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylabel("human − AI surprise gap (nats)")
    ax.set_title("Headline asymmetry is inherited by the generative model\nhuman is the less-predictable side (gap > 0)")
    ax.legend(fontsize=8)
    fig.suptitle("Generative MCMC — sampling whole conversations from the role-coupled chain", y=1.0)
    save(fig, "05_mcmc_generative.png")


def main():
    print("rendering figures -> results/figures/")
    plot_separability()
    plot_h2()
    plot_h3()
    plot_markov()
    plot_mcmc()
    print("done.")


if __name__ == "__main__":
    main()
