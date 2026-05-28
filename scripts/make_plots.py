#!/usr/bin/env python3
"""
Render every result to figures under `results/figures/`. One figure per
experiment, all on our 1-layer GPT (6.64M params, OpenWebText) + 4096-feature
SAE applied to real human/AI conversations.

Data sources:
  - results/data/exp1_{corpus}.json   Experiment 1: is L0 Poisson?
  - results/data/exp2_{corpus}.json   Experiment 2: WLLN + CLT for the L0 mean
  - results/data/exp3_{corpus}.json   Experiment 3: role distributions (entropy/KL)
  - results/data/mcmc_{corpus}.json   Experiment 4 + MCMC summary
  - results/mcmc/arrays_{corpus}.npz  Experiment 4 + MCMC arrays
  - SEPARABILITY (below)              classifier AUCs recorded in RESEARCH.md

Usage:  python scripts/make_plots.py
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
CORPORA = ["stripped", "wildchat"]
NICE = {"stripped": "ShareGPT (stripped)", "wildchat": "WildChat"}
COL = {"stripped": "#2c7fb8", "wildchat": "#d95f0e"}

# Recorded role-separability AUCs (RESEARCH.md, jobs 8687826 / 8687859). The
# synthetic-corpus control (role AUC 0.56) is the null: same generator for both
# roles -> role unlearnable by construction.
SEPARABILITY = {
    "A linear":      {"stripped": 0.918, "wildchat": 0.921},
    "A nonlinear":   {"stripped": 0.978, "wildchat": 0.981},
    "B generated":   {"stripped": 0.931, "wildchat": 0.934},
    "D group-aware": {"stripped": 0.914, "wildchat": 0.919},
}
SYNTHETIC_CONTROL = 0.56


def load_json(corpus, name):
    p = os.path.join(ROOT, "results", "data", f"{name}_{corpus}.json")
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
    ax.axhline(SYNTHETIC_CONTROL, ls=":", c="firebrick", lw=1.3,
               label=f"synthetic-corpus null ({SYNTHETIC_CONTROL:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(probes, rotation=15, ha="right")
    ax.set_ylabel("ROC AUC"); ax.set_ylim(0.45, 1.02)
    ax.set_title("Role separability of SAE turn-features (HTML-stripped)\n"
                 "human vs AI recoverable at AUC ~0.92 linear / ~0.98 nonlinear, "
                 "robust across two corpora; null = 0.56")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "01_separability.png")


def plot_exp1_poisson():
    d = load_json("stripped", "exp1")
    if d is None:
        print("  [skip] exp1_stripped.json missing"); return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # (a) L0 PMF: empirical vs Poisson null
    k = np.array(d["k_grid"]); emp = np.array(d["empirical_pmf"]); poi = np.array(d["poisson_pmf"])
    ax = axes[0]
    ax.plot(k, emp, label="empirical L0", color=COL["stripped"], lw=1.8)
    ax.plot(k, poi, label=f"Poisson(λ={d['lambda_hat']:.1f}) null", color="grey", ls="--")
    ax.set_xlim(0, min(200, k.max())); ax.set_xlabel("active features / turn (L0)"); ax.set_ylabel("probability")
    ax.set_title(f"L0 is super-Poissonian\nFano var/mean ≈ {d['fano_factor']:.1f} "
                 f"(Poisson = 1) → features co-fire")
    ax.legend(fontsize=8)

    # (b) Markov + Chebyshev tail bounds vs empirical
    ax = axes[1]
    t = np.array(d["tail_grid"])
    ax.semilogy(t, np.clip(d["empirical_tail"], 1e-6, 1), label="empirical", color="k", lw=2)
    ax.semilogy(t, np.clip(d["markov_tail"], 1e-6, 1), label="Markov", ls="--")
    ax.semilogy(t, np.clip(d["chebyshev_tail"], 1e-6, 1), label="Chebyshev", ls="--")
    ax.set_xlabel("t  (deviation above mean)"); ax.set_ylabel("P(L0 − μ ≥ t)")
    ax.set_title("Markov & Chebyshev tail bounds vs empirical\n(both valid; Chebyshev tighter for large t)")
    ax.legend(fontsize=8)

    # (c) summary text
    ax = axes[2]; ax.axis("off")
    lines = [
        "Experiment 1 — Is L0 Poisson?  (ShareGPT stripped)",
        "",
        f"  mean μ            = {d['mu']:.2f}",
        f"  variance          = {d['var']:.1f}",
        f"  Fano  var/mean    = {d['fano_factor']:.2f}   (Poisson = 1)",
        "",
        "Poisson-as-limit-of-Binomial null",
        f"  λ̂ = Σ p_i        = {d['lambda_hat']:.1f}",
        f"  Σpᵢ² approx error = {d['poisson_approx_error']:.1f}   (>>1 → Poisson void)",
        f"  χ² p-value        = {d['poisson_chi2_pvalue']:.1e}   → rejected",
        "",
        "Tail-sum identity  E[N]=Σ_k P(N≥k)",
        f"  {d['expectation_tail_sum']:.6f} vs {d['expectation_direct']:.6f} ✓",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    fig.suptitle("Experiment 1 — L0 is a well-behaved but strongly over-dispersed RV "
                 "(tail-sum · Poisson limit · Markov/Chebyshev)", y=1.04)
    save(fig, "02_exp1_poisson.png")


def plot_exp2_clt():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) WLLN: running sample mean converges
    ax = axes[0]
    for c in CORPORA:
        d = load_json(c, "exp2")
        if d is None: continue
        ax.plot(d["running_mean_n"], d["running_mean"], label=NICE[c], color=COL[c], lw=1.6)
        ax.axhline(d["mu"], ls=":", c=COL[c], lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("turns averaged (N)"); ax.set_ylabel("running mean of L0")
    ax.set_title("WLLN: the L0 sample mean concentrates on μ\n(running mean settles as N grows)")
    ax.legend(fontsize=8)

    # (b) CLT: standardised block means vs N(0,1)
    ax = axes[1]
    d = load_json("stripped", "exp2")
    if d is not None:
        z = np.array(d["standardised_block_means"])
        ax.hist(z, bins=15, density=True, alpha=0.65, color=COL["stripped"],
                label=f"block means (B={d['n_blocks']}×{d['block_size']})")
        xs = np.linspace(-3.5, 3.5, 200)
        ax.plot(xs, norm.pdf(xs), "k--", lw=1.6, label="N(0,1)")
        ax.set_xlabel("standardised block mean"); ax.set_ylabel("density")
        ax.set_title(f"CLT: block means are Gaussian\nKS p = {d['ks_pvalue']:.2f} "
                     f"(fails to reject normality)")
        ax.legend(fontsize=8)
    fig.suptitle("Experiment 2 — the L0 sample mean obeys WLLN + CLT "
                 "even though per-turn L0 is not Gaussian", y=1.03)
    save(fig, "03_exp2_clt.png")


def plot_exp3_roles():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    roles = ["human", "ai"]; x = np.arange(len(roles)); w = 0.38

    # (a) marginal entropy with delta-method error bars
    ax = axes[0]
    for i, c in enumerate(CORPORA):
        d = load_json(c, "exp3")
        if d is None: continue
        H = [d["entropy_human"]["H_hat"], d["entropy_ai"]["H_hat"]]
        err = [d["entropy_human"]["delta_se"], d["entropy_ai"]["delta_se"]]
        b = ax.bar(x + (i - 0.5) * w, H, w, yerr=err, capsize=4, label=NICE[c], color=COL[c])
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=8)
    ax.set_xticks(x); ax.set_xticklabels(roles)
    ax.set_ylabel("plug-in entropy (nats)"); ax.set_ylim(4.4, 5.0)
    ax.set_title("Marginal feature entropy ≈ equal by role\n(±delta-method 1σ) → not a 'mode-seeking' gap")
    ax.legend(fontsize=8)

    # (b) active feature vocabulary
    ax = axes[1]
    for i, c in enumerate(CORPORA):
        d = load_json(c, "exp3")
        if d is None: continue
        V = [d["entropy_human"]["support_size"], d["entropy_ai"]["support_size"]]
        b = ax.bar(x + (i - 0.5) * w, V, w, label=NICE[c], color=COL[c])
        ax.bar_label(b, fmt="%d", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(roles)
    ax.set_ylabel("# distinct active features")
    ax.set_title("AI uses a broader feature vocabulary")
    ax.legend(fontsize=8)

    # (c) KL asymmetry (Jensen => KL >= 0, so positive is meaningful)
    ax = axes[2]
    labels = ["KL(human‖ai)", "KL(ai‖human)"]; xx = np.arange(len(labels))
    for i, c in enumerate(CORPORA):
        d = load_json(c, "exp3")
        if d is None: continue
        v = [d["kl_human_to_ai"], d["kl_ai_to_human"]]
        b = ax.bar(xx + (i - 0.5) * w, v, w, label=NICE[c], color=COL[c])
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("KL divergence (nats)")
    ax.set_title("KL ≥ 0 by Jensen → positive, asymmetric\nrole divergence is genuine (small)")
    ax.legend(fontsize=8)
    fig.suptitle("Experiment 3 — roles differ in WHICH features fire, not in marginal entropy "
                 "(entropy · delta method · KL/Jensen)", y=1.04)
    save(fig, "04_exp3_roles.png")


def plot_exp4_markov():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) transition heatmap (stripped)
    a = load_arrays("stripped")
    if a is not None:
        P = a["P"]; pi = a["pi"]
        order = np.argsort(pi)[::-1]
        ax = axes[0, 0]
        im = ax.imshow(P[np.ix_(order, order)], cmap="viridis", aspect="auto")
        ax.set_title("Turn→turn transition matrix P (ShareGPT)\nargmax states, sorted by stationary mass")
        ax.set_xlabel("next state"); ax.set_ylabel("current state")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # (b) stationary distribution
    ax = axes[0, 1]
    for c in CORPORA:
        a = load_arrays(c)
        if a is None: continue
        pi = np.sort(a["pi"])[::-1]
        ax.plot(np.arange(pi.size), pi, marker="o", ms=3, label=NICE[c], color=COL[c])
        ax.axhline(1 / pi.size, ls=":", c=COL[c], lw=0.8)
    ax.set_xlabel("state (rank)"); ax.set_ylabel("stationary π")
    ax.set_title("Stationary distribution: spread over states\n(high entropy, a few mild attractors)")
    ax.legend(fontsize=8)

    # (c) mixing: TV(start→π) decays geometrically
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
    ax.set_title("Conversations forget their opening state\n(ergodic; TV→π decays geometrically)")
    ax.legend(fontsize=8)

    # (d) per-state conditional entropy (predictability spectrum)
    ax = axes[1, 1]
    for c in CORPORA:
        a = load_arrays(c)
        if a is None: continue
        H = np.sort(a["cond_entropy_rows"])
        ax.plot(np.arange(H.size), H, marker="o", ms=3, label=NICE[c], color=COL[c])
        ax.axhline(np.log(H.size), ls=":", c=COL[c], lw=0.8)
    ax.set_xlabel("state (sorted)"); ax.set_ylabel("H(next | state) (nats)")
    ax.set_title("Predictability spectrum across states\nsome near-deterministic, some wide-open")
    ax.legend(fontsize=8)
    fig.suptitle("Experiment 4 — conversation Markov chain on argmax states: "
                 "ergodic, mixing, interpretable (Chapman–Kolmogorov · stationary π)", y=1.0)
    save(fig, "05_exp4_markov.png")


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

    # (b) real vs synthetic state distribution (stripped)
    ax = axes[0, 1]
    a = load_arrays("stripped")
    if a is not None:
        pr = a["pmode_real"]; ps = a["pmode_synthetic"]
        order = np.argsort(pr)[::-1]
        xx = np.arange(pr.size)
        ax.bar(xx - 0.2, pr[order], 0.4, label="real", color=COL["stripped"])
        ax.bar(xx + 0.2, ps[order], 0.4, label="synthetic (MCMC)", color="#9ecae1")
        ax.set_xlabel("state (rank)"); ax.set_ylabel("frequency")
        ax.set_title("Generative check: state usage\nMarkov sim matches the marginal mix")
        ax.legend(fontsize=8)

    # (c) distinct-states-per-conversation real vs synthetic (the adequacy gap)
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
        ax.set_xlabel("# distinct states per conversation")
        ax.set_ylabel("density")
        ax.set_title("Where the memoryless model fails\nreal dialogue is stickier (revisits fewer states)")
        ax.legend(fontsize=8)

    # (d) surprise gap real vs synthetic, both corpora
    ax = axes[1, 1]
    labels, gr, gs = [], [], []
    for c in CORPORA:
        d = load_json(c, "mcmc")
        if d is None: continue
        g = d["generative_check"]
        labels.append(NICE[c]); gr.append(g["surprise_gap_real"]); gs.append(g["surprise_gap_synthetic"])
    xx = np.arange(len(labels)); w = 0.38
    b1 = ax.bar(xx - w / 2, gr, w, label="real", color="#2c7fb8")
    b2 = ax.bar(xx + w / 2, gs, w, label="synthetic (MCMC)", color="#9ecae1")
    ax.bar_label(b1, fmt="%+.3f", fontsize=8); ax.bar_label(b2, fmt="%+.3f", fontsize=8)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylabel("human − AI surprise gap (nats)")
    ax.set_title("Headline asymmetry is inherited by the generative model\nhuman is the less-predictable side (gap > 0)")
    ax.legend(fontsize=8)
    fig.suptitle("Generative MCMC — sampling whole conversations from the role-coupled chain "
                 "(Metropolis–Hastings · detailed balance)", y=1.0)
    save(fig, "06_mcmc_generative.png")


def main():
    print("rendering figures -> results/figures/")
    plot_separability()
    plot_exp1_poisson()
    plot_exp2_clt()
    plot_exp3_roles()
    plot_exp4_markov()
    plot_mcmc()
    print("done.")


if __name__ == "__main__":
    main()
