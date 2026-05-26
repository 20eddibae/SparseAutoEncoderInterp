# Results — figures and data

Everything here is rendered from our own 1-layer GPT (6.64M params, OpenWebText) →
SAE (4096 features / 8× expansion, the Bricken et al. 2023 A/1 config) applied to
real human/AI conversations. CPU-only, from the per-turn feature artifacts in
`../artifacts/`. Every experiment uses only the course probability vocabulary
(`../CLAUDE.md`); the Markov-chain state is the **dominant (argmax) SAE feature per
turn** — an order statistic, not a clustering output. Regenerate with:

```bash
sbatch slurm/09_experiments.sbatch        # the whole pipeline (both corpora + figures + tests)
# or by hand, per corpus:
python scripts/run_hypothesis.py --exp 1 --features artifacts/features_stripped.npz --out results/data/exp1_stripped.json
python scripts/run_hypothesis.py --exp 2 --features artifacts/features_stripped.npz --out results/data/exp2_stripped.json
python scripts/run_hypothesis.py --exp 3 --features artifacts/features_stripped.npz --out results/data/exp3_stripped.json
python scripts/mcmc_conversations.py --features artifacts/features_stripped.npz --k 32   # Exp 4 + MCMC
python scripts/make_plots.py
```

## Figures (`figures/`)

| File | What it shows |
|---|---|
| `01_separability.png` | **Control.** Role (human vs AI) separability of SAE turn-features: AUC ~0.92 linear / ~0.98 nonlinear, robust ShareGPT-stripped ≈ WildChat, vs the synthetic-corpus null (0.56). |
| `02_exp1_poisson.png` | **Experiment 1.** L0 (active features/turn): empirical vs Poisson-as-limit null (Fano var/mean ≈ 10 → super-Poissonian, features co-fire), Markov & Chebyshev tail bounds, tail-sum check, χ² rejection. |
| `03_exp2_clt.png` | **Experiment 2.** WLLN: the L0 running mean concentrates on μ. CLT: standardised block means vs N(0,1) + KS test. |
| `04_exp3_roles.png` | **Experiment 3.** Role entropy with delta-method error bars (AI slightly *higher*, resolved); AI uses a broader feature vocabulary; KL ≥ 0 by Jensen, small but positive. Separability is about *which* features fire. |
| `05_exp4_markov.png` | **Experiment 4.** Conversation chain on argmax states (M=32): transition matrix, stationary π (spread, not collapsed), geometric TV decay / mixing (ergodic, 2–4 turns), per-state predictability spectrum. |
| `06_mcmc_generative.png` | **Kept (H5/H6).** Metropolis–Hastings converges to π; synthetic conversations from the role-coupled chain match the marginal state mix and inherit the human>AI surprise gap, but visit *more* distinct states than real dialogue (the memoryless model's failure mode). |

## Data (`data/`, `mcmc/`)

- `data/exp{1,2,3}_{stripped,wildchat}.json` — full numeric output of `run_hypothesis.py`.
- `data/mcmc_{stripped,wildchat}.json` — Experiment 4 + role-coupled chain + MH + generative-check summary.
- `mcmc/arrays_{stripped,wildchat}.npz` — transition matrix, π, role kernels, MH trace, dwell/diversity arrays used by `make_plots.py`.

## Headline numbers

| Quantity | ShareGPT (stripped) | WildChat |
|---|---:|---:|
| role separability (linear / nonlinear AUC) | 0.918 / 0.978 | 0.921 / 0.981 |
| **Exp 1** L0 Fano var/mean (Poisson = 1) | 9.94 | 10.29 |
| **Exp 2** CLT KS p-value (block means) | 0.65 | 0.024 |
| **Exp 3** role entropy human / ai (nats) | 4.734 / 4.784 | 4.754 / 4.795 |
| **Exp 3** Δ entropy (± delta-method 1σ) | −0.049 ± 0.0004 | −0.041 ± 0.0005 |
| **Exp 4** pooled CK residual | 1.312 | 1.435 |
| **Exp 4** mixing time (TV ≤ 0.25) | 4 turns | 2 turns |
| **Exp 4** entropy rate (% of ln 32) | 2.459 (71%) | 2.604 (75%) |
| **Exp 4 / headline** human − AI surprise gap (nats) | **+0.532** | **+0.763** |
| H5/H6 role-coupled CK residual | 0.971 | 1.154 |
| H5/H6 MH TV(empirical, π) after 200k steps | 0.012 | 0.012 |
| H5/H6 synthetic surprise gap (generative check) | +0.518 | +0.751 |
| H5/H6 distinct states/conv — real vs synthetic | 3.67 vs 4.55 | 2.98 vs 3.39 |
