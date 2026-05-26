# Results — figures and data

Everything here is rendered from our own 1-layer GPT (6.64M params, OpenWebText) →
SAE (4096 features / 8× expansion, the Bricken et al. 2023 A/1 config) applied to
real human/AI conversations. CPU-only, from the per-turn feature artifacts in
`../artifacts/`. Regenerate with:

```bash
# JSON for the entropy / concentration hypotheses
python scripts/run_hypothesis.py --h 2 --features artifacts/features_stripped.npz --out results/data/h2_stripped.json
python scripts/run_hypothesis.py --h 3 --features artifacts/features_stripped.npz --out results/data/h3_stripped.json
#   (repeat with features_wildchat.npz)

# Markov chain + generative MCMC (writes results/data/mcmc_*.json and results/mcmc/arrays_*.npz)
python scripts/mcmc_conversations.py --features artifacts/features_stripped.npz --k 32
python scripts/mcmc_conversations.py --features artifacts/features_wildchat.npz --k 32

# all figures
python scripts/make_plots.py
```

## Figures (`figures/`)

| File | What it shows |
|---|---|
| `01_separability.png` | Role (human vs AI) separability of SAE turn-features: AUC ~0.92 linear / ~0.98 nonlinear, robust ShareGPT-stripped ≈ WildChat. The HTML-markup confound has been stripped (see RESEARCH.md). |
| `02_h3_concentration.png` | L0 (active features/turn): empirical vs Poisson null (var/mean ≈ 9.9 → super-Poissonian, features co-fire), concentration bounds, Poisson rejection, CLT for block means. |
| `03_h2_entropy.png` | Role entropy ≈ equal (4.73 vs 4.78 nats); AI uses a broader feature vocabulary; small asymmetric KL. Separability is about *which* features fire, not how many. |
| `04_markov_chain.png` | Conversation-mode chain (K=32): transition matrix, stationary π (spread, not collapsed), mixing (ergodic, ~4–10 turns), per-mode predictability spectrum. |
| `05_mcmc_generative.png` | **Generative MCMC.** Metropolis–Hastings converges to π; synthetic conversations from the role-coupled chain match the marginal mode mix and inherit the human>AI surprise gap, but visit *more* distinct modes than real dialogue (the memoryless model's failure mode). |

## Data (`data/`, `mcmc/`)

- `data/h{2,3}_{stripped,wildchat}.json` — full numeric output of `run_hypothesis.py`.
- `data/mcmc_{stripped,wildchat}.json` — role-coupled chain, MH, and generative-check summary.
- `mcmc/arrays_{stripped,wildchat}.npz` — transition matrix, π, role kernels, MH trace, dwell/diversity arrays used by `make_plots.py`.

## Headline numbers

| Quantity | ShareGPT (stripped) | WildChat |
|---|---:|---:|
| role separability (linear / nonlinear AUC) | 0.918 / 0.978 | 0.921 / 0.981 |
| L0 var/mean (Fano; Poisson = 1) | 9.9 | — |
| role marginal entropy (human / ai, nats) | 4.73 / 4.78 | 4.75 / 4.80 |
| pooled CK residual | 1.72 | 1.62 |
| **role-coupled CK residual** | **1.17** | **1.23** |
| mixing time (TV ≤ 0.25) | 10 turns | 4 turns |
| **human − AI surprise gap (nats)** | **+0.496** | **+0.675** |
| MH TV(empirical, π) after 200k steps | 0.0068 | 0.0049 |
| synthetic surprise gap (generative check) | +0.492 | +0.668 |
| distinct modes/conv — real vs synthetic | 3.95 vs 5.18 | 3.18 vs 3.85 |
