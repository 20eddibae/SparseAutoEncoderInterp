# Project plan & proof inventory

## Core framing
A human-AI conversation is a stochastic process on the SAE feature space, not on tokens. Each turn induces a sparse vector in `R^F` (F = 4096). The trajectory through feature space is the object of study. Four hypotheses, all tested with course-issued tools.

## Probability space
- Sample space `Ω`: sequences of turns. A *turn* is a pair (role ∈ {human, ai}, text).
- Random elements: `T_t = (role_t, text_t)`; `X_t = SAE(text_t)` ∈ `R^F`; `S_t = supp(X_t)` ∈ `{0,1}^F`.
- σ-algebra: Borel on `R^F` for `X_t`; power set on the coarsened states (see below).
- Probability measure: estimated empirically from the corpus (Monte Carlo over prompt distribution).

## Course technique → where it lives
| Technique | Module | Hypothesis |
|---|---|---|
| Bayes' theorem | `probability/bayes.py` | H2 |
| Tail-sum identity | `probability/tail_sum.py` | H3 |
| Linearity of expectation | implicit in `bayes.py` per-feature decomposition | H2, H3 |
| Poisson limit (Le Cam) + thinning | `probability/poisson.py` | H3 |
| Order statistics | `probability/order_stats.py` | H3, monosemanticity sanity check |
| Convolution (turn sums) | `coarsen.py` (n_active per turn) + `concentration.py` | H1, H3 |
| MGFs | `probability/mgf.py` | H3 |
| Conditional expectation, tower | `probability/markov.py` (transition kernel) | H1 |
| Markov / Chebyshev | `probability/concentration.py` | H3 |
| WLLN, Monte Carlo | `probability/clt.py` (`running_mean`) | All |
| CLT | `probability/clt.py` (`clt_diagnostic`) | H3 |
| Jensen's inequality | `probability/entropy.py` (`log_support_bound`) | H2 |
| Markov chains, CK, stationary, mixing | `probability/markov.py` | H1, H4 |

## The four hypotheses

**H1 — Markovianity (centerpiece).**
Conversation trajectories on the coarsened state space are first-order Markov.
Test: estimate `P̂` from 1-step transitions; compare `P̂²` to the empirical 2-step matrix `P̂2_emp`. Report Frobenius norm of the residual, max-element residual, and chi-squared statistic over the (state, state) contingency table.
Proof to include in the write-up: re-derive CK for our coarsened kernel (state = top-k SAE feature tuple). The proof is standard but should be done explicitly because the coarsening can break Markovianity even when the underlying chain is Markov — the test detects exactly that.

**H2 — Human vs AI entropy signature.**
Human prompts have higher feature-entropy than AI completions. Estimate `H(p̂_human)` and `H(p̂_ai)` via plug-in; attach Chebyshev half-widths.
Lemma to prove: variance of the plug-in entropy estimator is bounded by `(log N)^2 / N` (Antos-Kontoyiannis; the simple proof uses bounded differences, optional).

**H3 — Sub-Gaussian concentration.**
`N_active(turn)` is sub-Gaussian with proxy `σ_g²`. Fit `σ_g²` from the empirical MGF; compare Markov, Chebyshev, and sub-Gaussian bounds to the empirical tail.
Side observations: Poisson-thinning null is rejected; CLT diagnostic on block means.

**H4 — Topic-conditional stationarity.**
Coarsened chain has near-absorbing structure: conversations seeded with different topics converge to different stationary distributions. Per-topic stationary `π_topic` and pairwise KL.

## Repo map (analysis-only)
Code at `src/scf/` does not duplicate `sparse-dictionary-learning`. The bridge (`sdl_bridge.py`) imports `HookedGPT`, `GPTConfig`, and `AutoEncoder` from the sibling clone. Update `configs/default.yaml: sdl_repo` if you move the clone.

## Compute split
- **HPC, GPU:** transformer training (~6h A100), SAE training (~4h A100), MLP-activation generation (~2h A100). See `docs/HPC_PLAYBOOK.md`.
- **HPC, CPU:** OpenWebText tokenisation (~1h, ~30GB disk).
- **Laptop:** corpus feature extraction (~1h CPU, can use CUDA), all four hypothesis tests (minutes), plotting.

## Write-up scaffolding
For each hypothesis, the write-up should contain: (i) the empirical numbers, (ii) the small theorem that justifies the estimator, (iii) the proof, (iv) one figure. Total length target: ~12 pages.
