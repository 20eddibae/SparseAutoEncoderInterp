# Project plan & proof inventory

## Core framing
A human-AI conversation is a stochastic process on the SAE feature space, not on tokens. Each turn induces a sparse vector in `R^F` (F = 4096). The trajectory through feature space is the object of study. **Four experiments + one kept generative result, all using only the course probability vocabulary** (full topic list in `../CLAUDE.md`; each experiment maps to ≤2 topics, no outside tools).

## Probability space
- Sample space `Ω`: sequences of turns. A *turn* is a pair (role ∈ {human, ai}, text).
- Random elements: `T_t = (role_t, text_t)`; `X_t = SAE(text_t)` ∈ `R^F`; `S_t = supp(X_t)` ∈ `{0,1}^F`.
- σ-algebra: Borel on `R^F` for `X_t`; power set on the argmax state space (see below).
- Probability measure: estimated empirically from the corpus (Monte Carlo over prompt distribution).
- **State for the conversation chain:** `Z_t = argmax_j X_t[j]` over the informative features (always-on features excluded) — the dominant feature, an order statistic (items 18–19). Not a clustering output: no KMeans, no `K`.

## Course technique → where it lives
| Technique (course item) | Module | Experiment |
|---|---|---|
| Tail-sum identity (5/8) | `probability/tail_sum.py` | Exp 1 |
| Poisson-as-limit-of-Binomial + thinning, Le Cam (13/14) | `probability/poisson.py` | Exp 1 |
| Markov / Chebyshev inequalities (25/26) | `probability/concentration.py` | Exp 1 |
| WLLN (31) | `probability/clt.py` (`running_mean`) | Exp 2 |
| CLT (33) | `probability/clt.py` (`clt_diagnostic`) | Exp 2 |
| Entropy as `E[-log p]` + delta method (35) | `probability/entropy.py` (`entropy_with_delta_se`) | Exp 3 |
| KL + Jensen `KL ≥ 0` (28) | `probability/entropy.py` (`kl_divergence`) | Exp 3 |
| Bayes' theorem (3) — one supporting line | `probability/bayes.py` | Exp 3 |
| Order statistics: argmax state (18/19) | `coarsen.py` (`build_argmax_space`) | Exp 4 |
| Markov chain, Chapman–Kolmogorov, stationary π, geometric TV decay / mixing (36/37) | `probability/markov.py` | Exp 4 |
| MCMC / detailed balance / Metropolis–Hastings (38/40/41) | `scripts/mcmc_conversations.py` | H5/H6 (kept) |
| MGF uniqueness (23/24) — *reference only* (CLT proof) | `probability/mgf.py` | — |

## The four experiments

**Experiment 1 — Is L0 Poisson?**
`L0 = #features firing per turn` (a sum of 4096 Bernoullis). Verify the tail-sum identity `E[L0]=Σ_k P(L0≥k)`; fit the Poisson-as-limit-of-Binomial null and report the Fano factor var/mean (=1 iff Poisson); bound the upper tail with Markov and Chebyshev. Finding: super-Poissonian (Fano ≈ 10), Poisson rejected — features co-fire. *No MGF Gaussian fit, no CLT here.*

**Experiment 2 — Does the L0 sample mean concentrate?**
WLLN: the running sample mean of `L0` settles onto μ. CLT: standardised block means are ≈ N(0,1), checked by a KS test. The sample-mean machinery is valid even though per-turn `L0` is itself non-Gaussian.

**Experiment 3 — Are the roles different distributions?**
Per-role marginal feature distribution. Plug-in entropy `H = E[-log p]` with a **delta-method** standard error; KL both directions; **Jensen** gives `KL ≥ 0`, so a strictly positive KL is genuine evidence the distributions differ. Finding: entropies are within 1% but the delta-method bar resolves them (AI slightly *higher*, not lower); the discriminative signal is in *which* features fire (Bayes posterior ~1.0 for top features), reported as a single line.

**Experiment 4 — Conversation-mode dynamics (Markov chain only).**
State = dominant (argmax) feature per turn. Transition matrix = row-normalised counts (the MLE is a sample mean → WLLN). Chapman–Kolmogorov residual tests first-order Markovianity; stationary π is the left eigenvector; TV(start→π) decays geometrically (mixing time). Conditional entropy of the next state, split by role, gives the **human − AI surprise gap** (+0.53 / +0.76 nats). Proof to include: re-derive CK for the argmax kernel, and note the period-2 role alternation is why the pooled 1-step chain leaves a residual — which the kept role-coupled chain (H5/H6) models.

**Kept (H5/H6) — role-coupled chain + generative MCMC.**
Two role kernels `T_{h→a}`, `T_{a→h}`; their product lowers the CK residual (~0.3). A Metropolis–Hastings sampler (detailed balance) reproduces π to TV ≈ 0.012; forward simulation inherits the surprise gap but over-counts distinct states — real dialogue is stickier.

## Repo map (analysis-only)
Code at `src/scf/` does not duplicate `sparse-dictionary-learning`. The bridge (`sdl_bridge.py`) imports `HookedGPT`, `GPTConfig`, and `AutoEncoder` from the sibling clone. Update `configs/default.yaml: sdl_repo` if you move the clone.

## Compute split
- **HPC, GPU:** transformer training (~6h A100), SAE training (~4h A100), MLP-activation generation (~2h A100). See `docs/HPC_PLAYBOOK.md`.
- **HPC, CPU:** OpenWebText tokenisation (~1h, ~30GB disk).
- **Laptop / login node (CPU):** corpus feature extraction (~1h CPU, can use CUDA), all four experiments + MCMC (minutes; `sbatch slurm/09_experiments.sbatch`), plotting.

## Write-up scaffolding
For each experiment, the write-up should contain: (i) the empirical numbers, (ii) the small theorem that justifies the estimator, (iii) the proof, (iv) one figure. Every probabilistic claim must trace to a numbered course topic (`../CLAUDE.md`). Total length target: ~12 pages.
