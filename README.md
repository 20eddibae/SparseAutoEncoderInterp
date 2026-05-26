# Stochastic Conversation Features (SCF)

Honors Probability final project, Dartmouth Spring 2026.

We treat a human-AI conversation as a stochastic process on a sparse autoencoder (SAE) feature space, not on tokens. Each turn induces a sparse vector in `R^F` (F = 4096); a conversation is a trajectory through feature space. Every experiment reduces to a probability question on these trajectories, and **uses only the course probability vocabulary** (see `CLAUDE.md` for the full topic list — each experiment maps to ≤2 topics, no outside tools).

This repo is the *analysis* layer. It does not duplicate the SAE training code — it imports from a sibling clone of [`sparse-dictionary-learning`](https://github.com/shehper/sparse-dictionary-learning) via `scf.sdl_bridge`.

## Four experiments + a kept generative result

- **Experiment 1 — Is L0 Poisson?** `L0` = active features per turn. Tail-sum identity, Poisson-as-limit-of-Binomial fit + Fano factor, Markov & Chebyshev tail bounds. *(`scripts/run_hypothesis.py --exp 1`)*
- **Experiment 2 — Does the sample mean concentrate?** Block-means of `L0`: WLLN + CLT, KS test against a Gaussian. *(`--exp 2`)*
- **Experiment 3 — Are the roles different distributions?** Per-role marginal feature distribution: entropy with a delta-method error bar, KL both ways, Jensen as the reason `KL ≥ 0`. *(`--exp 3`)*
- **Experiment 4 — Conversation-mode dynamics.** A Markov chain whose **state is the dominant (argmax) SAE feature per turn** — an order statistic, *not* a clustering output (no KMeans, no `K`). Chapman–Kolmogorov, stationary `π` + geometric TV decay / mixing, conditional entropy split by role. *(`scripts/mcmc_conversations.py`)*
- **Kept: generative MCMC.** Role-coupled kernels, Metropolis–Hastings to `π`, and forward-sampled synthetic conversations. *(`scripts/mcmc_conversations.py`)*

Headline: at the dominant-feature level the conversation is a moderately-high-randomness ergodic chain, and the randomness is **asymmetric — the human is the less-predictable side** (human surprise exceeds AI self-surprise by ~+0.5/+0.8 nats on ShareGPT/WildChat). See `RESULTS.md`.

## Repo layout
```
configs/             yaml configs (paths to SDL repo, checkpoint dirs)
src/scf/
  sdl_bridge.py      load HookedGPT + AutoEncoder from sparse-dictionary-learning
  feature_extractor  turn text -> (binary, magnitude) feature vectors
  corpus/            sharegpt, synthetic self-play
  coarsen.py         argmax (dominant-feature) state space — an order statistic
  probability/       course toolbox: entropy, concentration, poisson,
                     clt, order_stats, bayes, markov  (mgf kept for reference)
  hypotheses/        exp1_poisson, exp2_clt, exp3_roles drivers
scripts/             run_hypothesis.py (exp 1-3), mcmc_conversations.py (exp 4 + MCMC),
                     markov_chain.py, make_plots.py, HPC shell scripts
tests/               unit tests for analysis math (run on CPU)
docs/                PROJECT.md and HPC_PLAYBOOK.md
```

## Workflow
1. **HPC (GPU)** — train the 1-layer transformer + SAE inside the sibling `sparse-dictionary-learning` clone. On Dartmouth Discovery: `bash slurm/submit_all.sh` chains 5 dependent Slurm jobs (`slurm/01_prepare_owt.sbatch` … `slurm/05_extract_features.sbatch`). See `slurm/SETUP.md` for the one-time conda env and `docs/HPC_PLAYBOOK.md` for everything else.
2. **Output of step 1**: `artifacts/features_*.npz` with per-turn SAE feature vectors. Move this to your laptop, or stay on the cluster.
3. **Laptop or login node (CPU)** — regenerate every result and figure with `sbatch slurm/09_experiments.sbatch`, or by hand: `python scripts/run_hypothesis.py --exp {1,2,3} --features artifacts/features_stripped.npz`, `python scripts/mcmc_conversations.py --features artifacts/features_stripped.npz --k 32`, `python scripts/make_plots.py`. Pure numpy/scipy, no GPU.

## Install
```
pip install -e .
```

## On Dartmouth Discovery — one-line bootstrap
```
git clone git@github.com:20eddibae/SparseAutoEncoderInterp.git stochastic-conversation-features
cd stochastic-conversation-features
bash bootstrap.sh all     # unpacks SDL, builds conda env via srun, submits 5-job pipeline
```
Subsequent runs: `bash bootstrap.sh submit`. See `docs/HPC_PLAYBOOK.md` for failure modes, `slurm/SETUP.md` for the manual breakdown, and `docs/PROJECT.md` for the proof inventory.
