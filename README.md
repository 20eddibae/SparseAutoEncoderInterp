# Stochastic Conversation Features (SCF)

Honors Probability final project, Dartmouth Spring 2026.

We treat a human-AI conversation as a stochastic process on a sparse autoencoder (SAE) feature space, not on tokens. Each turn induces a sparse binary vector in `{0,1}^F`; a conversation is a trajectory through feature space. The four hypotheses we test all reduce to probability questions on these trajectories.

This repo is the *analysis* layer. It does not duplicate the SAE training code — it imports from a sibling clone of [`sparse-dictionary-learning`](https://github.com/shehper/sparse-dictionary-learning) via `scf.sdl_bridge`.

## Hypotheses
- **H1 Markovianity** — `P̂² ≈ P̂_{2-step}` via Chapman-Kolmogorov; chi-squared discrepancy.
- **H2 Human vs AI signature** — plug-in entropy + Chebyshev error bound.
- **H3 Concentration** — `N_active` per turn is sub-Gaussian; MGF fit; Markov vs Chebyshev vs empirical tail.
- **H4 Stationarity by topic** — stationary distribution conditioned on seed topic; absorbing-class structure.

## Repo layout
```
configs/             yaml configs (paths to SDL repo, checkpoint dirs)
src/scf/
  sdl_bridge.py      load HookedGPT + AutoEncoder from sparse-dictionary-learning
  feature_extractor  turn text -> (binary, magnitude) feature vectors
  corpus/            sharegpt, synthetic self-play
  coarsen.py         top-k state / k-means co-activation cluster
  probability/       course toolbox: entropy, concentration, mgf, poisson,
                     clt, order_stats, bayes, markov
  hypotheses/        h1..h4 drivers
scripts/             extract_features.py, run_hypothesis.py, HPC shell scripts
tests/               unit tests for analysis math (run on CPU)
docs/                PROJECT.md and HPC_PLAYBOOK.md
```

## Workflow
1. **HPC (GPU)** — train the 1-layer transformer + SAE inside the sibling `sparse-dictionary-learning` clone. On Dartmouth Discovery: `bash slurm/submit_all.sh` chains 5 dependent Slurm jobs (`slurm/01_prepare_owt.sbatch` … `slurm/05_extract_features.sbatch`). See `slurm/SETUP.md` for the one-time conda env and `docs/HPC_PLAYBOOK.md` for everything else.
2. **Output of step 1**: `artifacts/features.npz` with per-turn SAE feature vectors. Move this to your laptop, or stay on the cluster.
3. **Laptop or login node** — `python scripts/run_hypothesis.py --h {1,2,3,4} --features artifacts/features.npz`. Pure numpy/scipy, no GPU.

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
