"""Experiment drivers (CPU-only). Each module exposes `run(features_npz_path, cfg)`.

  exp1_poisson  -- Experiment 1: is L0 Poisson?
  exp2_clt      -- Experiment 2: does the L0 sample mean concentrate? (WLLN+CLT)
  exp3_roles    -- Experiment 3: are the two roles different distributions?

Experiment 4 (conversation-mode Markov chain) + generative MCMC live in
scripts/mcmc_conversations.py and scripts/markov_chain.py.
"""
