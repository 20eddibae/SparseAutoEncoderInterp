#!/usr/bin/env bash
# Train the sparse autoencoder on the MLP activation dataset.
#
# Lightweight regime. Training length is DATA-DRIVEN: train.py sets
# num_steps = total_activations // batch_size (one epoch); it does NOT read
# max_steps. With 2e8 activations from step 3 that is ~24k steps.
# Resampling fires 4x at steps ~2k/4k/6k/8k (interval=2000), then ~16k steps to
# re-converge revived neurons -> keeps dead neurons low for interpretable features.
# L1 (3e-3), lr (3e-4), batch (8192), n_features (4096) match the original repo.
# Output: $SDL_REPO/autoencoder/out/openwebtext/<timestamp-autoencoder-openwebtext>/ckpt.pt

set -euo pipefail

SDL_REPO="${SDL_REPO:-/Users/eddiebae/CS/sparse-dictionary-learning}"

cd "$SDL_REPO/autoencoder"
# NOTE: train.py has no --max_steps; length = total_activations // batch_size.
python train.py \
    --dataset=openwebtext \
    --gpt_ckpt_dir=out \
    --n_features=4096 \
    --batch_size=8192 \
    --learning_rate=3e-4 \
    --l1_coeff=3e-3 \
    --resampling_interval=2000 \
    --num_resamples=4 \
    --eval_interval=2000 \
    --save_interval=5000 \
    --device=cuda \
    --wandb_log=True
