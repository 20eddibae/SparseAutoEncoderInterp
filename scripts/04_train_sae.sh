#!/usr/bin/env bash
# Train the sparse autoencoder on the MLP activation dataset.
#
# Lightweight regime: 5e4 steps (vs original 5e5). L1 coefficient and lr
# unchanged. Expect 5-10% dead neurons (vs ~5% in original).
#
# GPU: 1x A100 ~ 4h, 1x V100 ~ 8h.
# Output: $SDL_REPO/autoencoder/out/openwebtext/<timestamp-autoencoder-openwebtext>/ckpt.pt

set -euo pipefail

SDL_REPO="${SDL_REPO:-/Users/eddiebae/CS/sparse-dictionary-learning}"
MAX_STEPS="${MAX_STEPS:-50000}"

cd "$SDL_REPO/autoencoder"
python train.py \
    --dataset=openwebtext \
    --gpt_ckpt_dir=out \
    --n_features=4096 \
    --batch_size=8192 \
    --learning_rate=3e-4 \
    --l1_coeff=3e-3 \
    --resampling_interval=12500 \
    --num_resamples=3 \
    --eval_interval=5000 \
    --save_interval=10000 \
    --max_steps="$MAX_STEPS" \
    --device=cuda \
    --wandb_log=True
