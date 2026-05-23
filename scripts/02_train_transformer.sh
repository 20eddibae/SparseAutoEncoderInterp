#!/usr/bin/env bash
# Train the 1-layer transformer (d_model=128, n_head=4) on OpenWebText.
#
# Lightweight regime for the probability project: 40k iters (~1/5 of original
# 200k). Loss won't fully converge but features are stable enough for the
# statistical tests.
#
# GPU: 1x A100 ~ 6h, 1x V100 ~ 12h, 1x consumer 24GB ~ 18h.
# Output: $SDL_REPO/transformer/out/ckpt.pt

set -euo pipefail

SDL_REPO="${SDL_REPO:-/Users/eddiebae/CS/sparse-dictionary-learning}"
N_GPUS="${N_GPUS:-1}"
MAX_ITERS="${MAX_ITERS:-40000}"

cd "$SDL_REPO/transformer"

if [[ "$N_GPUS" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py \
        config/train_gpt2.py \
        --wandb_project=scf-transformer \
        --n_layer=1 --n_embd=128 --n_head=4 \
        --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS"
else
    python train.py config/train_gpt2.py \
        --wandb_project=scf-transformer \
        --n_layer=1 --n_embd=128 --n_head=4 \
        --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS"
fi
