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

# Resume from checkpoint if one exists, so Slurm preemption / requeue / timeout
# on the (slow) MIG GPU slice doesn't throw away progress. nanoGPT writes
# out/ckpt.pt during eval; always_save_checkpoint=True guarantees a recent
# checkpoint regardless of val-loss improvement.
CKPT_DIR="${GPT_CKPT_DIR:-out}"
if [[ -f "$CKPT_DIR/ckpt.pt" ]]; then
    INIT_FROM=resume
    echo "[train] found $CKPT_DIR/ckpt.pt -> resuming"
else
    INIT_FROM=scratch
    echo "[train] no checkpoint -> training from scratch"
fi

ARGS=(
    config/train_gpt2.py
    --wandb_project=scf-transformer
    --n_layer=1 --n_embd=128 --n_head=4
    --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS"
    --init_from="$INIT_FROM"
    --always_save_checkpoint=True
)

if [[ "$N_GPUS" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${ARGS[@]}"
else
    python train.py "${ARGS[@]}"
fi
