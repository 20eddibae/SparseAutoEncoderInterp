#!/usr/bin/env bash
# Collect MLP activations from the trained transformer to feed the SAE.
#
# Lightweight regime: 1M contexts * 200 tokens = 2e8 activations (~190GB f16).
# Original paper used 4B (20M contexts). 2e8 -> SAE trains ~24k steps (one epoch,
# 2e8/8192), enough for neuron resampling to fire 4x and re-converge.
#
# Output: $SDL_REPO/autoencoder/data/openwebtext/512/<partition_*.pt>
#         (that dir is symlinked to /dartfs-hpc/scratch -- $HOME has no room.)
# RAM: ~80GB peak with default 20 partitions; reduce num_contexts if tight.

set -euo pipefail

SDL_REPO="${SDL_REPO:-/Users/eddiebae/CS/sparse-dictionary-learning}"
NUM_CONTEXTS="${NUM_CONTEXTS:-1000000}"
NUM_PARTITIONS="${NUM_PARTITIONS:-20}"

cd "$SDL_REPO/autoencoder"
python -u prepare.py \
    --num_contexts="$NUM_CONTEXTS" \
    --num_sampled_tokens=200 \
    --num_partitions="$NUM_PARTITIONS" \
    --dataset=openwebtext \
    --gpt_ckpt_dir=out \
    --device=cuda
