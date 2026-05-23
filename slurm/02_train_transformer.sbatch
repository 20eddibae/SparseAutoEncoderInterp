#!/bin/bash -l
#SBATCH --job-name=scf-train-gpt
#SBATCH --partition=gpuq
#SBATCH --account=rc
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Train the 1-layer transformer (d_model=128, n_head=4) on OpenWebText.
# A100 80GB on gpuq partition: ~6h for 40k iters; 10h walltime budget is safe.
# Override: MAX_ITERS=80000 sbatch slurm/02_train_transformer.sbatch

source "$(dirname "$0")/00_env.sh"

export MAX_ITERS="${MAX_ITERS:-40000}"
echo "[job] MAX_ITERS=$MAX_ITERS"

bash "$SCF_REPO/scripts/02_train_transformer.sh"
