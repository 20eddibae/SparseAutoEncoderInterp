#!/bin/bash -l
#SBATCH --job-name=scf-train-sae
#SBATCH --partition=gpuq
#SBATCH --account=lsonglab
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Train the sparse autoencoder on the MLP activations dataset.
# A100 80GB on gpuq: ~4h for 50k steps; 8h walltime budget includes resampling.
# Override: MAX_STEPS=100000 sbatch slurm/04_train_sae.sbatch

source "${SLURM_SUBMIT_DIR:-/dartfs-hpc/rc/home/w/f0079dw/SparseAutoEncoderInterp}/slurm/00_env.sh"

export MAX_STEPS="${MAX_STEPS:-50000}"
echo "[job] MAX_STEPS=$MAX_STEPS"

bash "$SCF_REPO/scripts/04_train_sae.sh"
