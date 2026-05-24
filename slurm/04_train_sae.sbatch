#!/bin/bash -l
#SBATCH --job-name=scf-train-sae
#SBATCH --partition=gpuq
#SBATCH --account=lsonglab
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Train the sparse autoencoder on the MLP activations dataset.
# Length is DATA-DRIVEN (num_steps = total_activations // batch_size, ~24k steps
# from step 3's 2e8 activations); train.py has no max_steps. 12h walltime is
# ample. NOTE: autoencoder/train.py does NOT resume from its own checkpoints, so
# --requeue restarts from scratch on preemption (cheap: the job is short).

source "${SLURM_SUBMIT_DIR:-/dartfs-hpc/rc/home/w/f0079dw/SparseAutoEncoderInterp}/slurm/00_env.sh"

bash "$SCF_REPO/scripts/04_train_sae.sh"
