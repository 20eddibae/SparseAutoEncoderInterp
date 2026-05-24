#!/bin/bash -l
#SBATCH --job-name=scf-mlp-data
#SBATCH --partition=gpuq
#SBATCH --account=lsonglab
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Generate the SAE training dataset by running the trained transformer forward.
# prepare.py pre-allocates the FULL activation buffer in RAM as float32:
#   num_contexts * num_sampled_tokens(200) * n_ffwd(512) * 4 bytes.
#   500k contexts -> ~205GB ; 1M contexts -> ~410GB. num_partitions only splits
#   the buffer at save time, it does NOT lower peak RAM.
# 512G covers the 1M-context default below with headroom (gpuq nodes have ~1-1.5TB).
# Override: NUM_CONTEXTS=500000 sbatch slurm/03_generate_mlp_data.sbatch  # halves RAM

source "${SLURM_SUBMIT_DIR:-/dartfs-hpc/rc/home/w/f0079dw/SparseAutoEncoderInterp}/slurm/00_env.sh"

export NUM_CONTEXTS="${NUM_CONTEXTS:-1000000}"
export NUM_PARTITIONS="${NUM_PARTITIONS:-40}"
echo "[job] NUM_CONTEXTS=$NUM_CONTEXTS  NUM_PARTITIONS=$NUM_PARTITIONS"

bash "$SCF_REPO/scripts/03_generate_mlp_data.sh"
