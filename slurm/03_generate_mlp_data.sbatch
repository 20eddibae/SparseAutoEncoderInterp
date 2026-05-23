#!/bin/bash -l
#SBATCH --job-name=scf-mlp-data
#SBATCH --partition=gpuq
#SBATCH --account=rc
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Generate the SAE training dataset by running the trained transformer forward.
# 500k contexts -> ~100GB of float16 activations across 20 partitions.
# Memory: original paper used 1TB for 4M contexts; 256G is sufficient for 500k.
# Override: NUM_CONTEXTS=1000000 NUM_PARTITIONS=40 sbatch slurm/03_generate_mlp_data.sbatch

source "$(dirname "$0")/00_env.sh"

export NUM_CONTEXTS="${NUM_CONTEXTS:-500000}"
export NUM_PARTITIONS="${NUM_PARTITIONS:-20}"
echo "[job] NUM_CONTEXTS=$NUM_CONTEXTS  NUM_PARTITIONS=$NUM_PARTITIONS"

bash "$SCF_REPO/scripts/03_generate_mlp_data.sh"
