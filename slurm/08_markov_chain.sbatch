#!/bin/bash -l
#SBATCH --job-name=scf-markov
#SBATCH --partition=standard
#SBATCH --account=lsonglab
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=END,FAIL

# Turn-level conversation Markov chain on the already-extracted per-turn features.
# CPU-only (clustering + small eigendecomposition). Runs both trusted corpora.
source "${SLURM_SUBMIT_DIR:-/dartfs-hpc/rc/home/w/f0079dw/SparseAutoEncoderInterp}/slurm/00_env.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

K="${K:-32}"
for feats in artifacts/features_stripped.npz artifacts/features_wildchat.npz; do
    echo "######### markov chain: $feats  (k=$K) #########"
    python "$SCF_REPO/scripts/markov_chain.py" --features "$SCF_REPO/$feats" --k "$K"
    echo
done
