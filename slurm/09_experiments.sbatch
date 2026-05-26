#!/bin/bash -l
#SBATCH --job-name=scf-experiments
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

# Regenerate every CPU-only result from the already-extracted per-turn features.
#   Experiment 1 (Poisson), 2 (WLLN/CLT), 3 (roles)   -> run_hypothesis.py
#   Experiment 4 (Markov chain) + generative MCMC      -> mcmc_conversations.py
#   all figures                                        -> make_plots.py
# No GPU, no re-extraction; state = argmax (dominant) SAE feature per turn.
source "${SLURM_SUBMIT_DIR:-/dartfs-hpc/rc/home/w/f0079dw/SparseAutoEncoderInterp}/slurm/00_env.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$SCF_REPO"

run_corpus () {
    local tag="$1" feats="$2" cfg="$3"
    echo "######### $tag : $feats #########"
    for e in 1 2 3; do
        python scripts/run_hypothesis.py --exp "$e" --features "$feats" \
            --config "$cfg" --out "results/data/exp${e}_${tag}.json"
    done
    python scripts/mcmc_conversations.py --features "$feats" --k 32 --out-prefix results
}

run_corpus stripped  artifacts/features_stripped.npz  configs/discovery.yaml
run_corpus wildchat  artifacts/features_wildchat.npz  configs/discovery_wildchat.yaml

echo "######### figures #########"
python scripts/make_plots.py

echo "######### tests #########"
python tests/run_tests.py
echo "DONE_ALL"
