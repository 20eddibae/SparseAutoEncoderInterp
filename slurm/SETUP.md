# Discovery one-time setup

## 1. ssh in
```bash
ssh <netid>@discovery.dartmouth.edu
```

## 2. Clone both repos under your home
```bash
cd $HOME
git clone https://github.com/shehper/sparse-dictionary-learning.git
# replace <url> below with however you're sharing this project
git clone <url> stochastic-conversation-features
```

## 3. Create the conda environment (interactive node, not the login node)
The login node is shared and CPU-throttled — never install heavy packages there. Grab a small interactive job and install from inside it.

```bash
srun --account=rc --partition=standard --cpus-per-task=4 --mem=16G --time=01:00:00 --pty /bin/bash

module purge
module load python

# create a python 3.9 env to match sparse-dictionary-learning's requirements
conda create -n scf-env python=3.9 -y
source activate scf-env

# 1) SDL dependencies (transformer + SAE training stack)
cd $HOME/sparse-dictionary-learning
pip install -r requirements.txt

# 2) analysis-layer dependencies
cd $HOME/stochastic-conversation-features
pip install -e .

# verify torch + cuda
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Exit the interactive shell when done: `exit`.

## 4. Edit configs/discovery.yaml
The Discovery config is committed but the SAE checkpoint path is blank — fill it in after step 4 (SAE training) finishes. Until then steps 1-4 don't read the config; only `scripts/extract_features.py` (step 5) and the hypothesis drivers do.

## 5. Submit jobs
From `$HOME/stochastic-conversation-features`:
```bash
bash slurm/submit_all.sh
```
This chains all 5 jobs with `--dependency=afterok:`. Check status:
```bash
squeue --me
```
Logs land in `logs/<job>.<jobid>.out` and `.err`.

If a step has already finished and you only want to re-run later ones:
```bash
bash slurm/submit_all.sh --skip-train       # skip owt + transformer
bash slurm/submit_all.sh --skip-sae         # only re-run feature extraction
```

## 6. After the SAE finishes
Discovery will email you (`#SBATCH --mail-type=END`). Then:
```bash
ls $HOME/sparse-dictionary-learning/autoencoder/out/openwebtext/
# pick the timestamped subfolder, paste it into configs/discovery.yaml:
#   sae_ckpt_subdir: "<timestamp-autoencoder-openwebtext>"
```
The pre-submitted feature-extraction job (step 5) will use it.

## 7. Hypothesis tests
These are CPU-only and fast. Run interactively or in a tiny job:
```bash
for h in 1 2 3 4; do
  python scripts/run_hypothesis.py --h $h \
      --config configs/discovery.yaml \
      --features artifacts/features.npz \
      --out artifacts/h${h}.json
done
```
