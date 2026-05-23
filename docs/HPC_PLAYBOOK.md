# HPC playbook — Dartmouth Discovery

Discovery is Slurm-scheduled. GPUs live on the `gpuq` partition (A100 80GB per `sinfo -O gres -p gpuq`); CPU work runs on `standard`. All jobs use `--account=rc`.

This playbook is a thin wrapper: the real definitions are in `slurm/*.sbatch`. See `slurm/SETUP.md` for the one-time conda env creation.

## Layout on Discovery
```
$HOME/sparse-dictionary-learning/   # instrument (do not edit)
$HOME/stochastic-conversation-features/   # this repo (analysis layer)
$HOME/stochastic-conversation-features/logs/    # %x.%j.out + .err
$HOME/stochastic-conversation-features/artifacts/    # features.npz + hypothesis JSONs
```
If `$HOME` is short on quota, set `SCF_REPO` and `SDL_REPO` to a `/dartfs-hpc/rc/lab/...` or `/scratch/...` location before sourcing the env script.

## End-to-end submission

```bash
cd $HOME/stochastic-conversation-features
bash slurm/submit_all.sh
```

This `sbatch`-submits five jobs with `--dependency=afterok` so each waits for its predecessor:

| Step | Script | Partition | GPU | Walltime budget | Memory | Wallclock estimate |
|------|--------|-----------|-----|------------------|--------|--------------------|
| 1 | `01_prepare_owt.sbatch` | `standard` | – | 4h | 64G | ~1h |
| 2 | `02_train_transformer.sbatch` | `gpuq` | 1×A100 | 10h | 64G | ~6h (40k iters) |
| 3 | `03_generate_mlp_data.sbatch` | `gpuq` | 1×A100 | 6h | 256G | ~2h (500k contexts) |
| 4 | `04_train_sae.sbatch` | `gpuq` | 1×A100 | 8h | 96G | ~4h (50k steps) |
| 5 | `05_extract_features.sbatch` | `gpuq` | 1×A100 | 2h | 32G | ~30min (10k convs) |

Monitor with:
```bash
squeue --me
scontrol show job <jobid>
tail -f logs/scf-train-gpt.<jobid>.out
```
Cancel with `scancel <jobid>` or `scancel --me`.

## Re-running a single step

Use the skip flags to re-anchor the chain:
```bash
bash slurm/submit_all.sh --skip-owt        # OWT already tokenised
bash slurm/submit_all.sh --skip-train      # transformer ckpt exists
bash slurm/submit_all.sh --skip-mlp        # activations already on disk
bash slurm/submit_all.sh --skip-sae        # only re-extract features
```

Or submit one script directly:
```bash
MAX_STEPS=100000 sbatch slurm/04_train_sae.sbatch
```

## After the SAE finishes
1. Note the timestamped folder under `$HOME/sparse-dictionary-learning/autoencoder/out/openwebtext/`.
2. Paste it into `configs/discovery.yaml: sae_ckpt_subdir`.
3. The pre-queued step 5 (feature extraction) will pick it up once it runs. If you submitted step 5 *before* setting the SAE path, cancel and resubmit:
   ```bash
   scancel <step5-jobid>
   sbatch slurm/05_extract_features.sbatch
   ```

## Conversation corpus
Default `configs/discovery.yaml` points to ShareGPT at `$HOME/stochastic-conversation-features/data/raw/sharegpt.json`. Fetch on the login node (small, ~700MB):
```bash
mkdir -p data/raw && cd data/raw
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O sharegpt.json
```
To switch to synthetic self-play instead, set `corpus.name: synthetic` in the config.

## Hypothesis tests (CPU, fast)
Run from a login node or a small interactive job:
```bash
srun --account=rc --partition=standard --cpus-per-task=2 --mem=8G --time=30:00 --pty /bin/bash
source activate scf-env
cd $HOME/stochastic-conversation-features
for h in 1 2 3 4; do
  python scripts/run_hypothesis.py --h $h \
      --config configs/discovery.yaml \
      --features artifacts/features.npz \
      --out artifacts/h${h}.json
done
```

## Compute totals (default settings)
- 1× A100-hour budget: ~13h GPU billed (steps 2+3+4+5).
- CPU-only: ~1h (step 1) + minutes (hypotheses).
- Disk: ~30GB (OWT) + ~100GB (MLP activations) + ~1GB (SAE ckpt) = ~130GB peak.
- Wall time end-to-end if no queue waits: ~14h.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Job `CANCELLED ... DUE TO TIME LIMIT` | bump walltime past the wallclock estimate | edit `#SBATCH --time=...` in the sbatch script |
| `OOM-killed` on step 3 | bump `--mem` to 384G or increase `NUM_PARTITIONS` | both work; partitions = lower peak RAM |
| Step 5 fails: `sae_ckpt_subdir is empty` | step 4 finished but you didn't update `configs/discovery.yaml` | paste the subfolder and resubmit step 5 |
| `srun: error: Unable to allocate resources` | gpuq is full | check `sinfo -p gpuq`; try `preemptable` partitions for non-critical reruns |
| Conda env activation fails in job | `module load python` not loaded before `source activate` | already handled in `slurm/00_env.sh`; check `logs/*.err` for the actual line |
