# Research log

## 2026-05-24 — Are the SAE turn-features separable by role?

**Question.** After the full pipeline run (transformer → SAE → feature
extraction), do per-turn SAE feature vectors separate `human` from `ai`
turns?

**Artifact under test.** `artifacts/features.npz` — 120,000 turns × 4,096 SAE
features, balanced 60k human / 60k ai. Pooling `max`, `activation_threshold`
0.0. Produced by jobs 8680139 (GPT) → 8683696 (MLP data) → 8683697 (SAE,
24,413 steps) → 8684152 (extract).

### Feature sanity (job 8684368)
- No NaN/Inf. magnitude min/mean/max = 0 / 0.031 / 7.54.
- ~72.5 active features per turn (≈1.8% density) — healthy SAE sparsity.
- 1,568 / 4,096 dead features, 1 always-active. No all-zero turns.

### Separability results (jobs 8684368, 8684379)
Balanced subsamples, 25% held out. Baseline = 0.50.

| Probe | Label | Classifier | Test acc | AUC |
|-------|-------|------------|---------:|----:|
| A | role, all turns | logreg (magnitudes) | 0.544 | 0.564 |
| A | role, all turns | logreg (support) | 0.541 | 0.567 |
| A | role, all turns | grad. boosting (nonlinear) | 0.557 | 0.575 |
| B | role, **generated turns only** (turn_idx>0) | logreg | 0.507 | **0.509** |
| C | **seed vs generated** (turn 0 vs >0) | logreg | 0.901 | **0.966** |
| D | role, **group-aware** split by conv_id | logreg | 0.546 | 0.567 |

### Verdict: NOT separable by role — and that is expected, not a bug

Role is at chance once the first turn is excluded (**B: AUC 0.509**), and a
nonlinear model does not help (A: 0.575). Group-aware splitting changes
nothing (D ≈ A), so the weak A-signal is not conversation leakage.

The features themselves are fine: seed text vs model-generated text is
**strongly separable (C: AUC 0.966)**, so the SAE encodes real distributional
structure.

**Root cause.** `src/scf/corpus/synthetic.py` builds each conversation by
seeding turn 0 with a 32-token OpenWebText snippet, then generating *every*
subsequent turn — both `human` and `ai` — from the *same* transformer with the
*same* sampling (temp 1.0, top-k 40). Human and ai turns are therefore one
distribution; only turn 0 is real text. With `max_turns=12`, turn 0 is 1 of 6
human turns, which is exactly the small residual signal seen in probe A. The
corpus was switched from `sharegpt` to `synthetic` on 2026-05-23 because no
ShareGPT JSON was available (see `configs/discovery.yaml`).

### What to do to fix it

1. **Use a real human/AI corpus.** Provide a ShareGPT JSON and set
   `corpus.name: sharegpt` + `corpus.sharegpt_path` in `configs/discovery.yaml`
   (the loader already exists), then re-run extraction (`slurm/05`) and the
   separability check (`slurm/06`). Real human prompts vs assistant replies
   differ in distribution, so role should become learnable.
2. **If staying synthetic,** make the two roles distinguishable on purpose —
   e.g. different sampling for `human` vs `ai` turns, or seed every human turn
   from real text rather than only turn 0. Otherwise role is unlearnable by
   construction and "role separability" is not a meaningful metric here.
3. **Sanity probe to keep:** seed-vs-generated (probe C) is a good ongoing
   check that the transformer+SAE pipeline still encodes distributional signal,
   independent of the role question.

Reproduce: `sbatch slurm/06_check_separability.sbatch` (sanity + role probe,
writes `artifacts/separability.png`) and `sbatch slurm/07_extended_separability.sbatch`
(probes A–D).

## 2026-05-25 — Real corpus confirms it: models fine, corpus was the blocker

Swapped the corpus to a **real** human/AI dataset (RyokoAI/ShareGPT52K,
`sg_90k_part1.json`, 45,332 real human↔ChatGPT convs; `corpus.name: sharegpt`)
and re-ran extraction (job 8686447, 70,132 turns) + both separability checks
(8686448, 8686449). **Nothing about the transformer or SAE changed** — same
6.64M-param 1-layer GPT, same SAE checkpoint. Synthetic baseline preserved at
`artifacts/features_synthetic.npz`.

| Probe | Synthetic AUC | **Real (ShareGPT) AUC** |
|-------|--------------:|------------------------:|
| A role, all turns (linear) | 0.564 | **0.989** |
| A role, all turns (nonlinear GB) | 0.575 | **0.99998** |
| B role, generated turns only | 0.509 | **0.995** |
| C seed (turn0) vs generated | 0.966 | 0.820 |
| D role, group-aware split | 0.567 | **0.994** |

**Verdict: the lightweight models are sufficient — capacity was never the
issue.** Same models, swap synthetic→real corpus, and role goes 0.56→0.99.
Probe B is no longer ~chance (that assumption was synthetic-specific; the
script's printed "expect ~chance" note is now stale). D≈A rules out leakage.

**Caveat — the 0.99 is inflated by a formatting artifact.** In ShareGPT52K the
assistant (`gpt`) turns are HTML-wrapped (`<div class="markdown prose…">`):
354,892 `<div>` hits across ~363,740 gpt turns (~98%), while human turns are
plain text. The SAE can separate roles partly by detecting HTML-tag tokens, a
trivial tell rather than genuine human-vs-AI language. The top discriminative
feature (1295, coef +3.81, ~2× the next) is consistent with one near-trivial
cue. So the *true* semantic-role separability is somewhere below 0.99.

**To get the clean number:** re-run extraction on HTML-stripped text, or use a
clean-text corpus (WildChat — gated on HF, needs a token). The "do we need a
bigger model" question is already answered (no); this only refines *how*
separable real roles are. → **Done below (HTML-stripped re-extraction).**

## 2026-05-25 (later) — HTML-stripped re-extraction: the genuine number is ~0.92

The 0.99 above conflates two things: genuine human/AI language, and the fact
that ShareGPT52K assistant turns are HTML-wrapped (`<div class="markdown
prose">`) while human turns are plain. Quantifying the confound on the raw
`features.npz` (`scripts/evaluate_html_artifact.py`, which replays the loader to
tag each turn's raw text): HTML presence ≈ role almost perfectly — human 0.1%
HTML vs AI 99.9% HTML, with only **52 plain-AI and 51 HTML-human turns** out of
70,132. A single "contains an HTML tag" bit predicts role at **AUC 0.999**,
*beating* the full 4096-dim feature vector (0.9935).

That collinearity also means the obvious in-place control is unreliable:
dropping the top-N HTML-correlated features sends role to chance (top-200 →
0.508), but because role and HTML are ~collinear in the raw corpus, *any*
genuinely role-predictive feature also looks HTML-predictive and gets dropped.
So that test **overstates** the artifact and is not decisive.

**The decisive test is to remove the markup at the source and re-extract.**
Added `corpus.strip_html` (config) and a stdlib HTML→text stripper in
`src/scf/corpus/sharegpt.py` (scf-env has no bs4/lxml), set `strip_html: true`
in `configs/discovery.yaml`, and re-ran the pipeline — **same 6.64M-param
transformer, same SAE checkpoint, only the input text changed**:

- extraction job 8687825 → `artifacts/features_stripped.npz` (70,133 turns)
- probe job 8687826 (`slurm/07`, `FEATURES=…features_stripped.npz`)

| Probe | Raw / HTML AUC | **Stripped AUC** |
|-------|---------------:|-----------------:|
| A role, all turns (linear)       | 0.989    | **0.918** |
| A role, all turns (nonlinear GB) | 0.99998  | **0.978** |
| B role, generated turns only     | 0.995    | **0.931** |
| C seed (turn0) vs generated      | 0.820    | 0.757     |
| D role, group-aware split        | 0.994    | **0.914** |

**Verdict (refined).** Role is **genuinely** separable from real human/AI
*language* at **AUC ~0.92 linear / ~0.98 nonlinear**, and the signal is not a
first-turn cue (B=0.931 on generated-only turns) nor conversation leakage
(D=0.914, group-aware). The raw 0.99 was ~0.07 (linear) inflated by HTML markup,
and the near-perfect 0.99998 nonlinear was mostly that tell. The headline
conclusion is unchanged in *direction* — the lightweight models suffice, no
bigger model needed — but the honest magnitude is **~0.92, not 0.99**.

`artifacts/features.npz` is kept as the raw/HTML "before"; the clean run is
`artifacts/features_stripped.npz`. A residual cross-check (e.g. WildChat) could
confirm ~0.92 isn't itself a subtler ShareGPT-specific tell, but the formatting
confound — the only obvious one — is now removed.
