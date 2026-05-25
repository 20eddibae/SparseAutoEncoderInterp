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

### Cross-corpus confirmation — WildChat agrees to within 0.003

Ran the same models on a **completely independent** clean-text corpus
(allenai/WildChat-1M, English, real user↔GPT-3.5/4 turns, plain text — HTML in
only 0.4%/0.8% of human/ai turns). Fetched 10k convs via
`scripts/fetch_wildchat.py` → `data/raw/wildchat.json` (ShareGPT format, reuses
the loader); `configs/discovery_wildchat.yaml`; extraction job 8687858 →
`artifacts/features_wildchat.npz` (45,690 turns, balanced 22,739 human / 22,951
ai); probe job 8687859.

| Probe | ShareGPT stripped | **WildChat (clean)** |
|-------|------------------:|---------------------:|
| A role, all turns (linear)       | 0.918 | **0.921** |
| A role, all turns (nonlinear GB) | 0.978 | **0.981** |
| B role, generated turns only     | 0.931 | **0.934** |
| C seed (turn0) vs generated      | 0.757 | 0.786 |
| D role, group-aware split        | 0.914 | **0.919** |

Two independent scrapes with different formatting conventions agree to within
~0.003 on every role probe. A dataset-specific artifact would not reproduce
across corpora, so the **~0.92 linear / ~0.98 nonlinear is genuine human/AI
language separability** — confirmed, not just "formatting removed." Final
answer to "are the features separable by role": **yes, at AUC ~0.92, robust to
corpus, first-turn exclusion, and conversation-leakage controls.**

## 2026-05-25 — Mapping "Towards Monosemanticity" to the probability modules

Reference: Bricken et al., *Towards Monosemanticity: Decomposing Language Models
With Dictionary Learning*, Transformer Circuits Thread, 2023
(https://transformer-circuits.pub/2023/monosemantic-features/index.html).

**Why this paper is directly usable here:** its headline run "A/1" is *our exact
configuration* — a 1-layer transformer with `d_model=128`, `d_mlp=512`, and a
sparse autoencoder with **4096 features (8× expansion)**, trained on The Pile.
That is the same architecture and SAE width as this project. So the paper's
feature taxonomy (base64, DNA, Arabic/Hebrew script, HTML) and its measurement
methods transfer almost directly to our `artifacts/features_stripped.npz` and
`artifacts/features_wildchat.npz`. The mapping below gives each `src/scf/
probability/` module a concrete, paper-grounded experiment. Run everything on
the trusted corpora (stripped ShareGPT / WildChat), not the raw HTML one.

### `markov.py` + `coarsen.py` ← "Finite State Automata" (the headline match)
The paper's "finite-state automata" — a base64 feature that re-excites itself
(single-node loop), an all-caps-snake-case 2-node system (text node ↔ underscore
node), and a 4-node HTML system (`<tag>` → tag-name → tag-close → whitespace →
back to `<tag>`) — **are Markov chains over feature-states**, formed by one
feature raising the probability of tokens that make the next feature fire. Our
`coarsen.py` (top-k / cluster state) → `estimate_transition` → `chapman_
kolmogorov_test` → `stationary_distribution` → `mixing_time` pipeline recovers
exactly these objects. Concrete deliverable: build the feature-state transition
matrix and (a) flag **self-exciting features** (large diagonal `P[i,i]`) and
small cycles = their FSA; (b) hunt specifically for an HTML/base64 automaton; (c)
report stationary distribution and mixing time. This makes H1/H4 — and the
role-coupled H5/H6 below — a *reproduction* of the paper's hand-found automata on
the same model size, not just a course exercise.

### `poisson.py` ← L0 norm and feature-density histograms
The paper tracks the **L0 norm** (number of features active per token; target
< 10–20) and **feature-density histograms** (log-scale; the bimodal "ultralow-
density cluster" vs the interpretable high-density cluster; 168 dead + 292
ultralow of 4096). Our `N_active` per turn is a sum of 4096 Bernoulli firings →
Le Cam Poisson(λ = Σ pᵢ), which `poisson_thinning_test` already computes. Paper-
backed prediction: real data deviates **above** the Poisson null because
features are correlated (feature-splitting ⇒ co-firing). Also reproduce the
feature-density histogram and report dead / ultralow / high-density counts for
our SAE.

### `bayes.py` ← log-likelihood-ratio proxies (they use Bayes' rule explicitly)
The paper validates features with a computational proxy `log(P(s|context)/P(s))`
and derives `P(s|Arabic)` via **Bayes' rule**. Our `posterior_role_given_feature`
is the conjugate object `P(role | feature fires)`. Deliverable: build an LLR
proxy for a cleanly detectable context (code, base64, a Unicode script, or role)
and correlate it with feature activation — the paper's validity bar is Pearson
0.74 (Arabic) / 0.80 (DNA). Report the top per-role discriminative features and
their posteriors.

### `order_stats.py` + `concentration.py` ← expected-value plots & "big activations matter"
The paper argues "large feature activations have larger effects" and uses
**expected-value plots** (the activation distribution weighted by activation
magnitude) to show most of a feature's *impact* comes from its high activations.
`order_stats.py` (max activation, top1−top2 gap, top-1 fraction = a
monosemanticity/specificity score) plus `concentration.py` (Markov's inequality
`P(act ≥ t) ≤ μ/t`, Chebyshev, sub-Gaussian) formalize this: show that activation
*mass* concentrates in the interpretable high tail, and bound that tail.

### `mgf.py` + `clt.py` ← logit-weight bimodality ("interference" mode)
Throughout the paper, a feature's **logit-weight distribution is bimodal**: a
large central near-Gaussian "interference" mode plus a small "signal" mode of
context-specific tokens. Compute per-feature logit weights (SAE decoder ×
MLP-down × unembed, the path-expansion of the Framework), then use
`fit_normal_via_mgf` / the CLT diagnostic to test whether the central mode is
Gaussian interference — separating signal from noise quantitatively.

### `entropy.py` ← H2 role entropy + a documented caution
H2 (entropy of the per-role feature-firing distribution; predict AI lower →
"mode-seeking") is now runnable on a trusted corpus. Add the Markov **entropy
rate** `Σ πᵢ H(Pᵢ·)` as a single predictability scalar, and compare human- vs
ai-conditioned entropy rate. Caution to cite: the paper *tried* an information-
theoretic metric to select dictionaries and found it **did not** correlate with
interpretability — so treat entropy as descriptive, not as a quality target.

### `tail_sum.py` ← mean active-feature count sanity check
`E[N_active] = Σ_{k≥1} P(N_active ≥ k)` cross-checks the L0 mean from the Poisson
analysis via the tail-sum identity.

### Two cross-cutting probability ideas (not a single module)
- **Importance sampling / change of measure.** The paper resamples dead SAE
  neurons with probability ∝ loss², and reweights auto-interp examples by
  `feature_density × interval_probability`. A clean place to demonstrate
  importance weights / reweighted estimators.
- **"Model vs data" control.** The paper runs dictionary learning on a
  *random-weight* transformer to separate dataset structure from model
  computation. This is the exact analog of our **synthetic-corpus control**
  (role AUC 0.56): same logic on a different axis — cite as precedent for why
  the synthetic baseline matters.

### Suggested first deliverable
Implement the **FSA detector** (self-loops + small cycles in the feature-state
transition graph from `markov.py`/`coarsen.py`) on `features_stripped.npz`, then
the **role-coupled chain** (H5/H6): estimate `T_{h→a}` and `T_{a→h}` separately
and test (a) that modeling the alternation lowers the CK residual vs the pooled
chain in H1, and (b) that the two kernels differ (role-asymmetric dynamics).

## 2026-05-25 — Research question framing + first results (see RESULTS.md)

**Research question.** *How random is human–AI interaction?* Operationalized as
the entropy/predictability of an SAE-feature Markov chain over a conversation —
and, crucially, decomposed by who acts (human-surprise vs AI-self-surprise).

**Two chains (from the plan).** *Chain 1* = within-generation, token-level feature
dynamics (the paper's finite-state automata; needs per-token extraction — not yet
run). *Chain 2* = conversation-level, turns as time steps, states = clusters of
per-turn SAE activations (conversation "modes"). We run Chain 2 on the trusted
corpora; it is also where the human enters.

**First results are in `RESULTS.md`** (root). Headline: at the conversational-mode
level (K=32) the process is moderately-high randomness with stable structure
(entropy rate 74–84% of max; ergodic, mixing in 4–10 turns), and the randomness is
**asymmetric — the human is ~1.6–2× less predictable than the AI** (human surprise
exceeds AI self-surprise by +0.50/+0.68 nats on ShareGPT/WildChat). Probability
modules exercised so far: `markov`,`coarsen`,`entropy` (H1/H4 turn chain);
`poisson`,`concentration`,`clt`,`mgf`,`tail_sum` (H3 — L0 is super-Poissonian,
features co-fire); `entropy`,`bayes` (H2 — roles have ~equal marginal entropy, so
separability is about *which* features fire). Remaining: token-level FSA
(per-token extraction, GPU), and CPU-only `order_stats` / LLR proxy /
logit-weight MGF. See the RESULTS.md scoping table.
