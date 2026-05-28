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

**Root cause.** `src/scf/corpus/synthetic.py` generates *every* turn — both
`human` and `ai` — from the same transformer with the same sampling, so the two
roles are one distribution and only turn 0 is real text. Role was therefore
unlearnable by construction. Fix: use a real human/AI corpus (done in the next
section). Reproduce probes A–D: `sbatch slurm/07_extended_separability.sbatch`.

## 2026-05-25 — Real corpus, the HTML artifact, and the genuine number (~0.92)

Swapped to a **real** human/AI corpus (RyokoAI/ShareGPT52K) — same transformer,
same SAE, only the corpus changed. Raw ShareGPT *appeared* to separate role at
~0.99, **but that number is not valid**: ShareGPT52K assistant turns are
HTML-wrapped (`<div class="markdown prose">`) and human turns are plain, so the
SAE was largely detecting markup, not human-vs-AI language. We confirmed the
confound with `scripts/evaluate_html_artifact.py` (replays the loader to tag each
turn's raw text): HTML ≈ role almost perfectly (human 0.1% vs AI 99.9% HTML; only
52 plain-AI + 51 HTML-human turns of 70,132), and a single "contains-HTML" bit
predicts role at **AUC 0.999**, beating the full feature vector. The inflated raw
AUROCs are therefore discarded.

**Decisive fix — strip the markup at the source and re-extract.** Added
`corpus.strip_html` + a stdlib HTML→text stripper in `src/scf/corpus/sharegpt.py`,
set `strip_html: true` in `configs/discovery.yaml`, re-ran the pipeline (same
transformer + SAE, only the input text changed): extraction job 8687825 →
`artifacts/features_stripped.npz` (70,133 turns), probe job 8687826.

| Probe | **Stripped AUC** |
|-------|-----------------:|
| A role, all turns (linear)       | **0.918** |
| A role, all turns (nonlinear GB) | **0.978** |
| B role, generated turns only     | **0.931** |
| C seed (turn0) vs generated      | 0.757     |
| D role, group-aware split        | **0.914** |

Role is **genuinely** separable from human/AI *language* at **~0.92 linear /
~0.98 nonlinear**, not a first-turn cue (B) nor conversation leakage (D). Note: an
in-place "drop the top HTML-correlated features" test sent role to chance
(top-200 → 0.508) but is *not* decisive — role and HTML were collinear, so it also
drops genuine role features; the re-extraction is the trustworthy test.

**Cross-corpus confirmation — WildChat.** Independent clean-text corpus
(allenai/WildChat-1M, English, plain text), same models, via
`scripts/fetch_wildchat.py` → `configs/discovery_wildchat.yaml`; extraction job
8687858 → `artifacts/features_wildchat.npz` (45,690 turns), probe 8687859:

| Probe | ShareGPT stripped | WildChat |
|-------|------------------:|---------:|
| A linear       | 0.918 | 0.921 |
| A nonlinear GB | 0.978 | 0.981 |
| B generated    | 0.931 | 0.934 |
| C seed vs gen  | 0.757 | 0.786 |
| D group-aware  | 0.914 | 0.919 |

Two independent scrapes agree to within ~0.003 on every probe. **Final answer:
the features separate by role at AUC ~0.92, robust to corpus, first-turn
exclusion, and leakage controls** — the lightweight models suffice; the magnitude
is ~0.92, not the artifact-inflated 0.99.

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
Poisson(λ = Σ pᵢ) under the Poisson-limit-of-the-Binomial, which
`poisson_thinning_test` already computes. Paper-
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
`P(act ≥ t) ≤ μ/t` and Chebyshev) formalize this: show that activation
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

## 2026-05-26 — Simplification to keep everything inside the course vocabulary

We restructured the project so the *methods* live entirely inside the course's
probability vocabulary (the full topic list now lives in `CLAUDE.md`, and every
experiment maps to ≤2 topics; no new probability concepts are introduced). The
work is now **four experiments + one kept generative result**, each independently
defensible:

1. **Is L0 Poisson?** (tail-sum · Poisson-as-limit-of-Binomial · Markov/Chebyshev)
2. **Does the L0 sample mean concentrate?** (WLLN · CLT)
3. **Are the roles different distributions?** (entropy + delta-method error bar ·
   KL · Jensen)
4. **Conversation-mode dynamics** (Markov chain: CK · stationary π + geometric TV
   decay / mixing), plus the kept **role-coupled chain + generative MCMC** (H5/H6).

**The big change: KMeans is gone.** The conversation-chain state is now the
**dominant (argmax) SAE feature per turn** — an order statistic (course items
18–19), a deterministic function of the activation vector, *not* a clustering
output. This removes the only out-of-course step (KMeans is optimisation, not
probability) and the fragile `K=32` hyperparameter. Two course-justified
refinements keep it well-posed: (a) the always-on DC feature 2684 has firing
probability 1 (zero information) and is excluded from the argmax; (b) the M−1 most
frequent dominant features are named states and the rare tail is pooled into one
`other` state, where M is a *reporting granularity* (like a histogram bin count),
not a tuned hyperparameter — the headline is robust across M ∈ {16,32,48}.

**The headline survived and got cleaner.** With argmax states (M=32) the human−AI
surprise gap is **+0.53 / +0.76 nats** (ShareGPT / WildChat) — directionally
identical to, and a bit larger than, the old KMeans +0.50 / +0.68, and now a
property of a deterministic, course-native state. The role-coupled chain cuts the
CK residual ~0.3 and the generative MCMC inherits the gap (+0.52 / +0.75). The
delta-method error bar on Experiment 3 is tight enough to *resolve* the role
entropy difference (AI is slightly but significantly *higher*-entropy, −0.049 ±
0.0004) — sharper than the old "roughly equal under a loose Chebyshev bar."

**Dropped to stay in scope** (each replaced or shelved, not lost): the MGF
Gaussianity fit for L0 (the Poisson rejection already does that work); the
spectral-gap framing (replaced by "TV distance to π decays geometrically" — same
fact, course-native); the topic-conditional stationary test (old H4); and from the
forward plan the importance-sampling / Gibbs-measure gestures, the token-level FSA
(needs GPU per-token extraction), and the logit-weight bimodality (a different
model layer). The Bayes posterior is folded into Experiment 3 as one line. Kept as
designed: the two corpora, the tail-sum sanity check, and the synthetic-corpus
control (role AUC 0.56) framed as the proper null. `mgf.py` remains in the toolbox
for reference (CLT's proof uses MGF uniqueness) but is no longer a headline test.

Final figures: `results/figures/01–06*.png`. Final numbers: `RESULTS.md`.
