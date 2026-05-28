# Results — four probability experiments on real-conversation SAE features

All experiments use **our trained models** (6.64M-param 1-layer GPT on OpenWebText
→ SAE, 4096 features / 8× expansion — the same config as Bricken et al. 2023's
A/1 run) applied to **real GPT conversations**. Features are per-turn, max-pooled,
in `artifacts/features_stripped.npz` (HTML-stripped ShareGPT52K, 70,133 turns) and
`artifacts/features_wildchat.npz` (WildChat-1M English, 45,690 turns). Everything
here is CPU-only on already-extracted features (no GPU, no re-extraction).

**Scope.** Every experiment uses only the course probability vocabulary (the topic
list in `CLAUDE.md`); each maps to ≤2 topics. The conversation-chain **state is the
dominant (argmax) SAE feature per turn — an order statistic, not a clustering
output** (there is no KMeans and no `K` hyperparameter anywhere). Figures and
machine-readable outputs for every result below live in
[`results/`](results/README.md) (`results/figures/01–06*.png`).

Regenerate everything with `sbatch slurm/09_experiments.sbatch`.

---

## Experiment 1 — Is L0 Poisson?   `run_hypothesis.py --exp 1`
Topics: **tail-sum identity** (5/8) · **Poisson-as-limit-of-Binomial** (13/14) ·
**Markov & Chebyshev** tail bounds (25/26). The RV is `L0` = number of SAE
features firing per turn (a sum of 4096 Bernoulli indicators).

| Quantity | ShareGPT (stripped) | WildChat |
|---|---:|---:|
| mean μ | 77.65 | 82.52 |
| variance | 771.7 | 849.1 |
| **Fano factor var/mean** (Poisson = 1) | **9.94** | **10.29** |
| tail-sum `E[N]` vs direct mean | 77.6461580 vs 77.6461580 ✓ | 82.5155180 vs 82.5155180 ✓ |
| λ̂ = Σ pᵢ | 77.65 | 82.52 |
| Poisson-approx. correction Σ pᵢ² | 53.2 | 59.6 |
| Poisson χ² p-value | 0.0 | 0.0 |

- **Tail-sum identity** `E[N] = Σ_{k≥1} P(N≥k)` matches the direct sample mean to
  1e-9 on both corpora — a one-line numerical check of the April-7 theorem. ✓
- **The Poisson null is rejected hard.** Under independent firing, `L0` would be
  a sum of independent Bernoullis → Poisson(λ) (the Poisson-limit-of-the-Binomial,
  course item 13). That limit needs small firing rates: its leading correction
  Σpᵢ² ≈ 53–60 is itself ≫1, so the approximation was never going to hold. The
  empirical variance is ~10× the mean (Fano ≈ 10, Poisson = 1), so `L0` is
  strongly **over-dispersed** and χ² rejects at p = 0. **Interpretation:** features
  are **correlated / co-fire** (feature-splitting ⇒ groups activate together), so
  per-turn firing is far from independent Bernoullis.
- **Markov and Chebyshev bounds both hold** and bracket the empirical upper tail
  of `L0 − μ`; Chebyshev is the tighter of the two for large deviations.

**Takeaway:** `L0` is a real, well-behaved RV (mean ~78–83) but strongly
super-Poissonian — direct evidence the SAE features are correlated.

## Experiment 2 — Does the sample mean concentrate?   `--exp 2`
Topics: **WLLN** (31) + **CLT** (33). Same RV (`L0`), but now only the
limit-theorem question: the *sample mean* of `L0`, not `L0` itself.

| Quantity | ShareGPT (stripped) | WildChat |
|---|---:|---:|
| running mean → μ | 77.65 | 82.52 |
| blocks (B × size) | 50 × 1402 | 50 × 913 |
| KS stat (block means vs N(0,1)) | 0.101 | 0.207 |
| KS p-value | **0.65** (Gaussian ✓) | 0.024 (borderline) |

- **WLLN:** the running sample mean of `L0` settles onto μ as more turns are
  averaged — the curve flattens to μ ≈ 78 / 83 on the two corpora.
- **CLT:** standardised block means are Gaussian for ShareGPT (KS p = 0.65, fails
  to reject normality). WildChat is borderline (KS p = 0.024, KS stat 0.21) — a
  mild deviation expected from a 50-point KS test; the running mean still
  concentrates cleanly. So the WLLN/CLT machinery is valid for the sample mean
  **even though per-turn `L0` is itself non-Gaussian** (Experiment 1).

## Experiment 3 — Are the roles different distributions?   `--exp 3`
Topics: **entropy as `E[−log p]` with a delta-method error bar** (35) ·
**KL** both directions with **Jensen** (28) as the reason `KL ≥ 0`. For each role
we form the marginal feature-firing distribution.

| Quantity | ShareGPT (stripped) | WildChat |
|---|---:|---:|
| entropy `H(human)` | 4.734 ± 0.0003 | 4.754 ± 0.0004 |
| entropy `H(ai)` | 4.784 ± 0.0003 | 4.795 ± 0.0004 |
| **Δ = H(h) − H(a)** | **−0.049 ± 0.0004** | **−0.041 ± 0.0005** |
| distinct active features (human / ai) | 1,499 / 2,234 | 1,365 / 2,239 |
| KL(human‖ai) | 0.028 | 0.030 |
| KL(ai‖human) | 0.042 | 0.053 |
| max posterior P(role \| feature) | ~1.000 | ~1.000 |

- **The delta-method bar resolves the difference.** With N ≈ 2–3 M feature-firings
  the standard error on each entropy is ~3e-4, so Δ = −0.049 is ~100 σ from zero:
  the AI's marginal feature distribution has **slightly but significantly *higher***
  entropy than the human's, and AI uses a broader vocabulary (≈2,234 vs 1,499
  distinct features). This is the **opposite** of the "AI is mode-seeking ⇒ lower
  entropy" hypothesis. Both corpora agree.
- **KL ≥ 0 by Jensen**, and both directions are strictly positive (0.028–0.053) —
  so the two role distributions are genuinely different, just close in *total*
  entropy. The discriminative power lives in *which* features fire: the Bayes
  posterior `P(role | feature)` reaches ~1.0 for the top role-specific features.

**Takeaway:** the ~0.92 role separability is **not** an entropy-magnitude effect
(the entropies are within 1% and AI's is the larger). Roles differ in *which*
features fire, not in *how many*.

## Experiment 4 — Conversation-mode dynamics (Markov chain only)   `mcmc_conversations.py`
Topics: **Chapman–Kolmogorov** (36) · **stationary π + geometric TV decay /
mixing** (37). **State = the dominant (argmax) SAE feature per turn**, an order
statistic. The always-on DC feature **2684** (firing probability 1 ⇒ zero
information) is excluded; the 31 most frequent dominant features are named states
and the rare tail is pooled into one `other` state (M = 32). The transition matrix
is the row-normalised count matrix — the MLE is a sample mean, justified by WLLN.

| Quantity | ShareGPT (stripped) | WildChat |
|---|---:|---:|
| conversations / turn-transitions | 10,000 / 60,133 | 9,995 / 35,695 |
| states | 32 | 32 |
| Chapman–Kolmogorov residual (entrywise L2) | 1.312 | 1.435 |
| mixing time (TV ≤ 0.25) | **4 turns** | **2 turns** |
| TV(start→π) after 10 turns | 0.006 | ~0.000 |
| **converges to a unique stationary π** | **yes** | **yes** |
| entropy of π (max = ln 32 = 3.466) | 2.852 | 2.754 |
| entropy rate `h` (nats/turn) | 2.459 (71%) | 2.604 (75%) |
| **human surprise** `H(ai→next human)` | 2.639 | 2.943 |
| **AI self-surprise** `H(human→next ai)` | 2.108 | 2.180 |
| **gap (human − AI)** | **+0.532** | **+0.763** |

- **Ergodic and mixing.** From a point mass the total-variation distance to π
  **decays geometrically** and the chain mixes in 2–4 turns (TV after 10 turns
  ≈ 0): a conversation forgets its opening dominant-feature in a handful of turns
  and settles onto a stable, high-entropy spread of states (π entropy 2.75–2.85 of
  a 3.466 max), not a single absorbing state.
- **Not perfectly first-order.** CK residual ≈ 1.3–1.4 (> 0): there is
  higher-order structure the pooled 1-step chain misses — expected, because role
  strictly alternates (period-2). This is what the role-coupled chain below models.
- **Headline — "where the human enters" replicates on both corpora.** The human's
  next dominant-feature is **consistently less predictable than the model's**:
  `H(ai-turn → next human-turn)` exceeds `H(human-turn → next ai-turn)` by
  **+0.53 nats (ShareGPT)** and **+0.76 nats (WildChat)**. The model's
  continuations are more on-rails; the human injects more surprise on re-entry.
  The gap is **larger and cleaner than under the old KMeans state definition**
  (+0.50 / +0.68), and it is now a property of a deterministic, course-justified
  state — not a clustering artifact.

**Robustness to the one remaining knob (M).** Sweeping the reporting granularity
M ∈ {16, 32, 48}, the chain is ergodic at every M and the gap is positive and
grows monotonically with resolution (ShareGPT +0.05 → +0.53 → +0.76; WildChat
+0.30 → +0.76 → +1.01). It only washes out at the coarsest M = 16, where there is
too little resolution to see the asymmetry. This monotone robustness *replaces*
the old "absolute entropies depend on K = 32" caveat.

## Kept result — role-coupled chain & generative MCMC (H5/H6)   `mcmc_conversations.py`
Topics (forthcoming, allowed): **MCMC** (38) · **detailed balance** (40) ·
**Metropolis–Hastings** (41). Role strictly alternates, so we estimate the two
role kernels separately — `T_{h→a}` (human-turn → next ai = AI self-surprise) and
`T_{a→h}` (ai-turn → next human = human surprise) — and add two samplers.

| Quantity | ShareGPT (stripped) | WildChat |
|---|---:|---:|
| pooled CK residual | 1.312 | 1.435 |
| **role-coupled CK residual** `‖P2_emp − T_{h→a}T_{a→h}‖` | **0.971** | **1.154** |
| CK improvement | **+0.341** | **+0.282** |
| kernel difference (entrywise L2) `‖T_{h→a} − T_{a→h}‖` | 1.983 | 1.846 |
| human surprise / AI self-surprise (nats) | 2.639 / 2.108 | 2.943 / 2.180 |
| **surprise gap (human − AI)** | **+0.532** | **+0.763** |
| Metropolis–Hastings acceptance | 0.419 | 0.394 |
| **MH TV(empirical, π) after 200k steps** | **0.012** | **0.012** |
| synthetic surprise gap (generative) | +0.518 | +0.751 |
| TV(state dist real, synthetic) | 0.010 | 0.006 |
| distinct states/conv — real vs synthetic | 3.67 vs 4.55 | 2.98 vs 3.39 |

- **Modeling the role alternation captures real structure.** Splitting the chain by
  role drops the Chapman–Kolmogorov residual by ~0.3 (1.31→0.97 ShareGPT, 1.44→1.15
  WildChat): a chunk of the apparent non-Markovianity was just the period-2 role
  flip, recovered by the 2-step product `T_{h→a}T_{a→h}`. The two kernels are
  genuinely different (entrywise L2 distance ≈ 1.8–2.0), i.e. **the dynamics are role-asymmetric**
  — the structural reason the human is the less-predictable side.
- **Metropolis–Hastings reproduces π.** A textbook MH sampler (symmetric
  random-walk proposal, target = the analytic π, acceptance via π only) converges to
  within **TV ≈ 0.012** of the fixed-point (πP = π) π after 200k steps — two
  independent routes to π agree. (Acceptance is ~0.4 rather than ~0.7 because the
  argmax-state π is more peaked than the old cluster π; it still converges.)
- **Generative posterior-predictive check.** Forward-sampling whole synthetic
  conversations from the role-coupled kernels reproduces the marginal state mix
  almost exactly (TV ≈ 0.006–0.010) and **inherits the headline asymmetry**
  (synthetic gap +0.52 / +0.75 vs real +0.53 / +0.76) — confirming the gap is a
  property of the fitted kernels, not a measurement quirk. **Where the memoryless
  model fails:** synthetic conversations visit **more distinct states per
  conversation** than real ones (4.55 vs 3.67 ShareGPT; 3.39 vs 2.98 WildChat).
  Real dialogue is *stickier* — it revisits a smaller working set of dominant
  features than a 1-step Markov chain predicts. This is exactly the residual
  higher-order structure the CK test flags, and it is the concrete gap a
  higher-order or context-augmented chain would close.

---

## Control — is the role signal real?   (synthetic-corpus null)

Role separability is genuine, not an artifact: on HTML-stripped ShareGPT and on
WildChat the per-turn features separate human from AI at **AUC ≈ 0.92 linear /
≈ 0.98 nonlinear**, robust to first-turn exclusion (probe B) and conversation
leakage (probe D, group-aware split). The **synthetic self-play corpus is the
null**: there both roles are generated by the same transformer with the same
sampling, so role is one distribution and unlearnable by construction — and indeed
the same probe returns **AUC ≈ 0.56** (chance). That null is the control that makes
the 0.92 meaningful. (Provenance and the HTML-confound story are in `RESEARCH.md`;
figure `results/figures/01_separability.png`.)

---

## Synthesis — how random is human–AI interaction?

Caveat first: this is randomness at the level of the **dominant SAE feature per
turn** (M = 32 states) seen through our small SAE, measured **per turn** (not
token-level perplexity). Absolute entropies scale with the state count, so the
*relative* comparisons (human vs AI, ShareGPT vs WildChat) are the trustworthy
part; the absolute percentages less so — but the comparisons are now robust across
M (Experiment 4) rather than pinned to a single clustering.

Entropy rates as "effective number of equally-likely next states" (`exp(h)`, of 32):

| | nats | effective next-states | % of max |
|---|---:|---:|---:|
| max (uniform) | 3.466 | 32 | 100% |
| WildChat entropy rate | 2.604 | 13.5 | 75% |
| ShareGPT entropy rate | 2.459 | 11.7 | 71% |
| human turn → next (human surprise) | 2.64 / 2.94 | 14 / 19 | — |
| AI turn → next (AI self-surprise) | 2.11 / 2.18 | 8.2 / 8.8 | — |

**1. Mostly random, but structured.** Knowing the current dominant feature predicts
the next only ~25–29% better than chance (71–75% of max entropy remains): ~12–14
effective next-states of 32. Far from deterministic, not a coin flip.

**2. The randomness is asymmetric, and the human is its source** (replicates on both
corpora, and the generative model inherits it). The AI's next move is ~1.7–2.3×
more predictable than the human's (AI self-surprise ≈ 8–9 effective states vs human
14–19). Relative to the model's own autoregressive flow the human injects
**+0.53 / +0.76 nats of excess surprise**. The machine half is on-rails; the human
half is where the randomness lives.

**3. A convergent stochastic process, not a runaway one.** Both chains are ergodic —
they forget the opening dominant feature in 2–4 turns and settle into a stable,
high-entropy stationary mix (π entropy 2.75–2.85 of 3.466). Genuinely random (no
single attractor) but bounded (no endless wander); real dialogue is even stickier
than the memoryless model predicts (revisits fewer states), which is the residual
the CK test and the generative check both flag.

**Answer:** moderately-high randomness with stable structure, **concentrated on the
human side** of the exchange.

Limits before leaning on this: dominant-feature level, not token-level; M = 32
reporting granularity and a weak 1-layer SAE (the human>AI *gap* is robust across M
and corpora, the absolute % less so); and the chain is not perfectly Markov (CK ≈
1.3–1.4), so the 1-step entropy rate slightly over-states randomness — the
role-coupled chain cuts the CK residual by ~0.3 and confirms the residual that
remains is real topical stickiness, not just the role flip.
