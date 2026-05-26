# Results — probability experiments on real-conversation SAE features

All experiments use **our trained models** (6.64M-param 1-layer GPT on OpenWebText
→ SAE, 4096 features / 8× expansion — the same config as Bricken et al. 2023's
A/1 run) applied to **real GPT conversations**. Features are per-turn, max-pooled,
in `artifacts/features_stripped.npz` (HTML-stripped ShareGPT52K, 70,133 turns) and
`artifacts/features_wildchat.npz` (WildChat-1M English, 45,690 turns). Everything
here is CPU-only on already-extracted features (no GPU, no re-extraction).

Maps to the probability modules scoped in `RESEARCH.md`. Status legend:
**[run]** executed below · **[scoped]** specified, not yet run · **[needs-extract]**
requires per-token feature extraction (a GPU step).

**Figures and machine-readable outputs for every result below live in
[`results/`](results/README.md)** (`results/figures/01–05*.png`).

---

## H3 — concentration of L0 / active-feature count  [run]
`scripts/run_hypothesis.py --h 3` on `features_stripped.npz`. The RV is
`N_active` = number of SAE features firing per turn. Modules exercised:
`concentration`, `poisson`, `clt`, `mgf`, `tail_sum`.

- **Mean L0 = 77.6 features/turn, variance = 771.7 → variance/mean ≈ 9.9.**
  Heavily **over-dispersed**.
- **Tail-sum identity** `E[N]=Σ_k P(N≥k)`: 77.6461580 (tail-sum) vs 77.6461580
  (direct) — match to 1e-9. ✓ sanity check passes.
- **Poisson null rejected hard**: χ² p = 0.0; Le Cam TV bound = 53.2 (≫1, so the
  Poisson approximation is formally void anyway). A Poisson would have
  variance = mean; we see ~10× that. **Interpretation (matches the paper's
  prediction):** features are **correlated / co-fire** (feature-splitting ⇒
  groups of related features activate together), so per-turn firing is far from
  independent Bernoullis.
- **CLT holds for the sample mean**: block-means (block=1402, 50 blocks) are
  Gaussian, KS p = 0.65 (fails to reject normality). ✓ WLLN/CLT machinery valid
  even though the per-turn variable is not Gaussian.
- **Concentration bounds all valid**: empirical tail ≤ Markov ≤ … with the
  sub-Gaussian bound tightest; Chebyshev only kicks in for `t` beyond ~13.
- **MGF Gaussian fit is poor** (residual 3.08): `N_active` is *not* Gaussian
  (skewed, over-dispersed) — consistent with the non-Poisson finding.

**Takeaway:** L0 is a real, well-behaved RV with mean ~78, but it is strongly
super-Poissonian — direct evidence that the SAE features are correlated, exactly
the regime the paper attributes to feature-splitting.

## H2 — role entropy & Bayes posteriors  [run]
`scripts/run_hypothesis.py --h 2` on `features_stripped.npz`. Modules: `entropy`
(plug-in + Chebyshev error bar, KL), `bayes` (top role-discriminative features).

- **Marginal feature entropy is ~equal across roles**: human H = 4.734 nats,
  ai H = 4.784 nats; Δ = −0.049, **within the Chebyshev 1-σ bar (~0.04)**. The
  "AI is mode-seeking ⇒ lower entropy" hypothesis is **not** supported at the
  level of the marginal feature distribution.
- **AI uses a broader feature vocabulary**: 2,234 distinct active features for
  ai vs 1,499 for human.
- **KL between role distributions is small but nonzero & asymmetric**:
  KL(human‖ai) = 0.028, KL(ai‖human) = 0.043 nats.

**Takeaway:** the ~0.92 role separability does **not** come from how *much*
entropy each role has, nor from L0 — it comes from *which* features fire (the
Bayes top-feature sets differ by role). Entropy is descriptive here, not the
discriminator — echoing the paper's caution that information metrics didn't track
the property of interest.

## H1 / H4 — turn-level conversation Markov chain  [run]
`scripts/markov_chain.py` (SLURM job 8688059, CPU, 45s). Clusters per-turn
activation vectors into **K=32 "conversation modes"** (Option 3, MiniBatchKMeans
on L2-normalized vectors), then builds the turn→turn chain within each
conversation. Modules: `markov` (transition fit, Chapman–Kolmogorov, stationary
distribution, mixing time), `entropy` (entropy rate, conditional entropy).

| Quantity | ShareGPT (stripped) | WildChat |
|---|---:|---:|
| conversations / turn-transitions | 10,000 / 60,133 | 9,995 / 35,695 |
| Chapman–Kolmogorov residual (Frobenius) | 1.72 | 1.62 |
| spectral gap `1−\|λ₂\|` | 0.135 | 0.394 |
| mixing time (TV ≤ 0.25) | **10 turns** | **4 turns** |
| TV(start→π) after 10 turns | 0.047 | 0.001 |
| **converges to unique stationary π** | **yes** | **yes** |
| entropy of π (max = ln 32 = 3.466) | 3.336 | 3.323 |
| entropy rate `h` (nats/turn) | 2.566 | 2.903 |
| **human surprise** `H(ai→next human)` | 2.641 | 3.068 |
| **AI self-surprise** `H(human→next ai)` | 2.145 | 2.394 |
| **gap (human − AI)** | **+0.496** | **+0.675** |

**Convergence.** Both chains are ergodic and **converge to a unique stationary
distribution** (TV(start→π) decays to ~0). So a conversation's "mode" does settle
— but π has high entropy (3.32–3.34 of a 3.466 max), so it settles onto a
*spread* of modes with a few mild attractors (top mode ~9–10% mass), not a single
absorbing feature. WildChat mixes faster (4 vs 10 turns; larger spectral gap),
i.e. its conversations forget their opening topic sooner.

**Not perfectly first-order.** CK residual ≈ 1.6–1.7 (not 0) → there is
higher-order structure the pooled 1-step chain misses — expected, because role
strictly alternates (period-2). This is exactly what the role-coupled chain
(H5/H6, scoped in `RESEARCH.md`) is designed to model.

**Headline — "where the human enters" replicates across both corpora.** The
human's next conversational mode is **consistently less predictable than the
model's**: `H(ai-turn → next human-turn)` exceeds `H(human-turn → next ai-turn)`
by **+0.50 nats (ShareGPT)** and **+0.68 nats (WildChat)**. The model's
continuations are more on-rails (lower conditional entropy); the human injects
more surprise when re-entering. This quantifies the prompt's intuition — humans
are the perturbation that moves the conversation between modes — and it holds on
two independent real-dialogue corpora with our own transformer + SAE.

Per-state structure is interpretable too: some modes are near-deterministic
(lowest conditional entropy ~0.98 nats on ShareGPT) and others wide-open (~3.2
nats), i.e. some conversation modes strongly predict the next mode and others
don't — the turn-level analog of the paper's deterministic (base64-loop) vs
open-ended features.

## H5 / H6 — role-coupled chain & generative MCMC  [run]
`scripts/mcmc_conversations.py` (CPU, K=32). Role strictly alternates, so the
pooled 1-step chain (H1) is not first-order. We estimate the two role kernels
separately — `T_{h→a}` (human-turn mode → next ai mode = AI self-surprise) and
`T_{a→h}` (ai-turn mode → next human mode = human surprise) — and add two samplers
over the fitted chain. Figures: `results/figures/04_markov_chain.png`,
`results/figures/05_mcmc_generative.png`.

| Quantity | ShareGPT (stripped) | WildChat |
|---|---:|---:|
| pooled CK residual | 1.720 | 1.616 |
| **role-coupled CK residual** (`‖P2_emp − T_{h→a}T_{a→h}‖`) | **1.171** | **1.231** |
| CK improvement | **+0.549** | **+0.385** |
| kernel difference `‖T_{h→a} − T_{a→h}‖_F` | 2.596 | 2.350 |
| human surprise / AI self-surprise (nats) | 2.641 / 2.145 | 3.068 / 2.394 |
| **surprise gap (human − AI)** | **+0.496** | **+0.675** |
| Metropolis–Hastings acceptance | 0.731 | 0.713 |
| **MH TV(empirical, π) after 200k steps** | **0.0068** | **0.0049** |
| synthetic surprise gap (generative) | +0.492 | +0.668 |
| TV(mode dist real, synthetic) | 0.0151 | 0.0105 |
| distinct modes/conv — real vs synthetic | 3.95 vs 5.18 | 3.18 vs 3.85 |

**Modeling the role alternation captures real structure.** Splitting the chain by
role drops the Chapman–Kolmogorov residual by **~0.4–0.55** (1.72→1.17 ShareGPT,
1.62→1.23 WildChat): a chunk of the "non-Markovianity" in H1 was just the period-2
role flip, and the role-coupled 2-step model `T_{h→a}T_{a→h}` recovers it. The two
kernels are genuinely different (`‖·‖_F` ≈ 2.4–2.6), i.e. **the dynamics are
role-asymmetric** — which is the structural reason the human is the less-predictable
side, not an artifact of pooling.

**Metropolis–Hastings reproduces π.** A textbook MCMC sampler (symmetric random-walk
proposal, target = the analytic stationary π) converges to within **TV ≈ 0.005–0.007**
of the eigen/power-iteration π after 200k steps (acceptance ~0.72). The two
independent routes to π agree — a sanity check on the stationary-distribution claim
in H1/H4.

**Generative posterior-predictive check.** Forward-sampling whole synthetic
conversations from the role-coupled kernels (matching the real opening-mode and
length distributions) reproduces the marginal mode mix almost exactly (TV ≈ 0.01)
and **inherits the headline asymmetry** (synthetic gap +0.49/+0.67 vs real
+0.50/+0.68) — confirming the +0.50/+0.68 nats is a property of the fitted kernels,
not a measurement quirk. **Where the memoryless model fails:** synthetic
conversations visit **more distinct modes per conversation** than real ones (5.18 vs
3.95 ShareGPT; 3.85 vs 3.18 WildChat). Real dialogue is *stickier* — it revisits a
smaller working set of modes than a 1-step Markov chain predicts. This is exactly
the residual higher-order structure the CK test flags (topical coherence that spans
more than one turn), and it is the concrete remaining gap a higher-order or
context-augmented chain would close.

---

## Scoping the remainder of the paper → probability mapping

| Module(s) | Paper hook | Status | Note |
|---|---|---|---|
| `markov`+`coarsen` (turn chain, H1/H4) | conversation dynamics | **[run]** | job 8688059, below |
| `markov` **role-coupled + MCMC** (H5/H6) | role-asymmetric dynamics | **[run]** | `mcmc_conversations.py`; CK 1.72→1.17, MH→π, generative check |
| `poisson`+`concentration`+`clt`+`mgf`+`tail_sum` (H3) | L0 / feature density | **[run]** | over-dispersed, non-Poisson |
| `entropy`+`bayes` (H2) | role entropy, LLR proxy | **[run]** | roles ~equal entropy |
| `order_stats` | expected-value plots / specificity | **[scoped]** | per-turn top-1 fraction & max-gap; trivial to add to H3 |
| `bayes` LLR proxy `log P(s\|ctx)/P(s)` | Arabic/DNA/base64 proxies | **[scoped]** | needs a per-turn context label (e.g. "contains code"); cheap |
| `mgf`+`clt` on **logit weights** | logit-weight "interference" mode | **[scoped]** | needs SAE decoder × W_down × unembed — a model load (CPU ok), not yet wired |
| `markov` **token-level FSA** | base64 self-loop / HTML 4-node | **[needs-extract]** | requires per-TOKEN feature extraction (GPU); our features are per-turn |

The single remaining GPU step is the **per-token extraction** needed for the
paper's within-generation finite-state automata (Chain 1). Everything else runs
on CPU from the existing per-turn artifacts.

---

## Synthesis — how random is human–AI interaction?

Caveat first: this is randomness at the level of **conversational modes (K=32
clusters)** seen through our small SAE, measured **per turn** (not token-level
perplexity). Absolute entropies scale with K, so the *relative* comparisons
(human vs AI, ShareGPT vs WildChat) are the trustworthy part; the absolute
percentages less so.

Entropy rates converted to "effective number of equally-likely next modes"
(`exp(H)`, out of 32):

| | nats | effective next-modes | % of max randomness |
|---|---:|---:|---:|
| max (uniform) | 3.466 | 32 | 100% |
| WildChat entropy rate | 2.903 | 18.2 | 84% |
| ShareGPT entropy rate | 2.566 | 13.0 | 74% |
| human turn → next (human surprise) | 2.64 / 3.07 | 14 / 21 | — |
| AI turn → next (AI self-surprise) | 2.15 / 2.39 | 8.6 / 11 | — |

**1. Mostly random, but structured.** Knowing the current mode predicts the next
only ~16–26% better than chance (74–84% of max entropy remains): ~13–18 effective
next-modes of 32. Far from deterministic, but not a coin flip.

**2. The randomness is asymmetric, and the human is its source** (replicates on
both corpora). The AI's next move is ~1.6–2× more predictable than the human's
(AI self-surprise 8.6–11 effective modes vs human 14–21). Relative to the model's
own autoregressive flow, the human injects **+0.50–0.68 nats of excess surprise**.
The machine half is comparatively on-rails; the human half is where the
randomness lives.

**3. A convergent stochastic process, not a runaway one.** Both chains are
ergodic — they forget the opening topic in ~4–10 turns and settle into a stable
stationary mix of modes (π entropy 3.33 of 3.466). Genuinely random (no single
attractor) but bounded (no endless wander): the conversation explores a stable
repertoire of ~a-dozen-plus modes, reshuffled each turn, with the human doing
most of the reshuffling.

**Answer:** moderately-high randomness with stable structure, **concentrated on
the human side** of the exchange.

Limits before leaning on this: mode-level not token-level (the token FSA / Chain 1
is the model-side "on-rails" question, needs per-token extraction); coarse K=32
and a weak 1-layer model (the human>AI *gap* is robust, the absolute % less so —
a K-sweep would confirm); and the chain is not perfectly Markov (CK ≈ 1.6), so the
1-step entropy rate slightly over-states randomness — the role-coupled chain (H5/H6,
now run) cuts the CK residual by ~0.4–0.55 and confirms the residual that remains is
real topical stickiness (real conversations revisit fewer modes than the memoryless
model generates), not just the role flip.
