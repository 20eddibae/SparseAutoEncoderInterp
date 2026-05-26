# Brainstorm — exploring the transformer + SAE through the probability toolkit

> **Status (2026-05-26): historical exploration.** The project has since been
> simplified to **four experiments + one kept generative result**, all inside the
> course probability vocabulary (`CLAUDE.md`); see `RESULTS.md` / `RESEARCH.md`.
> What survived from the ideas below: the Poisson over-dispersion test (Exp 1), the
> WLLN/CLT block-mean check (Exp 2), role entropy + KL (Exp 3, now with a
> delta-method bar), and the turn-level Markov chain (Exp 4) — but with the **state
> redefined as the argmax (dominant) feature, an order statistic, not a clustering
> output (no KMeans, no K)**. What was dropped to stay in scope: the MGF Gaussianity
> fit, the Naive-Bayes-vs-logreg gap, the independence-surrogate counterfactual, the
> PCA/L²-projection idea, and the spectral framing — none are needed for the four
> experiments. The HTML-artifact triangulation (items 1–3 at the bottom) was already
> resolved (see `RESEARCH.md`, 2026-05-25): the genuine role AUC is ~0.92.

The SAE features are themselves a probability object: per turn we have a 4096-dim
sparse vector (`magnitudes` continuous + `support` binary) with `role`, `conv_id`,
and `turn_idx`. ~72–80 features active/turn (~1.8% density), 1568–1817 dead
features, one always-on, and role separability that is corpus-dependent
(synthetic ≈ 0.56, real ShareGPT ≈ 0.99 but inflated by an HTML-formatting
artifact). Almost every technique from the course maps onto a concrete experiment
on that object.

## Each class technique → a testable hypothesis about the SAE

**Conditional probability & Bayes (Apr 2–3).** Treat each feature *j* as an event.
Compute `P(active_j | human)` vs `P(active_j | ai)`; Bayes turns these into
`P(role | active_j)` — a per-feature "role detector" score. The logreg
coefficients (feature 1295 had coef +3.81, ~2× the next) are a linear stand-in.
→ *Hypothesis:* the role signal is carried by a handful of features, not spread
out. Test: rank features by likelihood ratio / Bayes factor; check how few you
need to recover most of the 0.99 AUC.

**Law of Total Probability / partitioning (Apr 2).** Decompose
`P(active_j) = Σ_t P(active_j | turn=t)·P(turn=t)`, or partition by "turn contains
HTML."
→ *Hypothesis (directly attacks our artifact):* conditioning on HTML-presence
collapses the role AUC. If `P(role | features, HTML-stripped)` drops sharply, the
0.99 was formatting; if it survives, there is genuine linguistic signal.

**Independence factorization & `E[g(X)h(Y)]=E[g]E[h]` (Apr 28–May 1).** The SAE
ideal is a disentangled basis — features should be near-independent. Test
pairwise: does `E[s_i s_j] = E[s_i]E[s_j]` hold for the support bits?
→ *Hypothesis:* features are strongly *dependent* (co-activation cliques),
revealing feature "circuits." Measure pairwise mutual information / the
co-activation matrix; dense off-diagonal structure = entanglement.

**Sums of independent RVs, MGFs, Poisson-as-limit (Apr 9, 30).** Active-feature
count per turn = sum of 4096 indicator variables. If features fired
independently, the count is Poisson-Binomial ≈ Poisson (mean ≈ 72.5).
→ *Hypothesis:* the empirical active-count distribution is **over-dispersed** vs
Poisson(72.5) → quantitative evidence of dependence. Compare variance to mean; a
variance ≫ mean rejects independence.

**CLT (May 14).** The logreg score is a weighted sum of many features → per class
it is approximately Gaussian, so AUC ↔ separation of two Gaussians, summarized by
d′ = (μ₁−μ₀)/σ.
→ *Hypothesis:* compute d′ for synthetic vs real; the artifact should show as one
or two features with enormous individual d′ rather than many small ones (a "thin"
vs "broad" signal).

**Markov / Chebyshev inequalities (May 12).** Use a concentration bound to set a
principled "dead feature" threshold: a feature with true rate p fires ≥1 time in
N turns with boundable probability.
→ *Hypothesis:* the ~1568–1817 dead features are *genuinely* dead (rate below the
Chebyshev bound), not merely rare — i.e., the SAE over-allocated capacity. This
tells you the effective dictionary size.

**Jensen / convexity (HW11).** The config uses `turn_pooling: max`. Jensen:
`E[max] ≥ max E` — max-pooling is exactly the convex-amplification of rare spikes.
→ *Hypothesis:* max-pooling **inflates the HTML artifact** (one `<div>` token
spikes a feature). Test mean- vs max-pooling on separability; if the artifact
theory holds, max-pool's AUC drops more than mean-pool's once HTML is stripped.

**MLE / Naive Bayes vs logreg.** Naive Bayes = MLE under the feature-independence
assumption.
→ *Hypothesis:* the gap (logreg AUC − Naive-Bayes AUC) *quantifies how much
feature dependence matters* for role. Small gap → independence is fine; large gap
→ the circuits in the independence test above are doing real work.

**Markov chains & Chapman-Kolmogorov (May 19).** Add the time axis we have
ignored: model the sequence of dominant features across `turn_idx` within a
conversation as a Markov chain.
→ *Hypothesis:* human-turn and ai-turn transition matrices differ; the stationary
distribution = "typical feature usage" per role. Chapman-Kolmogorov lets you test
the Markov assumption (do 2-step transitions equal the matrix squared?).

**Monte Carlo & permutation (May 14).** Build a null for separability: shuffle
`role` labels, recompute AUC many times → null distribution; bootstrap turns for a
CI on the real AUC.
→ *Hypothesis:* 0.99 is far outside the permutation null (trivially significant),
but the *honest* question is the effect size on HTML-stripped data — use the same
machinery there.

**Order statistics & min/max (Apr 16).** With `coarsen.k=5`, each turn is
summarized by its top-5 features = order statistics of the activation vector.
→ *Hypothesis:* the distribution of the per-turn *max* activation differs by role
(the HTML spike again). Plot the CDF of the max activation, split by role.

**Inverse transform / simulation (May 7).** Generate synthetic feature vectors
matching the empirical *marginals* but forcing independence (sample each feature
from its own fitted CDF).
→ *Hypothesis:* if separability on these independence-preserving surrogates ≈ real
separability, role lives in marginals; if it drops, role lives in the *dependence*
structure. A clean counterfactual.

**Expectation as L² projection + covariance.** `E[X]=argmin_c E[(X−c)²]` is the
same least-squares the SAE *decoder* solves when reconstructing activations.
→ *Hypothesis:* PCA on the feature covariance finds a dominant "role axis"; check
whether it aligns with the logreg direction and with reconstruction-error
directions.

## Where to start, given what we know

The HTML artifact is the live problem, so the highest-leverage experiments are the
ones that isolate signal from formatting:

1. **Partition-by-HTML** (Law of Total Probability) — re-extract on HTML-stripped
   text and recheck the role AUC.
2. **Mean-vs-max pooling** (Jensen) — does the artifact's contribution shrink
   under mean-pooling?
3. **Independence-surrogate counterfactual** (inverse transform) — is role in the
   marginals or the dependence structure?

Together they answer "is the 0.99 real?" using three different tools from the
course — the kind of triangulation the "local vs. global" theme rewards.
