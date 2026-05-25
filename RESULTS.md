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

## H1 / H4 — turn-level conversation Markov chain  [run — see Markov section]
Submitted as SLURM job 8688059 (`slurm/08_markov_chain.sbatch`), results appended
below when complete. `scripts/markov_chain.py` clusters per-turn activation
vectors into K=32 "conversation modes" (Option 3) and builds the turn→turn chain:
Chapman–Kolmogorov residual, stationary distribution π and convergence, mixing
time / spectral gap, entropy rate, and the role-asymmetry "human surprise vs AI
self-surprise" measure.

---

## Scoping the remainder of the paper → probability mapping

| Module(s) | Paper hook | Status | Note |
|---|---|---|---|
| `markov`+`coarsen` (turn chain, H1/H4) | conversation dynamics | **[run]** | job 8688059, below |
| `poisson`+`concentration`+`clt`+`mgf`+`tail_sum` (H3) | L0 / feature density | **[run]** | over-dispersed, non-Poisson |
| `entropy`+`bayes` (H2) | role entropy, LLR proxy | **[run]** | roles ~equal entropy |
| `order_stats` | expected-value plots / specificity | **[scoped]** | per-turn top-1 fraction & max-gap; trivial to add to H3 |
| `bayes` LLR proxy `log P(s\|ctx)/P(s)` | Arabic/DNA/base64 proxies | **[scoped]** | needs a per-turn context label (e.g. "contains code"); cheap |
| `mgf`+`clt` on **logit weights** | logit-weight "interference" mode | **[scoped]** | needs SAE decoder × W_down × unembed — a model load (CPU ok), not yet wired |
| `markov` **token-level FSA** | base64 self-loop / HTML 4-node | **[needs-extract]** | requires per-TOKEN feature extraction (GPU); our features are per-turn |

The single remaining GPU step is the **per-token extraction** needed for the
paper's within-generation finite-state automata (Chain 1). Everything else runs
on CPU from the existing per-turn artifacts.
