# IV. Inference

Estimating the point is half the work; the other half is quantifying the
uncertainty. `skxperiments` offers three inference paths with distinct
interpretations (randomization, Neyman, bootstrap), plus the correction for
multiple testing and power planning. This section derives what each one does.

---

## 13. Randomization-based inference (Fisher's sharp null)

### Intuition

We do not assume the data follow a normal. We assume the treatment had no effect
at all and ask: given that scenario, how rare would it be to observe a difference
as large as the one we observed, purely because of the draw? If it is very rare,
we reject the "zero effect" hypothesis.

### Formalization: the sharp null and the randomization distribution

Fisher's **sharp null** states that the effect is zero for **every** unit:

$$H_0:\quad Y_i(1) = Y_i(0) \quad \text{for all } i.$$

Under `H_0`, each unit's observed outcome would not depend on the arm: the value
`Y_i` is fixed. The only random thing is **which** allocation came out. So we
can, in principle, recompute the test statistic (for example, the difference in
means) for **each** of the `C(N, n_T)` possible allocations, holding the `Y_i`
fixed and just swapping the labels. This generates the **randomization
distribution** of the statistic under the null, with no distributional
assumption. The p-value is the fraction of that distribution as extreme as or
more extreme than the observed one.

When the number of allocations is too large to enumerate, we sample `m`
allocations by Monte Carlo. The p-value uses the Phipson & Smyth (2010)
correction:

$$p = \frac{B + 1}{m + 1},$$

where `B` is the number of permutations as extreme as or more extreme than the
observed one. The `+1` in the numerator and denominator includes the **observed
allocation itself** as one of the valid possibilities under the null. Without
that correction (`p = B/m`), the p-value could come out exactly zero, which is
impossible in a permutation test and causes serious errors in multiple-testing
analyses.

### Worked example

Ten rats, five receive the new feed (treatment) and five the old (control).
Under the sharp null, each rat's weight would be the same in either arm. Shuffle
the ten observed weights, draw five into a "fake" treatment group, and compute
the difference in means. Repeat `m = 1000` times. If in only `B = 4` of those
1000 draws the "fake" difference was as extreme as or more extreme than the real
one, the p-value is

$$p = \frac{4 + 1}{1000 + 1} \approx 0.005.$$

Since `0.5%` is rare, we reject the null.

> **In the library.** `RandomizationTest(estimator, n_permutations, ...)` uses
> exactly `(1 + n_extreme) / (1 + n_permutations)`. `draw()` re-randomizes by the
> same mechanism as the design (respecting blocking and re-randomization). The
> two-sided criterion is `|T_perm| ≥ |T_obs|`.

---

## 14. Neyman variance and the Wald interval

### Intuition

The Neyman variance measures the uncertainty of the effect considering only the
units in the experiment (finite population). It is **conservative** for a precise
reason: part of the true variance depends on something we never observe, and the
Neyman formula handles this by overestimating, so as not to promise more
precision than the data support.

### Formalization: the exact variance and why we drop a term

Under complete randomization, the exact variance of the difference-in-means
estimator for the SATE is (Neyman, 1923):

$$
\operatorname{Var}(\hat{\tau})
= \frac{S_1^2}{n_T} + \frac{S_0^2}{n_C} - \frac{S_{\tau}^2}{N},
$$

where:

- `S_1²` is the variance of the `Y_i(1)` across the `N` units,
- `S_0²` is the variance of the `Y_i(0)`,
- `S_τ²` is the variance of the **individual effects** `δ_i = Y_i(1) - Y_i(0)`.

The problem: `S_τ²` involves `Y_i(1)` and `Y_i(0)` **of the same unit**, which we
never observe together (the fundamental problem of [section I](01-foundations.md)).
So `S_τ²` is **not identifiable**. The Neyman estimator simply **drops** that
term:

$$\hat{V}_{\text{Neyman}} = \frac{s_T^2}{n_T} + \frac{s_C^2}{n_C},$$

with `s²` the sample variances within each arm (`ddof = 1`). Since `S_τ² ≥ 0`,
dropping the negative term makes `V̂`, on average, **greater than or equal to**
the true variance. Hence the name "conservative". There is a case where it is
**exact**: when the effect is **constant** (`δ_i = c` for all `i`), we have
`S_τ² = 0` and nothing is lost.

For **blocked** designs, the stratified form, consistent with the block weighting
of [topic 9](03-estimation.md), is

$$\hat{V} = \sum_b \left(\frac{N_b}{N}\right)^2 \hat{V}_b.$$

The Wald interval is

$$\hat{\tau} \pm z_{1-\alpha/2}\,\sqrt{\hat{V}},$$

with `z` from the **normal** (1.96 for 95%).

### Worked example: constant vs. heterogeneous effect

Scenario A (constant effect): the drug speeds recovery by exactly 2 days for
everyone. Then `S_τ² = 0` and the Neyman variance is exact; the CI has the
"right" width.

Scenario B (heterogeneous effect): the drug speeds recovery by 10 days for some
and delays it by 2 days for others. Now `S_τ² > 0`, and since the Neyman formula
ignores the `-S_τ²/N` term, it overestimates the variance. The CI comes out
**wider** than necessary: the method refuses to declare high precision when there
is hidden instability in the individual effects. This is a feature, not a defect.

> **In the library.** `NeymanCI(estimator)` accepts `DifferenceInMeans` and
> `BlockedDifferenceInMeans`, uses the **normal** quantile (not `t`) and the
> stratified form in the blocked case. It is conservative for the SATE (finite
> population) and approximately unbiased for the PATE (superpopulation), because
> in the superpopulation the covariance term of the potentials enters in a
> different form.

---

## 15. Bootstrap: percentile vs. BCa

### Intuition

The bootstrap approximates the sampling distribution of almost any statistic by
resampling the data itself with replacement, with no closed-form formula and no
normality. It is the **superpopulation** reading: we treat the observed units as
a sample from a larger population.

### Formalization

From the data, `B` resamples (with replacement) are generated and the statistic
is recomputed in each, obtaining the bootstrap distribution `θ̂^*`.

- **Percentile**: the CI is the `α/2` and `1 - α/2` quantiles of `θ̂^*` directly.
- **BCa** (*bias-corrected and accelerated*): adjusts the cut points by two
  parameters:
  - `z_0` (bias correction): `Φ⁻¹` of the **fraction of bootstrap replicates
    below the observed estimate**. It measures the median bias.
  - `a` (acceleration): captures the skewness, estimated by leave-one-out
    jackknife.

  The adjusted percentiles are

  $$\alpha_{\text{adjusted}} = \Phi\!\left(z_0 + \frac{z_0 + z^{(\alpha)}}{1 - a\,(z_0 + z^{(\alpha)})}\right),$$

  and the CI limit is the corresponding quantile of the bootstrap distribution.
  When `z_0 = 0` and `a = 0` (symmetric distribution, no bias), the BCa coincides
  with the percentile.

### Worked example

Suppose you want a CI for the correlation between entrance-exam score and college
performance, with only 15 students. The correlation is bounded to `[-1, 1]` and
its sampling distribution is skewed at that small `n`, so the normal
approximation of the t test is poor. You draw 1000 resamples of 15 students (with
replacement), compute the correlation in each, and use the BCa to correct the
bias and the skewness. If the observed correlation of `0.77` is pulled by an
outlier, the BCa CI reflects that instability more honestly than a Gaussian CI.

> **In the library.** `BootstrapCI(estimator, method="bca"|"percentile", ...)`
> **resamples within each arm** (and within block×arm), preserving the margins of
> the design. It is not a generic IID bootstrap: for experiments, resampling by
> arm is the correct scheme. It accepts any scalar estimator (including `Lin` and
> `CUPED`, which gives a CI for them). The degenerate BCa case (when all or none
> of the replicates fall below the observed value, leaving `z_0` undefined)
> raises an error suggesting `method="percentile"`.

---

## 16. Multiple testing: FWER vs. FDR

### Intuition

Testing many hypotheses inflates the chance that a false positive appears purely
by chance. Under independence, the probability of **at least one** false positive
in `m` tests at level `α` is

$$1 - (1 - \alpha)^m.$$

For `α = 0.05` and `m = 10`, this is already `1 - 0.95^{10} \approx 0.40`: a 40%
chance of a false discovery. The correction controls that risk, with two
philosophies.

### Formalization

- **FWER** (*family-wise error rate*): probability of at least one false positive
  in the family.
  - **Bonferroni**: reject if `p ≤ α/m`.
  - **Holm** (step-down): order `p_(1) ≤ ... ≤ p_(m)`; compare `p_(i)` to
    `α/(m - i + 1)`; reject while it passes and stop at the first that fails. Holm
    is uniformly more powerful than Bonferroni and controls the same FWER.
- **FDR** (*false discovery rate*): expected proportion of false positives among
  the rejections.
  - **Benjamini-Hochberg (BH)**: order the p-values; find the largest `i` with
    `p_(i) ≤ (i/m)\,q`; reject all up to that `i`.

### Worked example: five tests

Ordered p-values `[0.008, 0.012, 0.020, 0.041, 0.300]`, with `α = q = 0.05`,
`m = 5`.

- **Bonferroni** (threshold `0.05/5 = 0.01`): only `0.008` passes. **1
  rejection.**
- **Holm**: `p_(1)=0.008 ≤ 0.05/5=0.010` (reject); `p_(2)=0.012 ≤ 0.05/4=0.0125`
  (reject); `p_(3)=0.020 ≤ 0.05/3=0.0167`? No, `0.020 > 0.0167`, **stop**.
  **2 rejections.**
- **BH**: largest `i` with `p_(i) ≤ (i/5)·0.05`: `i=1` `0.008≤0.010` ✓; `i=2`
  `0.012≤0.020` ✓; `i=3` `0.020≤0.030` ✓; `i=4` `0.041≤0.040` ✗; `i=5`
  `0.300≤0.050` ✗. The largest valid `i` is `3`, so reject `p_(1..3)`.
  **3 rejections.**

The same set yields 1, 2 or 3 discoveries depending on the method: Bonferroni is
the most conservative, BH the most powerful.

> **In the library.** `MultipleTestingCorrection(method=..., alpha=...)` with
> `"holm"` (default), `"bonferroni"` and `"bh"`. `ExperimentComparison` applies
> the correction over a family of experiments. BH assumes positive dependence
> (the typical case); the Benjamini-Yekutieli variant for arbitrary dependence is
> a v2 item.

### When to use each

Use **FWER** when a single false positive is costly (critical decisions,
regulatory filings). Use **FDR** in exploration with many metrics, accepting a
small proportion of errors in exchange for more power.

---

## 17. Power analysis

### Intuition

Planning to have enough units to see the effect you are after. Four quantities
are tied together: fix three and the fourth is determined.

- Sample size `n`.
- Minimum detectable effect (MDE, `δ`).
- Power `1 - β` (chance of detecting an effect that exists), by convention 80% or
  90%.
- Level `α` (chance of a false positive), by convention 5%.

### Formalization

For a test of the difference of two means, the effect detectable at a given power
is

$$\delta = (z_{1-\alpha/2} + z_{1-\beta})\,\sigma\,\sqrt{\tfrac{1}{n_T} + \tfrac{1}{n_C}}.$$

Isolating the size with equal allocation (`n` per arm), one arrives at

$$n_{\text{per arm}} = \frac{2\,\sigma^2\,(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}.$$

With `α = 0.05` (`z = 1.96`) and power 80% (`z = 0.84`), `(1.96 + 0.84)^2 ≈ 7.85`,
and `2 × 7.85 ≈ 16`, which gives the rule of thumb

$$n_{\text{per arm}} \approx \frac{16\,\sigma^2}{\delta^2}.$$

### Worked example

MDE `δ = 0.2`, `σ = 1`, `α = 0.05` two-sided, power 80%, allocation 50/50.

$$
n_{\text{per arm}} \approx \frac{16 \cdot 1}{0.04} = 400,
\qquad n_{\text{total}} \approx 800.
$$

The exact form (the one the library uses) with `σ_eff = σ\sqrt{1/0.5 + 1/0.5} = 2σ`
rounds the result up:

$$n_{\text{total}} = \left\lceil \left(\frac{(z_{1-\alpha/2} + z_{1-\beta})\cdot 2}{0.2}\right)^2 \right\rceil.$$

With the rounded quantiles (`1.96` and `0.84`) the argument is `(2.80\cdot 2/0.2)^2
= 28^2 = 784`. With the exact quantiles (`1.95996` and `0.84162`) it is
`28.016^2 \approx 784.9`, and the ceiling gives `n_total = 785` (392 treated, 393
control), close to the `800` of the rule of thumb (the `16` rounds `15.7`).

> **In the library.** `power_analysis(n=, mde=, power=, std=, alpha=, allocation=, ...)`
> solves for one of the quantities given the others, by the **exact** form with
> the normal quantiles (the `16σ²/δ²` rule is just a quick check). Scope v1: two
> groups, continuous outcome, normal approximation. `σ²` usually comes from
> historical data or from an A/A test (see [V. Diagnostics](05-diagnostics.md)).

---

Related notebooks:
[`02_inference_three_ways`](../../../examples/for_starters/en/02_inference_three_ways.ipynb),
[`07_many_tests`](../../../examples/for_starters/en/07_many_tests.ipynb),
[`09_putting_it_together`](../../../examples/for_starters/en/09_putting_it_together.ipynb).
