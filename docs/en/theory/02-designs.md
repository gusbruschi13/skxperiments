# II. Designs

The design is the decision of **how** to assign the treatment. It determines the
comparability of the groups and, consequently, which inference is valid
afterward. This section covers the completely randomized design, blocking,
re-randomization, and the factorial design, with the mathematics of each and
numerical examples.

---

## 4. CRD (Completely Randomized Design)

### Intuition

It is the simplest design: units are assigned at random, with no imposed
structure. Randomization works as statistical insurance, since it distributes
equally, on average, all external factors (known or not) across the arms.

### Formalization: complete randomization vs. Bernoulli

There are two common ways to randomize two arms, and the distinction matters:

1. **Bernoulli**: each unit is treated independently with probability `p`. The
   number of treated units is random, `n_T \sim \text{Binomial}(N, p)`.
2. **Complete randomization**: the number of treated `n_T` is fixed and we draw
   **which** subset of size `n_T` receives the treatment, with all

   $$\binom{N}{n_T}$$

   subsets equally likely.

Complete randomization has two practical advantages: it guarantees the size of
each arm (without the risk of an unlucky Bernoulli draw leaving a tiny arm) and
it has slightly smaller variance for the difference in means. The set of
possible draws (the `C(N, n_T)` allocations) is also the basis of
randomization-based inference (see [IV. Inference](04-inference.md), topic 13).

In the classical DoE tradition, the analysis of a CRD is usually presented via
an additive model `y_{ti} = η + τ_t + ε_{ti}` and the ANOVA F test, under normal
and homoscedastic errors. That is a **parametric** and historical framing.

> **In the library (important).** Two points of alignment:
> 1. `skxperiments` does **complete randomization with a fixed number of
>    treated**: `CRD(p=0.5)` resolves `n_treated = round(p · N)` and draws
>    exactly that many; it is not Bernoulli per unit. Use `CRD(p=...)` or
>    `CRD(n_treated=...)`, mutually exclusive.
> 2. The library's inference is **randomization-based** or built on the Neyman
>    variance, and it does **not** assume normal errors nor use the ANOVA F test.
>    The additive model above is context, not a library assumption.

### Worked example

With `N = 8` units and `n_T = 4`, there are `C(8,4) = 70` possible allocations,
all with the same probability `1/70`. Randomization picks one of them. The
"validity" of the experiment comes precisely from that space of 70 allocations:
the distribution of the test statistic under the null is obtained by sweeping
those allocations (in `RandomizationTest`, by Monte Carlo sampling when the
number is large).

### When to use and when not to

Use the CRD when the units are homogeneous and there is no information to group
them; it is the starting point of any A/B test. Avoid it when there is a known
source of variation that could be controlled by blocking, or when the background
noise is so high that the CRD becomes insensitive to real effects.

---

## 5. Blocking and stratification

### Intuition

The rule of thumb is "block what you can, randomize what you cannot". If a known
categorical variable (region, device, batch) explains a good part of the
variation in the outcome, we group the units by it into **blocks** and randomize
**within** each block. That way the between-block variation leaves the
experimental error, and the test becomes more sensitive.

### Formalization: why precision increases

Decompose the total variance of the outcome into a **between-block** part and a
**within-block** part:

$$\sigma^2_{\text{total}} = \sigma^2_{\text{between}} + \sigma^2_{\text{within}}.$$

In the CRD, all the variation (including the between-block part) enters the
estimator's error. In blocking, we compare treatment and control **within** each
block, so the between-block variation is removed from the error. The precision
gain is larger the more the block variable explains the outcome (the larger
`sigma²_between`). In the limit where each block is a pair (one treated and one
control unit that are very similar), we have the **matched-pairs** design.

The estimator combines the per-block effects into a weighted average (topic 9).
One caveat: the classical additive model `y = η + β_i + τ_t + ε` assumes the
treatment effect is the same in all blocks (no treatment×block interaction). The
library's weighted estimator does **not** depend on that assumption: it
estimates the size-weighted average of the per-block effects, whatever the
heterogeneity across blocks.

### Worked example

Suppose two regions with very different baselines. Region A has a typical outcome
around 0, region B around 3, and the true effect is `+0.5` in both. In a naive
CRD, the enormous A vs. B difference enters the error and can "drown" the effect
of `0.5`. Blocking by region, we estimate `+0.5` within A and `+0.5` within B,
and combine: the A vs. B noise never enters the calculation, and the final
estimate is far more precise. This is exactly the scenario of the `05_blocking`
notebook.

> **In the library.** `BlockedCRD(block_col=..., p=...)` randomizes within each
> block preserving the proportion; `BlockedDifferenceInMeans` estimates the
> size-weighted SATE. Two alignments:
> - The estimator does **not require** "no interaction" treatment×block.
> - **Matched pairs** (one treated and one control per block) are a valid
>   limiting case for estimation, but `BootstrapCI` needs at least 2 units per
>   stratum (block×arm), so matched pairs raise `InsufficientDataError` in the
>   bootstrap.

---

## 6. Re-randomization and the Mahalanobis distance

### Intuition

Randomization balances the covariates on average, but a specific draw can come
out imbalanced by bad luck, especially with few units or many covariates.
Re-randomization is insurance against that bad luck: define a balance criterion
in advance, draw, and if the draw fails the criterion, **discard it and draw
again**, until it passes.

### Formalization: the Mahalanobis distance

We need to summarize the imbalance of **several** covariates in a single number,
accounting for their scale and correlation. The Mahalanobis distance does this
(Morgan & Rubin, 2012):

$$
M = (\bar{X}_T - \bar{X}_C)^{\top}\,
      \big[\operatorname{cov}(\bar{X}_T - \bar{X}_C)\big]^{-1}\,
      (\bar{X}_T - \bar{X}_C).
$$

Here `X̄_T - X̄_C` is the vector of the covariates' mean differences between the
arms, and the matrix in the middle is the covariance of that difference, which
under complete randomization equals `(1/n_T + 1/n_C)·S_X`, with `S_X` the sample
covariance of the covariates. The draw is accepted if `M ≤ a`, where `a` is the
threshold.

Under approximate normality, `M` follows a chi-square with `k` degrees of
freedom (`k` = number of covariates), so a natural threshold is
`a = chi2.ppf(p_a, df=k)`, where `p_a` is the fraction of draws you accept (for
example `p_a = 0.05` accepts the 5% most balanced). The smaller `p_a`, the
stricter the criterion and the greater the variance reduction in the covariates
(and, in turn, in the estimator, to the extent that the covariates predict the
outcome).

### The inference must respect the criterion

A subtle and important point: if you re-randomize in the design but analyze with
a standard test, the test becomes **conservative**. The correct analysis uses a
randomization test that **generates the permutations under the same acceptance
criterion** used in the design.

### Worked example

With two covariates (`x1`, `x2`) and `p_a = 0.05`, the threshold is
`a = chi2.ppf(0.05, df=2) ≈ 5.99`. You draw; if the large seedlings landed almost
all in the treatment, `M` exceeds `a` and the draw is discarded; you repeat until
`M ≤ 5.99`. The result is a draw in which `x1` and `x2` are balanced across the
arms, and the final effect is isolated from those variables.

> **In the library.** `ReRandomizedCRD(covariates=[...], threshold=..., p=...)`
> implements the accept/reject loop, caches the covariance matrix and **reuses**
> it in `draw()`. `RandomizationTest` respects the criterion by reusing the same
> matrix/threshold in the permutations, keeping the test valid (not
> conservative). Use it when there are many covariates; avoid it in sequential
> allocation (units arriving one at a time) or without access to the covariates
> before the intervention.

---

## 7. Factorial 2^K

### Intuition

Instead of testing one factor at a time, the factorial design varies `K` binary
factors at once and estimates, in a single experiment, the **main effects** (the
average impact of each factor) and the **interactions** (when the effect of one
factor depends on the level of another). It is efficient: with one round of
`2^K` cells, all the effects are extracted.

### Formalization: orthogonal contrasts

There are `2^K` cells (combinations of the levels). With coding `-1` (low) and
`+1` (high), each effect is an orthogonal **contrast** on the cell means. The
equivalent regression model, for two factors, is

$$
y = \beta_0 + \beta_1 x_A + \beta_2 x_B + \beta_{12}\,x_A x_B + \varepsilon,
\qquad x_A, x_B \in \{-1, +1\}.
$$

The orthogonality of the contrast matrix makes each effect estimated
independently of the others. The number of effects is `2^K - 1`: `K` main
effects, `C(K,2)` two-way interactions, and so on up to the `K`-way interaction.
The algebra of the contrasts is detailed in [III. Estimation](03-estimation.md),
topic 12.

### Worked example: a 2^2

Imagine two factors `A` and `B` (each low/high) and the response means per cell:

| `A` | `B` | mean |
|---|---|---|
| - | - | 20 |
| + | - | 30 |
| - | + | 24 |
| + | + | 38 |

- Main effect of A: mean at high minus mean at low of A,
  `(30+38)/2 - (20+24)/2 = 34 - 22 = 12`.
- Main effect of B: `(24+38)/2 - (20+30)/2 = 31 - 25 = 6`.
- Interaction AB: `[(38+20) - (30+24)]/2 = (58 - 54)/2 = 2`.

The positive interaction of `2` means that A and B together yield a little more
than the sum of the isolated effects would suggest. Without the factorial
design, testing one factor at a time, you would never see that term.

> **In the library.** `FactorialDesign(factors=[...], n_per_cell=...)` requires
> `n_per_cell · 2^K` units and uses **little-endian** coding of the cells
> (`cell = Σ x_j · 2^j`). `FactorialEstimator` returns the `2^K - 1` effects. For
> large `K`, the number of runs explodes (`2^{10} = 1024`); **fractional**
> factorials, which test a fraction of the cells by trading high-order
> interactions for efficiency, are a v2 item.

### When to use and when not to

Use it in the screening phase, to discover which factors matter and how they
interact, or in multivariate A/B tests. Avoid it when the response is strongly
nonlinear (two levels do not capture curvature, a situation where center points
are added or one moves to response-surface methods, both in v2).

---

Related notebooks:
[`04_balance_rerandomization`](../../../examples/for_starters/en/04_balance_rerandomization.ipynb),
[`05_blocking`](../../../examples/for_starters/en/05_blocking.ipynb),
[`06_factorial`](../../../examples/for_starters/en/06_factorial.ipynb).
