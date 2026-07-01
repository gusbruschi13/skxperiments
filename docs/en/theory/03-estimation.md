# III. Estimation

How to turn the observed data into an estimate of the effect. Every estimator in
this section produces **the point only**; the measure of uncertainty (standard
error, interval, p-value) comes from inference, covered in
[IV. Inference](04-inference.md). The separation is deliberate: in the library,
the estimator answers "what is the effect" and the inference class answers "how
sure".

---

## 8. Difference in means and unbiasedness

### Intuition

Under randomization, the groups are comparable on average, so the difference
between the treatment mean and the control mean, after the intervention,
estimates the effect.

### Formalization and proof of unbiasedness

The estimator is

$$\hat{\tau} = \bar{Y}_T - \bar{Y}_C.$$

Why is it unbiased for the SATE under complete randomization? The key is that,
in this design, **every** unit has the same probability of being treated,
`P(T_i = 1) = n_T / N`. The treated-arm mean is an average of the `Y_i(1)`
restricted to the units drawn for treatment. Taking the expected value over the
draw:

$$
\mathbb{E}[\bar{Y}_T]
= \mathbb{E}\!\left[\frac{1}{n_T}\sum_{i:\,T_i=1} Y_i(1)\right]
= \frac{1}{n_T}\sum_{i=1}^{N} \mathbb{P}(T_i=1)\,Y_i(1)
= \frac{1}{n_T}\cdot\frac{n_T}{N}\sum_{i=1}^{N} Y_i(1)
= \overline{Y(1)}.
$$

By the same argument, `E[Ȳ_C] = mean(Y(0))`. Hence

$$\mathbb{E}[\hat{\tau}] = \overline{Y(1)} - \overline{Y(0)} = \text{SATE}.$$

No model assumption was used: only the structure of the draw. This is the sense
of "the design dictates the inference".

### Worked example

In the science table of [section I](01-foundations.md) (SATE `= 2.33`), suppose
a draw that puts `{1, 4, 5}` in treatment and `{2, 3, 6}` in control.

- Treated observe `Y(1)`: `12, 9, 16`, mean `12.33`.
- Controls observe `Y(0)`: `8, 12, 7`, mean `9.00`.
- `τ̂ = 12.33 - 9.00 = 3.33`.

Another draw would give a different number (the estimate has variance). But the
mean of `τ̂` over the 20 possible draws (`C(6,3) = 20`) is exactly `2.33`, as the
proof above guarantees. The variance of that distribution is what inference
measures.

> **In the library.** `DifferenceInMeans(outcome_col=...)` for `CRDAssignment`
> (including re-randomized). It is the most transparent estimator and the
> starting point. Avoid it in observational studies (selection bias) and be
> careful in small samples (unbiased, but with high variance).

---

## 9. Stratified estimator (block weighting)

### Intuition

We estimate the effect **within** each block and combine by a weighted average
by block size. Because the comparison is internal to the block, the between-block
variation does not enter the error.

### Formalization

With blocks `b = 1, ..., B`, size `N_b` and internal effect `τ̂_b`,

$$
\hat{\tau}_{\text{strat}} = \sum_{b} \frac{N_b}{N}\,\hat{\tau}_b,
\qquad N = \sum_b N_b.
$$

Each `τ̂_b = Ȳ_{T,b} - Ȳ_{C,b}` is unbiased for the SATE of block `b` (by the
argument of topic 8 applied within the block). The weighted average is therefore
unbiased for the global SATE. The variance, by independence across blocks, is

$$
\operatorname{Var}(\hat{\tau}_{\text{strat}})
= \sum_b \left(\frac{N_b}{N}\right)^2 \operatorname{Var}(\hat{\tau}_b),
$$

a formula that reappears in the [stratified Neyman variance](04-inference.md).

### Worked example

Two blocks. Block A: `N_A = 40`, internal effect `τ̂_A = 0.4`. Block B:
`N_B = 60`, internal effect `τ̂_B = 0.7`. The weighted estimator is

$$
\hat{\tau}_{\text{strat}} = \frac{40}{100}(0.4) + \frac{60}{100}(0.7)
= 0.16 + 0.42 = 0.58.
$$

Note that it differs from a simple average `(0.4 + 0.7)/2 = 0.55`: the weighting
gives more weight to the larger block.

> **In the library.** `BlockedDifferenceInMeans` implements exactly that
> size-weighted average. It is equivalent to regression with interactions when
> the block variable is categorical. It requires each block to have at least one
> treated and one control unit (common support).

---

## 10. Regression adjustment (Lin)

### Intuition

Covariates correlated with the outcome can reduce the variance of the estimate.
Historically, there was a fear (Freedman) that using regression to adjust
experiments could **introduce bias** if the model was wrong. Lin (2013) shows
that, by including the **treatment×covariate interactions** with the covariates
**mean-centered**, the adjustment never worsens the asymptotic precision, even
when the linear model is only an approximation. The geometric intuition: allowing
each arm its own slope prevents a covariate-outcome relationship that differs
between the groups from contaminating the effect estimate.

### Formalization

The Lin estimator is the coefficient `β` in the OLS regression

$$
Y_i = \alpha + \beta\,T_i + \gamma\,(z_i - \bar{z}) + \delta\,T_i\,(z_i - \bar{z}) + \varepsilon_i,
$$

where `z_i - z̄` are the centered covariates and `T_i` the treatment indicator.
Centering is what makes `β` the ATE estimate (and not an effect at an arbitrary
point of the covariates). The interaction term `T_i (z_i - z̄)` lets each arm
have its own slope.

### Correct inference

The "do no harm" properties are asymptotic and depend on using a
heteroscedasticity-robust variance (the Huber-White sandwich estimator,
typically HC2). The plain difference in means and the Lin regression agree on the
expected value; Lin tends to have a smaller standard error when the covariates
predict the outcome.

> **In the library (alignment).** `LinEstimator(outcome_col, covariates)`
> produces the **point**. The "correct" Lin interval uses the robust variance,
> which does **not** come out of `NeymanCI` (its whitelist accepts only
> `DifferenceInMeans`/`BlockedDifferenceInMeans`). To get a CI with `Lin`, use
> `BootstrapCI` (which accepts any scalar estimator) or `RandomizationTest`. This
> is a point the user needs to know, so as not to look for the Neyman+Lin
> combination.

### When to use

In unequal allocation (for example 75% treatment, 25% control), where regression
without interaction fails more easily, and whenever you want more precision
safely. Avoid it in very small samples (there is a bias that vanishes with `n`)
or when the priority is the full transparency of the difference in means.

---

## 11. CUPED

### Intuition

CUPED (*Controlled-experiment Using Pre-Experiment Data*) uses a measurement of
the same user **prior** to the experiment to remove noise. If we know the user's
habitual behavior, we subtract the predictable part and are left with the part
the treatment actually moved.

### Formalization: control variates

CUPED is an application of the **control variates** method. Given the metric `Y`
and a pre-experiment covariate `X`, define the adjusted residual `Y - θX`. Its
variance is

$$
\operatorname{Var}(Y - \theta X)
= \operatorname{Var}(Y) - 2\theta\operatorname{Cov}(Y,X) + \theta^2\operatorname{Var}(X).
$$

Minimizing over `θ` (derivative set to zero):

$$
-2\operatorname{Cov}(Y,X) + 2\theta\operatorname{Var}(X) = 0
\quad\Longrightarrow\quad
\theta^{*} = \frac{\operatorname{Cov}(Y,X)}{\operatorname{Var}(X)}.
$$

Substituting back, the minimum variance is

$$\operatorname{Var}(Y - \theta^{*} X) = \operatorname{Var}(Y)\,(1 - \rho^2),$$

where `ρ = corr(Y, X)`. That is, the **variance gets multiplied by `1 - ρ²`**: a
correlation of `0.7` between past and present leaves the variance at
`1 - 0.49 = 0.51` of the original, a drop of about 49% (in practice, "half").
The adjusted-effect estimator is

$$\Delta_{\text{CUPED}} = (\bar{Y}_T - \bar{Y}_C) - \theta\,(\bar{X}_T - \bar{X}_C).$$

Since `X` is pre-experiment, under randomization `E(X_T) = E(X_C)`, so the
subtracted term has zero expectation and the adjustment **introduces no bias**.

### Worked example: Bing slowdown

In the classic example (Deng et al., 2013), the Bing team measured the impact of
a deliberate 250 ms delay on engagement. With an ordinary t test, it took two
weeks to reach significance. Using as a covariate the same users' activity in the
prior two weeks (with `ρ ≈ 0.7`, hence `1 - ρ² ≈ 0.5`), the variance dropped
about 50% and the effect became significant on the first day, with half the
users. The math above explains where that "50%" comes from: it is `1 - ρ²`.

> **In the library.** `CUPED(outcome_col, pre_experiment_col)` computes
> `θ = Cov(Y,X)/Var(X)` and exposes `theta` and `correlation` in `Results.extra`.
> v1 accepts only `CRDAssignment`. The covariate **must precede the treatment**
> (or be unaffected by it); using a covariate contaminated by the treatment
> introduces bias.

---

## 12. Factorial contrasts (±1 coding) and the scale of the effects

### Intuition

With levels coded `-1` (low) and `+1` (high), each effect is an average of
responses with `±1` signs, which isolates the change attributable to that factor
(or combination of factors).

### Formalization: the algebra of the contrasts

For a nonempty subset `S` of factors, the effect is

$$
\text{effect}_S = \frac{1}{2^{K-1}}
\sum_{\text{cells}} \bar{y}_{\text{cell}} \prod_{j \in S}(2x_j - 1),
$$

where `x_j` is the level of factor `j` in the cell (in `{0,1}`) and `2x_j - 1`
converts it to `±1`. Particular cases:

- Main effect of A: `ȳ(A+) - ȳ(A-)` (the difference of the means between the
  high and the low level of A).
- Interaction AB: `[ȳ(++) + ȳ(--) - ȳ(+-) - ȳ(-+)] / 2`.

The divisor `2^{K-1}` is the number of cells on each "side" of the contrast.

### The subtlety of scale: ±1 vs. {0,1}

This point confuses those who simulate data. Suppose you generate the outcome by
a model with `{0,1}` coding:

$$y = b_A\,A + b_B\,B + b_{AB}\,(A\,B) + \text{noise}, \qquad A,B \in \{0,1\}.$$

The library's `±1` contrast does **not** return `b_A` for the main effect of A.
Working through the four cell means, one obtains

$$
\text{effect}_A = b_A + \tfrac{1}{2} b_{AB},
\qquad \text{effect}_{AB} = \tfrac{1}{2} b_{AB}.
$$

That is, the estimated main effect is the average of A's effect over the two
levels of B, and the interaction comes with a factor of `1/2`. This is not a bug:
it is the definition of a factorial effect. When writing teaching examples with
simulation, make this explicit so the reader is not surprised by the difference
between the simulation's `b` and the estimated effects.

### A factorial effect is not Cohen's d

They are two distinct scales worth not confusing:

- The **factorial effect** (what the estimator returns) is in the **raw unit** of
  the response (for example, "+12 yield points").
- **Cohen's d** is a **standardized** effect size, `d = (m_A - m_B)/σ`, used in
  **power** analysis (see [IV. Inference](04-inference.md), topic 17), to say
  whether an effect is "small, medium or large" regardless of the unit.

> **In the library (correction).** `FactorialEstimator` returns the contrasts in
> **raw units**, with the `1/2^{K-1}` divisor on the higher-order terms. It does
> **not** standardize by `σ` and therefore does **not** return Cohen's d. The
> library's convention for main effects is the **full** high-minus-low difference
> (not the half, which some references call the regression coefficient).

---

Related notebooks:
[`03_reducing_variance`](../../../examples/for_starters/en/03_reducing_variance.ipynb),
[`06_factorial`](../../../examples/for_starters/en/06_factorial.ipynb).
