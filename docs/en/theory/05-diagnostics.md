# V. Diagnostics and practice

Before trusting a result, you must check whether the experiment was well
conducted. This section covers the three central diagnostics (balance, SRM and
A/A) and the composition of everything into a reproducible flow.

---

## 18. Covariate balance (SMD)

### Intuition

The treatment and control groups should have similar distributions of initial
characteristics (age, region, prior behavior). If they differ **before** the
intervention, the outcome becomes confounded. The SMD (standardized mean
difference) measures that imbalance on a scale independent of the variable's
unit, allowing "age in years" and "income in dollars" to be compared on the same
ruler.

### Formalization

For a continuous covariate,

$$\text{SMD} = \frac{\bar{x}_T - \bar{x}_C}{\sqrt{(s_T^2 + s_C^2)/2}},$$

where `s_T²` and `s_C²` are the covariate's variances in each group (with
`ddof = 1`). The denominator is the pooled standard deviation, which standardizes
the difference without being inflated by the sample size. The most common rule of
thumb (Stuart, 2010) is to treat `|SMD| > 0.1` as a relevant imbalance.

### Why not use a t test to assess balance

This is a subtle and important point. A t test (or a p-value) for "are the means
equal?" depends on the **sample size**: with a huge `n`, a tiny and irrelevant
difference becomes "significant"; with a small `n`, a real imbalance may not
reach significance. Worse: after matching, `n` drops and the p-value rises,
giving the false impression of balance. The SMD, being standardized and
independent of `n`, does not have this problem. That is why the recommendation is
to assess balance by the **magnitude of the SMD**, not by a p-value. When the
outcome is sensitive to dispersion, it is also worth checking the **variance
ratio** between the groups, since equal means with very different variances can
still bias.

### Worked example and the Love plot

Study of a sales training. At the start, the group that adopted it has 10 years
of average experience and the control 2 years. With standard deviations of, say,
4 and 3, the pooled denominator is `sqrt((16 + 9)/2) = sqrt(12.5) ≈ 3.54`, and
the initial SMD is

$$\frac{10 - 2}{3.54} \approx 2.26,$$

very high (well above `0.1`): the groups are not comparable. After balancing (by
blocking, re-randomization or, in observational studies, matching), the means
converge and the SMD drops close to zero. The **Love plot** draws, for each
covariate, the SMD before (open dot, far from zero) and after (filled dot, close
to zero), making the balance visible at a glance.

> **In the library.** `check_balance(assignment, covariates)` computes the SMD
> with the pooled standard deviation `sqrt((var_T + var_C)/2)` and `ddof = 1`.
> `BalanceReport` flags `|SMD| > 0.1` and exposes the table, which `plot_balance`
> draws as a Love plot. The library, by design, does **not** use a t test for
> balance. In randomized experiments, this is a diagnostic of the **concretely
> realized draw** (randomization guarantees balance on average, not in each
> draw).

---

## 19. SRM (Sample Ratio Mismatch)

### Intuition

If the observed proportion of units in each arm does not match the planned one,
something broke: a defective draw, data loss, asymmetric bot filtering. The SRM
is an alarm for an **implementation bug**, not a scientific hypothesis. That is
why it invalidates everything: if the allocation is wrong, no estimate built on
it is reliable.

### Formalization

We test the hypothesis that the observed ratio equals the planned one with a
goodness-of-fit chi-square. For two arms with `N` units and expected proportion
`p`, the expected counts are `E_T = N p` and `E_C = N(1-p)`, and the statistic is

$$\chi^2 = \sum_{g \in \{T, C\}} \frac{(O_g - E_g)^2}{E_g},$$

with 1 degree of freedom. SRM is flagged if the p-value is **very** low,
typically `p < 0.001`.

### Why the threshold is so strict

In large samples (millions of units), the Law of Large Numbers makes the ratio
almost perfect, so any systematic deviation produces a tiny p-value. The strict
threshold of `0.001` reflects that SRM usually comes from **selective loss** of
specific users (for example, the most active being classified as bots and
discarded), which is enough to flip the direction of the real effect. A loose
threshold (0.05) would generate too many false alarms and, at the same time, a
true SRM almost always clears `0.001` with room to spare.

### Worked example: Bing

Planned 50/50 split. Observed: `821,588` in control and `815,482` in treatment,
total `N = 1,637,070`. The expected counts are `N/2 = 818,535` each. The
statistic is

$$
\chi^2 = \frac{(821588 - 818535)^2}{818535} + \frac{(815482 - 818535)^2}{818535}
= 2 \cdot \frac{3053^2}{818535} \approx 22.8,
$$

whose p-value (chi-square, 1 d.f.) is about `1.8 \times 10^{-6}`. The count
difference looks small (ratio `0.993`), but the tiny p-value proves it was not
chance. The investigation revealed that the treatment caused a bug that made the
system classify real users as bots, excluding them. The SRM prevented a decision
based on corrupted data.

> **In the library.** `SRMTest(threshold=0.001)` uses `scipy.stats.chisquare`
> comparing observed and expected counts (proportion `p` from the design for two
> arms, or uniform cells in the factorial). Run it **always, first**, before
> analyzing; if it fails, debug the cause before looking at any other result.
> `ExperimentPipeline` runs `SRMTest` automatically.

---

## 20. A/A test: calibration and uniformity of the p-values

### Intuition

In an A/A test, both groups receive **the same** experience. It is a scale
calibrated at zero: since there is no real difference, the system should not flag
"significance" more than chance allows. If it does, there is an error in the
code, the statistics or the draw.

### Formalization: why the p-values are uniform

Under the true null (guaranteed in an A/A), the p-value of a well-calibrated test
follows a `Uniform[0, 1]` distribution. This is a consequence of the probability
integral transform: if the test statistic has the distribution assumed under the
null, then `p = F(T)` is uniform. The operational consequence is

$$\mathbb{P}(p \le \alpha \mid H_0) = \alpha.$$

In 1000 A/A tests, about 50 are expected with `p < 0.05` purely by luck. A
histogram of the p-values should be **flat**. A spike near zero indicates that
the system is miscalibrated and producing false positives in excess.

### Worked example

The team suspects that the standard error of the "clicks per user" metric is
underestimated because of bot users (outliers) and the fact that the
randomization unit (user) differs from the measurement unit (click). They
simulate 1000 A/A tests drawing past users into two identical groups (*offline
replay*, so as not to spend real traffic) and plot the histogram of the 1000
p-values. If 15% come out with `p < 0.05` (instead of the expected 5%), the
system is miscalibrated. After fixing the computation (for example, with the
Delta method for the ratio metric), they repeat the simulation until the
histogram is flat, proving that the Type I error rate returned to `α`.

> **In the library.** `AATest(design, inference, n_simulations, ...)`
> re-randomizes over data with no effect and checks the **false-positive rate**
> (binomial test against `α`) and the **uniformity** of the p-values
> (Kolmogorov-Smirnov). The case of a randomization unit different from the
> measurement unit, which violates independence and breaks uniformity, motivates
> the Delta method (a v2 item).

---

## 21. Composition: pipeline and report

### Intuition

After understanding each piece, we compose a flow that goes from design to
report, reproducibly and with the diagnostics running before the results are
read.

### What industry practice includes (context)

In mature experimentation platforms (Kohavi et al.), the full flow goes through
*data cooking* (joining logs, filtering bots, cleaning duplicates, computing
metrics), guardrail metrics and an OEC (*Overall Evaluation Criterion*) that
combines several metrics into a single index,

$$\text{OEC} = \sum_i w_i \cdot \text{Metric}_i,$$

with `w_i` the importance weights. A recurring principle (from Shewhart) is
"preserve the evidence": the report shows the SRM and the guardrails **before**
the gains, to avoid a biased reading.

### What the library does (scope)

`skxperiments` covers the statistical part **from the `Assignment` onward**, not
the ingestion and cleaning of data nor the OEC computation.

- `ExperimentPipeline(inference, diagnostics=[SRMTest(), ...])` runs the
  diagnostics and the inference over an `Assignment` and returns a
  `PipelineResult` that gathers the result, the diagnostics report and the flags.
  `SRMTest` runs by default, and a flag does not interrupt the estimation (unless
  you ask for `raise_on_flag=True`).
- `ExperimentComparison` compares independent experiments applying the
  multiple-testing correction across the family.
- `ExperimentReport` generates a self-contained HTML with the results table, the
  diagnostics and the embedded plots.

### Worked example

A button-color test at LinkedIn. `ExperimentPipeline` receives the `Assignment`
(already with the outcome), runs `SRMTest` (green), runs `NeymanCI` for the
effect, and packages everything. `ExperimentReport` shows, at the top, the SRM
diagnostic; if it passed, it displays the effect and the interval. If the effect
is `+1%` with a CI that excludes zero, the report evidences a robust result, with
the "diagnostic first, gain second" order that good practice recommends.

> **In the library (scope).** `ExperimentPipeline`/`ExperimentReport` handle the
> statistical composition and the report; *data cooking* and the OEC are platform
> **methodology**, outside what the library implements. The principle of
> surfacing the diagnostics before the gains is reflected in the design of
> `PipelineResult`.

---

Related notebooks:
[`08_diagnostics`](../../../examples/for_starters/en/08_diagnostics.ipynb),
[`09_putting_it_together`](../../../examples/for_starters/en/09_putting_it_together.ipynb).
