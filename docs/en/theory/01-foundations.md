# I. Foundations

This section establishes the vocabulary and the logic of causal inference under
the *potential outcomes* framework (Rubin Causal Model), which is the basis of
all of `skxperiments`. The central idea, which recurs in every module of the
library, is simple to state and powerful in its consequences: the **treatment
assignment mechanism** is the starting point of the analysis, not the
statistical model.

> These texts are theoretical. For hands-on use alongside the API, see the
> notebooks in [`examples/for_starters/en/`](../../../examples/for_starters/en/).

---

## 1. The fundamental problem and potential outcomes

### Intuition

To speak of a "causal effect" is to compare two worlds: the world in which the
unit received the treatment and the world in which it did not. For a patient,
the effect of a drug is the difference between the recovery time **with** the
drug and the recovery time **without** it, for the same patient, under the same
conditions. The obstacle is that, in reality, each patient either takes the drug
or does not. The other world, the counterfactual, is never observed.

### Formalization

For each unit `i` we define two potential outcomes:

- `Y_i(1)`: the outcome that would be observed if `i` received the treatment.
- `Y_i(0)`: the outcome that would be observed if `i` did not receive it.

The individual causal effect is the difference between them:

$$\delta_i = Y_i(1) - Y_i(0).$$

The assignment is represented by `T_i` (1 for treated, 0 for control). The
observed outcome is just one of the two potentials, selected by `T_i`:

$$Y_i = T_i\,Y_i(1) + (1 - T_i)\,Y_i(0).$$

The **fundamental problem of causal inference** (Holland, 1986) is that, for
each unit, we observe `Y_i(1)` or `Y_i(0)`, never both. So `delta_i` is never
directly measurable. Under this lens, causal inference is a **missing data**
problem: half of the "science table" (the table with `Y(0)` and `Y(1)` for
every unit) is always absent.

The way out is to abandon the individual effect and estimate **averages** (see
topic 2), because averages of unobserved quantities can, under randomization, be
estimated without bias from what we do observe.

### Assumptions

For this structure to be well defined, we assume two conditions, gathered under
the acronym SUTVA (*Stable Unit Treatment Value Assumption*):

1. **No interference**: the outcome of one unit does not depend on the treatment
   assigned to other units. `Y_i(t)` depends only on the treatment of `i`.
2. **No hidden versions of the treatment** (consistency): there is a single,
   well-defined "treatment", so that the observed outcome is exactly the
   potential outcome of the arm received.

SUTVA fails, for example, in social networks and marketplaces, where the
treatment of one user "leaks" to another. When that happens, the standard
estimators become biased (this is a v2 theme in the library).

### Worked example: the science table

Consider six units with the following table of potential outcomes (which in
practice we never observe in full):

| unit | `Y(0)` | `Y(1)` | `delta` |
|---|---|---|---|
| 1 | 10 | 12 | 2 |
| 2 | 8 | 11 | 3 |
| 3 | 12 | 13 | 1 |
| 4 | 9 | 9 | 0 |
| 5 | 11 | 16 | 5 |
| 6 | 7 | 10 | 3 |

The true average effect over these six units is

$$\text{SATE} = \frac{2+3+1+0+5+3}{6} = \frac{14}{6} \approx 2.33.$$

Note that it is also the difference of the means of the two potentials:
`mean(Y(1)) = 71/6 ≈ 11.83` and `mean(Y(0)) = 57/6 = 9.5`, whose difference is
`2.33`. This identity reappears in topic 8: it is why the difference in means
estimates the average effect.

The point to internalize: we will never see both columns. If unit 5 is treated,
we observe `16` and lose the `11`; its effect of `5` is forever an inference,
not a measurement.

> **In the library.** `skxperiments.core.potential_outcomes.PotentialOutcomes`
> represents `Y(0)`, `Y(1)`, the individual effect and the ATE, for teaching
> purposes (in a real experiment you would never have both columns). "Strong
> ignorability", which in observational studies must be **assumed** (given a set
> of covariates, assignment is independent of the potentials), in the library is
> **guaranteed by design**: randomization produces that independence by
> construction. That is why the library starts from the `Assignment`, not from a
> selection model. Pearl's formulation via graphs and do-calculus is logically
> equivalent; the library adopts the potential-outcomes language because it is
> the most direct one for experiments.

---

## 2. Estimands: ATE, SATE vs. PATE

### Intuition

Since the individual effect is inaccessible, we define the target of the
inference (the "estimand") as an **average effect**. But an average over which
set? Over the units actually in the experiment, or over a larger population of
which they are a sample? This choice is not cosmetic: it changes the source of
randomness, the variance formula, and the interpretation of the confidence
interval.

### Formalization

The *Average Treatment Effect* is the mean of the individual effects:

$$
\text{ATE} = \frac{1}{n}\sum_{i=1}^{n} \big(Y_i(1) - Y_i(0)\big)
            = \overline{Y(1)} - \overline{Y(0)}.
$$

From here, two readings:

- **SATE** (*Sample Average Treatment Effect*): the mean of the effects for the
  `n` **specific** units in the study. Here the potential outcomes are treated
  as fixed numbers, and the **only** source of randomness is the draw of the
  treatment. This is the **finite-population** view (Neyman, Fisher).
- **PATE** (*Population Average Treatment Effect*): the average effect in a
  **superpopulation**, of which the `n` units are a random sample. There are two
  sources of randomness: the sampling of the units and the draw of the
  treatment.

Formally, the PATE is the expected value of the SATE when we resample the units,
and in the large-`n` limit the two coincide in value. The difference shows up in
the **variance**: the uncertainty about the PATE includes the variability of
which units entered the study, which does not exist in the finite-population
view.

### When to use each

- Use **SATE** (finite population) when the interest is exactly in these units.
  Classic example: the effect of a fertilizer on **this** field divided into
  plots. There is no relevant "superpopulation of fields".
- Use **PATE** (superpopulation) when the goal is to generalize. It is the
  default in online A/B tests: today's users are seen as a sample of a
  continuous stream of future users.

There is also the ATT (*Average Treatment Effect on the Treated*), the mean
effect restricted to the treated units, useful when the effect is heterogeneous
and the decision is about who actually receives the treatment.

### Worked example

In the science table of topic 1, the SATE is `2.33`: it is the average effect
**in those six units**. If they were a sample from a large population of similar
units, and we wanted to predict the average effect in any future sample, we
would be after the PATE. The point estimate would be the same `2.33`, but the
confidence interval for the PATE would be wider, because it must absorb the
uncertainty of "these six could have been another six".

> **In the library.** The distinction maps directly onto the choice of
> inference: **SATE → `NeymanCI`** (finite population) and **PATE →
> `BootstrapCI`** (superpopulation). Choosing between the two is choosing **about
> whom** you want to conclude. `ExperimentComparison` and
> `MultipleTestingCorrection` operate at the level of the estimated effects,
> regardless of that choice.

---

## 3. Why randomize

### Intuition

If we let the units choose (or an uncontrolled process decide) who receives the
treatment, the groups tend to differ in characteristics that also affect the
outcome. The observed difference between the groups then mixes the treatment
effect with those pre-existing differences. This is **confounding**. Randomizing
breaks the link between the unit's characteristics and the assignment, making
the groups comparable on average.

### Formalization: selection bias

Decompose the observed difference in means. Let the treatment group be `T=1` and
the control group `T=0`. What we observe is

$$
\mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]
= \underbrace{\mathbb{E}[Y(1) \mid T=1] - \mathbb{E}[Y(0) \mid T=1]}_{\text{ATT (true effect on the treated)}}
+ \underbrace{\mathbb{E}[Y(0) \mid T=1] - \mathbb{E}[Y(0) \mid T=0]}_{\text{selection bias}}.
$$

The second term is the **selection bias**: the baseline (`Y(0)`) difference
between those who were treated and those who were not, **in the absence** of
treatment. In observational studies it is usually nonzero (those who seek the
treatment were already different). Randomization forces

$$T \perp \big(Y(0), Y(1)\big),$$

that is, the assignment is independent of the potential outcomes. This zeroes
the selection bias (`E[Y(0)|T=1] = E[Y(0)|T=0]`) and equates ATT and ATE, so
that the difference in means comes to estimate the causal effect.

### Worked example: confounding vs. randomization

Return to the science table. Suppose a **self-selection** process in which the
units with the higher baseline `Y(0)` tend to seek the treatment. Say the
treated are `{1, 3, 5}` (with `Y(0)` of 10, 12 and 11) and the controls are
`{2, 4, 6}` (with `Y(0)` of 8, 9 and 7).

- Treated observe `Y(1)`: `12, 13, 16`, mean `13.67`.
- Controls observe `Y(0)`: `8, 9, 7`, mean `8.00`.
- Observed difference: `13.67 - 8.00 = 5.67`.

But the true average effect is `2.33`. The excess of `3.3` is pure selection
bias: the treated already started from a higher baseline (`E[Y(0)|T=1] = 11`
against `E[Y(0)|T=0] = 8`). The "evidence" of an enormous effect is, in large
part, the difference that already existed beforehand.

Now randomize. Under the draw, any unit has the same chance of landing in each
arm, so `E[Y(0)|T=1] = E[Y(0)|T=0] = 9.5` (the overall mean of `Y(0)`), the bias
disappears, and the difference in means estimates `2.33`. The Netflix example is
the same mechanism: without randomizing, the *heavy users* (who would churn less
anyway) dominate the new-interface group and inflate the apparent effect.

### Limit: balance only on average

Randomization guarantees comparability **on average**, over all possible draws.
A specific draw, especially in a small sample, can come out imbalanced by bad
luck (all the *heavy users* in one arm). This motivates **re-randomization**
(topic 6): restricting the acceptable draws to a balance criterion.

> **In the library.** The design is the starting point: `CRD`, `BlockedCRD`,
> `ReRandomizedCRD` and `FactorialDesign` produce an `Assignment`, which is the
> contract consumed by the estimators. Because randomization guarantees
> ignorability, the library does not need to (and does not try to) model a
> selection process. To diagnose the balance of a concrete draw, use
> `check_balance` and `BalanceReport` (see [V. Diagnostics](05-diagnostics.md)).

---

Related notebooks:
[`00_why_randomize`](../../../examples/for_starters/en/00_why_randomize.ipynb),
[`01_first_experiment`](../../../examples/for_starters/en/01_first_experiment.ipynb).
