# skxperiments documentation

Library for **experimental design and causal inference** under the *potential
outcomes* framework (Rubin Causal Model). The core idea: **the treatment
assignment mechanism is the starting point**, not the statistical model.

This documentation teaches the concepts alongside the API. It is didactic
material for people new to experimentation, not just a reference.

> **Status:** we start with Markdown and notebooks. The plan is to grow into a
> site (Quarto or mkdocs) later; the content is already portable.

---

## Two traditions of "DoE"

The term *Design of Experiments* carries two traditions that are easy to
conflate:

1. **Classical DoE** (Fisher, Box, Montgomery): factorial, response surface,
   process optimization, ANOVA. The question is *which combination of factors
   optimizes a response*.
2. **Causal inference and A/B testing** (Rubin, Imbens): potential outcomes,
   randomization, average treatment effect (ATE). The question is *what is the
   effect of a treatment*.

`skxperiments` lives mostly in **tradition 2**, with `FactorialDesign`
bridging to tradition 1. Keep that boundary in mind: a "factorial interaction"
as an **effect** (what the lib estimates) is a different thing from the
cell-mean *interaction plot* of process optimization.

---

## Learning path (notebooks)

Each notebook teaches an experimentation concept and the matching API. They
live in [`examples/en/`](../../examples/en/) and run in CI (nbmake).

| # | Notebook | Concept |
|---|---|---|
| 01 | Your first experiment | potential outcomes; CRD, DifferenceInMeans, RandomizationTest |
| 02 | Inference three ways | randomization vs. Neyman (finite pop.) vs. bootstrap (superpop.) |
| 03 | Reducing variance | covariates with Lin and CUPED |
| 04 | Balance and rerandomization | `check_balance`, `ReRandomizedCRD` |
| 05 | Blocking | `BlockedCRD` when you have strata |
| 06 | Factorial | 2^K, main effects and interactions (bridge to classical DoE) |
| 07 | Many tests | multiple-testing correction (FWER vs. FDR), `ExperimentComparison` |
| 08 | Trust your experiment | diagnostics: SRM, A/A, balance |
| 09 | Putting it together | `power_analysis`, `ExperimentPipeline`, `ExperimentReport` |

*(Notebooks 02 to 09 land in the next steps.)*

---

## Glossary (work in progress)

- **ATE** (*Average Treatment Effect*): the average effect of the treatment in
  the population of interest.
- **Potential outcomes** (`Y(0)` and `Y(1)`): the two results a unit *would*
  have under control and under treatment; only one is observed.
- **Finite population vs. superpopulation**: inferring about *these* units
  (finite; Neyman) vs. about a larger population they are sampled from
  (superpopulation; bootstrap).
- **Sharp null**: Fisher's strong null hypothesis, **zero effect for every**
  unit (tested by `RandomizationTest`).
- **SMD** (*Standardized Mean Difference*): standardized difference in means,
  used to check covariate balance.
- **SRM** (*Sample Ratio Mismatch*): the observed allocation departs from the
  intended one, an implementation-bug alarm.
- **FWER vs. FDR**: controlling the chance of **any** false positive
  (family-wise) vs. the **proportion** of false positives among discoveries.

*(To be expanded in Step 6, with decision guides: which design, estimator, and
inference to choose.)*
