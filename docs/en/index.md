# skxperiments documentation

Library for **experimental design and causal inference** under the *potential
outcomes* framework (Rubin Causal Model). The core idea: **the treatment
assignment mechanism is the starting point**, not the statistical model.

This documentation teaches the concepts alongside the API. It is didactic
material for people new to experimentation, not just a reference.

> **Status:** we start with Markdown and notebooks. The plan is to grow into a
> site (Quarto or mkdocs) later; the content is already portable.

**Guides:** [how to choose](choosing.md) (design, estimator, inference),
[glossary](glossary.md), and the [theory series](theory/01-foundations.md)
(concepts and math).

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
live in the **for_starters** track (simulated data, didactic) at
[`examples/for_starters/en/`](../../examples/for_starters/en/) and run in CI
(nbmake). A **real-data** track will follow.

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

## Theory series

Five chapters that build the concepts and the mathematics behind the library,
derived with worked numeric examples. They pair with the notebooks above: the
notebooks show the API, the theory explains the why. Each idea ends with an
"In the library" callout that ties the math back to the code.

| Chapter | Covers |
|---|---|
| [I. Foundations](theory/01-foundations.md) | potential outcomes, estimands (SATE/PATE), why randomize |
| [II. Designs](theory/02-designs.md) | CRD, blocking, re-randomization (Mahalanobis), factorial 2^K |
| [III. Estimation](theory/03-estimation.md) | difference in means, stratified, Lin, CUPED, factorial contrasts |
| [IV. Inference](theory/04-inference.md) | randomization test, Neyman, bootstrap, multiple testing, power |
| [V. Diagnostics](theory/05-diagnostics.md) | balance (SMD), SRM, A/A, pipeline and report |

---

## Glossary

The core terms (potential outcomes, ATE, finite population vs.
superpopulation, sharp null, SMD, SRM, FWER/FDR) live in
[`glossary.md`](glossary.md).
