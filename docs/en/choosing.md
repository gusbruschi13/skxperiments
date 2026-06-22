# How to choose

A quick guide to assembling an experiment: design, estimator, inference,
and diagnostics. When in doubt, start with the simplest option (`CRD` +
`DifferenceInMeans` + `NeymanCI`) and move up as needed.

## 1. Which design?

| Situation | Use |
|---|---|
| Simple A/B, no structure | `CRD` |
| A known categorical that affects the outcome (region, device) | `BlockedCRD` |
| Want guaranteed balance on continuous covariates | `ReRandomizedCRD` |
| Testing several factors at once | `FactorialDesign` |

## 2. Which estimator?

| Situation | Use |
|---|---|
| CRD, no covariates | `DifferenceInMeans` |
| CRD, with measured covariates | `LinEstimator` |
| CRD, with a pre-experiment metric | `CUPED` |
| Blocked | `BlockedDifferenceInMeans` |
| Factorial | `FactorialEstimator` |

## 3. Which inference?

| I want... | Use |
|---|---|
| a design-based p-value with no assumptions | `RandomizationTest` |
| a finite-population CI (about **these** units) | `NeymanCI` |
| a superpopulation CI (a larger population) | `BootstrapCI` |
| to compare several effects or experiments | `MultipleTestingCorrection`, `ExperimentComparison` |

Note: in v1, `NeymanCI` covers `DifferenceInMeans` and
`BlockedDifferenceInMeans`. To get an interval with `LinEstimator` or
`CUPED`, use `BootstrapCI` (which accepts any scalar estimator) or
`RandomizationTest`.

## 4. Always: diagnostics

- **SRM** (`SRMTest`): run it **before** analyzing; it catches collection bugs.
- **Balance** (`check_balance`, `BalanceReport`): check the covariates.
- **A/A** (`AATest`): validate calibration when building a new pipeline.

## Recommended flow

`power_analysis` (plan the n) → `design.randomize` → collect the outcomes →
`ExperimentPipeline` (automatic diagnostics + inference) →
`ExperimentReport`.

See the notebook track in [`index.md`](index.md) for each step in detail,
and the [`glossary.md`](glossary.md) for the terms.
