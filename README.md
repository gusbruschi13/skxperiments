# skxperiments

> Randomization-based experimental design and causal inference, sklearn-style.

![CI](https://github.com/username/skxperiments/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)

A Python library for designing randomized experiments and estimating causal effects under the
potential outcomes framework (Rubin Causal Model). Treatment assignment is the starting point;
statistical models come second.

## Status

Active development. Phases 0–3 complete; Phases 4.1–4.2 complete; Phases 4.3–4.5 in planning.
See [Project status](#project-status) below for details.

## Installation

```bash
pip install skxperiments
```

Requires Python 3.10+. Dependencies: `numpy`, `pandas`, `scipy`.

## Quick start

```python
import numpy as np
import pandas as pd
from skxperiments.design.crd import CRD
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.inference import RandomizationTest

# 1. Generate a synthetic dataset
rng = np.random.default_rng(42)
df = pd.DataFrame({
    "x": rng.normal(0.0, 1.0, 200),
    "y": rng.normal(0.0, 1.0, 200),
})

# 2. Design: completely randomized assignment, 50/50 split
design = CRD(p=0.5, seed=42)
assignment = design.randomize(df)

# 3. Point estimate of the ATE
estimator = DifferenceInMeans(outcome_col="y")
result = estimator.fit(assignment).estimate()
print(result.ate)

# 4. Randomization-based p-value (Fisher's sharp null)
rt = RandomizationTest(estimator=estimator, n_permutations=10_000, seed=0)
result = rt.fit(assignment).estimate()
print(result.ate, result.p_value)
```

For variance reduction with covariates, use `LinEstimator` (Lin 2013) or `CUPED` (Deng et al.
2013). For blocked or factorial designs, use `BlockedCRD` + `BlockedDifferenceInMeans` or
`FactorialDesign` + `FactorialEstimator`. For rerandomization on Mahalanobis distance,
use `ReRandomizedCRD` (Morgan & Rubin 2012). `RandomizationTest` works with all of these
(except `FactorialAssignment` in v1). To control the family-wise error rate or false
discovery rate when reporting multiple effects, wrap the result in
`MultipleTestingCorrection`.

## Design philosophy

1. **The assignment mechanism is primary**, not the statistical model.
2. **API in scikit-learn style**: parameters in `__init__`, data in `fit()`, learned attributes
   end with `_`.
3. **`Assignment` is the contract** between designs and estimators — estimators receive
   `Assignment` objects, not loose DataFrames.
4. **Randomization-based inference is the default**; classical t-tests are not.
5. **Finite-population vs. superpopulation inference are distinguished explicitly.**
6. **Fail fast** with clear messages when designs and estimators are incompatible.
7. **No side effects**: `fit()` and `randomize()` never mutate input DataFrames.

## Project status

| Phase | Module | Status |
|---|---|---|
| 0 | Scaffold, exceptions, CI | ✓ Complete |
| 1 | Core (`Assignment`, `Results`, base classes) | ✓ Complete |
| 2 | Designs (CRD, BlockedCRD, ReRandomizedCRD, FactorialDesign, balance, power) | ✓ Complete |
| 3 | Estimators (DIM, BlockedDIM, Factorial, Lin, CUPED) | ✓ Complete |
| 4 | Inference (RandomizationTest, MultipleTestingCorrection, NeymanCI, BootstrapCI, sequential) | 🚧 In progress (4.1, 4.2 complete) |
| 5 | Diagnostics (SRM, A/A test, balance report) | Planned |
| 6 | Pipeline composition | Planned |
| 7 | Visualization and reporting | Planned |

Test coverage: 528 tests, all passing on CI.

See [`ROADMAP.md`](ROADMAP.md) for deferred features and v2 plans, and `CHANGELOG.md` for the
full history of changes.

## What's implemented

### Designs (`skxperiments.design`)

- **`CRD`** — Completely randomized design.
- **`BlockedCRD`** — Independent randomization within blocks.
- **`ReRandomizedCRD`** — Mahalanobis acceptance criterion with cached covariance matrix; loop with `max_attempts`.
- **`FactorialDesign`** — 2^K factorial design with equal cell sizes; little-endian cell encoding.
- **`check_balance(assignment, covariates)`** — Standardized mean differences (SMD), pooled std with `ddof=1`.
- **`power_analysis(...)`** — Sample size, MDE, or power for two-sample mean comparisons.

### Estimators (`skxperiments.estimators`)

- **`DifferenceInMeans`** — Simple ATE for `CRDAssignment`.
- **`BlockedDifferenceInMeans`** — Size-weighted ATE for `BlockedAssignment`.
- **`FactorialEstimator`** — All 2^K − 1 effects (main effects and interactions of all orders) for `FactorialAssignment`. Returns `Results` in multi-effect mode.
- **`LinEstimator`** — Covariate-adjusted ATE via OLS with treatment-covariate interaction (Lin 2013).
- **`CUPED`** — Variance reduction with a pre-experiment covariate (Deng et al. 2013).

All estimators return `Results` with point estimates only; standard errors and confidence
intervals come from inference classes in `skxperiments.inference`.

### Inference (`skxperiments.inference`)

- **`RandomizationTest`** — Fisher's sharp null hypothesis test via Monte Carlo permutations.
  Uses `Assignment.draw()` to respect the original randomization mechanism (including
  rerandomization Mahalanobis criterion and within-block proportions). P-value via the
  Phipson & Smyth (2010) continuity correction. Three alternatives: `"two-sided"`,
  `"greater"`, `"less"`. Works with `DifferenceInMeans`, `BlockedDifferenceInMeans`,
  `LinEstimator`, and `CUPED`.
- **`MultipleTestingCorrection`** — Bonferroni, Holm (FWER) and Benjamini-Hochberg (FDR)
  correction over a family of p-values. Accepts a multi-effect `Results` (typical from
  `FactorialEstimator` after inference) or a list of scalar `Results` (for comparing
  independent experiments). Clips corrected p-values to `[0, 1]`; preserves originals
  in `Results.extra["original_p_values"]`. Default method is Holm.

## What's coming

### Phase 4 — Inference (continued)

`RandomizationTest` and `MultipleTestingCorrection` are implemented. The remaining
inference classes will produce confidence intervals beyond the permutation p-value:

- **`NeymanCI`** — Finite-population variance for CRD and blocked CRD; CUPED-specific
  variance via internal branch.
- **`BootstrapCI`** — Percentile, BCa (superpopulation inference).
- **`SequentialTest`** — mSPRT and always-valid intervals (under evaluation; may be
  deferred to v2 per `ROADMAP.md`).

### Phase 5 — Diagnostics

`SRMTest`, `AATest`, `BalanceReport`. The pipeline (Phase 6) will run `SRMTest` automatically
before any estimation.

### Phase 6 — Pipeline

`ExperimentPipeline` composes design + estimator + inference with automatic SRM checking.
`ExperimentComparison` aggregates multiple pipelines for forest plots and joint multiple-testing
correction.

### Phase 7 — Reporting

Plots (`plot_balance`, `plot_forest`, `plot_effect`, `plot_interaction`, `plot_power_curve`,
`plot_null_distribution`, `plot_srm`) and HTML reports (`ExperimentReport`). `matplotlib` and
`plotly` are optional dependencies.

## Contributing

Contributions are welcome. Please open an issue to discuss substantial changes before submitting
a pull request. The architecture has documented design decisions that should be respected — see
[`ROADMAP.md`](ROADMAP.md), the project notes in `CHANGELOG.md`, and the docstrings of base
classes (`BaseAssignment`, `BaseEstimator`, `Results`) for the contracts new code must follow.

Run the test suite with:

```bash
pytest tests/ -v
```

Skip slow statistical tests:

```bash
pytest tests/ -v -m "not slow"
```

## License

MIT.

## References

The implementations follow standard textbook formulations:

- Imbens, G. W., & Rubin, D. B. (2015). *Causal inference for statistics, social, and biomedical
  sciences: An introduction.* Cambridge University Press.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data: Reexamining
  Freedman's critique. *Annals of Applied Statistics*, 7(1), 295–318.
- Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate balance in
  experiments. *Annals of Statistics*, 40(2), 1263–1282.
- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the sensitivity of online
  controlled experiments by utilizing pre-experiment data. *WSDM 2013*.
- Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for experimenters: Design,
  innovation, and discovery* (2nd ed.). Wiley.
- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Routledge.
- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of baseline covariates
  between treatment groups in propensity-score matched samples. *Statistics in Medicine*.
- Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero: calculating
  exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics
  and Molecular Biology*, 9(1).
- Fisher, R. A. (1935). *The Design of Experiments*. Oliver and Boyd.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian
  Journal of Statistics*, 6(2), 65–70.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and
  powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*,
  57(1), 289–300.