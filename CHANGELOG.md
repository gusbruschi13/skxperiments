# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Phase 4: inference classes (`RandomizationTest`, `NeymanCI`, `BootstrapCI`,
  `MultipleTestingCorrection`, `SequentialTest`).
- Phase 5: diagnostics (`SRMTest`, `AATest`, `BalanceReport`).
- Phase 6: `ExperimentPipeline` and `ExperimentComparison`.
- Phase 7: visualization and HTML reporting.

## [0.1.0-dev] - 2026-05-31

### Added — Phase 0: Scaffold

- Project scaffold: `pyproject.toml`, `README.md`, `.pre-commit-config.yaml`, GitHub Actions CI.
- Package structure: `core`, `design`, `estimators`, `inference`, `diagnostics`, `reporting`.
- Custom exceptions hierarchy: `SkxperimentsError`, `DesignEstimatorMismatch`, `NotFittedError`,
  `InsufficientDataError`, `InvalidDesignError`.

### Added — Phase 1: Core

- `PotentialOutcomes` class: unit-level Y(0), Y(1), ITE, ATE.
- `BaseAssignment` (ABC) with abstract `draw(seed)` method for randomization-based inference;
  `CRDAssignment`, `BlockedAssignment`, `FactorialAssignment` concrete subclasses.
- `Results` class with mutually exclusive scalar (`ate`) and multi-effect
  (`effects: dict[tuple[str, ...], float]`) modes; auto-populated metadata fields
  (`estimator_name`, `design_name`, `n_obs`, `n_treated`, `n_control`).
- `BaseDesign`, `BaseEstimator`, `BaseInference` abstract base classes;
  `_validate_assignment_type` accepting `type | tuple[type, ...]`.
- `DiagnosticsReport` dataclass.

### Added — Phase 2: Designs

- **`CRD`**: completely randomized design with `n_treated` (absolute) or `p` (proportion),
  mutually exclusive.
- **`BlockedCRD`**: independent randomization within blocks; `BlockedAssignment` carries
  `block_col_` and `block_sizes_`.
- **`ReRandomizedCRD`**: Mahalanobis acceptance criterion (Morgan & Rubin 2012); covariance
  matrix cached in `CRDAssignment.rerandomization_metadata` and reused in `draw()` without
  recomputation; rejects on singular covariance.
- **`FactorialDesign`**: 2^K with equal cell sizes; little-endian cell encoding
  (`cell_idx = sum(factor_value * 2^i for i, factor in enumerate(factors))`);
  `FactorialAssignment.cell_ids(**factor_values)` for cell selection.
- **`check_balance(assignment, covariates)`**: standardized mean differences with pooled std
  `sqrt((var_t + var_c) / 2)`, `ddof=1` (Austin 2009, Stuart 2010).
- **`power_analysis(...)`**: keyword-only function resolving one of `n`, `mde`, or `power`
  given the other two; normal approximation; supports unequal allocation.

### Added — Phase 3: Estimators

- **`DifferenceInMeans`**: simple ATE for `CRDAssignment` (including rerandomized);
  rejects `BlockedAssignment` and `FactorialAssignment`.
- **`BlockedDifferenceInMeans`**: size-weighted SATE for `BlockedAssignment`
  (Imbens & Rubin 2015); unbiased without within-block variance assumptions.
- **`FactorialEstimator`**: all 2^K − 1 orthogonal contrasts (main effects and interactions
  of all orders) for `FactorialAssignment`; alphabetical effect keys via
  `itertools.combinations(sorted(factor_cols), r)`; returns `Results` in multi-effect mode.
- **`LinEstimator`**: OLS of Y on `[1, T, X_centered, T * X_centered]` (Lin 2013);
  accepts `CRDAssignment` or `BlockedAssignment`; rejects constant covariates.
  `inference_mode` is documentational metadata propagated to `Results.extra`.
- **`CUPED`**: pre-experiment covariate adjustment (Deng et al. 2013); `theta_` and
  `correlation_` propagated via `Results.extra`. v1 accepts only `CRDAssignment`;
  blocked extension planned for v2.

### Architectural decisions

The library has 19 numbered architectural decisions documented in source comments and base
class docstrings. Notable ones:

- The `Assignment` object carries a reference to the generating design (`design_`) so
  `draw()` can replay the randomization mechanism for inference.
- The outcome variable is **not** part of the `Assignment` contract; estimators receive
  `outcome_col` as a parameter and resolve it against `assignment.data_` at fit time.
- `Results` is the uniform output object; estimators auto-populate `estimator_name` and
  `design_name`. Inference classes (Phase 4) will populate `inference_name`.
- `inference_mode` (`finite_population` vs. `superpopulation`) is metadata in Phase 3;
  it will live as a `BaseInference` attribute in Phase 4.

### Tests

~452 tests across `tests/core/`, `tests/design/`, `tests/estimators/`, and
`tests/integration/`. All passing on CI. Tests follow a strict pattern of grouping classes
(`TestXxxCreation`, `TestXxxValidation`, `TestXxxNumerics`, etc.) with seeded helpers.