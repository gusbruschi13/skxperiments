# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 4.2: Multiple testing correction

- **`MultipleTestingCorrection`** (`skxperiments.inference.multiple`):
  utility class for applying Bonferroni, Holm, or Benjamini-Hochberg
  correction to a family of p-values. Standalone class (not
  `BaseInference`); operates on post-processed p-values rather than
  on `Assignment` objects. API: `MultipleTestingCorrection(method,
  alpha).correct(results) -> Results | list[Results]`. Accepts
  `Results` in multi-effect mode (`p_value: dict`) or `list[Results]`
  in scalar mode; output preserves input format. Default
  `method="holm"` (uniformly more powerful than Bonferroni for FWER).
  All methods clip corrected p-values to `[0, 1]`.
- **FWER vs. FDR control**: the docstring explicitly distinguishes
  Bonferroni and Holm (FWER, family-wise error rate) from BH (FDR,
  false discovery rate); the choice depends on the inferential goal.
- **Double-correction detection**: applying `correct()` to a
  `Results` whose `extra` already contains any of the 4 reserved
  keys raises `InvalidDesignError` pointing the user at the original
  uncorrected `Results`.
- **`alpha` override**: `MultipleTestingCorrection(alpha=...)`
  overrides the input `Results`' `alpha` in the output.
- **Reserved keys schema in `Results.extra`** extended with
  `correction_method`, `original_p_values`, `family_wise_alpha`,
  `n_tests`. Documented in the `Results` class docstring.

### Added — Phase 4.1: Randomization-based inference

- **`RandomizationTest`** (`skxperiments.inference.randomization_test`): Fisher's sharp null
  hypothesis test via Monte Carlo permutations. Materializes the `BaseAssignment.draw()`
  contract by routing each permutation through the original randomization mechanism — under
  rerandomization, the cached Mahalanobis covariance matrix is reused via
  `CRDAssignment.rerandomization_metadata`; under blocking, within-block proportions are
  preserved automatically. Always refits the estimator on the original assignment at the
  start of `fit()`; prior estimator state is discarded. Three alternatives:
  `"two-sided"` (criterion `|T_perm| >= |T_obs|`), `"greater"`, `"less"`. P-value uses the
  Phipson & Smyth (2010) continuity correction `(1 + n_extreme) / (1 + n_permutations)`,
  guaranteeing a Monte Carlo p-value strictly greater than zero. Reproducibility: same
  `seed` produces identical `null_distribution_`.
- **`BaseInference.estimate()`** abstract method: subclasses must now implement both
  `fit()` and `estimate()`, mirroring `BaseEstimator`.
- **`BaseInference._validate_assignment_type()`**: thin wrapper exposing the same
  validation surface as `BaseEstimator`. Underlying logic extracted to a module-level
  helper `_check_assignment_type` in `skxperiments.core.base` so both ABCs share a
  single source of truth for the `DesignEstimatorMismatch` message format.
- Reserved keys schema in `Results.extra` documented in the `Results` class docstring:
  `inference_mode`, `theta`, `correlation` (written by Phase 3 estimators);
  `n_permutations`, `null_distribution`, `alternative` (written by `RandomizationTest`).
- `pytest` marker `slow` registered in `pyproject.toml` for tests that run statistical
  property checks.

### Added — Roadmap

- **`ROADMAP.md`**: centralized tracking of deferred features, decisions
  to revisit, and v2 plans. Organized by phase with What / Why deferred /
  Trigger structure. Linked from `README.md`.

### Tests — Phase 4.2

- 25 tests for `MultipleTestingCorrection` across 7 grouping classes
  covering creation validations, input validation (rejects scalar
  single, multi-effect without p_value, empty list, mixed list, double
  correction), Bonferroni (known values, clipping, ordering),
  Holm (known values, monotonicity, dominance over Bonferroni,
  clipping), Benjamini-Hochberg (known values, monotonicity, FDR
  control via 1000-rep simulation marked `slow`), multi-effect input
  (effects/metadata preservation, alpha override), and list input
  (order preservation, per-Results metadata, family metadata in each
  element).

### Tests — Phase 4.1

- 36 tests for `RandomizationTest` across 10 grouping classes covering creation,
  validation, fit/estimate behavior, statistical properties (slow), reproducibility,
  rerandomization (Mahalanobis preservation under draws), blocking (per-block
  treatment count preservation), integration with all four Phase 3 estimators
  (`DifferenceInMeans`, `BlockedDifferenceInMeans`, `LinEstimator`, `CUPED`), and
  alternative hypothesis behavior. `FactorialAssignment` and multi-effect estimators
  are explicitly rejected (deferred to v2).
- 15 new tests in `tests/core/test_base.py` covering the extended `BaseInference`
  contract and snapshot tests pinning the `DesignEstimatorMismatch` message format
  after the `_check_assignment_type` refactor.

### Planned

- Phase 4.3: `NeymanCI` for finite-population variance under CRD and blocked CRD;
  CUPED-specific variance via internal branch.
- Phase 4.4: `BootstrapCI` (percentile, BCa); explicitly superpopulation inference.
- Phase 4.5: `SequentialTest` (mSPRT, always-valid intervals) — under evaluation;
  may be deferred to v2 (see `ROADMAP.md`).
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