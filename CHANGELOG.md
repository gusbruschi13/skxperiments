# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Packaging

- Published `0.1.0.dev0` to PyPI as a developmental pre-release
  (`pip install --pre skxperiments`; the `viz` extra adds matplotlib).
- Added packaging metadata (authors, keywords, classifiers, project URLs)
  and `.github/workflows/publish.yml` for trusted publishing (OIDC) on a
  published GitHub Release. Release flow documented in `RELEASING.md`.

### Added — Phase 7: Visualization and reporting

- **Plots** (`skxperiments.reporting.plots`): `plot_balance`, `plot_srm`,
  `plot_null_distribution` (diagnostic) and `plot_effect`, `plot_forest`,
  `plot_interaction`, `plot_power_curve` (result). Each accepts an
  optional `ax` and returns the `matplotlib.axes.Axes` it drew on.
- **`ExperimentReport`** (`skxperiments.reporting.summary`): renders a
  `PipelineResult` as a self-contained static HTML page — results table,
  diagnostics summary, and the relevant plots embedded inline as base64
  PNGs. No template engine. `include_plots=False` produces the page
  without matplotlib. `to_html()` / `save(path)`.
- **Optional `matplotlib` dependency**: added a `viz` extra
  (`pip install skxperiments[viz]`) and included matplotlib in `dev`.
  Plotting is imported lazily; calling a plot without matplotlib raises a
  clear `ImportError` pointing at the extra. Importing the package, or
  building a report with `include_plots=False`, needs no optional
  dependency.

### Added — Phase 6: Pipeline and comparison

- **`ExperimentPipeline`** (`skxperiments.pipeline`): composes an inference
  procedure (which already wraps an estimator) with a set of diagnostics
  and runs them against a single `Assignment`. The design travels with the
  assignment (`assignment.design_`), so neither design nor estimator is a
  separate argument. Diagnostics run best-effort (one raising a
  `SkxperimentsError` is recorded as a warning and skipped); a flagged
  diagnostic is surfaced but does not stop estimation unless
  `raise_on_flag=True`. Default diagnostics: `[SRMTest()]`. Returns a
  `PipelineResult` bundling the inference `Results`, a merged
  `DiagnosticsReport`, and per-diagnostic results.
- **`ExperimentComparison`** (`skxperiments.pipeline`): compares several
  independent experiments by collecting each scalar ATE/p-value and
  applying `MultipleTestingCorrection` across the family. Accepts a dict
  of `PipelineResult` or scalar `Results`. Returns a `ComparisonResult`
  with the corrected `Results` per experiment and a comparison table
  (ATE, SE, CI, original/corrected p-value, significance) ready for the
  Phase 7 forest plot. Multi-effect/subgroup comparison is deferred to v2.

### Added — Phase 5: Diagnostics

- **`SRMTest`** (`skxperiments.diagnostics.srm`): Sample Ratio Mismatch
  check via Pearson's chi-squared, comparing observed arm/cell counts to
  the design's intended allocation. Two-arm (CRD/Blocked) and factorial
  designs; expected allocation inferred from `design_.p` or a uniform
  split over cells, or supplied explicitly. Default threshold 0.001 (the
  industry SRM convention). Returns `SRMResult`.
- **`BalanceReport`** (`skxperiments.diagnostics.balance_report`): wraps
  `check_balance` to report the standardized mean difference (SMD) per
  covariate and flag covariates with `|SMD| > threshold` (default 0.1,
  Austin 2009). Constant covariates (undefined SMD) are surfaced as
  warnings. Two-arm only; rejects factorial. Returns `BalanceResult`
  (the table is exposed via `to_dataframe` for the Phase 7 Love plot).
- **`AATest`** (`skxperiments.diagnostics.aa_test`): re-randomizes a
  design on fixed data and runs a wrapped inference each time, checking
  calibration. The false-positive rate is compared to `alpha` with an
  exact binomial test (flag below `meta_threshold`, default 0.001), and
  p-value uniformity is reported via a Kolmogorov-Smirnov test (secondary
  warning). Returns `AAResult`.
- Each diagnostic exposes a dedicated frozen result dataclass with
  `summary`/`to_dict` and a `to_diagnostics_report()` mapping into the
  existing `DiagnosticsReport` (for the Phase 6 pipeline). `Results` is
  left untouched: diagnostics are not estimands.

### Added — Phase 4.4: Bootstrap confidence intervals

- **`BootstrapCI`** (`skxperiments.inference.bootstrap`): `BaseInference`
  subclass that approximates the sampling distribution of an estimator's
  ATE by resampling units with replacement within each treatment arm
  (within each block-by-arm stratum for blocked designs), then forms a
  `"percentile"` or `"bca"` (default) confidence interval. The bootstrap
  is the library's explicit **superpopulation** method: it ignores the
  randomization mechanism and always reports
  `inference_mode="superpopulation"`.
- **Estimator-agnostic**: any estimator producing a scalar `Results.ate`
  is supported (`DifferenceInMeans`, `BlockedDifferenceInMeans`,
  `LinEstimator`, `CUPED`); each resample is turned back into an
  `Assignment` and the estimator is refitted. Multi-effect estimators are
  rejected.
- **BCa**: bias-correction `z0` from the bootstrap distribution and
  acceleration `a` from a leave-one-out jackknife (Efron 1987). The
  degenerate case (bias-correction undefined) raises `InvalidDesignError`
  suggesting `method="percentile"`.
- **Output**: bootstrap standard error, percentile/BCa interval, and an
  approximate two-sided bootstrap p-value (achieved significance level).
- **Fail fast**: `InsufficientDataError` when any arm (CRD) or block-by-arm
  stratum (blocked) has fewer than 2 units; matched-pair blocked designs
  are not supported by the within-stratum bootstrap in v1 (see `ROADMAP.md`).
- **Reserved keys schema in `Results.extra`** extended with `method`,
  `n_resamples`, `bootstrap_distribution`, and (BCa only)
  `bias_correction` and `acceleration`. Documented in the `Results`
  docstring.

### Added — Phase 4.3: Neyman confidence intervals

- **`NeymanCI`** (`skxperiments.inference.neyman`): `BaseInference`
  subclass that wraps a scalar estimator and builds a two-sided Wald
  confidence interval and p-value from Neyman's variance estimator,
  dispatched by assignment type. CRD uses the conservative
  `s_t^2 / n_t + s_c^2 / n_c` (`ddof=1`); blocked designs use the
  stratified `sum_b (N_b / N)^2 * V_b`, consistent with the size-weighted
  ATE of `BlockedDifferenceInMeans` (Imbens & Rubin 2015, ch. 6 and 9).
  The interval uses the normal quantile (`scipy.stats.norm`).
- **Estimator whitelist**: v1 accepts only `DifferenceInMeans` and
  `BlockedDifferenceInMeans`, validated at construction with
  `DesignEstimatorMismatch`. `CUPED` and `LinEstimator` support is
  deferred (see `ROADMAP.md`).
- **Finite-population scope**: an estimator reporting
  `inference_mode="superpopulation"` is rejected with a message
  redirecting to `BootstrapCI` (Phase 4.4). The Neyman formula is
  identical under both interpretations; the restriction is a scope choice.
- **Fail fast**: `InsufficientDataError` when any arm (CRD) or any arm
  within a block (blocked) has fewer than 2 observations.
- **Reserved keys schema in `Results.extra`** extended with
  `variance_type` (`"neyman"` or `"neyman_stratified"`); `NeymanCI` also
  propagates `inference_mode`. Documented in the `Results` docstring.

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

### Tests — Phase 7

- 32 tests across `tests/reporting/`: 8 for the diagnostic plots, 14 for
  the result plots, and 10 for `ExperimentReport`. All run headless (Agg
  backend) and assert on figure artefacts (tick labels, bar/line counts,
  histogram bins) and HTML substrings rather than pixels; the
  optional-dependency guard is exercised by monkeypatching matplotlib
  out of `sys.modules`.

### Tests — Phase 6

- 33 tests across `tests/test_pipeline.py` (16) and
  `tests/test_comparison.py` (17): pipeline creation/validation, clean
  and flagged runs, `raise_on_flag`, best-effort diagnostic errors, a
  custom `BalanceReport` diagnostic, empty diagnostics, and the result
  surface; comparison creation/validation, Bonferroni known values,
  mixed `PipelineResult`/`Results` input, order preservation,
  multi-effect and missing-p-value rejection, and the result surface.

### Tests — Phase 5

- 61 tests across `tests/diagnostics/`: 23 for `SRMTest` (creation/
  validation, design-inferred and explicit expectations, factorial,
  the result surface, unsupported input), 20 for `BalanceReport`
  (balanced/imbalanced detection, threshold control, covariate subsets,
  constant covariates, blocked designs, factorial rejection, NaN
  propagation, result surface), and 18 for `AATest` (creation/validation,
  calibrated-pipeline calibration, reproducibility, miscalibration flag,
  no-p-value rejection, result surface, and a slow nested run with
  `RandomizationTest`).

### Tests — Phase 4.4

- 36 tests for `BootstrapCI` across 10 grouping classes covering creation
  and parameter validation, assignment-type and multi-effect rejection,
  insufficient-data fail-fast (including matched-pair blocks), the
  degenerate BCa case, fitted attributes, estimate output (percentile and
  BCa extras, superpopulation override), reproducibility, percentile-vs-BCa
  agreement on symmetric data (slow), blocked designs, estimator-agnostic
  smoke tests (`LinEstimator`, `CUPED`), assignment immutability, and slow
  numerics (Monte Carlo coverage and agreement of the bootstrap SE with the
  `NeymanCI` SE).

### Tests — Phase 4.3

- 30 tests for `NeymanCI` across 8 grouping classes covering creation and
  alpha validation, estimator-whitelist and assignment-type rejection
  (factorial, multi-effect, superpopulation), insufficient-data fail-fast,
  hand-checked CRD and stratified-blocked variance, the Wald CI/p-value,
  rerandomization acceptance, assignment immutability, and slow Monte
  Carlo coverage (CRD and blocked, near the nominal 95%).

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

### Status

The v1 feature set (Phases 0–7) is complete: design, estimation,
randomization/finite-population/superpopulation inference, multiple-testing
correction, diagnostics, pipeline composition, and reporting.

Deferred to v2 (see `ROADMAP.md`): `SequentialTest` (mSPRT/always-valid),
Benjamini-Yekutieli correction, CUPED/Lin variance in `NeymanCI`,
studentized bootstrap, block-resampling bootstrap, subgroup comparison,
a plotly backend, and interactive dashboards.

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