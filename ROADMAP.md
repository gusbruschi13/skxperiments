# Roadmap

This document tracks deferred features, architectural decisions to revisit,
and technical debt accumulated during the development of skxperiments. It
is the canonical place to record "v2" items so they don't get lost in
commit messages or chat logs.

Items are grouped by phase. Each item lists:

- **What**: the deferred feature or decision.
- **Why deferred**: rationale at the time of deferral.
- **Trigger**: condition under which it should be reconsidered.

---

## Core

### Mutate-vs-construct contract for `Assignment`

- **What**: `Assignment` immutability is a convention, not enforced. The
  constructor is public, allowing tests and users to build new
  `Assignment` objects with modified `data_`.
- **Why deferred**: Enforcing immutability via `frozen` dataclasses or
  `__setattr__` overrides adds complexity for little gain in a library
  where users are expected to be sophisticated.
- **Trigger**: First reported bug caused by accidental mutation in user
  code or in `ExperimentPipeline` (Phase 6).

### `treated_ids` / `control_ids` contract for factorial designs

- **What**: `FactorialAssignment.treated_ids()` and `control_ids()` raise
  `NotImplementedError`. Users must call `cell_ids(**factor_values)` for
  cell selection.
- **Why deferred**: Factorial treatment is multi-valued, not binary. The
  current ABC enforces a binary semantics that doesn't generalize.
- **Trigger**: When `FactorialEstimator` users frequently need to slice
  by single-factor groups; consider redesigning `BaseAssignment` to
  separate "binary treatment" from "multi-cell" semantics.

---

## Designs (Phase 2)

### `BlockedDifferenceInMeans` weighting parameter

- **What**: Currently size-weighted, hard-coded. No `weighting` parameter
  in `__init__`.
- **Why deferred**: Size-weighted is the standard SATE estimator under
  blocked CRD; alternatives (precision-weighted, equal-weighted) are
  rare in practice.
- **Trigger**: Demand for precision-weighted estimation, which requires
  variance estimates per block (which are not produced by point
  estimators today).

### `power_analysis` extensions

- **What**: v1 supports two-group continuous outcomes only. Blocked,
  factorial, cluster designs and binary outcomes are not supported.
- **Why deferred**: Scope control. Each design type has its own power
  formula and most users only need the two-group case.
- **Trigger**: Add per-design as `BlockedCRD.power_analysis()` etc., or
  extend `power_analysis()` with a `design` parameter once Phase 6
  pipeline is mature.

### Rerandomization with non-Mahalanobis criteria

- **What**: `ReRandomizedCRD` uses Mahalanobis distance. Other criteria
  (e.g., per-covariate t-test thresholds) are not supported.
- **Why deferred**: Mahalanobis is the canonical choice (Morgan & Rubin
  2012). Other criteria add API surface for marginal gain.
- **Trigger**: Specific user request with literature backing.

---

## Estimators (Phase 3)

### `CUPED` with `BlockedAssignment`

- **What**: v1 rejects `BlockedAssignment` via `DesignEstimatorMismatch`.
- **Why deferred**: CUPED with blocking requires either pooling pre-period
  data across blocks (loses block structure) or estimating theta per
  block (small-sample issues). Both have trade-offs requiring a careful
  decision.
- **Trigger**: User demand or evidence from literature on best practice.

### `LinEstimator` with `FactorialAssignment`

- **What**: Currently rejects `FactorialAssignment`. Lin's adjustment
  with multiple treatment arms is non-trivial.
- **Why deferred**: Conceptual: with K factors, the OLS specification
  must include K main-effect treatments and their interactions, which
  changes the estimand from "ATE" to "factor-specific effects". This
  duplicates `FactorialEstimator` with covariates.
- **Trigger**: When `FactorialEstimator` users want covariate adjustment.
  Likely a new class `LinFactorialEstimator` rather than extending
  `LinEstimator`.

### Multiple covariates in CUPED

- **What**: `CUPED` accepts a single `pre_experiment_col`. Multiple
  pre-period covariates would require a multivariate theta.
- **Why deferred**: Multivariate CUPED is asymptotically equivalent to
  `LinEstimator` with the pre-period covariates. Users wanting
  multivariate adjustment should use `LinEstimator`.
- **Trigger**: Probably never; document the equivalence in CUPED's
  docstring as the canonical answer.

---

## Inference (Phase 4)

### `NeymanCI` with `CUPED` and `LinEstimator`

- **What**: `NeymanCI` v1 accepts only `DifferenceInMeans` and
  `BlockedDifferenceInMeans`, validated by a whitelist at construction.
  `CUPED` and `LinEstimator` are rejected via `DesignEstimatorMismatch`.
- **Why deferred**: Each implies a different variance estimator — CUPED
  needs Neyman's variance on the adjusted outcome
  `Y - theta * (X_pre - mean(X_pre))`, and Lin needs the HC-robust
  regression variance — neither of which is the two-sample/stratified
  formula `NeymanCI` v1 implements. Sniffing `Results.extra["theta"]` to
  switch formulas was considered and rejected as implicit coupling.
- **Trigger**: When users need analytical CIs for covariate-adjusted
  estimators. Likely a per-estimator variance contract (estimators expose
  their own variance) or a dedicated path, rather than special-casing
  inside `NeymanCI`.

### `RandomizationTest` with `FactorialAssignment`

- **What**: v1 rejects `FactorialAssignment`. Multi-effect estimators
  produce `Results.effects` (a dict), not a scalar `ate`.
- **Why deferred**: Permuting under a factorial design requires deciding
  which sharp null is being tested (joint null of all effects vs. null
  of a specific effect). Different permutation schemes are needed for
  each. Non-trivial design decision.
- **Trigger**: Phase 4.4 or Phase 6, when users explicitly ask for
  randomization-based p-values on factorial effects. Likely a new class
  `FactorialRandomizationTest` with explicit `effect_key` parameter.

### `RandomizationTest` `"two-sided-conservative"` alternative

- **What**: Current two-sided uses `|T_perm| >= |T_obs|`. A conservative
  variant `2 * min(p_greater, p_less)` would be useful under strong null
  asymmetry.
- **Why deferred**: For typical CRD with balanced sample sizes and
  Fisher's sharp null, the null distribution is approximately symmetric
  and the two definitions agree. The conservative variant matters only
  in edge cases (highly imbalanced blocks, restrictive rerandomization
  thresholds).
- **Trigger**: User report of misleading p-values under asymmetric null,
  or when adding sequential tests where asymmetry is more common.

### Exact enumeration of permutations for small N

- **What**: `RandomizationTest` always uses Monte Carlo. For small N,
  exact enumeration of all `binom(n, n_treated)` permutations is feasible.
- **Why deferred**: Monte Carlo with `n_permutations=10_000` matches
  exact enumeration to 4 decimal places for any N where exact would
  matter. Implementation cost > value.
- **Trigger**: Users with N < 20 and need to publish exact p-values.

### `BootstrapCI` for matched-pair / small-stratum blocked designs

- **What**: `BootstrapCI` resamples within each block-by-arm stratum and
  requires at least 2 units per stratum. Matched-pair blocked designs
  (1 treated + 1 control per block) therefore raise
  `InsufficientDataError`.
- **Why deferred**: A within-stratum bootstrap is degenerate when a
  stratum holds a single unit. The fix is a block-level (cluster)
  bootstrap that resamples whole blocks with replacement — a different
  resampling scheme with its own bias/variance trade-offs.
- **Trigger**: User demand for bootstrap CIs on matched-pair or
  small-block designs. Likely a `block_resample=True` option or a
  dedicated cluster-bootstrap path.

### `BootstrapCI` studentized variant

- **What**: v1 supports percentile and BCa. Studentized requires the
  estimator to expose a standard error, which Phase 3 estimators don't.
- **Why deferred**: Phase 4.4 shipped `BootstrapCI` with percentile and
  BCa only. Studentized requires a per-resample standard error (either a
  nested bootstrap, which is slow, or an analytical SE recomputed on each
  resample, e.g. the Neyman variance). It was cut to keep 4.4 scoped.
- **Trigger**: Fast-follow to Phase 4.4 (or v2). The analytical-SE path
  is now viable since `NeymanCI` computes a closed-form variance that
  could be evaluated per resample.

### `MultipleTestingCorrection` with Benjamini-Yekutieli

- **What**: v1 supports Bonferroni, Holm, BH (Benjamini-Hochberg). BY
  (Benjamini-Yekutieli, FDR under arbitrary dependence) is not included.
- **Why deferred**: BY is conservative and rarely used in practice. BH
  under positive dependence (the typical case in factorial designs) is
  the standard.
- **Trigger**: User request with citation.

### `SequentialTest` (mSPRT, always-valid intervals)

- **What**: Originally planned as Phase 4.5; **deferred to v2** (decided
  at the end of Phase 4.4).
- **Why deferred**: Sequential tests require distributional assumptions
  (subgaussianity, normality) that break the "design dictates inference"
  philosophy of the rest of the library. They are also inherently coupled
  to streaming/online use cases, which don't fit the one-shot
  `fit(assignment) -> estimate()` contract — there is no time/arrival
  dimension on a fixed finite-population `Assignment`. Both point to a
  dedicated design rather than a fourth sibling of the existing inference
  classes.
- **Trigger**: A v2 effort with its own API design (a streaming
  `update()`/`peek()` interface, or an `order_col` arrival convention),
  possibly as a separate `skxperiments-sequential` package.

---

## Diagnostics (Phase 5)

### `NoveltyDetection`

- **What**: Originally planned in the project notes; not in scope for
  Phase 5 v1.
- **Why deferred**: Requires temporal data structure that the rest of
  the library doesn't model. Would require extending `Assignment` or
  introducing a new `TimeSeriesAssignment` type.
- **Trigger**: After Phase 6 pipeline is stable; novelty detection is a
  pipeline-level concern, not an estimation concern.

---

## Pipeline (Phase 6)

### Subgroup analysis in `ExperimentComparison`

- **What**: `ExperimentComparison` (shipped in Phase 6) compares scalar
  ATEs across independent experiments and rejects multi-effect results.
  Subgroup analysis (effects within slices of one experiment) is not in
  scope.
- **Why deferred**: Subgroup analysis raises multiple-testing issues that
  interact with `MultipleTestingCorrection` and need careful API
  thought. Independent-experiment comparison is the simpler case.
- **Trigger**: Now that `ExperimentComparison` v1 has shipped, revisit
  when users ask for subgroup support — likely a dedicated entry point
  that takes one experiment plus a subgroup column.

---

## Reporting (Phase 7)

### Interactive dashboards

- **What**: v1 of `reporting` produces static HTML reports.
- **Why deferred**: Interactive dashboards (Streamlit, Dash) are a
  separate concern with their own deployment story.
- **Trigger**: Outside the library's core scope. Could become a
  companion package `skxperiments-dashboards`.

---

## Cross-cutting

### Performance optimization of resampling inference (`RandomizationTest`, `BootstrapCI`, `AATest`)

- **What**: These loop in pure Python, refitting `estimator.fit()` once
  per resample — `n_permutations` calls to `assignment.draw()` for
  `RandomizationTest`, and `n_resamples` (plus an `n_obs` leave-one-out
  jackknife for BCa) assignment reconstructions for `BootstrapCI`. No
  vectorization, no parallelism, no caching of design matrices.
  `AATest` makes it worse: wrapping a resampling inference nests the
  loops, `O(n_simulations x n_resamples)` fits — hence the guidance to
  prefer the analytic `NeymanCI` for routine A/A calibration.
- **Why deferred**: Premature. Profiling on real workloads should drive
  optimization, not speculation.
- **Trigger**: Real reports of `RandomizationTest` being too slow.
  Likely solutions: (1) cache design matrix in estimator and only update
  the treatment column per permutation; (2) parallelize via `joblib` for
  `n_permutations >= 1000`; (3) special-case `DifferenceInMeans` to skip
  the estimator-fit overhead.

### Type-checked Assignment polymorphism

- **What**: `BaseEstimator.fit(assignment: Any)` uses `Any` because
  different estimators accept different `Assignment` subclasses.
  Runtime validation via `_validate_assignment_type`.
- **Why deferred**: Generic typing (`fit(assignment: AssignmentT)` with
  `AssignmentT` bound per subclass) would let mypy catch incompatible
  combinations statically, but adds noise to type signatures and
  generic variance issues.
- **Trigger**: When the project starts publishing `py.typed` and users
  rely on mypy for the public API.

### `inference_mode` as `BaseInference` attribute

- **What**: Currently a parameter passed to estimators (`LinEstimator`)
  and propagated via `Results.extra`. The original architectural plan
  was for `inference_mode` to live on `BaseInference` instances.
- **Why deferred**: Decided in Phase 3 that the read-side (Phase 4
  inference classes) will consume it from `Results.extra`. The "live on
  BaseInference" plan was redundant.
- **Trigger**: Phase 4.3 (`NeymanCI`) consumed `inference_mode` from
  `Results.extra` as planned and did not need to override it (it only
  reads the value to reject `"superpopulation"`). Reconsider a
  first-class attribute only if a later inference class needs to set it.

### CI-CD: required status checks with `paths-ignore`

- **What**: CI uses `paths-ignore` for docs-only changes. If GitHub
  required-status-checks are enabled, this creates "pending" PRs that
  can't merge.
- **Why deferred**: Repo is currently solo dev; required status checks
  not enabled.
- **Trigger**: When opening contributions to outsiders. Solution:
  replace `paths-ignore` with a "skip job" pattern that returns success
  immediately for ignored paths.

---

## How to use this document

When deferring a decision during development:

1. Add an entry under the relevant phase section.
2. Fill in **What**, **Why deferred**, and **Trigger**.
3. Reference the entry in the docstring of the affected class with a
   `TODO (see ROADMAP.md)`.
4. Mention in the relevant `CHANGELOG.md` entry under "Planned" or as
   a note.

When tackling a deferred item:

1. Move the entry from this document to the `CHANGELOG.md` under the
   appropriate `Added` or `Changed` section.
2. Remove the `TODO (see ROADMAP.md)` from the affected docstrings.