"""Tests for skxperiments.inference.bootstrap.BootstrapCI."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
    FactorialAssignment,
)
from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InsufficientDataError,
    InvalidDesignError,
    NotFittedError,
)
from skxperiments.core.results import Results
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.crd import CRD
from skxperiments.estimators.blocked_difference_in_means import (
    BlockedDifferenceInMeans,
)
from skxperiments.estimators.cuped import CUPED
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.estimators.lin_estimator import LinEstimator
from skxperiments.inference.bootstrap import BootstrapCI
from skxperiments.inference.neyman import NeymanCI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crd_assignment_with_ate(
    n: int,
    seed: int,
    true_ate: float = 0.0,
    p: float = 0.5,
) -> CRDAssignment:
    """Build a CRDAssignment with outcome Y = X + true_ate * T + noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(0.0, 1.0, n)})

    design = CRD(p=p, seed=seed)
    assignment = design.randomize(df)

    t = assignment.data_[assignment.treatment_col_].values.astype(float)
    x = assignment.data_["x"].values

    df_with_y = assignment.data_.copy()
    df_with_y["y"] = x + true_ate * t + rng.normal(0.0, 1.0, n)

    return CRDAssignment(
        data=df_with_y,
        treatment_col=assignment.treatment_col_,
        design=design,
        seed=seed,
    )


def _make_blocked_assignment_with_ate(
    n_per_block: int,
    n_blocks: int,
    seed: int,
    true_ate: float = 0.0,
) -> BlockedAssignment:
    """Build a BlockedAssignment with outcome Y = X + true_ate * T + noise."""
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks

    df = pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n),
            "block": np.repeat(np.arange(n_blocks), n_per_block),
        }
    )

    design = BlockedCRD(block_col="block", p=0.5, seed=seed)
    assignment = design.randomize(df)

    t = assignment.data_[assignment.treatment_col_].values.astype(float)
    x = assignment.data_["x"].values

    df_with_y = assignment.data_.copy()
    df_with_y["y"] = x + true_ate * t + rng.normal(0.0, 1.0, n)

    return BlockedAssignment(
        data=df_with_y,
        treatment_col=assignment.treatment_col_,
        design=design,
        block_col=assignment.block_col_,
        block_sizes=assignment.block_sizes_,
        seed=seed,
    )


def _make_cuped_assignment(
    n: int,
    seed: int,
    true_ate: float = 0.0,
    correlation: float = 0.7,
) -> CRDAssignment:
    """Build a CRDAssignment with y and a correlated pre-period y_pre."""
    rng = np.random.default_rng(seed)
    y_pre = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)
    y_baseline = correlation * y_pre + np.sqrt(1.0 - correlation**2) * noise

    df = pd.DataFrame({"y_pre": y_pre, "_y_base": y_baseline})

    design = CRD(p=0.5, seed=seed)
    assignment = design.randomize(df)

    t = assignment.data_[assignment.treatment_col_].values.astype(float)
    df_with_y = assignment.data_.copy()
    df_with_y["y"] = df_with_y["_y_base"].values + true_ate * t
    df_with_y = df_with_y.drop(columns=["_y_base"])

    return CRDAssignment(
        data=df_with_y,
        treatment_col=assignment.treatment_col_,
        design=design,
        seed=seed,
    )


def _fixed_crd_assignment(
    treatment: list[int],
    y: list[float],
) -> CRDAssignment:
    """Build a deterministic CRDAssignment from explicit treatment/outcome."""
    df = pd.DataFrame({"treatment": treatment, "y": y})
    return CRDAssignment(data=df, treatment_col="treatment", design=None)


def _fixed_blocked_assignment(
    block: list[int],
    treatment: list[int],
    y: list[float],
) -> BlockedAssignment:
    """Build a deterministic BlockedAssignment for fail-fast tests."""
    df = pd.DataFrame({"block": block, "treatment": treatment, "y": y})
    block_sizes = df.groupby("block").size().to_dict()
    return BlockedAssignment(
        data=df,
        treatment_col="treatment",
        design=None,
        block_col="block",
        block_sizes=block_sizes,
    )


def _make_factorial_assignment(n: int, seed: int) -> FactorialAssignment:
    """Build a FactorialAssignment directly (K=2, little-endian cells)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    cell = a + 2 * b
    df = pd.DataFrame(
        {"A": a, "B": b, "_cell": cell, "y": rng.normal(0.0, 1.0, n)}
    )
    cell_sizes = {int(c): int((cell == c).sum()) for c in range(4)}
    return FactorialAssignment(
        data=df,
        design=None,
        factor_cols=["A", "B"],
        cell_sizes=cell_sizes,
        seed=seed,
    )


class _MultiEffectDIM(DifferenceInMeans):
    """DifferenceInMeans subclass returning a multi-effect Results."""

    def estimate(self) -> Results:
        return Results(
            effects={("A",): 1.0},
            estimator_name=type(self).__name__,
        )


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


class TestBootstrapCICreation:
    """Tests for BootstrapCI.__init__ validations."""

    def test_basic_creation(self) -> None:
        """Defaults: method='bca', n_resamples=10_000, alpha=0.05."""
        ci = BootstrapCI(estimator=DifferenceInMeans(outcome_col="y"))
        assert ci.method == "bca"
        assert ci.n_resamples == 10_000
        assert ci.alpha == 0.05
        assert ci.seed is None

    def test_custom_parameters(self) -> None:
        """Custom method/n_resamples/alpha/seed are stored."""
        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"),
            method="percentile",
            n_resamples=500,
            alpha=0.10,
            seed=7,
        )
        assert ci.method == "percentile"
        assert ci.n_resamples == 500
        assert ci.alpha == 0.10
        assert ci.seed == 7

    def test_invalid_estimator_raises(self) -> None:
        """estimator must be a BaseEstimator instance."""
        with pytest.raises(InvalidDesignError, match="BaseEstimator"):
            BootstrapCI(estimator="not an estimator")  # type: ignore[arg-type]

    def test_invalid_method_raises(self) -> None:
        """method must be one of the supported options."""
        with pytest.raises(InvalidDesignError, match="method"):
            BootstrapCI(
                estimator=DifferenceInMeans(outcome_col="y"),
                method="studentized",
            )

    def test_invalid_n_resamples_zero_raises(self) -> None:
        """n_resamples == 0 is rejected."""
        with pytest.raises(InvalidDesignError):
            BootstrapCI(
                estimator=DifferenceInMeans(outcome_col="y"), n_resamples=0
            )

    def test_invalid_n_resamples_negative_raises(self) -> None:
        """n_resamples < 0 is rejected."""
        with pytest.raises(InvalidDesignError):
            BootstrapCI(
                estimator=DifferenceInMeans(outcome_col="y"), n_resamples=-5
            )

    def test_invalid_n_resamples_non_int_raises(self) -> None:
        """n_resamples must be int (float rejected)."""
        with pytest.raises(InvalidDesignError):
            BootstrapCI(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_resamples=100.0,  # type: ignore[arg-type]
            )

    def test_invalid_n_resamples_bool_raises(self) -> None:
        """n_resamples=True is rejected (bool subclasses int)."""
        with pytest.raises(InvalidDesignError):
            BootstrapCI(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_resamples=True,  # type: ignore[arg-type]
            )

    def test_invalid_alpha_range_raises(self) -> None:
        """alpha must be strictly in (0, 1)."""
        with pytest.raises(InvalidDesignError, match="alpha"):
            BootstrapCI(
                estimator=DifferenceInMeans(outcome_col="y"), alpha=1.0
            )

    def test_invalid_alpha_bool_raises(self) -> None:
        """alpha=True is rejected."""
        with pytest.raises(InvalidDesignError, match="alpha"):
            BootstrapCI(
                estimator=DifferenceInMeans(outcome_col="y"),
                alpha=True,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# 2. Validation in fit/estimate
# ---------------------------------------------------------------------------


class TestBootstrapCIValidation:
    """Tests for BootstrapCI.fit guards."""

    def test_rejects_factorial_assignment(self) -> None:
        """fit(FactorialAssignment) raises DesignEstimatorMismatch."""
        assignment = _make_factorial_assignment(n=40, seed=0)
        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"), n_resamples=100
        )
        with pytest.raises(DesignEstimatorMismatch):
            ci.fit(assignment)

    def test_rejects_multi_effect_estimator(self) -> None:
        """An estimator producing Results.effects (no ate) is rejected."""
        assignment = _make_crd_assignment_with_ate(n=30, seed=0)
        ci = BootstrapCI(
            estimator=_MultiEffectDIM(outcome_col="y"), n_resamples=100
        )
        with pytest.raises(
            InvalidDesignError, match="scalar|multi-effect|Results.ate"
        ):
            ci.fit(assignment)

    def test_estimate_before_fit_raises_not_fitted(self) -> None:
        """estimate() before fit() raises NotFittedError."""
        ci = BootstrapCI(estimator=DifferenceInMeans(outcome_col="y"))
        with pytest.raises(NotFittedError):
            ci.estimate()

    def test_insufficient_data_crd_arm(self) -> None:
        """A CRD arm with fewer than 2 units raises InsufficientDataError."""
        assignment = _fixed_crd_assignment(
            treatment=[1, 0, 0, 0], y=[1.0, 2.0, 3.0, 4.0]
        )
        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"), n_resamples=100
        )
        with pytest.raises(InsufficientDataError, match="treated arm"):
            ci.fit(assignment)

    def test_insufficient_data_matched_pair_blocked(self) -> None:
        """Matched-pair blocks (1 per arm) raise InsufficientDataError."""
        assignment = _fixed_blocked_assignment(
            block=[0, 0, 1, 1],
            treatment=[1, 0, 1, 0],
            y=[1.0, 2.0, 3.0, 4.0],
        )
        ci = BootstrapCI(
            estimator=BlockedDifferenceInMeans(outcome_col="y"),
            n_resamples=100,
        )
        with pytest.raises(InsufficientDataError, match="block"):
            ci.fit(assignment)

    def test_bca_undefined_on_constant_outcome(self) -> None:
        """BCa raises when the bootstrap distribution is degenerate."""
        # Outcomes constant within each arm -> every resample yields the
        # same ATE, so prop_less is 0 and z0 is undefined.
        assignment = _fixed_crd_assignment(
            treatment=[1, 1, 1, 0, 0, 0],
            y=[5.0, 5.0, 5.0, 2.0, 2.0, 2.0],
        )
        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"),
            method="bca",
            n_resamples=200,
            seed=0,
        )
        with pytest.raises(InvalidDesignError, match="percentile"):
            ci.fit(assignment)


# ---------------------------------------------------------------------------
# 3. Fit behavior
# ---------------------------------------------------------------------------


class TestBootstrapCIFit:
    """Tests for BootstrapCI.fit behavior and fitted attributes."""

    def test_fit_returns_self(self) -> None:
        """fit returns self."""
        assignment = _make_crd_assignment_with_ate(n=40, seed=0)
        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_resamples=200,
            seed=0,
        )
        assert ci.fit(assignment) is ci

    def test_fitted_attributes_exist(self) -> None:
        """After fit, fitted attributes have the right types."""
        assignment = _make_crd_assignment_with_ate(n=40, seed=0)
        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_resamples=200,
            seed=0,
        )
        ci.fit(assignment)

        assert isinstance(ci.assignment_, CRDAssignment)
        assert isinstance(ci.observed_statistic_, float)
        assert isinstance(ci.bootstrap_distribution_, np.ndarray)
        assert isinstance(ci.se_, float)
        assert isinstance(ci.ci_, tuple)
        assert isinstance(ci.p_value_, float)

    def test_distribution_length(self) -> None:
        """bootstrap_distribution_ has length n_resamples."""
        assignment = _make_crd_assignment_with_ate(n=40, seed=0)
        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_resamples=300,
            seed=0,
        )
        ci.fit(assignment)
        assert len(ci.bootstrap_distribution_) == 300

    def test_observed_matches_difference_in_means(self) -> None:
        """observed_statistic_ equals DIM.fit(assignment).estimate().ate."""
        assignment = _make_crd_assignment_with_ate(
            n=50, seed=0, true_ate=1.5
        )
        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_resamples=200,
            seed=0,
        )
        ci.fit(assignment)

        dim = DifferenceInMeans(outcome_col="y")
        expected = dim.fit(assignment).estimate().ate
        assert ci.observed_statistic_ == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 4. Estimate behavior
# ---------------------------------------------------------------------------


class TestBootstrapCIEstimate:
    """Tests for BootstrapCI.estimate output."""

    def _fitted(self, method: str = "bca") -> BootstrapCI:
        assignment = _make_crd_assignment_with_ate(
            n=60, seed=0, true_ate=1.0
        )
        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"),
            method=method,
            n_resamples=400,
            seed=0,
        )
        ci.fit(assignment)
        return ci

    def test_estimate_returns_results(self) -> None:
        """estimate() returns a Results instance."""
        assert isinstance(self._fitted().estimate(), Results)

    def test_inference_name(self) -> None:
        """inference_name == 'BootstrapCI'."""
        assert self._fitted().estimate().inference_name == "BootstrapCI"

    def test_se_positive_and_ci_contains_ate(self) -> None:
        """se > 0 and the CI brackets the observed ATE."""
        result = self._fitted().estimate()
        assert result.se > 0.0
        assert result.ci[0] <= result.ate <= result.ci[1]

    def test_extra_common_keys(self) -> None:
        """extra carries method, n_resamples, distribution, inference_mode."""
        result = self._fitted().estimate()
        assert result.extra["method"] == "bca"
        assert result.extra["n_resamples"] == 400
        assert isinstance(result.extra["bootstrap_distribution"], np.ndarray)
        assert len(result.extra["bootstrap_distribution"]) == 400
        assert result.extra["inference_mode"] == "superpopulation"

    def test_bca_extra_has_correction_and_acceleration(self) -> None:
        """BCa output exposes bias_correction and acceleration."""
        result = self._fitted(method="bca").estimate()
        assert "bias_correction" in result.extra
        assert "acceleration" in result.extra
        assert isinstance(result.extra["bias_correction"], float)
        assert isinstance(result.extra["acceleration"], float)

    def test_percentile_extra_has_no_bca_keys(self) -> None:
        """Percentile output omits the BCa-specific keys."""
        result = self._fitted(method="percentile").estimate()
        assert "bias_correction" not in result.extra
        assert "acceleration" not in result.extra

    def test_overrides_inference_mode_to_superpopulation(self) -> None:
        """inference_mode is superpopulation even if the estimator differs."""
        # LinEstimator writes inference_mode='finite_population'; BootstrapCI
        # must override it.
        assignment = _make_crd_assignment_with_ate(
            n=50, seed=0, true_ate=1.0
        )
        ci = BootstrapCI(
            estimator=LinEstimator(outcome_col="y", covariates=["x"]),
            method="percentile",
            n_resamples=200,
            seed=0,
        )
        result = ci.fit(assignment).estimate()
        assert result.extra["inference_mode"] == "superpopulation"


# ---------------------------------------------------------------------------
# 5. Reproducibility
# ---------------------------------------------------------------------------


class TestBootstrapCIReproducibility:
    """Tests for deterministic behavior given a fixed seed."""

    def test_same_seed_same_interval(self) -> None:
        """Two BootstrapCIs with the same seed produce identical output."""
        assignment = _make_crd_assignment_with_ate(n=40, seed=0)

        def _run() -> BootstrapCI:
            ci = BootstrapCI(
                estimator=DifferenceInMeans(outcome_col="y"),
                method="bca",
                n_resamples=300,
                seed=123,
            )
            return ci.fit(assignment)

        a = _run()
        b = _run()
        assert np.array_equal(
            a.bootstrap_distribution_, b.bootstrap_distribution_
        )
        assert a.ci_ == b.ci_
        assert a.se_ == b.se_


# ---------------------------------------------------------------------------
# 6. Methods
# ---------------------------------------------------------------------------


class TestBootstrapCIMethods:
    """Tests comparing percentile and BCa."""

    def test_percentile_constant_data(self) -> None:
        """On constant-within-arm data, percentile CI collapses to the ATE."""
        assignment = _fixed_crd_assignment(
            treatment=[1, 1, 1, 0, 0, 0],
            y=[5.0, 5.0, 5.0, 2.0, 2.0, 2.0],
        )
        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"),
            method="percentile",
            n_resamples=200,
            seed=0,
        )
        result = ci.fit(assignment).estimate()
        assert result.ci[0] == pytest.approx(3.0)
        assert result.ci[1] == pytest.approx(3.0)
        assert result.se == pytest.approx(0.0)

    @pytest.mark.slow
    def test_bca_close_to_percentile_on_symmetric_data(self) -> None:
        """With negligible bias/skew, BCa and percentile intervals agree."""
        assignment = _make_crd_assignment_with_ate(
            n=200, seed=1, true_ate=1.0
        )

        def _ci(method: str) -> tuple[float, float]:
            est = BootstrapCI(
                estimator=DifferenceInMeans(outcome_col="y"),
                method=method,
                n_resamples=3000,
                seed=2,
            )
            return est.fit(assignment).estimate().ci

        lo_p, hi_p = _ci("percentile")
        lo_b, hi_b = _ci("bca")
        width = hi_p - lo_p
        assert abs(lo_b - lo_p) < 0.15 * width
        assert abs(hi_b - hi_p) < 0.15 * width


# ---------------------------------------------------------------------------
# 7. Blocked
# ---------------------------------------------------------------------------


class TestBootstrapCIBlocked:
    """Tests for BootstrapCI with BlockedAssignment."""

    def test_accepts_blocked_assignment(self) -> None:
        """fit() accepts a BlockedAssignment and yields a finite interval."""
        assignment = _make_blocked_assignment_with_ate(
            n_per_block=10, n_blocks=4, seed=0, true_ate=0.5
        )
        ci = BootstrapCI(
            estimator=BlockedDifferenceInMeans(outcome_col="y"),
            method="bca",
            n_resamples=300,
            seed=0,
        )
        result = ci.fit(assignment).estimate()
        assert np.isfinite(result.ci[0])
        assert np.isfinite(result.ci[1])
        assert result.ci[0] <= result.ate <= result.ci[1]


# ---------------------------------------------------------------------------
# 8. Estimator-agnostic smoke
# ---------------------------------------------------------------------------


class TestBootstrapCIEstimatorAgnostic:
    """BootstrapCI works with any scalar estimator."""

    def test_works_with_lin_estimator(self) -> None:
        assignment = _make_crd_assignment_with_ate(
            n=60, seed=0, true_ate=1.0
        )
        ci = BootstrapCI(
            estimator=LinEstimator(outcome_col="y", covariates=["x"]),
            n_resamples=200,
            seed=0,
        )
        result = ci.fit(assignment).estimate()
        assert np.isfinite(result.se)

    def test_works_with_cuped(self) -> None:
        assignment = _make_cuped_assignment(
            n=60, seed=0, true_ate=1.0, correlation=0.7
        )
        ci = BootstrapCI(
            estimator=CUPED(outcome_col="y", pre_experiment_col="y_pre"),
            n_resamples=200,
            seed=0,
        )
        result = ci.fit(assignment).estimate()
        assert np.isfinite(result.se)


# ---------------------------------------------------------------------------
# 9. Immutability
# ---------------------------------------------------------------------------


class TestBootstrapCIImmutability:
    """fit must not mutate the assignment's data."""

    def test_fit_does_not_mutate_assignment(self) -> None:
        """assignment.data_ is unchanged after fit/estimate."""
        assignment = _make_crd_assignment_with_ate(
            n=40, seed=0, true_ate=1.0
        )
        before = assignment.data_.copy()

        ci = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_resamples=200,
            seed=0,
        )
        ci.fit(assignment).estimate()

        pd.testing.assert_frame_equal(assignment.data_, before)


# ---------------------------------------------------------------------------
# 10. Numerics (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBootstrapCINumerics:
    """Statistical property tests for BootstrapCI (slow)."""

    def test_crd_coverage_near_nominal(self) -> None:
        """A 95% percentile interval covers the true ATE in ~95% of reps."""
        n_reps = 120
        true_ate = 1.0
        covered = 0
        for i in range(n_reps):
            assignment = _make_crd_assignment_with_ate(
                n=60, seed=i, true_ate=true_ate
            )
            ci = BootstrapCI(
                estimator=DifferenceInMeans(outcome_col="y"),
                method="percentile",
                n_resamples=499,
                seed=10_000 + i,
            )
            result = ci.fit(assignment).estimate()
            if result.ci[0] <= true_ate <= result.ci[1]:
                covered += 1

        coverage = covered / n_reps
        assert coverage >= 0.88, (
            f"Empirical coverage {coverage:.3f} below tolerance."
        )

    def test_bootstrap_se_agrees_with_neyman(self) -> None:
        """Bootstrap SE matches the Neyman SE on a large CRD sample."""
        assignment = _make_crd_assignment_with_ate(
            n=400, seed=3, true_ate=1.0
        )

        boot = BootstrapCI(
            estimator=DifferenceInMeans(outcome_col="y"),
            method="percentile",
            n_resamples=4000,
            seed=4,
        )
        boot_se = boot.fit(assignment).estimate().se

        neyman = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        neyman_se = neyman.fit(assignment).estimate().se

        assert boot_se == pytest.approx(neyman_se, rel=0.10)
