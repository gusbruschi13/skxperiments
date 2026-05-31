"""Tests for skxperiments.estimators.cuped."""

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
    InvalidDesignError,
    NotFittedError,
)
from skxperiments.core.results import Results
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.crd import CRD
from skxperiments.design.factorial import FactorialDesign
from skxperiments.estimators.cuped import CUPED
from skxperiments.estimators.difference_in_means import DifferenceInMeans


# --- Helpers ---


def _make_crd_with_pre_experiment(
    n: int = 200,
    seed: int = 42,
    true_ate: float = 0.5,
    correlation: float = 0.7,
) -> CRDAssignment:
    """Build CRDAssignment with pre-experiment covariate correlated
    with outcome.

    x_pre ~ N(0, 1)
    y_baseline = correlation * x_pre + sqrt(1 - correlation**2) * epsilon
    y = y_baseline + true_ate * T

    With this construction, both x_pre and y_baseline have unit
    variance (asymptotically), so theta_ ~ correlation and
    correlation_ ~ correlation.

    # Mutação de data_ aceitável apenas em fixture de teste.
    """
    rng = np.random.default_rng(seed)
    x_pre = rng.normal(size=n)
    epsilon = rng.normal(size=n)

    # Outcome with controlled correlation between y and x_pre.
    y_baseline = (
        correlation * x_pre
        + np.sqrt(max(1.0 - correlation**2, 0.0)) * epsilon
    )

    df = pd.DataFrame(
        {
            "y_pre": x_pre,
            "y": y_baseline,
        }
    )

    design = CRD(p=0.5, seed=seed)
    assignment = design.randomize(df)

    treated = assignment.treated_ids()
    assignment.data_.iloc[
        treated, assignment.data_.columns.get_loc("y")
    ] += true_ate
    return assignment


# --- Tests ---


class TestCUPEDCreation:
    """Tests for CUPED instantiation."""

    def test_basic_creation(self) -> None:
        """Should instantiate with required arguments."""
        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        assert estimator.outcome_col == "y"
        assert estimator.pre_experiment_col == "y_pre"

    def test_columns_stored_verbatim(self) -> None:
        """Column names stored unchanged."""
        estimator = CUPED(
            outcome_col="metric_post", pre_experiment_col="metric_pre"
        )
        assert estimator.outcome_col == "metric_post"
        assert estimator.pre_experiment_col == "metric_pre"


class TestCUPEDInitValidation:
    """Tests for validation in __init__."""

    def test_same_column_raises(self) -> None:
        """outcome_col == pre_experiment_col must raise."""
        with pytest.raises(InvalidDesignError, match="must differ"):
            CUPED(outcome_col="y", pre_experiment_col="y")

    def test_outcome_col_non_string_raises(self) -> None:
        """Non-string outcome_col must raise."""
        with pytest.raises(InvalidDesignError, match="outcome_col"):
            CUPED(
                outcome_col=42,  # type: ignore[arg-type]
                pre_experiment_col="y_pre",
            )

    def test_outcome_col_empty_raises(self) -> None:
        """Empty outcome_col must raise."""
        with pytest.raises(InvalidDesignError, match="outcome_col"):
            CUPED(outcome_col="", pre_experiment_col="y_pre")

    def test_pre_experiment_col_non_string_raises(self) -> None:
        """Non-string pre_experiment_col must raise."""
        with pytest.raises(InvalidDesignError, match="pre_experiment_col"):
            CUPED(
                outcome_col="y",
                pre_experiment_col=None,  # type: ignore[arg-type]
            )

    def test_pre_experiment_col_empty_raises(self) -> None:
        """Empty pre_experiment_col must raise."""
        with pytest.raises(InvalidDesignError, match="pre_experiment_col"):
            CUPED(outcome_col="y", pre_experiment_col="")


class TestCUPEDFit:
    """Tests for CUPED.fit."""

    def test_fit_returns_self(self) -> None:
        """fit must return self for chaining."""
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        returned = estimator.fit(assignment)
        assert returned is estimator

    def test_stores_assignment(self) -> None:
        """fit stores assignment_."""
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)
        assert estimator.assignment_ is assignment

    def test_stores_ate(self) -> None:
        """fit computes and stores ate_ as a float."""
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)
        assert isinstance(estimator.ate_, float)

    def test_stores_theta_as_float(self) -> None:
        """fit computes and stores theta_ as a float."""
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)
        assert isinstance(estimator.theta_, float)

    def test_stores_correlation_as_float(self) -> None:
        """fit computes and stores correlation_ as a float."""
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)
        assert isinstance(estimator.correlation_, float)

    def test_correlation_in_valid_range(self) -> None:
        """correlation_ must be in [-1, 1]."""
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)
        assert -1.0 <= estimator.correlation_ <= 1.0


class TestCUPEDEstimate:
    """Tests for CUPED.estimate."""

    def test_returns_results(self) -> None:
        """estimate must return a Results instance."""
        assignment = _make_crd_with_pre_experiment()
        result = (
            CUPED(outcome_col="y", pre_experiment_col="y_pre")
            .fit(assignment)
            .estimate()
        )
        assert isinstance(result, Results)

    def test_ate_matches_manual_formula(self) -> None:
        """Results.ate must equal dim_y - theta * dim_x within
        rel=1e-12.
        """
        assignment = _make_crd_with_pre_experiment(
            n=300, seed=99, true_ate=1.0, correlation=0.5
        )
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)

        # Recompute manually.
        data = assignment.data_
        y = data["y"].values.astype(float)
        x = data["y_pre"].values.astype(float)
        treated = assignment.treated_ids()
        control = assignment.control_ids()

        cov_yx = float(np.cov(y, x, ddof=1)[0, 1])
        var_x = float(np.var(x, ddof=1))
        theta = cov_yx / var_x

        dim_y = float(y[treated].mean() - y[control].mean())
        dim_x = float(x[treated].mean() - x[control].mean())
        expected_ate = dim_y - theta * dim_x

        result = estimator.estimate()
        assert result.ate == pytest.approx(expected_ate, rel=1e-12)

    def test_extra_theta_propagated(self) -> None:
        """extra['theta'] must equal theta_."""
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)
        result = estimator.estimate()
        assert result.extra is not None
        assert result.extra["theta"] == estimator.theta_

    def test_extra_correlation_propagated(self) -> None:
        """extra['correlation'] must equal correlation_."""
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)
        result = estimator.estimate()
        assert result.extra is not None
        assert result.extra["correlation"] == estimator.correlation_

    def test_estimator_name(self) -> None:
        """estimator_name must be 'CUPED'."""
        assignment = _make_crd_with_pre_experiment()
        result = (
            CUPED(outcome_col="y", pre_experiment_col="y_pre")
            .fit(assignment)
            .estimate()
        )
        assert result.estimator_name == "CUPED"

    def test_design_name_for_crd(self) -> None:
        """design_name must be 'CRD' for CRDAssignment from CRD."""
        assignment = _make_crd_with_pre_experiment()
        result = (
            CUPED(outcome_col="y", pre_experiment_col="y_pre")
            .fit(assignment)
            .estimate()
        )
        assert result.design_name == "CRD"

    def test_inference_fields_are_none(self) -> None:
        """se, ci, p_value must be None."""
        assignment = _make_crd_with_pre_experiment()
        result = (
            CUPED(outcome_col="y", pre_experiment_col="y_pre")
            .fit(assignment)
            .estimate()
        )
        assert result.se is None
        assert result.ci is None
        assert result.p_value is None


class TestCUPEDValidation:
    """Tests for validation in fit."""

    def test_rejects_blocked_assignment(self) -> None:
        """fit must reject BlockedAssignment with v2 message."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=40),
                "block": ["A"] * 20 + ["B"] * 20,
                "y": rng.normal(size=40),
                "y_pre": rng.normal(size=40),
            }
        )
        assignment = BlockedCRD(
            block_col="block", p=0.5, seed=0
        ).randomize(df)
        assert isinstance(assignment, BlockedAssignment)

        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        with pytest.raises(DesignEstimatorMismatch) as exc:
            estimator.fit(assignment)
        # Error from _validate_assignment_type lists CRDAssignment.
        assert "CRDAssignment" in str(exc.value)

    def test_rejects_factorial_assignment(self) -> None:
        """fit must reject FactorialAssignment."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=40),
                "y": rng.normal(size=40),
                "y_pre": rng.normal(size=40),
            }
        )
        assignment = FactorialDesign(
            factors=["A", "B"], n_per_cell=10, seed=42
        ).randomize(df)
        assert isinstance(assignment, FactorialAssignment)

        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        with pytest.raises(DesignEstimatorMismatch, match="CRDAssignment"):
            estimator.fit(assignment)

    def test_missing_outcome_col_raises(self) -> None:
        """fit raises if outcome_col absent."""
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(
            outcome_col="missing", pre_experiment_col="y_pre"
        )
        with pytest.raises(InvalidDesignError, match="not found"):
            estimator.fit(assignment)

    def test_non_numeric_outcome_raises(self) -> None:
        """fit raises if outcome is non-numeric."""
        assignment = _make_crd_with_pre_experiment()
        assignment.data_["y"] = ["a"] * len(assignment.data_)
        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        with pytest.raises(InvalidDesignError, match="numeric"):
            estimator.fit(assignment)

    def test_outcome_with_nan_raises(self) -> None:
        """fit raises if outcome contains NaN."""
        assignment = _make_crd_with_pre_experiment()
        assignment.data_.iat[
            0, assignment.data_.columns.get_loc("y")
        ] = np.nan
        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        with pytest.raises(InvalidDesignError, match="NaN"):
            estimator.fit(assignment)

    def test_missing_pre_experiment_col_raises(self) -> None:
        """fit raises if pre_experiment_col absent."""
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(outcome_col="y", pre_experiment_col="missing")
        with pytest.raises(InvalidDesignError, match="not found"):
            estimator.fit(assignment)

    def test_non_numeric_pre_experiment_col_raises(self) -> None:
        """fit raises if pre_experiment_col is non-numeric."""
        assignment = _make_crd_with_pre_experiment()
        assignment.data_["y_pre"] = ["a"] * len(assignment.data_)
        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        with pytest.raises(InvalidDesignError, match="numeric"):
            estimator.fit(assignment)

    def test_pre_experiment_col_with_nan_raises(self) -> None:
        """fit raises if pre_experiment_col contains NaN."""
        assignment = _make_crd_with_pre_experiment()
        assignment.data_.iat[
            0, assignment.data_.columns.get_loc("y_pre")
        ] = np.nan
        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        with pytest.raises(InvalidDesignError, match="NaN"):
            estimator.fit(assignment)

    def test_constant_pre_experiment_col_raises(self) -> None:
        """fit raises if pre_experiment_col has zero variance."""
        assignment = _make_crd_with_pre_experiment()
        assignment.data_["y_pre"] = 5.0  # constant
        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        with pytest.raises(InvalidDesignError, match="zero variance"):
            estimator.fit(assignment)

    def test_estimate_before_fit_raises(self) -> None:
        """estimate raises NotFittedError if fit not called."""
        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        with pytest.raises(NotFittedError):
            estimator.estimate()


class TestCUPEDNumerics:
    """Tests for numerical correctness."""

    def test_equivalence_with_manual_formula(self) -> None:
        """ate_ must equal dim_y - theta * dim_x to within rel=1e-12."""
        assignment = _make_crd_with_pre_experiment(
            n=300, seed=11, true_ate=0.7, correlation=0.6
        )
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)

        data = assignment.data_
        y = data["y"].values.astype(float)
        x = data["y_pre"].values.astype(float)

        cov_yx = float(np.cov(y, x, ddof=1)[0, 1])
        var_x = float(np.var(x, ddof=1))
        theta = cov_yx / var_x

        treated = assignment.treated_ids()
        control = assignment.control_ids()
        dim_y = float(y[treated].mean() - y[control].mean())
        dim_x = float(x[treated].mean() - x[control].mean())
        expected = dim_y - theta * dim_x

        assert estimator.ate_ == pytest.approx(expected, rel=1e-12)
        # theta_ also matches.
        assert estimator.theta_ == pytest.approx(theta, rel=1e-12)

    def test_variance_reduction_vs_dim(self) -> None:
        """Monte Carlo: with correlation=0.7, CUPED must reduce
        variance vs. DifferenceInMeans by at least 30%.

        Each repetition uses seed=i for i in range(100) for
        reproducibility in CI.
        """
        n_reps = 100
        n = 200
        true_ate = 0.5
        correlation = 0.7

        cuped_ates = np.empty(n_reps)
        dim_ates = np.empty(n_reps)
        for i in range(n_reps):
            assignment = _make_crd_with_pre_experiment(
                n=n,
                seed=i,
                true_ate=true_ate,
                correlation=correlation,
            )
            cuped_ates[i] = (
                CUPED(outcome_col="y", pre_experiment_col="y_pre")
                .fit(assignment)
                .ate_
            )
            dim_ates[i] = (
                DifferenceInMeans(outcome_col="y").fit(assignment).ate_
            )

        var_cuped = float(np.var(cuped_ates, ddof=1))
        var_dim = float(np.var(dim_ates, ddof=1))
        # With correlation=0.7, theoretical reduction ~ 1 - 0.49 = 51%;
        # 30% is a generous lower bound for finite-sample noise.
        assert var_cuped < 0.7 * var_dim, (
            f"Variance reduction insufficient: var_cuped={var_cuped:.4f}, "
            f"var_dim={var_dim:.4f}"
        )

    def test_collapses_to_dim_with_zero_correlation(self) -> None:
        """With correlation=0, CUPED.ate_ ~ DifferenceInMeans.ate_
        within abs=0.1.
        """
        assignment = _make_crd_with_pre_experiment(
            n=500, seed=7, true_ate=1.0, correlation=0.0
        )
        cuped_ate = (
            CUPED(outcome_col="y", pre_experiment_col="y_pre")
            .fit(assignment)
            .ate_
        )
        dim_ate = (
            DifferenceInMeans(outcome_col="y").fit(assignment).ate_
        )
        assert abs(cuped_ate - dim_ate) < 0.1

    def test_equivalence_with_simple_regression(self) -> None:
        """CUPED with one covariate is mathematically equivalent to
        OLS of Y on [1, T, X_pre] without interaction. The coefficient
        of T from that regression must equal CUPED.ate_ to within
        rel=1e-10.

        Note: this equivalence does NOT hold for LinEstimator, which
        adds the interaction term T * X_centered.
        """
        assignment = _make_crd_with_pre_experiment(
            n=300, seed=42, true_ate=0.5, correlation=0.5
        )
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)

        data = assignment.data_
        y = data["y"].values.astype(float)
        x = data["y_pre"].values.astype(float).reshape(-1, 1)
        T = (
            data[assignment.treatment_col_]
            .values.astype(float)
            .reshape(-1, 1)
        )

        n = len(y)
        intercept = np.ones((n, 1))
        # No interaction term, no centering needed for OLS coefficient
        # of T to match CUPED (in the limit; in finite sample, this
        # is also exact when both methods use the same data).
        design_matrix = np.hstack([intercept, T, x])
        coef, *_ = np.linalg.lstsq(design_matrix, y, rcond=None)
        # coef[1] is the T coefficient.

        assert estimator.ate_ == pytest.approx(float(coef[1]), rel=1e-10)

    def test_theta_magnitude_with_known_correlation(self) -> None:
        """With correlation=0.7 and unit variances, theta_ ~ 0.7
        within abs=0.15 (loose tolerance for sample variability at n=200).
        """
        assignment = _make_crd_with_pre_experiment(
            n=200, seed=42, true_ate=0.5, correlation=0.7
        )
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)
        assert estimator.theta_ == pytest.approx(0.7, abs=0.15)

    def test_invariance_under_repeated_estimate(self) -> None:
        """fit().estimate() called twice on the same assignment yields
        identical ATE.
        """
        assignment = _make_crd_with_pre_experiment()
        estimator = CUPED(
            outcome_col="y", pre_experiment_col="y_pre"
        ).fit(assignment)
        r1 = estimator.estimate()
        r2 = estimator.estimate()
        assert r1.ate == r2.ate