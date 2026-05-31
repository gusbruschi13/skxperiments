"""Tests for skxperiments.estimators.lin_estimator."""

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
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.estimators.lin_estimator import LinEstimator


# --- Helpers ---


def _make_crd_with_covariates(
    n: int = 200,
    seed: int = 42,
    true_ate: float = 0.5,
    cov_effect: float = 0.3,
) -> CRDAssignment:
    """Build CRDAssignment with covariate that predicts outcome.

    Outcome model:
        y = true_ate * T + cov_effect * x + epsilon

    where epsilon ~ N(0, 1). With cov_effect > 0, x is correlated
    with y; LinEstimator should reduce variance compared to
    DifferenceInMeans.

    # Mutação de data_ aceitável apenas em fixture de teste.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    epsilon = rng.normal(size=n)

    df = pd.DataFrame(
        {
            "x": x,
            "y": cov_effect * x + epsilon,  # baseline; T effect added later
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


class TestLinEstimatorCreation:
    """Tests for LinEstimator instantiation."""

    def test_basic_creation(self) -> None:
        """Should instantiate with required arguments."""
        estimator = LinEstimator(outcome_col="y", covariates=["x"])
        assert estimator.outcome_col == "y"
        assert estimator.covariates == ["x"]

    def test_inference_mode_default(self) -> None:
        """inference_mode default must be 'finite_population'."""
        estimator = LinEstimator(outcome_col="y", covariates=["x"])
        assert estimator.inference_mode == "finite_population"

    def test_inference_mode_explicit(self) -> None:
        """inference_mode explicitly set must be stored."""
        estimator = LinEstimator(
            outcome_col="y",
            covariates=["x"],
            inference_mode="superpopulation",
        )
        assert estimator.inference_mode == "superpopulation"

    def test_covariates_list_stored_verbatim(self) -> None:
        """covariates list must be stored unchanged."""
        estimator = LinEstimator(
            outcome_col="y", covariates=["x1", "x2", "x3"]
        )
        assert estimator.covariates == ["x1", "x2", "x3"]


class TestLinEstimatorInitValidation:
    """Tests for validation in __init__."""

    def test_empty_covariates_raises(self) -> None:
        """Empty list must raise with DifferenceInMeans suggestion."""
        with pytest.raises(InvalidDesignError, match="DifferenceInMeans"):
            LinEstimator(outcome_col="y", covariates=[])

    def test_non_list_covariates_raises(self) -> None:
        """Non-list covariates must raise."""
        with pytest.raises(InvalidDesignError, match="list of strings"):
            LinEstimator(
                outcome_col="y", covariates="x"  # type: ignore[arg-type]
            )

    def test_non_string_covariate_element_raises(self) -> None:
        """Non-string element in covariates must raise."""
        with pytest.raises(InvalidDesignError, match="strings"):
            LinEstimator(
                outcome_col="y",
                covariates=["x", 42],  # type: ignore[list-item]
            )

    def test_invalid_inference_mode_raises(self) -> None:
        """Invalid inference_mode must raise."""
        with pytest.raises(InvalidDesignError, match="inference_mode"):
            LinEstimator(
                outcome_col="y",
                covariates=["x"],
                inference_mode="bayesian",
            )


class TestLinEstimatorFit:
    """Tests for LinEstimator.fit."""

    def test_fit_returns_self(self) -> None:
        """fit must return self for chaining."""
        assignment = _make_crd_with_covariates()
        estimator = LinEstimator(outcome_col="y", covariates=["x"])
        returned = estimator.fit(assignment)
        assert returned is estimator

    def test_accepts_crd_assignment(self) -> None:
        """fit must accept CRDAssignment."""
        assignment = _make_crd_with_covariates()
        estimator = LinEstimator(
            outcome_col="y", covariates=["x"]
        ).fit(assignment)
        assert isinstance(estimator.ate_, float)

    def test_accepts_blocked_assignment(self) -> None:
        """fit must accept BlockedAssignment."""
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "x": rng.normal(size=n),
                "block": ["A"] * 100 + ["B"] * 100,
                "y": rng.normal(size=n),
            }
        )
        assignment = BlockedCRD(
            block_col="block", p=0.5, seed=0
        ).randomize(df)
        assert isinstance(assignment, BlockedAssignment)

        estimator = LinEstimator(
            outcome_col="y", covariates=["x"]
        ).fit(assignment)
        assert isinstance(estimator.ate_, float)

    def test_stores_assignment(self) -> None:
        """fit stores assignment_."""
        assignment = _make_crd_with_covariates()
        estimator = LinEstimator(
            outcome_col="y", covariates=["x"]
        ).fit(assignment)
        assert estimator.assignment_ is assignment

    def test_stores_inference_mode(self) -> None:
        """fit stores inference_mode_ as a copy of inference_mode."""
        assignment = _make_crd_with_covariates()
        estimator = LinEstimator(
            outcome_col="y",
            covariates=["x"],
            inference_mode="superpopulation",
        ).fit(assignment)
        assert estimator.inference_mode_ == "superpopulation"

    def test_coefficients_shape_one_covariate(self) -> None:
        """coefficients_ shape must be (2 + 2*K,) = (4,) for K=1."""
        assignment = _make_crd_with_covariates()
        estimator = LinEstimator(
            outcome_col="y", covariates=["x"]
        ).fit(assignment)
        assert estimator.coefficients_.shape == (4,)

    def test_coefficients_shape_multiple_covariates(self) -> None:
        """coefficients_ shape must be (2 + 2*K,) for K covariates."""
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "x3": rng.normal(size=n),
                "y": rng.normal(size=n),
            }
        )
        assignment = CRD(p=0.5, seed=0).randomize(df)
        estimator = LinEstimator(
            outcome_col="y", covariates=["x1", "x2", "x3"]
        ).fit(assignment)
        # 2 + 2 * 3 = 8
        assert estimator.coefficients_.shape == (8,)


class TestLinEstimatorEstimate:
    """Tests for LinEstimator.estimate."""

    def test_returns_results(self) -> None:
        """estimate must return a Results instance."""
        assignment = _make_crd_with_covariates()
        result = (
            LinEstimator(outcome_col="y", covariates=["x"])
            .fit(assignment)
            .estimate()
        )
        assert isinstance(result, Results)

    def test_ate_matches_coefficients_index_1(self) -> None:
        """Results.ate must equal coefficients_[1] within tight tolerance."""
        assignment = _make_crd_with_covariates()
        estimator = LinEstimator(
            outcome_col="y", covariates=["x"]
        ).fit(assignment)
        result = estimator.estimate()
        assert result.ate == pytest.approx(
            float(estimator.coefficients_[1]), rel=1e-12
        )

    def test_extra_inference_mode_propagated(self) -> None:
        """extra['inference_mode'] must be propagated."""
        assignment = _make_crd_with_covariates()
        result = (
            LinEstimator(
                outcome_col="y",
                covariates=["x"],
                inference_mode="superpopulation",
            )
            .fit(assignment)
            .estimate()
        )
        assert result.extra is not None
        assert result.extra["inference_mode"] == "superpopulation"

    def test_estimator_name(self) -> None:
        """estimator_name must be 'LinEstimator'."""
        assignment = _make_crd_with_covariates()
        result = (
            LinEstimator(outcome_col="y", covariates=["x"])
            .fit(assignment)
            .estimate()
        )
        assert result.estimator_name == "LinEstimator"

    def test_design_name_for_crd(self) -> None:
        """design_name must be 'CRD' for CRDAssignment."""
        assignment = _make_crd_with_covariates()
        result = (
            LinEstimator(outcome_col="y", covariates=["x"])
            .fit(assignment)
            .estimate()
        )
        assert result.design_name == "CRD"

    def test_design_name_for_blocked(self) -> None:
        """design_name must be 'BlockedCRD' for BlockedAssignment."""
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "x": rng.normal(size=n),
                "block": ["A"] * 100 + ["B"] * 100,
                "y": rng.normal(size=n),
            }
        )
        assignment = BlockedCRD(
            block_col="block", p=0.5, seed=0
        ).randomize(df)
        result = (
            LinEstimator(outcome_col="y", covariates=["x"])
            .fit(assignment)
            .estimate()
        )
        assert result.design_name == "BlockedCRD"

    def test_inference_fields_are_none(self) -> None:
        """se, ci, p_value must be None."""
        assignment = _make_crd_with_covariates()
        result = (
            LinEstimator(outcome_col="y", covariates=["x"])
            .fit(assignment)
            .estimate()
        )
        assert result.se is None
        assert result.ci is None
        assert result.p_value is None


class TestLinEstimatorValidation:
    """Tests for validation in fit."""

    def test_rejects_factorial_assignment(self) -> None:
        """fit must reject FactorialAssignment."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=40),
                "y": rng.normal(size=40),
            }
        )
        assignment = FactorialDesign(
            factors=["A", "B"], n_per_cell=10, seed=42
        ).randomize(df)
        assert isinstance(assignment, FactorialAssignment)

        estimator = LinEstimator(outcome_col="y", covariates=["x"])
        with pytest.raises(DesignEstimatorMismatch) as exc:
            estimator.fit(assignment)
        # Message must mention both accepted types.
        assert "CRDAssignment" in str(exc.value)
        assert "BlockedAssignment" in str(exc.value)

    def test_missing_outcome_col_raises(self) -> None:
        """fit raises if outcome_col absent."""
        assignment = _make_crd_with_covariates()
        estimator = LinEstimator(outcome_col="missing", covariates=["x"])
        with pytest.raises(InvalidDesignError, match="not found"):
            estimator.fit(assignment)

    def test_non_numeric_outcome_raises(self) -> None:
        """fit raises if outcome is non-numeric."""
        assignment = _make_crd_with_covariates()
        assignment.data_["y"] = ["a"] * len(assignment.data_)
        estimator = LinEstimator(outcome_col="y", covariates=["x"])
        with pytest.raises(InvalidDesignError, match="numeric"):
            estimator.fit(assignment)

    def test_outcome_with_nan_raises(self) -> None:
        """fit raises if outcome contains NaN."""
        assignment = _make_crd_with_covariates()
        assignment.data_.iat[
            0, assignment.data_.columns.get_loc("y")
        ] = np.nan
        estimator = LinEstimator(outcome_col="y", covariates=["x"])
        with pytest.raises(InvalidDesignError, match="NaN"):
            estimator.fit(assignment)

    def test_missing_covariate_raises(self) -> None:
        """fit raises with absent covariate, naming it."""
        assignment = _make_crd_with_covariates()
        estimator = LinEstimator(
            outcome_col="y", covariates=["x", "missing_cov"]
        )
        with pytest.raises(InvalidDesignError, match="missing_cov"):
            estimator.fit(assignment)

    def test_non_numeric_covariate_raises(self) -> None:
        """fit raises with non-numeric covariate."""
        assignment = _make_crd_with_covariates()
        assignment.data_["category"] = ["a"] * len(assignment.data_)
        estimator = LinEstimator(
            outcome_col="y", covariates=["category"]
        )
        with pytest.raises(InvalidDesignError, match="numeric"):
            estimator.fit(assignment)

    def test_covariate_with_nan_raises(self) -> None:
        """fit raises with NaN in covariate."""
        assignment = _make_crd_with_covariates()
        assignment.data_.iat[
            0, assignment.data_.columns.get_loc("x")
        ] = np.nan
        estimator = LinEstimator(outcome_col="y", covariates=["x"])
        with pytest.raises(InvalidDesignError, match="NaN"):
            estimator.fit(assignment)

    def test_constant_covariate_raises(self) -> None:
        """fit raises with constant covariate, naming it."""
        assignment = _make_crd_with_covariates()
        assignment.data_["constant"] = 5.0
        estimator = LinEstimator(
            outcome_col="y", covariates=["x", "constant"]
        )
        with pytest.raises(InvalidDesignError, match="constant"):
            estimator.fit(assignment)

    def test_estimate_before_fit_raises(self) -> None:
        """estimate raises NotFittedError if fit not called."""
        estimator = LinEstimator(outcome_col="y", covariates=["x"])
        with pytest.raises(NotFittedError):
            estimator.estimate()


class TestLinEstimatorNumerics:
    """Tests for numerical correctness."""

    def test_equivalence_with_manual_ols(self) -> None:
        """ate_ must equal coef[1] from manual OLS via lstsq."""
        assignment = _make_crd_with_covariates(n=300, seed=99)
        estimator = LinEstimator(
            outcome_col="y", covariates=["x"]
        ).fit(assignment)

        data = assignment.data_
        X = data[["x"]].values.astype(float)
        X_centered = X - X.mean(axis=0)
        T = (
            data[assignment.treatment_col_]
            .values.astype(float)
            .reshape(-1, 1)
        )
        y = data["y"].values.astype(float)

        n = len(y)
        intercept = np.ones((n, 1))
        interaction = T * X_centered
        design_matrix = np.hstack(
            [intercept, T, X_centered, interaction]
        )
        manual_coef, *_ = np.linalg.lstsq(
            design_matrix, y, rcond=None
        )
        assert estimator.ate_ == pytest.approx(
            float(manual_coef[1]), rel=1e-12
        )

    def test_variance_reduction_vs_dim(self) -> None:
        """Monte Carlo: with strong covariate, LinEstimator must
        reduce variance vs. DifferenceInMeans by at least 30%.

        Each repetition uses seed=i for i in range(100) for
        reproducibility in CI.
        """
        n_reps = 100
        n = 200
        true_ate = 0.5
        cov_effect = 2.0  # strong covariate

        lin_ates = np.empty(n_reps)
        dim_ates = np.empty(n_reps)
        for i in range(n_reps):
            assignment = _make_crd_with_covariates(
                n=n,
                seed=i,
                true_ate=true_ate,
                cov_effect=cov_effect,
            )
            lin_ates[i] = (
                LinEstimator(outcome_col="y", covariates=["x"])
                .fit(assignment)
                .ate_
            )
            dim_ates[i] = (
                DifferenceInMeans(outcome_col="y").fit(assignment).ate_
            )

        var_lin = float(np.var(lin_ates, ddof=1))
        var_dim = float(np.var(dim_ates, ddof=1))
        # Lin must reduce variance by at least 30%.
        # With cov_effect=2.0, theoretical reduction ~ 80%; 30% is
        # a generous safety margin for finite-sample noise.
        assert var_lin < 0.7 * var_dim, (
            f"Variance reduction insufficient: var_lin={var_lin:.4f}, "
            f"var_dim={var_dim:.4f}"
        )

    def test_collapses_to_dim_with_pure_noise_covariate(self) -> None:
        """With cov_effect=0, LinEstimator and DifferenceInMeans
        produce ATEs that are close (within abs=0.1).
        """
        assignment = _make_crd_with_covariates(
            n=500, seed=7, true_ate=1.0, cov_effect=0.0
        )
        lin_ate = (
            LinEstimator(outcome_col="y", covariates=["x"])
            .fit(assignment)
            .ate_
        )
        dim_ate = (
            DifferenceInMeans(outcome_col="y").fit(assignment).ate_
        )
        assert abs(lin_ate - dim_ate) < 0.1

    def test_blocked_assignment_runs_without_error(self) -> None:
        """LinEstimator on BlockedAssignment runs and produces a
        float ATE. (Does not test agreement with
        BlockedDifferenceInMeans — Lin treats data as a single
        sample, by design.)
        """
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame(
            {
                "x": rng.normal(size=n),
                "block": ["A"] * 150 + ["B"] * 150,
                "y": rng.normal(size=n),
            }
        )
        assignment = BlockedCRD(
            block_col="block", p=0.5, seed=0
        ).randomize(df)
        estimator = LinEstimator(
            outcome_col="y", covariates=["x"]
        ).fit(assignment)
        assert isinstance(estimator.ate_, float)

    def test_invariance_under_repeated_estimate(self) -> None:
        """fit().estimate() called twice on the same assignment must
        yield identical ATE.
        """
        assignment = _make_crd_with_covariates()
        estimator = LinEstimator(
            outcome_col="y", covariates=["x"]
        ).fit(assignment)
        r1 = estimator.estimate()
        r2 = estimator.estimate()
        assert r1.ate == r2.ate