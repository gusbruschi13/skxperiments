"""Tests for skxperiments.estimators.difference_in_means."""

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
from skxperiments.design.rerandomized_crd import ReRandomizedCRD
from skxperiments.estimators.difference_in_means import DifferenceInMeans


# --- Helpers ---


def _make_assignment(
    n: int = 100,
    seed: int = 42,
    true_ate: float = 0.5,
) -> CRDAssignment:
    """Build a CRDAssignment whose outcome embeds a known ATE.

    Notes
    -----
    This helper mutates ``assignment.data_`` after randomization to
    inject the true ATE additively into the treated group's outcome.
    This is a deliberate exception to the project's no-side-effects
    principle, allowed exclusively in test fixtures. Production code
    must never mutate Assignment data.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    design = CRD(p=0.5, seed=seed)
    assignment = design.randomize(df)

    treated = assignment.treated_ids()
    assignment.data_.iloc[
        treated,
        assignment.data_.columns.get_loc("y"),
    ] += true_ate
    return assignment


# --- Tests ---


class TestDifferenceInMeansCreation:
    """Tests for DifferenceInMeans instantiation."""

    def test_basic_creation(self) -> None:
        """Should instantiate with outcome_col."""
        estimator = DifferenceInMeans(outcome_col="y")
        assert estimator.outcome_col == "y"

    def test_outcome_col_stored_as_is(self) -> None:
        """outcome_col must be stored verbatim, no transformation."""
        estimator = DifferenceInMeans(outcome_col="my_metric")
        assert estimator.outcome_col == "my_metric"


class TestDifferenceInMeansFit:
    """Tests for DifferenceInMeans.fit."""

    def test_fit_returns_self(self) -> None:
        """fit must return self for chaining."""
        assignment = _make_assignment()
        estimator = DifferenceInMeans(outcome_col="y")
        returned = estimator.fit(assignment)
        assert returned is estimator

    def test_stores_assignment(self) -> None:
        """fit must store the assignment as assignment_."""
        assignment = _make_assignment()
        estimator = DifferenceInMeans(outcome_col="y")
        estimator.fit(assignment)
        assert estimator.assignment_ is assignment

    def test_stores_ate(self) -> None:
        """fit must compute and store ate_ as a float."""
        assignment = _make_assignment()
        estimator = DifferenceInMeans(outcome_col="y")
        estimator.fit(assignment)
        assert isinstance(estimator.ate_, float)

    def test_accepts_rerandomized_crd_assignment(self) -> None:
        """fit must accept CRDAssignment from ReRandomizedCRD without
        special handling.
        """
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=200),
                "x2": rng.normal(size=200),
                "y": rng.normal(size=200),
            }
        )
        design = ReRandomizedCRD(
            covariates=["x1", "x2"],
            threshold=10.0,
            p=0.5,
            seed=42,
        )
        assignment = design.randomize(df)
        # Sanity: rerandomized assignment is a CRDAssignment instance
        # with metadata populated.
        assert isinstance(assignment, CRDAssignment)
        assert assignment.rerandomization_metadata is not None

        estimator = DifferenceInMeans(outcome_col="y")
        # Should not raise.
        estimator.fit(assignment)
        assert isinstance(estimator.ate_, float)


class TestDifferenceInMeansEstimate:
    """Tests for DifferenceInMeans.estimate."""

    def test_returns_results(self) -> None:
        """estimate must return a Results instance."""
        assignment = _make_assignment()
        estimator = DifferenceInMeans(outcome_col="y").fit(assignment)
        result = estimator.estimate()
        assert isinstance(result, Results)

    def test_ate_matches_manual_computation(self) -> None:
        """Results.ate must equal y[treated].mean() - y[control].mean()."""
        assignment = _make_assignment()
        estimator = DifferenceInMeans(outcome_col="y").fit(assignment)
        result = estimator.estimate()

        y = assignment.data_["y"].values
        expected = float(
            y[assignment.treated_ids()].mean()
            - y[assignment.control_ids()].mean()
        )
        assert result.ate == pytest.approx(expected, rel=1e-12)

    def test_estimator_name_is_class_name(self) -> None:
        """estimator_name must be 'DifferenceInMeans'."""
        assignment = _make_assignment()
        result = (
            DifferenceInMeans(outcome_col="y").fit(assignment).estimate()
        )
        assert result.estimator_name == "DifferenceInMeans"

    def test_design_name_is_crd(self) -> None:
        """design_name must reflect the design that produced the
        assignment.
        """
        assignment = _make_assignment()
        result = (
            DifferenceInMeans(outcome_col="y").fit(assignment).estimate()
        )
        assert result.design_name == "CRD"

    def test_n_metadata_populated(self) -> None:
        """n_obs, n_treated, n_control must be populated from assignment."""
        assignment = _make_assignment(n=120)
        result = (
            DifferenceInMeans(outcome_col="y").fit(assignment).estimate()
        )
        assert result.n_obs == 120
        assert result.n_treated == assignment.n_treated_
        assert result.n_control == assignment.n_control_
        assert result.n_treated + result.n_control == result.n_obs

    def test_inference_fields_are_none(self) -> None:
        """se, ci, p_value must be None (inference is Phase 4)."""
        assignment = _make_assignment()
        result = (
            DifferenceInMeans(outcome_col="y").fit(assignment).estimate()
        )
        assert result.se is None
        assert result.ci is None
        assert result.p_value is None

    def test_inference_name_is_none(self) -> None:
        """inference_name must be None when no inference was applied."""
        assignment = _make_assignment()
        result = (
            DifferenceInMeans(outcome_col="y").fit(assignment).estimate()
        )
        assert result.inference_name is None


class TestDifferenceInMeansValidation:
    """Tests for input validation in DifferenceInMeans."""

    def test_rejects_blocked_assignment(self) -> None:
        """fit must reject BlockedAssignment with DesignEstimatorMismatch."""
        df = pd.DataFrame(
            {
                "x": np.arange(40, dtype=float),
                "region": ["A"] * 20 + ["B"] * 20,
                "y": np.arange(40, dtype=float),
            }
        )
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        blocked_assignment = design.randomize(df)
        assert isinstance(blocked_assignment, BlockedAssignment)

        estimator = DifferenceInMeans(outcome_col="y")
        with pytest.raises(DesignEstimatorMismatch, match="CRDAssignment"):
            estimator.fit(blocked_assignment)

    def test_rejects_factorial_assignment(self) -> None:
        """fit must reject FactorialAssignment with DesignEstimatorMismatch."""
        df = pd.DataFrame(
            {
                "x": np.arange(40, dtype=float),
                "y": np.arange(40, dtype=float),
            }
        )
        design = FactorialDesign(
            factors=["A", "B"], n_per_cell=10, seed=42
        )
        factorial_assignment = design.randomize(df)
        assert isinstance(factorial_assignment, FactorialAssignment)

        estimator = DifferenceInMeans(outcome_col="y")
        with pytest.raises(DesignEstimatorMismatch, match="CRDAssignment"):
            estimator.fit(factorial_assignment)

    def test_missing_outcome_col_raises(self) -> None:
        """fit must raise InvalidDesignError if outcome_col is absent."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=40)})
        # Build a CRDAssignment that has no 'y' column.
        design = CRD(p=0.5, seed=0)
        assignment = design.randomize(df)

        estimator = DifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="not found"):
            estimator.fit(assignment)

    def test_non_numeric_outcome_raises(self) -> None:
        """fit must raise InvalidDesignError if outcome is not numeric."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=40),
                "y": ["a"] * 40,
            }
        )
        design = CRD(p=0.5, seed=0)
        assignment = design.randomize(df)

        estimator = DifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="numeric"):
            estimator.fit(assignment)

    def test_outcome_with_nan_raises(self) -> None:
        """fit must raise InvalidDesignError if outcome has NaN."""
        assignment = _make_assignment()
        # Inject NaN after randomization.
        assignment.data_.iloc[
            0, assignment.data_.columns.get_loc("y")
        ] = np.nan

        estimator = DifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="NaN"):
            estimator.fit(assignment)

    def test_estimate_before_fit_raises(self) -> None:
        """estimate must raise NotFittedError if fit was not called."""
        estimator = DifferenceInMeans(outcome_col="y")
        with pytest.raises(NotFittedError):
            estimator.estimate()


class TestDifferenceInMeansNumerics:
    """Numerical correctness tests."""

    def test_recovers_injected_ate(self) -> None:
        """With true_ate=1.0 and large n, recovered ATE must be close
        to 1.0 within statistical tolerance.
        """
        assignment = _make_assignment(n=1000, seed=123, true_ate=1.0)
        result = (
            DifferenceInMeans(outcome_col="y").fit(assignment).estimate()
        )
        # SE under N(0,1) noise with n=1000, p=0.5 is ~ sqrt(2/500) = 0.063.
        # |ate - 1.0| < 0.2 is roughly 3 SE.
        assert abs(result.ate - 1.0) < 0.2

    def test_zero_true_ate_yields_small_estimate(self) -> None:
        """With true_ate=0 and large n, |ate| should be small."""
        assignment = _make_assignment(n=1000, seed=42, true_ate=0.0)
        result = (
            DifferenceInMeans(outcome_col="y").fit(assignment).estimate()
        )
        assert abs(result.ate) < 0.2

    def test_invariance_under_repeated_estimate_calls(self) -> None:
        """Calling estimate() repeatedly on the same fit must return
        identical ATE values.
        """
        assignment = _make_assignment()
        estimator = DifferenceInMeans(outcome_col="y").fit(assignment)
        r1 = estimator.estimate()
        r2 = estimator.estimate()
        assert r1.ate == r2.ate

    def test_invariance_under_repeated_fit_estimate(self) -> None:
        """fit().estimate() called twice on the same assignment must
        yield identical ATE values.
        """
        assignment = _make_assignment()
        r1 = (
            DifferenceInMeans(outcome_col="y").fit(assignment).estimate()
        )
        r2 = (
            DifferenceInMeans(outcome_col="y").fit(assignment).estimate()
        )
        assert r1.ate == r2.ate