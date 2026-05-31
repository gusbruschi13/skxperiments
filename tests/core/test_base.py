"""Tests for skxperiments.core.base."""

import io
import sys

import pandas as pd
import pytest

from skxperiments.core.assignment import CRDAssignment
from skxperiments.core.base import (
    BaseDesign,
    BaseEstimator,
    BaseInference,
    DiagnosticsReport,
)
from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InvalidDesignError,
    NotFittedError,
)
from skxperiments.core.results import Results


# --- Concrete subclasses for testing ---


class ConcreteDesign(BaseDesign):
    """Concrete design for testing."""

    def __init__(self, n_treated: int = 5, seed: int | None = None) -> None:
        self.n_treated = n_treated
        self.seed = seed

    def randomize(self, df: pd.DataFrame) -> CRDAssignment:
        """Dummy randomize."""
        df_copy = df.copy()
        df_copy["treatment"] = 0
        df_copy.iloc[: self.n_treated, df_copy.columns.get_loc("treatment")] = 1
        return CRDAssignment(
            data=df_copy, treatment_col="treatment", design=self, seed=self.seed
        )


class ConcreteEstimator(BaseEstimator):
    """Concrete estimator for testing."""

    def __init__(self, outcome_col: str = "y") -> None:
        self.outcome_col = outcome_col

    def fit(self, assignment: CRDAssignment) -> "ConcreteEstimator":
        """Dummy fit."""
        self.assignment_ = assignment
        return self

    def estimate(self) -> Results:
        """Dummy estimate."""
        self._check_is_fitted()
        return Results(ate=0.0)


class ConcreteInference(BaseInference):
    """Concrete inference for testing."""

    def __init__(self, n_permutations: int = 1000) -> None:
        self.n_permutations = n_permutations

    def fit(self, assignment: CRDAssignment) -> "ConcreteInference":
        """Dummy fit."""
        self.result_ = Results(ate=0.0, p_value=0.5)
        return self


# --- Tests ---


class TestBaseDesignGetParams:
    """Tests for BaseDesign.get_params."""

    def test_returns_correct_params(self) -> None:
        """get_params should return init parameters."""
        design = ConcreteDesign(n_treated=10, seed=42)
        params = design.get_params()
        assert params == {"n_treated": 10, "seed": 42}

    def test_returns_default_params(self) -> None:
        """get_params should return defaults when not explicitly set."""
        design = ConcreteDesign()
        params = design.get_params()
        assert params == {"n_treated": 5, "seed": None}


class TestBaseDesignSetParams:
    """Tests for BaseDesign.set_params."""

    def test_updates_value(self) -> None:
        """set_params should update the parameter value."""
        design = ConcreteDesign(n_treated=5)
        design.set_params(n_treated=10)
        assert design.n_treated == 10

    def test_returns_self(self) -> None:
        """set_params should return self."""
        design = ConcreteDesign()
        result = design.set_params(n_treated=3)
        assert result is design

    def test_invalid_param_raises(self) -> None:
        """set_params should raise InvalidDesignError for unknown params."""
        design = ConcreteDesign()
        with pytest.raises(InvalidDesignError):
            design.set_params(nonexistent_param=42)


class TestBaseDesignRepr:
    """Tests for BaseDesign.__repr__."""

    def test_repr_format(self) -> None:
        """__repr__ should show class name and params."""
        design = ConcreteDesign(n_treated=5, seed=42)
        r = repr(design)
        assert "ConcreteDesign(" in r
        assert "n_treated=5" in r
        assert "seed=42" in r


class TestBaseEstimatorCheckIsFitted:
    """Tests for BaseEstimator._check_is_fitted."""

    def test_raises_before_fit(self) -> None:
        """Should raise NotFittedError before fit is called."""
        estimator = ConcreteEstimator()
        with pytest.raises(NotFittedError):
            estimator._check_is_fitted()

    def test_does_not_raise_after_fit(self) -> None:
        """Should not raise after fit is called."""
        estimator = ConcreteEstimator()
        df = pd.DataFrame({"y": [1, 2, 3, 4], "treatment": [1, 0, 1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        estimator.fit(assignment)
        # Should not raise
        estimator._check_is_fitted()


class TestBaseEstimatorValidateAssignmentType:
    """Tests for BaseEstimator._validate_assignment_type."""

    def test_raises_on_wrong_type(self) -> None:
        """Should raise DesignEstimatorMismatch with wrong type."""
        estimator = ConcreteEstimator()
        with pytest.raises(DesignEstimatorMismatch):
            estimator._validate_assignment_type("not an assignment", CRDAssignment)

    def test_does_not_raise_on_correct_type(self) -> None:
        """Should not raise with correct type."""
        estimator = ConcreteEstimator()
        df = pd.DataFrame({"treatment": [1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        # Should not raise
        estimator._validate_assignment_type(assignment, CRDAssignment)

    def test_accepts_tuple_of_types(self) -> None:
        """Should accept a tuple of expected types."""
        estimator = ConcreteEstimator()
        df = pd.DataFrame({"treatment": [1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        # Should not raise
        estimator._validate_assignment_type(assignment, (CRDAssignment,))
    
    def test_rejects_with_tuple_lists_all_expected_types(self) -> None:
        """Error message should mention all expected types when tuple is passed."""
        estimator = ConcreteEstimator()
        with pytest.raises(DesignEstimatorMismatch, match="CRDAssignment"):
            estimator._validate_assignment_type(
                "not an assignment", (CRDAssignment,)
            )


class TestBaseEstimatorGetSetParams:
    """Tests for BaseEstimator get_params and set_params."""

    def test_get_params(self) -> None:
        """get_params should return init parameters."""
        estimator = ConcreteEstimator(outcome_col="outcome")
        params = estimator.get_params()
        assert params == {"outcome_col": "outcome"}

    def test_set_params(self) -> None:
        """set_params should update parameter."""
        estimator = ConcreteEstimator()
        estimator.set_params(outcome_col="new_col")
        assert estimator.outcome_col == "new_col"

    def test_set_params_invalid_raises(self) -> None:
        """set_params should raise for unknown params."""
        estimator = ConcreteEstimator()
        with pytest.raises(InvalidDesignError):
            estimator.set_params(bad_param=True)


class TestBaseInference:
    """Tests for BaseInference."""

    def test_get_params(self) -> None:
        """get_params should return init parameters."""
        inf = ConcreteInference(n_permutations=5000)
        params = inf.get_params()
        assert params == {"n_permutations": 5000}

    def test_check_is_fitted_raises_before_fit(self) -> None:
        """Should raise NotFittedError before fit."""
        inf = ConcreteInference()
        with pytest.raises(NotFittedError):
            inf._check_is_fitted()

    def test_check_is_fitted_ok_after_fit(self) -> None:
        """Should not raise after fit."""
        inf = ConcreteInference()
        df = pd.DataFrame({"treatment": [1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        inf.fit(assignment)
        # Should not raise
        inf._check_is_fitted()

    def test_repr(self) -> None:
        """__repr__ should show class name and params."""
        inf = ConcreteInference(n_permutations=2000)
        r = repr(inf)
        assert "ConcreteInference(" in r
        assert "n_permutations=2000" in r


class TestDiagnosticsReport:
    """Tests for DiagnosticsReport."""

    def test_empty_report_summary(self, capsys: pytest.CaptureFixture) -> None:
        """Should print no issues message when empty."""
        report = DiagnosticsReport()
        report.summary()
        captured = capsys.readouterr()
        assert "✅ No issues found." in captured.out

    def test_report_with_flags(self, capsys: pytest.CaptureFixture) -> None:
        """Should print flags with ❌ prefix."""
        report = DiagnosticsReport(flags=["Imbalance detected"])
        report.summary()
        captured = capsys.readouterr()
        assert "❌ Imbalance detected" in captured.out

    def test_report_with_warnings(self, capsys: pytest.CaptureFixture) -> None:
        """Should print warnings with ⚠️ prefix."""
        report = DiagnosticsReport(warnings=["Small sample size"])
        report.summary()
        captured = capsys.readouterr()
        assert "⚠️ Small sample size" in captured.out

    def test_repr_format(self) -> None:
        """__repr__ should show counts."""
        report = DiagnosticsReport(
            flags=["a", "b"], warnings=["c"]
        )
        r = repr(report)
        assert r == "DiagnosticsReport(flags=2, warnings=1)"

    def test_default_empty_lists(self) -> None:
        """Default flags and warnings should be empty lists."""
        report = DiagnosticsReport()
        assert report.flags == []
        assert report.warnings == []