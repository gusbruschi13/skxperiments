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

    def estimate(self) -> Results:
        """Dummy estimate.

        Required since Phase 4 prep made BaseInference.estimate abstract.
        Returns the Results stored during fit.
        """
        self._check_is_fitted()
        return self.result_


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


# ---------------------------------------------------------------------------
# Phase 4 prep additions
# ---------------------------------------------------------------------------


# Sentinel types for assignment-type validation tests (decoupled from
# the real CRDAssignment / BlockedAssignment to keep these tests focused
# on the ABC contract, not on Assignment construction).


class _FakeCRDAssignment:
    """Sentinel type used only to test type checks."""


class _FakeBlockedAssignment:
    """Sentinel type used only to test type checks."""


class _FakeFactorialAssignment:
    """Sentinel type used only to test type checks."""


class TestBaseInferenceContract:
    """Tests for the abstract method contract of BaseInference.

    Phase 4 prep: BaseInference now requires both `fit` and `estimate`
    to be implemented by subclasses.
    """

    def test_subclass_with_both_methods_instantiates(self) -> None:
        """A subclass implementing both fit and estimate instantiates."""
        inf = ConcreteInference()
        assert isinstance(inf, BaseInference)

    def test_subclass_missing_estimate_fails(self) -> None:
        """A subclass implementing only fit cannot be instantiated."""

        class _MissingEstimate(BaseInference):
            def fit(self, assignment):  # type: ignore[no-untyped-def]
                return self

        with pytest.raises(TypeError, match="abstract"):
            _MissingEstimate()  # type: ignore[abstract]

    def test_subclass_missing_fit_fails(self) -> None:
        """A subclass implementing only estimate cannot be instantiated."""

        class _MissingFit(BaseInference):
            def estimate(self) -> Results:
                return Results(ate=0.0)

        with pytest.raises(TypeError, match="abstract"):
            _MissingFit()  # type: ignore[abstract]

    def test_subclass_missing_both_fails(self) -> None:
        """A subclass implementing neither fit nor estimate cannot instantiate."""

        class _MissingBoth(BaseInference):
            pass

        with pytest.raises(TypeError, match="abstract"):
            _MissingBoth()  # type: ignore[abstract]

    def test_estimate_returns_results(self) -> None:
        """A fitted subclass produces a Results from estimate()."""
        inf = ConcreteInference()
        df = pd.DataFrame({"treatment": [1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        inf.fit(assignment)
        result = inf.estimate()
        assert isinstance(result, Results)


class TestBaseInferenceValidateAssignmentType:
    """Tests for BaseInference._validate_assignment_type.

    Mirrors TestBaseEstimatorValidateAssignmentType to confirm parity
    between the two ABCs after the Phase 4 prep refactor.
    """

    def test_does_not_raise_on_correct_type(self) -> None:
        """No exception when assignment matches the expected type."""
        inf = ConcreteInference()
        assignment = _FakeCRDAssignment()
        inf._validate_assignment_type(assignment, _FakeCRDAssignment)

    def test_does_not_raise_when_type_in_tuple(self) -> None:
        """No exception when assignment matches one type in a tuple."""
        inf = ConcreteInference()
        assignment = _FakeBlockedAssignment()
        inf._validate_assignment_type(
            assignment,
            (_FakeCRDAssignment, _FakeBlockedAssignment),
        )

    def test_raises_on_single_type_mismatch(self) -> None:
        """Raises DesignEstimatorMismatch when single type does not match."""
        inf = ConcreteInference()
        assignment = _FakeFactorialAssignment()
        with pytest.raises(DesignEstimatorMismatch):
            inf._validate_assignment_type(assignment, _FakeCRDAssignment)

    def test_raises_on_tuple_type_mismatch(self) -> None:
        """Raises DesignEstimatorMismatch when none of tuple types match."""
        inf = ConcreteInference()
        assignment = _FakeFactorialAssignment()
        with pytest.raises(DesignEstimatorMismatch):
            inf._validate_assignment_type(
                assignment,
                (_FakeCRDAssignment, _FakeBlockedAssignment),
            )

    def test_error_message_lists_all_types_in_tuple(self) -> None:
        """Error message must include every expected type name when tuple."""
        inf = ConcreteInference()
        assignment = _FakeFactorialAssignment()
        with pytest.raises(DesignEstimatorMismatch) as excinfo:
            inf._validate_assignment_type(
                assignment,
                (_FakeCRDAssignment, _FakeBlockedAssignment),
            )
        msg = str(excinfo.value)
        assert "_FakeCRDAssignment" in msg
        assert "_FakeBlockedAssignment" in msg

    def test_error_message_includes_caller_class_name(self) -> None:
        """Error message must include the inference class name."""
        inf = ConcreteInference()
        assignment = _FakeFactorialAssignment()
        with pytest.raises(DesignEstimatorMismatch) as excinfo:
            inf._validate_assignment_type(assignment, _FakeCRDAssignment)
        assert "ConcreteInference" in str(excinfo.value)


class TestValidateAssignmentTypeSnapshot:
    """Snapshot tests for the DesignEstimatorMismatch error message.

    Phase 4 prep refactored the validation logic into a module-level
    helper. These tests pin the error message format to catch silent
    string drift during the refactor or in future changes. Both
    BaseEstimator and BaseInference must produce identical message
    structure.
    """

    def test_estimator_single_type_message_contains_key_parts(self) -> None:
        """Error message for single expected type contains the basics."""
        estimator = ConcreteEstimator()
        with pytest.raises(DesignEstimatorMismatch) as excinfo:
            estimator._validate_assignment_type(
                "not an assignment", CRDAssignment
            )
        msg = str(excinfo.value)
        assert "ConcreteEstimator" in msg
        assert "CRDAssignment" in msg

    def test_estimator_tuple_message_lists_all_types(self) -> None:
        """Error message for tuple expected types lists each name."""
        estimator = ConcreteEstimator()
        with pytest.raises(DesignEstimatorMismatch) as excinfo:
            estimator._validate_assignment_type(
                "not an assignment",
                (CRDAssignment, _FakeBlockedAssignment),
            )
        msg = str(excinfo.value)
        assert "CRDAssignment" in msg
        assert "_FakeBlockedAssignment" in msg

    def test_inference_single_type_message_contains_key_parts(self) -> None:
        """BaseInference produces the same message structure."""
        inf = ConcreteInference()
        with pytest.raises(DesignEstimatorMismatch) as excinfo:
            inf._validate_assignment_type(
                "not an assignment", _FakeCRDAssignment
            )
        msg = str(excinfo.value)
        assert "ConcreteInference" in msg
        assert "_FakeCRDAssignment" in msg

    def test_estimator_and_inference_messages_have_same_structure(
        self,
    ) -> None:
        """Both ABCs delegate to the same helper, so messages match shape."""
        estimator = ConcreteEstimator()
        inf = ConcreteInference()

        with pytest.raises(DesignEstimatorMismatch) as exc_est:
            estimator._validate_assignment_type(
                "not an assignment", _FakeCRDAssignment
            )
        with pytest.raises(DesignEstimatorMismatch) as exc_inf:
            inf._validate_assignment_type(
                "not an assignment", _FakeCRDAssignment
            )

        msg_est = str(exc_est.value)
        msg_inf = str(exc_inf.value)

        # Both must mention the expected type and the received type.
        assert "_FakeCRDAssignment" in msg_est
        assert "_FakeCRDAssignment" in msg_inf
        assert "str" in msg_est  # received_type for "not an assignment"
        assert "str" in msg_inf

        # Each carries its own caller class name.
        assert "ConcreteEstimator" in msg_est
        assert "ConcreteInference" in msg_inf