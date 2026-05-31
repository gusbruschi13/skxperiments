"""Tests for skxperiments.core.assignment."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import BaseAssignment, CRDAssignment
from skxperiments.core.exceptions import InvalidDesignError


class TestCRDAssignmentCreation:
    """Tests for CRDAssignment instantiation."""

    def test_does_not_modify_original_dataframe(self) -> None:
        """CRDAssignment should not modify the original DataFrame."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "treatment": [1, 0, 1, 0]})
        original_cols = list(df.columns)
        original_values = df.copy()

        _ = CRDAssignment(
            data=df.copy(),
            treatment_col="treatment",
            design=None,
            seed=42,
        )

        assert list(df.columns) == original_cols
        pd.testing.assert_frame_equal(df, original_values)

    def test_basic_creation(self) -> None:
        """Should create CRDAssignment with valid data."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "treatment": [1, 0, 1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None, seed=42
        )
        assert assignment.n_units_ == 4
        assert assignment.n_treated_ == 2
        assert assignment.n_control_ == 2

    def test_seed_stored(self) -> None:
        """Seed should be stored as attribute."""
        df = pd.DataFrame({"treatment": [1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None, seed=123
        )
        assert assignment.seed_ == 123

    def test_seed_none(self) -> None:
        """Seed can be None."""
        df = pd.DataFrame({"treatment": [1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None, seed=None
        )
        assert assignment.seed_ is None


class TestCRDAssignmentIds:
    """Tests for treated_ids and control_ids."""

    def test_treated_ids_correct(self) -> None:
        """treated_ids should return positions where treatment == 1."""
        df = pd.DataFrame({"treatment": [0, 1, 0, 1, 1]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        expected = np.array([1, 3, 4])
        np.testing.assert_array_equal(assignment.treated_ids(), expected)

    def test_control_ids_correct(self) -> None:
        """control_ids should return positions where treatment == 0."""
        df = pd.DataFrame({"treatment": [0, 1, 0, 1, 1]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        expected = np.array([0, 2])
        np.testing.assert_array_equal(assignment.control_ids(), expected)

    def test_ids_cover_all_units_no_overlap(self) -> None:
        """treated_ids + control_ids should cover all units without overlap."""
        df = pd.DataFrame({"treatment": [1, 0, 1, 0, 0, 1]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        treated = set(assignment.treated_ids().tolist())
        control = set(assignment.control_ids().tolist())

        # No overlap
        assert treated & control == set()

        # Cover all units
        assert treated | control == set(range(6))

    def test_n_treated_plus_n_control_equals_n_units(self) -> None:
        """n_treated_ + n_control_ should equal n_units_."""
        df = pd.DataFrame({"treatment": [1, 0, 1, 0, 1, 1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        assert assignment.n_treated_ + assignment.n_control_ == assignment.n_units_


class TestCRDAssignmentValidation:
    """Tests for validation in CRDAssignment."""

    def test_invalid_treatment_values_raises(self) -> None:
        """Should raise InvalidDesignError if treatment has values != 0 or 1."""
        df = pd.DataFrame({"treatment": [0, 1, 2]})
        with pytest.raises(InvalidDesignError):
            CRDAssignment(
                data=df, treatment_col="treatment", design=None
            )

    def test_invalid_treatment_negative_raises(self) -> None:
        """Should raise InvalidDesignError with negative treatment values."""
        df = pd.DataFrame({"treatment": [0, -1, 1]})
        with pytest.raises(InvalidDesignError):
            CRDAssignment(
                data=df, treatment_col="treatment", design=None
            )

    def test_float_treatment_values_raises(self) -> None:
        """Should raise InvalidDesignError with float treatment values."""
        df = pd.DataFrame({"treatment": [0.0, 0.5, 1.0]})
        with pytest.raises(InvalidDesignError):
            CRDAssignment(
                data=df, treatment_col="treatment", design=None
            )

    def test_missing_treatment_col_raises(self) -> None:
        """Should raise InvalidDesignError if treatment_col not in DataFrame."""
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        with pytest.raises(InvalidDesignError, match="not found in DataFrame"):
            CRDAssignment(
                data=df,
                treatment_col="treatment",
                design=None,
                seed=42,
            )


class TestCRDAssignmentRepr:
    """Tests for __repr__."""

    def test_repr_format(self) -> None:
        """__repr__ should follow expected format."""
        df = pd.DataFrame({"treatment": [1, 0, 1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        r = repr(assignment)
        assert r == "CRDAssignment(n_treated=2, n_control=2)"

    def test_repr_asymmetric(self) -> None:
        """__repr__ should reflect actual counts."""
        df = pd.DataFrame({"treatment": [1, 1, 1, 0]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None
        )
        r = repr(assignment)
        assert "n_treated=3" in r
        assert "n_control=1" in r


class TestCRDAssignmentDraw:
    """Tests for CRDAssignment.draw."""

    def test_implements_draw_method(self) -> None:
        """CRDAssignment should expose a callable draw method."""
        df = pd.DataFrame({"treatment": [0, 1, 0, 1]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None, seed=0
        )
        assert hasattr(assignment, "draw")
        assert callable(assignment.draw)

    def test_draw_raises_without_design(self) -> None:
        """draw() should raise InvalidDesignError when design_ is None."""
        df = pd.DataFrame({"treatment": [0, 1, 0, 1]})
        assignment = CRDAssignment(
            data=df, treatment_col="treatment", design=None, seed=0
        )
        with pytest.raises(InvalidDesignError, match="without a reference"):
            assignment.draw(seed=1)