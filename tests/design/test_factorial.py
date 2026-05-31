"""Tests for skxperiments.design.factorial."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import FactorialAssignment
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)
from skxperiments.design.factorial import FactorialDesign


# --- Helpers ---


def _df_for_design(n_per_cell: int, k: int) -> pd.DataFrame:
    """Build a DataFrame with the exact size required by a 2^K design."""
    n = n_per_cell * (2**k)
    return pd.DataFrame({"x": np.arange(n, dtype=float)})


# --- Tests ---


class TestFactorialDesignCreation:
    """Tests for FactorialDesign instantiation."""

    def test_creation_k2(self) -> None:
        """Should instantiate with K=2 factors."""
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assert design.factors == ["A", "B"]
        assert design.n_per_cell == 10
        assert design.seed == 42

    def test_creation_k3(self) -> None:
        """Should instantiate with K=3 factors."""
        design = FactorialDesign(
            factors=["A", "B", "C"], n_per_cell=5, seed=42
        )
        assert design.factors == ["A", "B", "C"]

    def test_empty_factors_raises(self) -> None:
        """Should raise InvalidDesignError for empty factors list."""
        with pytest.raises(InvalidDesignError, match="non-empty"):
            FactorialDesign(factors=[], n_per_cell=10)

    def test_duplicate_factors_raises(self) -> None:
        """Should raise InvalidDesignError for duplicate factor names."""
        with pytest.raises(InvalidDesignError, match="unique|Duplicates"):
            FactorialDesign(factors=["A", "B", "A"], n_per_cell=10)

    def test_n_per_cell_zero_raises(self) -> None:
        """Should raise InvalidDesignError for n_per_cell == 0."""
        with pytest.raises(InvalidDesignError, match="positive integer"):
            FactorialDesign(factors=["A", "B"], n_per_cell=0)

    def test_n_per_cell_negative_raises(self) -> None:
        """Should raise InvalidDesignError for negative n_per_cell."""
        with pytest.raises(InvalidDesignError, match="positive integer"):
            FactorialDesign(factors=["A", "B"], n_per_cell=-1)


class TestFactorialDesignRandomize:
    """Tests for FactorialDesign.randomize."""

    def test_returns_factorial_assignment(self) -> None:
        """Should return a FactorialAssignment instance."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        assert isinstance(assignment, FactorialAssignment)

    def test_does_not_modify_input_dataframe(self) -> None:
        """Should not modify the input DataFrame."""
        df = _df_for_design(n_per_cell=10, k=2)
        snapshot = df.copy()
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        _ = design.randomize(df)
        pd.testing.assert_frame_equal(df, snapshot)

    def test_data_contains_factor_and_cell_columns(self) -> None:
        """data_ must contain factor columns and '_cell'."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        for col in ["A", "B", "_cell"]:
            assert col in assignment.data_.columns

    def test_factor_values_are_binary(self) -> None:
        """Factor columns must contain only 0 and 1."""
        df = _df_for_design(n_per_cell=10, k=3)
        design = FactorialDesign(
            factors=["A", "B", "C"], n_per_cell=10, seed=42
        )
        assignment = design.randomize(df)
        for col in ["A", "B", "C"]:
            unique = set(assignment.data_[col].unique().tolist())
            assert unique.issubset({0, 1})

    def test_seed_reproducibility(self) -> None:
        """Same seed must produce identical assignments."""
        df = _df_for_design(n_per_cell=10, k=2)
        d1 = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        d2 = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        a1 = d1.randomize(df)
        a2 = d2.randomize(df)
        np.testing.assert_array_equal(
            a1.data_["_cell"].values, a2.data_["_cell"].values
        )

    def test_design_reference_set(self) -> None:
        """Assignment must hold a reference to the generating design."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        assert assignment.design_ is design


class TestFactorialDesignCellSizes:
    """Tests for cell sizing and encoding convention."""

    def test_each_cell_has_exact_size(self) -> None:
        """Each of the 2^K cells must have exactly n_per_cell units."""
        df = _df_for_design(n_per_cell=7, k=3)
        design = FactorialDesign(
            factors=["A", "B", "C"], n_per_cell=7, seed=42
        )
        assignment = design.randomize(df)
        for cell_idx in range(2**3):
            size = int((assignment.data_["_cell"] == cell_idx).sum())
            assert size == 7

    def test_cell_sizes_attribute_correct(self) -> None:
        """cell_sizes_ must reflect actual cell sizes."""
        df = _df_for_design(n_per_cell=5, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=5, seed=42)
        assignment = design.randomize(df)
        assert assignment.cell_sizes_ == {0: 5, 1: 5, 2: 5, 3: 5}

    def test_n_cells_is_2_to_k(self) -> None:
        """n_cells_ must equal 2^K."""
        df = _df_for_design(n_per_cell=3, k=4)
        design = FactorialDesign(
            factors=["A", "B", "C", "D"], n_per_cell=3, seed=42
        )
        assignment = design.randomize(df)
        assert assignment.n_cells_ == 16

    def test_encoding_convention_k2(self) -> None:
        """Verify little-endian encoding for K=2 with factors ['A', 'B'].

        A=0, B=0 -> cell 0
        A=1, B=0 -> cell 1
        A=0, B=1 -> cell 2
        A=1, B=1 -> cell 3
        """
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        data = assignment.data_

        # For each cell, all units in that cell must have the
        # corresponding (A, B) values.
        cell_to_factors = {
            0: (0, 0),
            1: (1, 0),
            2: (0, 1),
            3: (1, 1),
        }
        for cell_idx, (a_val, b_val) in cell_to_factors.items():
            mask = data["_cell"] == cell_idx
            assert (data.loc[mask, "A"] == a_val).all()
            assert (data.loc[mask, "B"] == b_val).all()


class TestFactorialDesignValidation:
    """Tests for validation in randomize."""

    def test_wrong_size_raises(self) -> None:
        """Should raise InsufficientDataError if df size != n_per_cell * 2^K."""
        df = pd.DataFrame({"x": range(10)})
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        with pytest.raises(InsufficientDataError):
            design.randomize(df)

    def test_factor_column_collision_raises(self) -> None:
        """Should raise InvalidDesignError if df contains a factor column."""
        df = pd.DataFrame({"x": range(40), "A": 0})
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        with pytest.raises(InvalidDesignError, match="reserved"):
            design.randomize(df)

    def test_cell_column_collision_raises(self) -> None:
        """Should raise InvalidDesignError if df already contains '_cell'."""
        df = pd.DataFrame({"x": range(40), "_cell": 0})
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        with pytest.raises(InvalidDesignError, match="reserved"):
            design.randomize(df)


class TestFactorialAssignmentAPI:
    """Tests for FactorialAssignment public API."""

    def test_treated_ids_raises(self) -> None:
        """treated_ids must raise NotImplementedError."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        with pytest.raises(
            NotImplementedError, match="cell_ids"
        ):
            assignment.treated_ids()

    def test_control_ids_raises(self) -> None:
        """control_ids must raise NotImplementedError."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        with pytest.raises(
            NotImplementedError, match="cell_ids"
        ):
            assignment.control_ids()

    def test_cell_ids_returns_correct_indices(self) -> None:
        """cell_ids must return iloc positions of units matching factors."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        data = assignment.data_

        ids = assignment.cell_ids(A=1, B=0)
        # All returned positions must satisfy A=1, B=0.
        assert (data.iloc[ids]["A"] == 1).all()
        assert (data.iloc[ids]["B"] == 0).all()
        assert len(ids) == 10  # n_per_cell

    def test_cell_ids_returns_ndarray(self) -> None:
        """cell_ids must return np.ndarray."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        ids = assignment.cell_ids(A=0, B=0)
        assert isinstance(ids, np.ndarray)

    def test_cell_ids_unknown_factor_raises(self) -> None:
        """cell_ids with unknown factor must raise InvalidDesignError."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        with pytest.raises(InvalidDesignError, match="Unknown factor"):
            assignment.cell_ids(A=1, Z=0)

    def test_cell_ids_invalid_value_raises(self) -> None:
        """cell_ids with non-binary value must raise InvalidDesignError."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        with pytest.raises(InvalidDesignError, match="0 or 1"):
            assignment.cell_ids(A=2, B=0)

    def test_cell_ids_missing_factor_raises(self) -> None:
        """cell_ids without all factors must raise InvalidDesignError."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        assignment = design.randomize(df)
        with pytest.raises(InvalidDesignError, match="Missing"):
            assignment.cell_ids(A=1)


class TestFactorialDrawIntegration:
    """Tests for FactorialAssignment.draw."""

    def test_draw_returns_factorial_assignment(self) -> None:
        """draw must return a FactorialAssignment."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        a1 = design.randomize(df)
        a2 = a1.draw(seed=99)
        assert isinstance(a2, FactorialAssignment)

    def test_draw_preserves_cell_sizes(self) -> None:
        """draw must preserve cell_sizes_ exactly."""
        df = _df_for_design(n_per_cell=7, k=3)
        design = FactorialDesign(
            factors=["A", "B", "C"], n_per_cell=7, seed=42
        )
        a1 = design.randomize(df)
        a2 = a1.draw(seed=99)
        assert a2.cell_sizes_ == a1.cell_sizes_

    def test_draw_changes_cell_assignment(self) -> None:
        """draw with a different seed must change cell assignment."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        a1 = design.randomize(df)
        a2 = a1.draw(seed=999)
        assert not np.array_equal(
            a1.data_["_cell"].values, a2.data_["_cell"].values
        )

    def test_draw_does_not_mutate_original(self) -> None:
        """draw must not mutate original assignment data."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        a1 = design.randomize(df)
        snapshot = a1.data_.copy()
        _ = a1.draw(seed=999)
        pd.testing.assert_frame_equal(a1.data_, snapshot)

    def test_draw_seed_reproducibility(self) -> None:
        """Same draw seed must produce identical assignments."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        a1 = design.randomize(df)
        a2 = a1.draw(seed=7)
        a3 = a1.draw(seed=7)
        np.testing.assert_array_equal(
            a2.data_["_cell"].values, a3.data_["_cell"].values
        )

    def test_draw_raises_without_design(self) -> None:
        """draw must raise InvalidDesignError when design_ is None."""
        df = _df_for_design(n_per_cell=10, k=2)
        design = FactorialDesign(factors=["A", "B"], n_per_cell=10, seed=42)
        a1 = design.randomize(df)
        a1.design_ = None
        with pytest.raises(InvalidDesignError, match="without a reference"):
            a1.draw(seed=1)