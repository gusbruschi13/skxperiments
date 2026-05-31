"""Tests for skxperiments.design.crd."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import CRDAssignment
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)
from skxperiments.design.crd import CRD


class TestCRDCreation:
    """Tests for CRD instantiation."""

    def test_creation_with_n_treated(self) -> None:
        design = CRD(n_treated=50, seed=42)
        assert design.n_treated == 50
        assert design.p is None

    def test_creation_with_p(self) -> None:
        design = CRD(p=0.5, seed=42)
        assert design.p == 0.5
        assert design.n_treated is None

    def test_default_treatment_col(self) -> None:
        design = CRD(p=0.5)
        assert design.treatment_col == "treatment"

    def test_custom_treatment_col(self) -> None:
        design = CRD(p=0.5, treatment_col="T")
        assert design.treatment_col == "T"

    def test_both_n_treated_and_p_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="exactly one"):
            CRD(n_treated=50, p=0.5)

    def test_neither_n_treated_nor_p_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="exactly one"):
            CRD()

    def test_n_treated_zero_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="positive integer"):
            CRD(n_treated=0)

    def test_n_treated_negative_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="positive integer"):
            CRD(n_treated=-5)

    def test_p_zero_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="in \\(0, 1\\)"):
            CRD(p=0.0)

    def test_p_one_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="in \\(0, 1\\)"):
            CRD(p=1.0)

    def test_p_negative_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="in \\(0, 1\\)"):
            CRD(p=-0.1)

    def test_p_above_one_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="in \\(0, 1\\)"):
            CRD(p=1.5)


class TestCRDRandomize:
    """Tests for CRD.randomize."""

    def test_returns_crd_assignment(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        assignment = CRD(p=0.5, seed=42).randomize(df)
        assert isinstance(assignment, CRDAssignment)

    def test_does_not_modify_input(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        snapshot = df.copy()
        _ = CRD(p=0.5, seed=42).randomize(df)
        pd.testing.assert_frame_equal(df, snapshot)

    def test_treatment_column_added(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        assignment = CRD(p=0.5, seed=42).randomize(df)
        assert "treatment" in assignment.data_.columns

    def test_n_treated_with_p(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        assignment = CRD(p=0.5, seed=42).randomize(df)
        assert assignment.n_treated_ == 50
        assert assignment.n_control_ == 50

    def test_n_treated_with_absolute_count(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        assignment = CRD(n_treated=30, seed=42).randomize(df)
        assert assignment.n_treated_ == 30
        assert assignment.n_control_ == 70

    def test_seed_reproducibility(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        a1 = CRD(p=0.5, seed=42).randomize(df)
        a2 = CRD(p=0.5, seed=42).randomize(df)
        np.testing.assert_array_equal(
            a1.data_["treatment"].values, a2.data_["treatment"].values
        )

    def test_different_seeds_differ(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        a1 = CRD(p=0.5, seed=1).randomize(df)
        a2 = CRD(p=0.5, seed=2).randomize(df)
        assert not np.array_equal(
            a1.data_["treatment"].values, a2.data_["treatment"].values
        )

    def test_design_reference_set(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        design = CRD(p=0.5, seed=42)
        assignment = design.randomize(df)
        assert assignment.design_ is design


class TestCRDValidation:
    """Tests for randomize validation."""

    def test_treatment_col_already_exists_raises(self) -> None:
        df = pd.DataFrame({"x": range(100), "treatment": 0})
        with pytest.raises(InvalidDesignError, match="already exists"):
            CRD(p=0.5).randomize(df)

    def test_too_few_units_raises(self) -> None:
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(InsufficientDataError):
            CRD(p=0.5).randomize(df)

    def test_n_treated_exceeds_n_total_raises(self) -> None:
        df = pd.DataFrame({"x": range(10)})
        with pytest.raises(InsufficientDataError):
            CRD(n_treated=20).randomize(df)

    def test_p_rounds_to_zero_raises(self) -> None:
        # n=2, p=0.1 -> round(0.2) = 0 treated
        df = pd.DataFrame({"x": range(2)})
        with pytest.raises(InvalidDesignError, match="strictly between"):
            CRD(p=0.1).randomize(df)

    def test_p_rounds_to_n_raises(self) -> None:
        # n=2, p=0.9 -> round(1.8) = 2 treated, all of them
        df = pd.DataFrame({"x": range(2)})
        with pytest.raises(InvalidDesignError, match="strictly between"):
            CRD(p=0.9).randomize(df)


class TestCRDDrawIntegration:
    """Tests for CRDAssignment.draw via CRD."""

    def test_draw_returns_crd_assignment(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        a1 = CRD(p=0.5, seed=42).randomize(df)
        a2 = a1.draw(seed=99)
        assert isinstance(a2, CRDAssignment)

    def test_draw_preserves_n_treated(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        a1 = CRD(p=0.5, seed=42).randomize(df)
        a2 = a1.draw(seed=99)
        assert a2.n_treated_ == a1.n_treated_

    def test_draw_changes_treatment_vector(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        a1 = CRD(p=0.5, seed=42).randomize(df)
        a2 = a1.draw(seed=999)
        assert not np.array_equal(
            a1.data_["treatment"].values, a2.data_["treatment"].values
        )

    def test_draw_seed_reproducibility(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        a1 = CRD(p=0.5, seed=42).randomize(df)
        a2 = a1.draw(seed=7)
        a3 = a1.draw(seed=7)
        np.testing.assert_array_equal(
            a2.data_["treatment"].values, a3.data_["treatment"].values
        )

    def test_draw_does_not_mutate_original(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        a1 = CRD(p=0.5, seed=42).randomize(df)
        snapshot = a1.data_.copy()
        _ = a1.draw(seed=999)
        pd.testing.assert_frame_equal(a1.data_, snapshot)

    def test_draw_raises_without_design(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        a1 = CRD(p=0.5, seed=42).randomize(df)
        a1.design_ = None
        with pytest.raises(InvalidDesignError, match="without a reference"):
            a1.draw(seed=1)