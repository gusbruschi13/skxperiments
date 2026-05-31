"""Tests for skxperiments.design.balance."""

import math

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import CRDAssignment
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)
from skxperiments.design.balance import check_balance


# --- Fixtures / helpers ---


def _make_assignment(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
) -> CRDAssignment:
    """Build a CRDAssignment directly from a DataFrame.

    Used because design.crd.CRD does not exist yet at this phase.
    """
    return CRDAssignment(
        data=df.copy(),
        treatment_col=treatment_col,
        design=None,
        seed=None,
    )


def _balanced_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic DataFrame with two numeric covariates and a
    randomized treatment vector with equal proportions.
    """
    rng = np.random.default_rng(seed)
    treatment = np.zeros(n, dtype=int)
    treatment[: n // 2] = 1
    rng.shuffle(treatment)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "treatment": treatment,
        }
    )


# --- Tests ---


class TestCheckBalanceBasic:
    """Tests for basic structure of the output."""

    def test_returns_expected_columns(self) -> None:
        """Output must have exactly the documented columns."""
        df = _balanced_df()
        assignment = _make_assignment(df)
        result = check_balance(assignment)
        assert list(result.columns) == [
            "covariate",
            "mean_treated",
            "mean_control",
            "std_pooled",
            "smd",
        ]

    def test_n_rows_matches_n_covariates(self) -> None:
        """Number of rows must equal number of covariates passed."""
        df = _balanced_df()
        assignment = _make_assignment(df)
        result = check_balance(assignment, covariates=["x1", "x2"])
        assert len(result) == 2

    def test_treatment_col_never_in_output(self) -> None:
        """The treatment column must never appear as a covariate row."""
        df = _balanced_df()
        assignment = _make_assignment(df)
        result = check_balance(assignment)
        assert "treatment" not in result["covariate"].values

    def test_returns_dataframe(self) -> None:
        """Output must be a pandas DataFrame."""
        df = _balanced_df()
        assignment = _make_assignment(df)
        result = check_balance(assignment)
        assert isinstance(result, pd.DataFrame)

    def test_default_index_is_rangeindex(self) -> None:
        """Output index must be a default RangeIndex."""
        df = _balanced_df()
        assignment = _make_assignment(df)
        result = check_balance(assignment)
        assert isinstance(result.index, pd.RangeIndex)


class TestCheckBalanceCovariateSelection:
    """Tests for how covariates are selected."""

    def test_none_selects_all_numeric_in_order(self) -> None:
        """covariates=None selects all numeric columns in DataFrame order."""
        df = pd.DataFrame(
            {
                "x2": [1.0, 2.0, 3.0, 4.0],
                "x1": [0.5, 1.5, 2.5, 3.5],
                "treatment": [1, 0, 1, 0],
            }
        )
        assignment = _make_assignment(df)
        result = check_balance(assignment)
        # Order must follow DataFrame column order, not alphabetical.
        assert list(result["covariate"].values) == ["x2", "x1"]

    def test_explicit_list_preserves_order(self) -> None:
        """Explicit covariates list must preserve the given order."""
        df = _balanced_df()
        assignment = _make_assignment(df)
        result = check_balance(assignment, covariates=["x2", "x1"])
        assert list(result["covariate"].values) == ["x2", "x1"]

    def test_treatment_excluded_when_numeric(self) -> None:
        """Even if treatment is numeric, it must be excluded when
        covariates=None.
        """
        df = _balanced_df()
        assignment = _make_assignment(df)
        result = check_balance(assignment)
        assert "treatment" not in result["covariate"].values

    def test_non_numeric_columns_excluded_by_default(self) -> None:
        """Non-numeric columns must be excluded when covariates=None."""
        df = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 4.0],
                "category": ["a", "b", "a", "b"],
                "treatment": [1, 0, 1, 0],
            }
        )
        assignment = _make_assignment(df)
        result = check_balance(assignment)
        assert list(result["covariate"].values) == ["x1"]

    def test_boolean_column_treated_as_numeric(self) -> None:
        """Boolean columns must be considered numeric when covariates=None."""
        df = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 4.0],
                "is_premium": [True, False, True, False],
                "treatment": [1, 0, 1, 0],
            }
        )
        assignment = _make_assignment(df)
        result = check_balance(assignment)
        assert "is_premium" in result["covariate"].values


class TestCheckBalanceValidation:
    """Tests for validation logic."""

    def test_raises_on_unknown_covariate(self) -> None:
        """Should raise InvalidDesignError if a covariate name is unknown."""
        df = _balanced_df()
        assignment = _make_assignment(df)
        with pytest.raises(InvalidDesignError, match="not found"):
            check_balance(assignment, covariates=["x1", "nonexistent"])

    def test_raises_on_nan_in_covariate(self) -> None:
        """Should raise InvalidDesignError if a selected covariate has NaN."""
        df = pd.DataFrame(
            {
                "x1": [1.0, np.nan, 3.0, 4.0],
                "treatment": [1, 0, 1, 0],
            }
        )
        assignment = _make_assignment(df)
        with pytest.raises(InvalidDesignError, match="NaN"):
            check_balance(assignment, covariates=["x1"])

    def test_raises_on_nan_when_default_selection(self) -> None:
        """Should raise on NaN even when covariates=None auto-selects."""
        df = pd.DataFrame(
            {
                "x1": [1.0, np.nan, 3.0, 4.0],
                "treatment": [1, 0, 1, 0],
            }
        )
        assignment = _make_assignment(df)
        with pytest.raises(InvalidDesignError, match="NaN"):
            check_balance(assignment)

    def test_raises_when_no_numeric_columns(self) -> None:
        """Should raise InsufficientDataError when no numeric covariate
        is available after excluding the treatment column.
        """
        df = pd.DataFrame(
            {
                "category": ["a", "b", "a", "b"],
                "treatment": [1, 0, 1, 0],
            }
        )
        assignment = _make_assignment(df)
        with pytest.raises(InsufficientDataError):
            check_balance(assignment)


class TestCheckBalanceNumerics:
    """Tests for numerical correctness."""

    def test_means_match_groupby(self) -> None:
        """mean_treated and mean_control must match groupby results."""
        df = _balanced_df()
        assignment = _make_assignment(df)
        result = check_balance(assignment, covariates=["x1"])

        expected = df.groupby("treatment")["x1"].mean()
        row = result.iloc[0]
        assert math.isclose(row["mean_treated"], expected.loc[1])
        assert math.isclose(row["mean_control"], expected.loc[0])

    def test_std_pooled_matches_formula(self) -> None:
        """std_pooled must equal sqrt((var_t + var_c) / 2) with ddof=1."""
        df = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 10.0, 12.0, 14.0],
                "treatment": [1, 1, 1, 0, 0, 0],
            }
        )
        assignment = _make_assignment(df)
        result = check_balance(assignment, covariates=["x1"])

        treated = np.array([1.0, 2.0, 3.0])
        control = np.array([10.0, 12.0, 14.0])
        var_t = np.var(treated, ddof=1)
        var_c = np.var(control, ddof=1)
        expected_std = float(np.sqrt((var_t + var_c) / 2.0))

        assert math.isclose(result.iloc[0]["std_pooled"], expected_std)

    def test_smd_matches_formula(self) -> None:
        """smd must equal (mean_t - mean_c) / std_pooled."""
        df = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 10.0, 12.0, 14.0],
                "treatment": [1, 1, 1, 0, 0, 0],
            }
        )
        assignment = _make_assignment(df)
        result = check_balance(assignment, covariates=["x1"])

        row = result.iloc[0]
        expected_smd = (row["mean_treated"] - row["mean_control"]) / row["std_pooled"]
        assert math.isclose(row["smd"], expected_smd)

    def test_std_pooled_never_negative(self) -> None:
        """std_pooled must be non-negative for all rows."""
        df = _balanced_df()
        assignment = _make_assignment(df)
        result = check_balance(assignment)
        assert (result["std_pooled"] >= 0).all()

    def test_smd_is_nan_when_std_pooled_zero(self) -> None:
        """SMD must be NaN when both groups have constant value
        (std_pooled == 0), without raising.
        """
        df = pd.DataFrame(
            {
                "x_constant": [5.0, 5.0, 5.0, 5.0],
                "treatment": [1, 0, 1, 0],
            }
        )
        assignment = _make_assignment(df)
        result = check_balance(assignment, covariates=["x_constant"])

        row = result.iloc[0]
        assert row["std_pooled"] == 0.0
        assert math.isnan(row["smd"])

    def test_smd_close_to_zero_with_random_assignment(self) -> None:
        """For a randomized treatment over synthetic data, SMD should be
        small in absolute value.
        """
        df = _balanced_df(n=2000, seed=123)
        assignment = _make_assignment(df)
        result = check_balance(assignment)
        # Loose threshold: with N=2000 and random assignment, |SMD|
        # should be well under 0.1 with very high probability.
        assert (result["smd"].abs() < 0.1).all()

    def test_does_not_modify_input_data(self) -> None:
        """check_balance must not mutate assignment.data_."""
        df = _balanced_df()
        assignment = _make_assignment(df)
        snapshot = assignment.data_.copy()
        check_balance(assignment)
        pd.testing.assert_frame_equal(assignment.data_, snapshot)