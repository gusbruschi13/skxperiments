"""Tests for skxperiments.design.rerandomized_crd."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from skxperiments.core.assignment import CRDAssignment
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)
from skxperiments.design.rerandomized_crd import ReRandomizedCRD


# --- Helpers ---


def _df_with_covariates(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a DataFrame with two numeric covariates."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )


def _mahalanobis_distance(
    df: pd.DataFrame,
    treatment: np.ndarray,
    covariates: list[str],
    cov_matrix: np.ndarray,
) -> float:
    """Compute scaled Mahalanobis distance between treatment groups."""
    n_t = int(treatment.sum())
    n_c = len(treatment) - n_t
    scaling = 1.0 / n_t + 1.0 / n_c
    cov_values = df[covariates].values
    mean_t = cov_values[treatment == 1].mean(axis=0)
    mean_c = cov_values[treatment == 0].mean(axis=0)
    d = mean_t - mean_c
    return float(d @ np.linalg.inv(scaling * cov_matrix) @ d)


# --- Tests ---


class TestReRandomizedCRDCreation:
    """Tests for ReRandomizedCRD instantiation."""

    def test_basic_creation_with_n_treated(self) -> None:
        """Should instantiate with n_treated."""
        design = ReRandomizedCRD(
            covariates=["x1"], threshold=5.0, n_treated=50, seed=42
        )
        assert design.n_treated == 50
        assert design.p is None

    def test_basic_creation_with_p(self) -> None:
        """Should instantiate with p."""
        design = ReRandomizedCRD(
            covariates=["x1"], threshold=5.0, p=0.5, seed=42
        )
        assert design.p == 0.5
        assert design.n_treated is None

    def test_both_n_treated_and_p_raises(self) -> None:
        """Should raise InvalidDesignError when both are provided."""
        with pytest.raises(InvalidDesignError, match="exactly one"):
            ReRandomizedCRD(
                covariates=["x1"],
                threshold=5.0,
                n_treated=50,
                p=0.5,
            )

    def test_neither_n_treated_nor_p_raises(self) -> None:
        """Should raise InvalidDesignError when neither is provided."""
        with pytest.raises(InvalidDesignError, match="exactly one"):
            ReRandomizedCRD(covariates=["x1"], threshold=5.0)

    def test_empty_covariates_raises(self) -> None:
        """Should raise InvalidDesignError for empty covariates list."""
        with pytest.raises(InvalidDesignError, match="non-empty"):
            ReRandomizedCRD(covariates=[], threshold=5.0, p=0.5)

    def test_threshold_zero_raises(self) -> None:
        """Should raise InvalidDesignError for threshold == 0."""
        with pytest.raises(InvalidDesignError, match="threshold"):
            ReRandomizedCRD(covariates=["x1"], threshold=0.0, p=0.5)

    def test_threshold_negative_raises(self) -> None:
        """Should raise InvalidDesignError for negative threshold."""
        with pytest.raises(InvalidDesignError, match="threshold"):
            ReRandomizedCRD(covariates=["x1"], threshold=-1.0, p=0.5)


class TestReRandomizedCRDRandomize:
    """Tests for ReRandomizedCRD.randomize."""

    def test_returns_crd_assignment(self) -> None:
        """Should return a CRDAssignment instance."""
        df = _df_with_covariates()
        design = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=10.0, p=0.5, seed=42
        )
        assignment = design.randomize(df)
        assert isinstance(assignment, CRDAssignment)

    def test_metadata_populated(self) -> None:
        """rerandomization_metadata must contain all expected keys."""
        df = _df_with_covariates()
        design = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=10.0, p=0.5, seed=42
        )
        assignment = design.randomize(df)
        meta = assignment.rerandomization_metadata
        assert meta is not None
        assert set(meta.keys()) == {
            "covariates",
            "threshold",
            "cov_matrix",
            "attempts",
            "scaling_factor",
        }
        assert meta["covariates"] == ["x1", "x2"]
        assert meta["threshold"] == 10.0
        assert isinstance(meta["cov_matrix"], np.ndarray)
        assert meta["cov_matrix"].shape == (2, 2)
        assert isinstance(meta["attempts"], int)
        assert meta["attempts"] >= 1
        assert isinstance(meta["scaling_factor"], float)

    def test_distance_below_threshold(self) -> None:
        """Mahalanobis distance of returned assignment must be <= threshold."""
        df = _df_with_covariates(n=200)
        threshold = 5.0
        design = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=threshold, p=0.5, seed=42
        )
        assignment = design.randomize(df)
        meta = assignment.rerandomization_metadata
        treatment = assignment.data_["treatment"].values
        distance = _mahalanobis_distance(
            df, treatment, ["x1", "x2"], meta["cov_matrix"]
        )
        assert distance <= threshold

    def test_seed_reproducibility(self) -> None:
        """Same seed must produce identical assignments."""
        df = _df_with_covariates()
        d1 = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=10.0, p=0.5, seed=42
        )
        d2 = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=10.0, p=0.5, seed=42
        )
        a1 = d1.randomize(df)
        a2 = d2.randomize(df)
        np.testing.assert_array_equal(
            a1.data_["treatment"].values, a2.data_["treatment"].values
        )

    def test_does_not_modify_input_dataframe(self) -> None:
        """Input DataFrame must not be modified."""
        df = _df_with_covariates()
        snapshot = df.copy()
        design = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=10.0, p=0.5, seed=42
        )
        _ = design.randomize(df)
        pd.testing.assert_frame_equal(df, snapshot)

    def test_design_reference_set(self) -> None:
        """Assignment must hold a reference to the generating design."""
        df = _df_with_covariates()
        design = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=10.0, p=0.5, seed=42
        )
        assignment = design.randomize(df)
        assert assignment.design_ is design


class TestReRandomizedCRDValidation:
    """Tests for validation in randomize."""

    def test_missing_covariate_raises(self) -> None:
        """Should raise InvalidDesignError for nonexistent covariate."""
        df = _df_with_covariates()
        design = ReRandomizedCRD(
            covariates=["x1", "missing"], threshold=10.0, p=0.5, seed=42
        )
        with pytest.raises(InvalidDesignError, match="not found"):
            design.randomize(df)

    def test_non_numeric_covariate_raises(self) -> None:
        """Should raise InvalidDesignError for non-numeric covariate."""
        df = _df_with_covariates()
        df["category"] = ["a"] * len(df)
        design = ReRandomizedCRD(
            covariates=["category"], threshold=10.0, p=0.5, seed=42
        )
        with pytest.raises(InvalidDesignError, match="numeric"):
            design.randomize(df)

    def test_nan_in_covariate_raises(self) -> None:
        """Should raise InvalidDesignError for NaN in covariate."""
        df = _df_with_covariates()
        df.loc[0, "x1"] = np.nan
        design = ReRandomizedCRD(
            covariates=["x1"], threshold=10.0, p=0.5, seed=42
        )
        with pytest.raises(InvalidDesignError, match="NaN"):
            design.randomize(df)

    def test_singular_covariance_matrix_raises(self) -> None:
        """Should raise InvalidDesignError for collinear covariates."""
        df = _df_with_covariates()
        df["x1_dup"] = df["x1"] * 2.0
        design = ReRandomizedCRD(
            covariates=["x1", "x1_dup"], threshold=10.0, p=0.5, seed=42
        )
        with pytest.raises(InvalidDesignError, match="singular|collinear"):
            design.randomize(df)

    def test_max_attempts_reached_raises(self) -> None:
        """Should raise InvalidDesignError when max_attempts is reached."""
        df = _df_with_covariates()
        design = ReRandomizedCRD(
            covariates=["x1", "x2"],
            threshold=1e-10,  # essentially impossible
            p=0.5,
            seed=42,
            max_attempts=5,
        )
        with pytest.raises(InvalidDesignError, match="failed to find"):
            design.randomize(df)

    def test_insufficient_data_raises(self) -> None:
        """Should raise InsufficientDataError when n < n_treated."""
        df = _df_with_covariates(n=10)
        design = ReRandomizedCRD(
            covariates=["x1", "x2"],
            threshold=10.0,
            n_treated=20,
            seed=42,
        )
        with pytest.raises(InsufficientDataError):
            design.randomize(df)

    def test_treatment_col_already_exists_raises(self) -> None:
        """Should raise if treatment_col already in DataFrame."""
        df = _df_with_covariates()
        df["treatment"] = 0
        design = ReRandomizedCRD(
            covariates=["x1"], threshold=10.0, p=0.5, seed=42
        )
        with pytest.raises(InvalidDesignError, match="already exists"):
            design.randomize(df)


class TestReRandomizedCRDDrawIntegration:
    """Critical tests: draw must respect the rerandomization criterion."""

    def test_all_draws_satisfy_criterion(self) -> None:
        """All draws must produce assignments with M <= threshold."""
        df = _df_with_covariates(n=200, seed=0)
        threshold = 3.0
        design = ReRandomizedCRD(
            covariates=["x1", "x2"],
            threshold=threshold,
            p=0.5,
            seed=42,
        )
        a_orig = design.randomize(df)

        for i in range(20):
            a_new = a_orig.draw(seed=1000 + i)
            assert a_new.rerandomization_metadata is not None
            cov_matrix = a_new.rerandomization_metadata["cov_matrix"]
            treatment = a_new.data_["treatment"].values
            distance = _mahalanobis_distance(
                df, treatment, ["x1", "x2"], cov_matrix
            )
            assert distance <= threshold, (
                f"Draw {i} produced distance {distance} > threshold "
                f"{threshold}"
            )

    def test_cov_matrix_identical_across_draws(self) -> None:
        """Each draw's cov_matrix must equal the original (no recompute)."""
        df = _df_with_covariates(n=200, seed=0)
        design = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=10.0, p=0.5, seed=42
        )
        a_orig = design.randomize(df)
        original_cov = a_orig.rerandomization_metadata["cov_matrix"]

        for i in range(5):
            a_new = a_orig.draw(seed=2000 + i)
            new_cov = a_new.rerandomization_metadata["cov_matrix"]
            np.testing.assert_array_equal(new_cov, original_cov)

    def test_each_draw_returns_rerandomized_assignment(self) -> None:
        """Each draw must return a CRDAssignment with metadata."""
        df = _df_with_covariates(n=200, seed=0)
        design = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=10.0, p=0.5, seed=42
        )
        a_orig = design.randomize(df)

        for i in range(5):
            a_new = a_orig.draw(seed=3000 + i)
            assert isinstance(a_new, CRDAssignment)
            assert a_new.rerandomization_metadata is not None

    def test_rejection_loop_is_exercised_under_strict_threshold(self) -> None:
        """Under a strict threshold, at least one draw must require
        more than one attempt — confirming the loop is not skipped.
        """
        df = _df_with_covariates(n=200, seed=0)
        # Threshold accepting ~1% of CRD randomizations under chi2(k=2).
        strict_threshold = float(chi2.ppf(0.01, df=2))
        design = ReRandomizedCRD(
            covariates=["x1", "x2"],
            threshold=strict_threshold,
            p=0.5,
            seed=42,
            max_attempts=10_000,
        )
        a_orig = design.randomize(df)

        attempts_per_draw = []
        for i in range(20):
            a_new = a_orig.draw(seed=4000 + i)
            attempts_per_draw.append(
                a_new.rerandomization_metadata["attempts"]
            )

        # With ~1% acceptance, expected attempts per draw ~ 100.
        # Asserting at least one draw required >1 attempt is a safe
        # lower bound: probability of all 20 draws accepting on first
        # try is ~0.01^20, vanishingly small.
        assert max(attempts_per_draw) > 1

    def test_draw_does_not_mutate_original(self) -> None:
        """draw must not mutate the original assignment's data."""
        df = _df_with_covariates(n=200)
        design = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=10.0, p=0.5, seed=42
        )
        a_orig = design.randomize(df)
        snapshot = a_orig.data_.copy()
        _ = a_orig.draw(seed=999)
        pd.testing.assert_frame_equal(a_orig.data_, snapshot)

    def test_draw_seed_reproducibility(self) -> None:
        """Same draw seed must produce identical assignments."""
        df = _df_with_covariates(n=200)
        design = ReRandomizedCRD(
            covariates=["x1", "x2"], threshold=10.0, p=0.5, seed=42
        )
        a_orig = design.randomize(df)
        a1 = a_orig.draw(seed=7)
        a2 = a_orig.draw(seed=7)
        np.testing.assert_array_equal(
            a1.data_["treatment"].values, a2.data_["treatment"].values
        )