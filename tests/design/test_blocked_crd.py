"""Tests for skxperiments.design.blocked_crd."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import BlockedAssignment
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)
from skxperiments.design.balance import check_balance
from skxperiments.design.blocked_crd import BlockedCRD


# --- Helpers ---


def _df_two_blocks(n_per_block: int = 50, seed: int = 0) -> pd.DataFrame:
    """Build a DataFrame with two blocks of equal size and one numeric
    covariate.
    """
    rng = np.random.default_rng(seed)
    n_total = 2 * n_per_block
    return pd.DataFrame(
        {
            "x": rng.normal(size=n_total),
            "region": ["A"] * n_per_block + ["B"] * n_per_block,
        }
    )


# --- Tests ---


class TestBlockedCRDCreation:
    """Tests for BlockedCRD instantiation."""

    def test_basic_creation(self) -> None:
        """Should instantiate with valid parameters."""
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        assert design.block_col == "region"
        assert design.p == 0.5
        assert design.seed == 42
        assert design.treatment_col == "treatment"

    def test_custom_treatment_col(self) -> None:
        """Should accept custom treatment_col."""
        design = BlockedCRD(block_col="region", p=0.5, treatment_col="T")
        assert design.treatment_col == "T"

    def test_p_none_raises(self) -> None:
        """Should raise InvalidDesignError when p is None."""
        with pytest.raises(InvalidDesignError):
            BlockedCRD(block_col="region", p=None)

    def test_p_zero_raises(self) -> None:
        """Should raise InvalidDesignError when p == 0."""
        with pytest.raises(InvalidDesignError):
            BlockedCRD(block_col="region", p=0.0)

    def test_p_one_raises(self) -> None:
        """Should raise InvalidDesignError when p == 1."""
        with pytest.raises(InvalidDesignError):
            BlockedCRD(block_col="region", p=1.0)

    def test_p_negative_raises(self) -> None:
        """Should raise InvalidDesignError when p is negative."""
        with pytest.raises(InvalidDesignError):
            BlockedCRD(block_col="region", p=-0.1)

    def test_p_above_one_raises(self) -> None:
        """Should raise InvalidDesignError when p > 1."""
        with pytest.raises(InvalidDesignError):
            BlockedCRD(block_col="region", p=1.5)


class TestBlockedCRDRandomize:
    """Tests for BlockedCRD.randomize."""

    def test_returns_blocked_assignment(self) -> None:
        """Should return a BlockedAssignment instance."""
        df = _df_two_blocks()
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        assignment = design.randomize(df)
        assert isinstance(assignment, BlockedAssignment)

    def test_does_not_modify_input_dataframe(self) -> None:
        """Should not modify the input DataFrame."""
        df = _df_two_blocks()
        snapshot = df.copy()
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        _ = design.randomize(df)
        pd.testing.assert_frame_equal(df, snapshot)

    def test_treatment_column_added_to_output(self) -> None:
        """Output assignment must carry the treatment column."""
        df = _df_two_blocks()
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        assignment = design.randomize(df)
        assert "treatment" in assignment.data_.columns

    def test_proportion_per_block_matches_p(self) -> None:
        """Number of treated per block must equal round(p * n_block)."""
        df = _df_two_blocks(n_per_block=50)
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        assignment = design.randomize(df)
        data = assignment.data_
        for label, n_block in assignment.block_sizes_.items():
            expected = int(round(0.5 * n_block))
            actual = int(
                ((data["region"] == label) & (data["treatment"] == 1)).sum()
            )
            assert actual == expected

    def test_proportion_per_block_with_uneven_blocks(self) -> None:
        """Proportion must hold per block when block sizes differ."""
        df = pd.DataFrame(
            {
                "x": list(range(30)),
                "region": ["A"] * 10 + ["B"] * 20,
            }
        )
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        assignment = design.randomize(df)
        data = assignment.data_
        treated_a = int(((data["region"] == "A") & (data["treatment"] == 1)).sum())
        treated_b = int(((data["region"] == "B") & (data["treatment"] == 1)).sum())
        assert treated_a == 5
        assert treated_b == 10

    def test_seed_reproducibility(self) -> None:
        """Same seed must produce identical treatment vectors."""
        df = _df_two_blocks()
        d1 = BlockedCRD(block_col="region", p=0.5, seed=42)
        d2 = BlockedCRD(block_col="region", p=0.5, seed=42)
        a1 = d1.randomize(df)
        a2 = d2.randomize(df)
        np.testing.assert_array_equal(
            a1.data_["treatment"].values, a2.data_["treatment"].values
        )

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds should usually produce different results."""
        df = _df_two_blocks()
        a1 = BlockedCRD(block_col="region", p=0.5, seed=1).randomize(df)
        a2 = BlockedCRD(block_col="region", p=0.5, seed=2).randomize(df)
        # Probability of identical vectors with N=100 is negligible.
        assert not np.array_equal(
            a1.data_["treatment"].values, a2.data_["treatment"].values
        )

    def test_block_sizes_reflect_actual_sizes(self) -> None:
        """block_sizes_ must match the count of units per block."""
        df = pd.DataFrame(
            {
                "x": list(range(30)),
                "region": ["A"] * 10 + ["B"] * 20,
            }
        )
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        assignment = design.randomize(df)
        assert assignment.block_sizes_ == {"A": 10, "B": 20}

    def test_assignment_design_reference_set(self) -> None:
        """Assignment must hold a reference to the generating design."""
        df = _df_two_blocks()
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        assignment = design.randomize(df)
        assert assignment.design_ is design


class TestBlockedCRDValidation:
    """Tests for validation in BlockedCRD."""

    def test_missing_block_col_raises(self) -> None:
        """Should raise InvalidDesignError when block_col is missing."""
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        with pytest.raises(InvalidDesignError, match="not found"):
            design.randomize(df)

    def test_treatment_col_already_exists_raises(self) -> None:
        """Should raise if treatment_col already in DataFrame."""
        df = pd.DataFrame(
            {
                "region": ["A", "A", "B", "B"],
                "treatment": [0, 1, 0, 1],
            }
        )
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        with pytest.raises(InvalidDesignError, match="already exists"):
            design.randomize(df)

    def test_block_too_small_raises(self) -> None:
        """Should raise InsufficientDataError if a block has fewer
        than 2 units.
        """
        df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "region": ["A", "A", "B"],
            }
        )
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        with pytest.raises(InsufficientDataError):
            design.randomize(df)

    def test_rounding_yields_zero_treated_raises(self) -> None:
        """Should raise InvalidDesignError if rounding yields 0 treated
        in some block.
        """
        # Block of size 2 with p=0.1 -> round(0.2) = 0 treated.
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "region": ["A", "A", "B", "B"],
            }
        )
        design = BlockedCRD(block_col="region", p=0.1, seed=42)
        with pytest.raises(InvalidDesignError, match="treated"):
            design.randomize(df)

    def test_rounding_yields_all_treated_raises(self) -> None:
        """Should raise InvalidDesignError if rounding yields n treated
        in some block (no controls).
        """
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "region": ["A", "A", "B", "B"],
            }
        )
        design = BlockedCRD(block_col="region", p=0.9, seed=42)
        with pytest.raises(InvalidDesignError, match="treated"):
            design.randomize(df)


class TestBlockedCRDBalanceIntegration:
    """Integration with check_balance."""

    def test_check_balance_runs_without_error(self) -> None:
        """check_balance must run successfully on a BlockedAssignment."""
        df = _df_two_blocks(n_per_block=100, seed=0)
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        assignment = design.randomize(df)
        result = check_balance(assignment, covariates=["x"])
        assert isinstance(result, pd.DataFrame)

    def test_block_indicator_perfectly_balanced(self) -> None:
        """A numeric indicator of the block must have SMD == 0 after
        blocked randomization (treatment is balanced exactly within
        every block, so the proportion of any block-derived dummy is
        identical across treatment arms).
        """
        df = _df_two_blocks(n_per_block=100, seed=0)
        # Add a numeric dummy that is a deterministic function of the block.
        df["region_is_A"] = (df["region"] == "A").astype(int)

        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        assignment = design.randomize(df)

        result = check_balance(assignment, covariates=["region_is_A"])
        smd = result.iloc[0]["smd"]
        # With p=0.5 and equal block sizes, treated/control means of
        # region_is_A must coincide exactly.
        assert smd == pytest.approx(0.0, abs=1e-12)


class TestBlockedCRDDrawIntegration:
    """Tests for BlockedAssignment.draw."""

    def test_draw_returns_blocked_assignment(self) -> None:
        """draw must return a BlockedAssignment."""
        df = _df_two_blocks()
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        a1 = design.randomize(df)
        a2 = a1.draw(seed=99)
        assert isinstance(a2, BlockedAssignment)

    def test_draw_preserves_block_sizes(self) -> None:
        """draw must preserve block_sizes_."""
        df = _df_two_blocks()
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        a1 = design.randomize(df)
        a2 = a1.draw(seed=99)
        assert a2.block_sizes_ == a1.block_sizes_

    def test_draw_changes_treatment_vector(self) -> None:
        """draw with a different seed must change the treatment vector."""
        df = _df_two_blocks(n_per_block=50)
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        a1 = design.randomize(df)
        a2 = a1.draw(seed=999)
        assert not np.array_equal(
            a1.data_["treatment"].values, a2.data_["treatment"].values
        )

    def test_draw_preserves_proportion_per_block(self) -> None:
        """draw must preserve the within-block treatment proportion."""
        df = _df_two_blocks(n_per_block=50)
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        a1 = design.randomize(df)
        a2 = a1.draw(seed=999)

        for label, n_block in a2.block_sizes_.items():
            expected = int(round(0.5 * n_block))
            actual = int(
                (
                    (a2.data_["region"] == label)
                    & (a2.data_["treatment"] == 1)
                ).sum()
            )
            assert actual == expected

    def test_draw_raises_without_design(self) -> None:
        """draw must raise InvalidDesignError when design_ is None."""
        df = _df_two_blocks()
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        a1 = design.randomize(df)

        # Manually drop the design reference to simulate orphan assignment.
        a1.design_ = None

        with pytest.raises(InvalidDesignError, match="without a reference"):
            a1.draw(seed=1)

    def test_draw_does_not_mutate_original(self) -> None:
        """draw must not mutate the original assignment's data."""
        df = _df_two_blocks()
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        a1 = design.randomize(df)
        snapshot = a1.data_.copy()
        _ = a1.draw(seed=999)
        pd.testing.assert_frame_equal(a1.data_, snapshot)

    def test_draw_seed_reproducibility(self) -> None:
        """Same draw seed must produce identical assignments."""
        df = _df_two_blocks()
        design = BlockedCRD(block_col="region", p=0.5, seed=42)
        a1 = design.randomize(df)
        a2 = a1.draw(seed=7)
        a3 = a1.draw(seed=7)
        np.testing.assert_array_equal(
            a2.data_["treatment"].values, a3.data_["treatment"].values
        )