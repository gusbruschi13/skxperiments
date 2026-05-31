"""Tests for skxperiments.estimators.blocked_difference_in_means."""

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
from skxperiments.estimators.blocked_difference_in_means import (
    BlockedDifferenceInMeans,
)
from skxperiments.estimators.difference_in_means import DifferenceInMeans


# --- Helpers ---


def _make_blocked_assignment(
    n_per_block: dict[str, int] | None = None,
    seed: int = 42,
    true_ate: float = 0.5,
    p: float = 0.5,
) -> BlockedAssignment:
    """Build a BlockedAssignment whose outcome embeds a known ATE.

    Parameters
    ----------
    n_per_block : dict[str, int] or None
        Mapping from block label to number of units. Default
        ``{"A": 50, "B": 50}``.
    seed : int
        Seed for random number generation and for BlockedCRD.
    true_ate : float
        Additive ATE injected into treated units' outcome.
    p : float
        Treatment proportion within each block.

    Notes
    -----
    Mutação de ``data_`` aceitável apenas em fixture de teste.
    """
    if n_per_block is None:
        n_per_block = {"A": 50, "B": 50}

    rng = np.random.default_rng(seed)

    block_labels: list[str] = []
    for b, n in n_per_block.items():
        block_labels.extend([b] * n)
    n_total = len(block_labels)

    df = pd.DataFrame(
        {
            "x": rng.normal(size=n_total),
            "block": block_labels,
            "y": rng.normal(size=n_total),
        }
    )

    design = BlockedCRD(block_col="block", p=p, seed=seed)
    assignment = design.randomize(df)

    treated = assignment.treated_ids()
    assignment.data_.iloc[
        treated, assignment.data_.columns.get_loc("y")
    ] += true_ate
    return assignment


# --- Tests ---


class TestBlockedDifferenceInMeansCreation:
    """Tests for BlockedDifferenceInMeans instantiation."""

    def test_basic_creation(self) -> None:
        """Should instantiate with outcome_col."""
        estimator = BlockedDifferenceInMeans(outcome_col="y")
        assert estimator.outcome_col == "y"

    def test_outcome_col_stored_as_is(self) -> None:
        """outcome_col must be stored verbatim."""
        estimator = BlockedDifferenceInMeans(outcome_col="my_metric")
        assert estimator.outcome_col == "my_metric"


class TestBlockedDifferenceInMeansFit:
    """Tests for BlockedDifferenceInMeans.fit."""

    def test_fit_returns_self(self) -> None:
        """fit must return self for chaining."""
        assignment = _make_blocked_assignment()
        estimator = BlockedDifferenceInMeans(outcome_col="y")
        returned = estimator.fit(assignment)
        assert returned is estimator

    def test_stores_assignment(self) -> None:
        """fit must store the assignment as assignment_."""
        assignment = _make_blocked_assignment()
        estimator = BlockedDifferenceInMeans(outcome_col="y").fit(assignment)
        assert estimator.assignment_ is assignment

    def test_stores_ate(self) -> None:
        """fit must compute and store ate_ as a float."""
        assignment = _make_blocked_assignment()
        estimator = BlockedDifferenceInMeans(outcome_col="y").fit(assignment)
        assert isinstance(estimator.ate_, float)

    def test_stores_block_ates(self) -> None:
        """fit must store block_ates_ as a dict with one entry per block."""
        assignment = _make_blocked_assignment(
            n_per_block={"A": 40, "B": 60}
        )
        estimator = BlockedDifferenceInMeans(outcome_col="y").fit(assignment)
        assert isinstance(estimator.block_ates_, dict)
        assert set(estimator.block_ates_.keys()) == {"A", "B"}

    def test_block_ate_matches_manual(self) -> None:
        """Each entry in block_ates_ must equal mean(Y_t_b) - mean(Y_c_b)."""
        assignment = _make_blocked_assignment(
            n_per_block={"A": 40, "B": 60}
        )
        estimator = BlockedDifferenceInMeans(outcome_col="y").fit(assignment)

        data = assignment.data_
        for block_val in ("A", "B"):
            mask = data["block"] == block_val
            block_y = data.loc[mask, "y"]
            block_t = data.loc[mask, assignment.treatment_col_]
            expected = float(
                block_y[block_t == 1].mean() - block_y[block_t == 0].mean()
            )
            assert estimator.block_ates_[block_val] == pytest.approx(
                expected, rel=1e-12
            )


class TestBlockedDifferenceInMeansEstimate:
    """Tests for BlockedDifferenceInMeans.estimate."""

    def test_returns_results(self) -> None:
        """estimate must return a Results instance."""
        assignment = _make_blocked_assignment()
        result = (
            BlockedDifferenceInMeans(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert isinstance(result, Results)

    def test_ate_matches_weighted_sum(self) -> None:
        """Results.ate must equal sum((n_b/N) * ate_b)."""
        assignment = _make_blocked_assignment(
            n_per_block={"A": 30, "B": 70}
        )
        estimator = BlockedDifferenceInMeans(outcome_col="y").fit(assignment)
        result = estimator.estimate()

        N = assignment.n_units_
        expected = 0.0
        for block_val, n_b in assignment.block_sizes_.items():
            expected += (n_b / N) * estimator.block_ates_[block_val]
        assert result.ate == pytest.approx(expected, rel=1e-12)

    def test_estimator_name(self) -> None:
        """estimator_name must be 'BlockedDifferenceInMeans'."""
        assignment = _make_blocked_assignment()
        result = (
            BlockedDifferenceInMeans(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.estimator_name == "BlockedDifferenceInMeans"

    def test_design_name(self) -> None:
        """design_name must be 'BlockedCRD'."""
        assignment = _make_blocked_assignment()
        result = (
            BlockedDifferenceInMeans(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.design_name == "BlockedCRD"

    def test_n_metadata_populated(self) -> None:
        """n_obs, n_treated, n_control must come from the assignment."""
        assignment = _make_blocked_assignment(
            n_per_block={"A": 20, "B": 30}
        )
        result = (
            BlockedDifferenceInMeans(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.n_obs == 50
        assert result.n_treated == assignment.n_treated_
        assert result.n_control == assignment.n_control_
        assert result.n_treated + result.n_control == result.n_obs

    def test_inference_fields_are_none(self) -> None:
        """se, ci, p_value must be None (inference is Phase 4)."""
        assignment = _make_blocked_assignment()
        result = (
            BlockedDifferenceInMeans(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.se is None
        assert result.ci is None
        assert result.p_value is None

    def test_inference_name_is_none(self) -> None:
        """inference_name must be None when no inference was applied."""
        assignment = _make_blocked_assignment()
        result = (
            BlockedDifferenceInMeans(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.inference_name is None


class TestBlockedDifferenceInMeansValidation:
    """Tests for input validation."""

    def test_rejects_crd_assignment(self) -> None:
        """fit must reject CRDAssignment with DesignEstimatorMismatch."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=40),
                "y": rng.normal(size=40),
            }
        )
        crd_assignment = CRD(p=0.5, seed=0).randomize(df)
        assert isinstance(crd_assignment, CRDAssignment)

        estimator = BlockedDifferenceInMeans(outcome_col="y")
        with pytest.raises(
            DesignEstimatorMismatch, match="BlockedAssignment"
        ):
            estimator.fit(crd_assignment)

    def test_rejects_factorial_assignment(self) -> None:
        """fit must reject FactorialAssignment with DesignEstimatorMismatch."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=40),
                "y": rng.normal(size=40),
            }
        )
        fact_assignment = FactorialDesign(
            factors=["A", "B"], n_per_cell=10, seed=42
        ).randomize(df)
        assert isinstance(fact_assignment, FactorialAssignment)

        estimator = BlockedDifferenceInMeans(outcome_col="y")
        with pytest.raises(
            DesignEstimatorMismatch, match="BlockedAssignment"
        ):
            estimator.fit(fact_assignment)

    def test_missing_outcome_col_raises(self) -> None:
        """fit must raise InvalidDesignError if outcome_col absent."""
        assignment = _make_blocked_assignment()
        estimator = BlockedDifferenceInMeans(outcome_col="missing")
        with pytest.raises(InvalidDesignError, match="not found"):
            estimator.fit(assignment)

    def test_non_numeric_outcome_raises(self) -> None:
        """fit must raise InvalidDesignError if outcome is non-numeric."""
        # Build a DataFrame with a string outcome.
        df = pd.DataFrame(
            {
                "x": np.arange(40, dtype=float),
                "block": ["A"] * 20 + ["B"] * 20,
                "y": ["x"] * 40,
            }
        )
        assignment = BlockedCRD(
            block_col="block", p=0.5, seed=42
        ).randomize(df)

        estimator = BlockedDifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="numeric"):
            estimator.fit(assignment)

    def test_outcome_with_nan_raises(self) -> None:
        """fit must raise InvalidDesignError if outcome contains NaN."""
        assignment = _make_blocked_assignment()
        assignment.data_.iloc[
            0, assignment.data_.columns.get_loc("y")
        ] = np.nan

        estimator = BlockedDifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="NaN"):
            estimator.fit(assignment)

    def test_block_without_treated_raises(self) -> None:
        """fit must raise InvalidDesignError naming the offending block
        when a block has zero treated units.
        """
        # Build a BlockedAssignment manually (no randomize) where block
        # 'A' has all controls and block 'B' has all treated, except
        # we force block 'A' to have zero treated.
        df = pd.DataFrame(
            {
                "block": ["A"] * 10 + ["B"] * 10,
                "treatment": [0] * 10 + [1] * 5 + [0] * 5,
                "y": np.arange(20, dtype=float),
            }
        )
        assignment = BlockedAssignment(
            data=df.copy(),
            treatment_col="treatment",
            design=None,
            block_col="block",
            block_sizes={"A": 10, "B": 10},
        )

        estimator = BlockedDifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="'A'"):
            estimator.fit(assignment)

    def test_block_without_controls_raises(self) -> None:
        """fit must raise InvalidDesignError naming the offending block
        when a block has zero control units.
        """
        df = pd.DataFrame(
            {
                "block": ["A"] * 10 + ["B"] * 10,
                "treatment": [1] * 10 + [1] * 5 + [0] * 5,
                "y": np.arange(20, dtype=float),
            }
        )
        assignment = BlockedAssignment(
            data=df.copy(),
            treatment_col="treatment",
            design=None,
            block_col="block",
            block_sizes={"A": 10, "B": 10},
        )

        estimator = BlockedDifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="'A'"):
            estimator.fit(assignment)

    def test_estimate_before_fit_raises(self) -> None:
        """estimate must raise NotFittedError if fit was not called."""
        estimator = BlockedDifferenceInMeans(outcome_col="y")
        with pytest.raises(NotFittedError):
            estimator.estimate()


class TestBlockedDifferenceInMeansNumerics:
    """Numerical correctness tests."""

    def test_recovers_injected_ate_balanced_blocks(self) -> None:
        """With balanced blocks and large n, recovered ATE must be
        within statistical tolerance of the injected true ATE.
        """
        assignment = _make_blocked_assignment(
            n_per_block={"A": 250, "B": 250},
            seed=123,
            true_ate=1.0,
        )
        result = (
            BlockedDifferenceInMeans(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert abs(result.ate - 1.0) < 0.2

    def test_size_weighting_not_simple_mean(self) -> None:
        """Two unequal blocks with different per-block ATEs: aggregated
        ATE must equal the size-weighted sum, NOT the simple mean of
        per-block ATEs.

        Construction:
            Block A: n_A = 30, true ATE = 0.0
            Block B: n_B = 70, true ATE = 1.0

        Size-weighted:  (30/100)*0.0 + (70/100)*1.0 = 0.7
        Simple mean:    (0.0 + 1.0) / 2 = 0.5
        """
        # Build the DataFrame manually so each block has a deterministic
        # per-block ATE, free of stochastic noise.
        n_a, n_b = 30, 70
        # Block A: 15 treated with y=0, 15 control with y=0 -> ATE_A=0
        # Block B: 35 treated with y=1, 35 control with y=0 -> ATE_B=1
        df = pd.DataFrame(
            {
                "block": ["A"] * n_a + ["B"] * n_b,
                "treatment": (
                    [1] * 15 + [0] * 15 + [1] * 35 + [0] * 35
                ),
                "y": (
                    [0.0] * 15 + [0.0] * 15
                    + [1.0] * 35 + [0.0] * 35
                ),
            }
        )
        assignment = BlockedAssignment(
            data=df.copy(),
            treatment_col="treatment",
            design=None,
            block_col="block",
            block_sizes={"A": n_a, "B": n_b},
        )

        estimator = BlockedDifferenceInMeans(outcome_col="y").fit(assignment)
        # Per-block ATEs are exact by construction.
        assert estimator.block_ates_["A"] == pytest.approx(0.0, abs=1e-12)
        assert estimator.block_ates_["B"] == pytest.approx(1.0, abs=1e-12)

        # Aggregated ATE must be size-weighted, not simple mean.
        result = estimator.estimate()
        assert result.ate == pytest.approx(0.7, abs=1e-12)
        # Sanity check: not the simple mean.
        assert result.ate != pytest.approx(0.5, abs=1e-3)

    def test_collapses_to_dim_when_blocks_uniform(self) -> None:
        """When all blocks have the same size and the same proportion
        of treated, BlockedDifferenceInMeans.ate_ must coincide with
        DifferenceInMeans.ate_ on the same treatment vector.
        """
        df = pd.DataFrame(
            {
                "block": ["A"] * 20 + ["B"] * 20,
                "treatment": [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10,
                "y": np.arange(40, dtype=float),
            }
        )
        blocked = BlockedAssignment(
            data=df.copy(),
            treatment_col="treatment",
            design=None,
            block_col="block",
            block_sizes={"A": 20, "B": 20},
        )
        crd = CRDAssignment(
            data=df.drop(columns=["block"]).copy(),
            treatment_col="treatment",
            design=None,
        )
        bdim_ate = (
            BlockedDifferenceInMeans("y").fit(blocked).estimate().ate
        )
        dim_ate = DifferenceInMeans("y").fit(crd).estimate().ate
        assert bdim_ate == pytest.approx(dim_ate, abs=1e-10)