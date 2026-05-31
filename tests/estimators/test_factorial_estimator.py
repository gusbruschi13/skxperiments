"""Tests for skxperiments.estimators.factorial_estimator."""

import itertools

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
from skxperiments.estimators.factorial_estimator import FactorialEstimator


# --- Helpers ---


def _make_factorial_assignment(
    factors: list[str] | None = None,
    n_per_cell: int = 200,
    seed: int = 42,
    effect_map: dict[tuple[str, ...], float] | None = None,
) -> FactorialAssignment:
    """Build a FactorialAssignment whose outcome embeds known effects.

    For each subset in ``effect_map``, injects an additive contribution
    to the outcome of every unit. The injection per unit is
    ``magnitude * sign(cell, subset) / 2**(K-1)``, which compensates
    the estimator's normalization by ``1/2**(K-1)`` so that the
    recovered effect equals exactly ``magnitude``.

    With ``n_per_cell`` units per cell and no noise, recovery is exact
    (within floating-point tolerance).

    # Injeção aditiva de efeitos no outcome — aceitável apenas em
    # fixture de teste.
    """
    if factors is None:
        factors = ["A", "B"]

    K = len(factors)
    n_total = n_per_cell * (2**K)

    df = pd.DataFrame(
        {
            "x": np.zeros(n_total),
            "y": np.zeros(n_total),
        }
    )

    design = FactorialDesign(
        factors=factors, n_per_cell=n_per_cell, seed=seed
    )
    assignment = design.randomize(df)

    if effect_map:
        normalization = 2 ** (K - 1)
        for subset, magnitude in effect_map.items():
            subset_indices = [factors.index(f) for f in subset]
            for unit_iloc in range(n_total):
                cell_idx = int(assignment.data_.iloc[unit_iloc]["_cell"])
                sign = 1
                for j in subset_indices:
                    x_j = (cell_idx >> j) & 1
                    sign *= 2 * x_j - 1
                assignment.data_.iat[
                    unit_iloc,
                    assignment.data_.columns.get_loc("y"),
                ] += sign * magnitude / normalization

    return assignment


# --- Tests ---


class TestFactorialEstimatorCreation:
    """Tests for FactorialEstimator instantiation."""

    def test_basic_creation(self) -> None:
        """Should instantiate with outcome_col."""
        estimator = FactorialEstimator(outcome_col="y")
        assert estimator.outcome_col == "y"

    def test_outcome_col_stored_as_is(self) -> None:
        """outcome_col stored verbatim."""
        estimator = FactorialEstimator(outcome_col="metric")
        assert estimator.outcome_col == "metric"


class TestFactorialEstimatorFit:
    """Tests for FactorialEstimator.fit."""

    def test_fit_returns_self(self) -> None:
        """fit must return self for chaining."""
        assignment = _make_factorial_assignment()
        estimator = FactorialEstimator(outcome_col="y")
        returned = estimator.fit(assignment)
        assert returned is estimator

    def test_stores_assignment(self) -> None:
        """fit must store the assignment as assignment_."""
        assignment = _make_factorial_assignment()
        estimator = FactorialEstimator(outcome_col="y").fit(assignment)
        assert estimator.assignment_ is assignment

    def test_stores_effects(self) -> None:
        """fit must store effects_ as a dict."""
        assignment = _make_factorial_assignment()
        estimator = FactorialEstimator(outcome_col="y").fit(assignment)
        assert isinstance(estimator.effects_, dict)

    def test_effects_count_for_k2(self) -> None:
        """For K=2, effects_ must have exactly 3 keys."""
        assignment = _make_factorial_assignment(factors=["A", "B"])
        estimator = FactorialEstimator(outcome_col="y").fit(assignment)
        assert len(estimator.effects_) == 3

    def test_effects_count_for_k3(self) -> None:
        """For K=3, effects_ must have exactly 7 keys."""
        assignment = _make_factorial_assignment(
            factors=["A", "B", "C"], n_per_cell=50
        )
        estimator = FactorialEstimator(outcome_col="y").fit(assignment)
        assert len(estimator.effects_) == 7


class TestFactorialEstimatorEstimate:
    """Tests for FactorialEstimator.estimate."""

    def test_returns_results(self) -> None:
        """estimate must return a Results instance."""
        assignment = _make_factorial_assignment()
        result = (
            FactorialEstimator(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert isinstance(result, Results)

    def test_ate_is_none(self) -> None:
        """ate must be None in multi-effect mode."""
        assignment = _make_factorial_assignment()
        result = (
            FactorialEstimator(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.ate is None

    def test_effects_populated(self) -> None:
        """effects must be populated."""
        assignment = _make_factorial_assignment()
        result = (
            FactorialEstimator(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.effects is not None
        assert len(result.effects) == 3  # K=2

    def test_n_treated_and_control_are_none(self) -> None:
        """n_treated and n_control must be None for factorial."""
        assignment = _make_factorial_assignment()
        result = (
            FactorialEstimator(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.n_treated is None
        assert result.n_control is None

    def test_estimator_name(self) -> None:
        """estimator_name must be 'FactorialEstimator'."""
        assignment = _make_factorial_assignment()
        result = (
            FactorialEstimator(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.estimator_name == "FactorialEstimator"

    def test_design_name(self) -> None:
        """design_name must be 'FactorialDesign'."""
        assignment = _make_factorial_assignment()
        result = (
            FactorialEstimator(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.design_name == "FactorialDesign"

    def test_n_obs_correct(self) -> None:
        """n_obs must equal total units in the assignment."""
        assignment = _make_factorial_assignment(
            factors=["A", "B"], n_per_cell=100
        )
        result = (
            FactorialEstimator(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.n_obs == 400  # 100 * 2^2

    def test_inference_fields_are_none(self) -> None:
        """se, ci, p_value must be None (inference is Phase 4)."""
        assignment = _make_factorial_assignment()
        result = (
            FactorialEstimator(outcome_col="y")
            .fit(assignment)
            .estimate()
        )
        assert result.se is None
        assert result.ci is None
        assert result.p_value is None


class TestFactorialEstimatorValidation:
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

        estimator = FactorialEstimator(outcome_col="y")
        with pytest.raises(
            DesignEstimatorMismatch, match="FactorialAssignment"
        ):
            estimator.fit(crd_assignment)

    def test_rejects_blocked_assignment(self) -> None:
        """fit must reject BlockedAssignment with DesignEstimatorMismatch."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=40),
                "block": ["A"] * 20 + ["B"] * 20,
                "y": rng.normal(size=40),
            }
        )
        blocked_assignment = BlockedCRD(
            block_col="block", p=0.5, seed=0
        ).randomize(df)
        assert isinstance(blocked_assignment, BlockedAssignment)

        estimator = FactorialEstimator(outcome_col="y")
        with pytest.raises(
            DesignEstimatorMismatch, match="FactorialAssignment"
        ):
            estimator.fit(blocked_assignment)

    def test_missing_outcome_col_raises(self) -> None:
        """fit must raise InvalidDesignError if outcome_col absent."""
        assignment = _make_factorial_assignment()
        estimator = FactorialEstimator(outcome_col="missing")
        with pytest.raises(InvalidDesignError, match="not found"):
            estimator.fit(assignment)

    def test_non_numeric_outcome_raises(self) -> None:
        """fit must raise InvalidDesignError for non-numeric outcome."""
        # Build a factorial assignment, then overwrite y with strings.
        assignment = _make_factorial_assignment()
        assignment.data_["y"] = ["x"] * len(assignment.data_)

        estimator = FactorialEstimator(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="numeric"):
            estimator.fit(assignment)

    def test_outcome_with_nan_raises(self) -> None:
        """fit must raise InvalidDesignError for NaN in outcome."""
        assignment = _make_factorial_assignment()
        assignment.data_.iat[
            0, assignment.data_.columns.get_loc("y")
        ] = np.nan

        estimator = FactorialEstimator(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="NaN"):
            estimator.fit(assignment)

    def test_estimate_before_fit_raises(self) -> None:
        """estimate must raise NotFittedError if fit was not called."""
        estimator = FactorialEstimator(outcome_col="y")
        with pytest.raises(NotFittedError):
            estimator.estimate()


class TestFactorialEstimatorNumerics:
    """Tests for numerical correctness and conventions."""

    def test_exact_recovery_k2(self) -> None:
        """K=2 with three effects injected, no noise: exact recovery
        within abs=1e-10.
        """
        injected = {
            ("A",): 1.0,
            ("B",): 0.5,
            ("A", "B"): 0.2,
        }
        assignment = _make_factorial_assignment(
            factors=["A", "B"],
            n_per_cell=200,
            effect_map=injected,
        )
        estimator = FactorialEstimator(outcome_col="y").fit(assignment)

        for key, value in injected.items():
            assert estimator.effects_[key] == pytest.approx(
                value, abs=1e-10
            ), f"Mismatch at {key}: got {estimator.effects_[key]}"

    def test_orthogonality_only_a_injected(self) -> None:
        """When only A's main effect is injected, B and AB must be
        ~0 within abs=1e-10.
        """
        assignment = _make_factorial_assignment(
            factors=["A", "B"],
            n_per_cell=200,
            effect_map={("A",): 1.0},
        )
        estimator = FactorialEstimator(outcome_col="y").fit(assignment)

        assert estimator.effects_[("A",)] == pytest.approx(1.0, abs=1e-10)
        assert estimator.effects_[("B",)] == pytest.approx(0.0, abs=1e-10)
        assert estimator.effects_[("A", "B")] == pytest.approx(
            0.0, abs=1e-10
        )

    def test_main_effect_sign_convention(self) -> None:
        """Outcome y=0 except where A=1, where y=1.0.
        Effect of A must be 1.0; B and AB must be 0.
        """
        df = pd.DataFrame({"x": np.zeros(800), "y": np.zeros(800)})
        design = FactorialDesign(
            factors=["A", "B"], n_per_cell=200, seed=42
        )
        assignment = design.randomize(df)

        a1_mask = assignment.data_["A"] == 1
        assignment.data_.loc[a1_mask, "y"] = 1.0  # mutação aceitável em teste

        estimator = FactorialEstimator(outcome_col="y").fit(assignment)

        assert estimator.effects_[("A",)] == pytest.approx(1.0, abs=1e-10)
        assert estimator.effects_[("B",)] == pytest.approx(0.0, abs=1e-10)
        assert estimator.effects_[("A", "B")] == pytest.approx(
            0.0, abs=1e-10
        )

    def test_key_order_k2(self) -> None:
        """For K=2 with factors ['A', 'B'], the key order must be
        [('A',), ('B',), ('A', 'B')].
        """
        assignment = _make_factorial_assignment(factors=["A", "B"])
        estimator = FactorialEstimator(outcome_col="y").fit(assignment)
        assert list(estimator.effects_.keys()) == [
            ("A",),
            ("B",),
            ("A", "B"),
        ]

    def test_key_order_k3(self) -> None:
        """For K=3 with factors ['A', 'B', 'C'], the key order must be
        [('A',), ('B',), ('C',), ('A','B'), ('A','C'), ('B','C'),
         ('A','B','C')].
        """
        assignment = _make_factorial_assignment(
            factors=["A", "B", "C"], n_per_cell=50
        )
        estimator = FactorialEstimator(outcome_col="y").fit(assignment)
        expected = [
            ("A",),
            ("B",),
            ("C",),
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
            ("A", "B", "C"),
        ]
        assert list(estimator.effects_.keys()) == expected

    def test_keys_alphabetical_regardless_of_factor_order(self) -> None:
        """Effect keys must be alphabetical even when FactorialDesign
        was constructed with factors in non-alphabetical order.
        """
        # Build df with the right size for K=2.
        df = pd.DataFrame({"x": np.zeros(800), "y": np.zeros(800)})
        design = FactorialDesign(
            factors=["B", "A"], n_per_cell=200, seed=42
        )
        assignment = design.randomize(df)
        estimator = FactorialEstimator(outcome_col="y").fit(assignment)

        assert list(estimator.effects_.keys()) == [
            ("A",),
            ("B",),
            ("A", "B"),
        ]

    def test_to_dataframe_output(self) -> None:
        """to_dataframe must produce 2^K - 1 rows with effect/estimate
        columns matching effects_.
        """
        injected = {
            ("A",): 1.0,
            ("B",): 0.5,
            ("A", "B"): 0.2,
        }
        assignment = _make_factorial_assignment(
            factors=["A", "B"],
            n_per_cell=200,
            effect_map=injected,
        )
        estimator = FactorialEstimator(outcome_col="y").fit(assignment)
        result = estimator.estimate()
        df = result.to_dataframe()

        assert len(df) == 3  # 2^2 - 1
        assert "effect" in df.columns
        assert "estimate" in df.columns

        # Order of rows must match effects_ keys.
        assert list(df["effect"].values) == list(estimator.effects_.keys())

        # Estimates must match effects_ values.
        for _, row in df.iterrows():
            key = row["effect"]
            assert row["estimate"] == pytest.approx(
                estimator.effects_[key], abs=1e-10
            )