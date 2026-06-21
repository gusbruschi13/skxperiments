"""Tests for skxperiments.inference.neyman.NeymanCI."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
    FactorialAssignment,
)
from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InsufficientDataError,
    InvalidDesignError,
    NotFittedError,
)
from skxperiments.core.results import Results
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.crd import CRD
from skxperiments.design.rerandomized_crd import ReRandomizedCRD
from skxperiments.estimators.blocked_difference_in_means import (
    BlockedDifferenceInMeans,
)
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.estimators.lin_estimator import LinEstimator
from skxperiments.inference import NeymanCI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crd_assignment_with_ate(
    n: int,
    seed: int,
    true_ate: float = 0.0,
    p: float = 0.5,
) -> CRDAssignment:
    """Build a CRDAssignment with outcome Y = X + true_ate * T + noise.

    The treatment effect is constant across units, so the SATE equals
    ``true_ate`` exactly; this makes Neyman's (conservative) interval
    exact and lets coverage tests target the nominal level.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(0.0, 1.0, n)})

    design = CRD(p=p, seed=seed)
    assignment = design.randomize(df)

    t = assignment.data_[assignment.treatment_col_].values.astype(float)
    x = assignment.data_["x"].values

    df_with_y = assignment.data_.copy()
    df_with_y["y"] = x + true_ate * t + rng.normal(0.0, 1.0, n)

    return CRDAssignment(
        data=df_with_y,
        treatment_col=assignment.treatment_col_,
        design=design,
        seed=seed,
    )


def _make_blocked_assignment_with_ate(
    n_per_block: int,
    n_blocks: int,
    seed: int,
    true_ate: float = 0.0,
) -> BlockedAssignment:
    """Build a BlockedAssignment with outcome Y = X + true_ate * T + noise."""
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks

    df = pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n),
            "block": np.repeat(np.arange(n_blocks), n_per_block),
        }
    )

    design = BlockedCRD(block_col="block", p=0.5, seed=seed)
    assignment = design.randomize(df)

    t = assignment.data_[assignment.treatment_col_].values.astype(float)
    x = assignment.data_["x"].values

    df_with_y = assignment.data_.copy()
    df_with_y["y"] = x + true_ate * t + rng.normal(0.0, 1.0, n)

    return BlockedAssignment(
        data=df_with_y,
        treatment_col=assignment.treatment_col_,
        design=design,
        block_col=assignment.block_col_,
        block_sizes=assignment.block_sizes_,
        seed=seed,
    )


def _make_rerandomized_assignment_with_ate(
    n: int,
    seed: int,
    threshold: float,
    true_ate: float = 0.0,
) -> CRDAssignment:
    """Build a rerandomized CRDAssignment via ReRandomizedCRD."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(0.0, 1.0, n),
            "x2": rng.normal(0.0, 1.0, n),
        }
    )

    design = ReRandomizedCRD(
        covariates=["x1", "x2"],
        threshold=threshold,
        p=0.5,
        seed=seed,
        max_attempts=10_000,
    )
    assignment = design.randomize(df)

    t = assignment.data_[assignment.treatment_col_].values.astype(float)
    x1 = assignment.data_["x1"].values
    x2 = assignment.data_["x2"].values

    df_with_y = assignment.data_.copy()
    df_with_y["y"] = x1 + x2 + true_ate * t + rng.normal(0.0, 1.0, n)

    return CRDAssignment(
        data=df_with_y,
        treatment_col=assignment.treatment_col_,
        design=design,
        seed=seed,
        rerandomization_metadata=assignment.rerandomization_metadata,
    )


def _fixed_crd_assignment(
    treatment: list[int],
    y: list[float],
) -> CRDAssignment:
    """Build a deterministic CRDAssignment from explicit treatment/outcome.

    Used for hand-checkable variance tests. ``design=None`` is acceptable:
    NeymanCI never calls ``draw()`` and the estimator tolerates a missing
    design reference (``design_name`` resolves to None).
    """
    df = pd.DataFrame({"treatment": treatment, "y": y})
    return CRDAssignment(data=df, treatment_col="treatment", design=None)


def _fixed_blocked_assignment(
    block: list[int],
    treatment: list[int],
    y: list[float],
) -> BlockedAssignment:
    """Build a deterministic BlockedAssignment for hand-checkable tests."""
    df = pd.DataFrame({"block": block, "treatment": treatment, "y": y})
    block_sizes = df.groupby("block").size().to_dict()
    return BlockedAssignment(
        data=df,
        treatment_col="treatment",
        design=None,
        block_col="block",
        block_sizes=block_sizes,
    )


def _make_factorial_assignment(n: int, seed: int) -> FactorialAssignment:
    """Build a FactorialAssignment directly (K=2, little-endian cells)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    cell = a + 2 * b
    df = pd.DataFrame(
        {"A": a, "B": b, "_cell": cell, "y": rng.normal(0.0, 1.0, n)}
    )
    cell_sizes = {int(c): int((cell == c).sum()) for c in range(4)}
    return FactorialAssignment(
        data=df,
        design=None,
        factor_cols=["A", "B"],
        cell_sizes=cell_sizes,
        seed=seed,
    )


class _MultiEffectDIM(DifferenceInMeans):
    """DifferenceInMeans subclass returning a multi-effect Results.

    Passes NeymanCI's estimator whitelist (it *is* a DifferenceInMeans)
    but produces ``Results.effects`` instead of ``Results.ate``, to
    exercise the scalar-only guard in ``NeymanCI.fit``.
    """

    def estimate(self) -> Results:
        return Results(
            effects={("A",): 1.0},
            estimator_name=type(self).__name__,
        )


class _SuperpopDIM(DifferenceInMeans):
    """DifferenceInMeans subclass reporting a superpopulation inference_mode.

    Passes the whitelist but injects ``inference_mode='superpopulation'``
    into ``Results.extra`` to exercise NeymanCI's finite-population guard.
    """

    def estimate(self) -> Results:
        base = super().estimate()
        base.extra = {"inference_mode": "superpopulation"}
        return base


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


class TestNeymanCICreation:
    """Tests for NeymanCI.__init__ validations."""

    def test_basic_creation(self) -> None:
        """Default alpha is 0.05; estimator is stored."""
        dim = DifferenceInMeans(outcome_col="y")
        ci = NeymanCI(estimator=dim)
        assert ci.alpha == 0.05
        assert ci.estimator is dim

    def test_custom_alpha(self) -> None:
        """A custom alpha in (0, 1) is accepted."""
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"), alpha=0.10)
        assert ci.alpha == 0.10

    def test_rejects_non_whitelisted_estimator(self) -> None:
        """An estimator outside the whitelist raises DesignEstimatorMismatch."""
        lin = LinEstimator(outcome_col="y", covariates=["x"])
        with pytest.raises(DesignEstimatorMismatch, match="NeymanCI"):
            NeymanCI(estimator=lin)

    def test_rejects_non_estimator(self) -> None:
        """A non-estimator object is rejected at construction."""
        with pytest.raises(DesignEstimatorMismatch):
            NeymanCI(estimator="not an estimator")  # type: ignore[arg-type]

    def test_rejects_alpha_out_of_range(self) -> None:
        """alpha must be strictly in (0, 1)."""
        dim = DifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="alpha"):
            NeymanCI(estimator=dim, alpha=0.0)
        with pytest.raises(InvalidDesignError, match="alpha"):
            NeymanCI(estimator=dim, alpha=1.0)

    def test_rejects_alpha_non_float(self) -> None:
        """alpha must be numeric."""
        dim = DifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="alpha"):
            NeymanCI(estimator=dim, alpha="0.05")  # type: ignore[arg-type]

    def test_rejects_alpha_bool(self) -> None:
        """alpha=True is rejected (bool is a subclass of int)."""
        dim = DifferenceInMeans(outcome_col="y")
        with pytest.raises(InvalidDesignError, match="alpha"):
            NeymanCI(estimator=dim, alpha=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Validation in fit/estimate
# ---------------------------------------------------------------------------


class TestNeymanCIValidation:
    """Tests for NeymanCI.fit type checks and guards."""

    def test_rejects_factorial_assignment(self) -> None:
        """fit(FactorialAssignment) raises DesignEstimatorMismatch."""
        assignment = _make_factorial_assignment(n=40, seed=0)
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        with pytest.raises(DesignEstimatorMismatch):
            ci.fit(assignment)

    def test_rejects_multi_effect_estimator(self) -> None:
        """An estimator producing Results.effects (no ate) is rejected."""
        assignment = _make_crd_assignment_with_ate(n=20, seed=0)
        ci = NeymanCI(estimator=_MultiEffectDIM(outcome_col="y"))
        with pytest.raises(
            InvalidDesignError, match="scalar|multi-effect|Results.ate"
        ):
            ci.fit(assignment)

    def test_rejects_superpopulation_inference_mode(self) -> None:
        """inference_mode='superpopulation' is rejected, redirecting to bootstrap."""
        assignment = _make_crd_assignment_with_ate(n=20, seed=0)
        ci = NeymanCI(estimator=_SuperpopDIM(outcome_col="y"))
        with pytest.raises(InvalidDesignError, match="BootstrapCI"):
            ci.fit(assignment)

    def test_estimate_before_fit_raises_not_fitted(self) -> None:
        """estimate() before fit() raises NotFittedError."""
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        with pytest.raises(NotFittedError):
            ci.estimate()

    def test_insufficient_data_crd_arm(self) -> None:
        """A CRD arm with fewer than 2 units raises InsufficientDataError."""
        assignment = _fixed_crd_assignment(
            treatment=[1, 0, 0, 0], y=[1.0, 2.0, 3.0, 4.0]
        )
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        with pytest.raises(InsufficientDataError, match="treated arm"):
            ci.fit(assignment)

    def test_insufficient_data_blocked_arm(self) -> None:
        """A block arm with fewer than 2 units raises InsufficientDataError."""
        assignment = _fixed_blocked_assignment(
            block=[0, 0, 1, 1],
            treatment=[1, 0, 1, 0],
            y=[1.0, 2.0, 3.0, 4.0],
        )
        ci = NeymanCI(estimator=BlockedDifferenceInMeans(outcome_col="y"))
        with pytest.raises(InsufficientDataError, match="block"):
            ci.fit(assignment)


# ---------------------------------------------------------------------------
# 3. Fit behavior
# ---------------------------------------------------------------------------


class TestNeymanCIFit:
    """Tests for NeymanCI.fit behavior and fitted attributes."""

    def test_fit_returns_self(self) -> None:
        """fit returns self."""
        assignment = _make_crd_assignment_with_ate(n=40, seed=0)
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        assert ci.fit(assignment) is ci

    def test_fitted_attributes_exist(self) -> None:
        """After fit, assignment_ and variance_ are populated."""
        assignment = _make_crd_assignment_with_ate(n=40, seed=0)
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        ci.fit(assignment)

        assert isinstance(ci.assignment_, CRDAssignment)
        assert isinstance(ci.variance_, float)
        assert ci.variance_ > 0.0

    def test_ate_matches_difference_in_means(self) -> None:
        """The point estimate equals DIM.fit(assignment).estimate().ate."""
        assignment = _make_crd_assignment_with_ate(
            n=60, seed=0, true_ate=1.5
        )
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        result = ci.fit(assignment).estimate()

        dim = DifferenceInMeans(outcome_col="y")
        expected = dim.fit(assignment).estimate().ate

        assert result.ate == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 4. Estimate behavior (CRD), hand-checkable
# ---------------------------------------------------------------------------


class TestNeymanCIEstimateCRD:
    """Tests for NeymanCI.estimate output under CRD."""

    def _hand_assignment(self) -> CRDAssignment:
        # y_t = [2, 4, 6] -> var(ddof=1)=4, /3 -> 4/3
        # y_c = [1, 2, 3] -> var(ddof=1)=1, /3 -> 1/3
        # V = 5/3 ; ATE = 4 - 2 = 2
        return _fixed_crd_assignment(
            treatment=[1, 1, 1, 0, 0, 0],
            y=[2.0, 4.0, 6.0, 1.0, 2.0, 3.0],
        )

    def test_variance_hand_value(self) -> None:
        """Neyman variance matches the hand computation 5/3."""
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        ci.fit(self._hand_assignment())
        assert ci.variance_ == pytest.approx(5.0 / 3.0)

    def test_ate_and_se(self) -> None:
        """ATE is 2.0 and SE is sqrt(5/3)."""
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        result = ci.fit(self._hand_assignment()).estimate()
        assert result.ate == pytest.approx(2.0)
        assert result.se == pytest.approx(np.sqrt(5.0 / 3.0))

    def test_ci_symmetric_around_ate(self) -> None:
        """The Wald CI is symmetric around the ATE with the right width."""
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"), alpha=0.05)
        result = ci.fit(self._hand_assignment()).estimate()

        se = np.sqrt(5.0 / 3.0)
        z = stats.norm.ppf(0.975)
        assert result.ci[0] == pytest.approx(2.0 - z * se)
        assert result.ci[1] == pytest.approx(2.0 + z * se)
        midpoint = (result.ci[0] + result.ci[1]) / 2.0
        assert midpoint == pytest.approx(result.ate)

    def test_p_value_wald(self) -> None:
        """The two-sided Wald p-value matches the normal-approximation value."""
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        result = ci.fit(self._hand_assignment()).estimate()

        se = np.sqrt(5.0 / 3.0)
        expected_p = 2.0 * (1.0 - stats.norm.cdf(2.0 / se))
        assert result.p_value == pytest.approx(expected_p)

    def test_results_metadata(self) -> None:
        """inference_name, variance_type and extra keys are correct."""
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        result = ci.fit(self._hand_assignment()).estimate()

        assert result.inference_name == "NeymanCI"
        assert result.estimator_name == "DifferenceInMeans"
        assert result.extra is not None
        assert result.extra["variance_type"] == "neyman"
        assert result.extra["inference_mode"] == "finite_population"

    def test_significant_when_ci_excludes_zero(self) -> None:
        """A strong effect yields p < alpha and a CI excluding zero."""
        assignment = _fixed_crd_assignment(
            treatment=[1, 1, 1, 1, 0, 0, 0, 0],
            y=[10.0, 11.0, 12.0, 13.0, 0.0, 1.0, 2.0, 3.0],
        )
        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        result = ci.fit(assignment).estimate()

        assert result.p_value < 0.05
        assert result.ci[0] > 0.0
        assert result.is_significant()


# ---------------------------------------------------------------------------
# 5. Blocked, hand-checkable
# ---------------------------------------------------------------------------


class TestNeymanCIBlocked:
    """Tests for NeymanCI with BlockedAssignment."""

    def _hand_assignment(self) -> BlockedAssignment:
        # Block 0: y_t=[3,5] (var 2 -> /2 = 1), y_c=[1,3] (var 2 -> /2 = 1)
        #          V_0 = 2 ; ATE_0 = 4 - 2 = 2
        # Block 1: y_t=[10,14] (var 8 -> /2 = 4), y_c=[6,8] (var 2 -> /2 = 1)
        #          V_1 = 5 ; ATE_1 = 12 - 7 = 5
        # weight = (4/8)^2 = 0.25 each ; V = 0.25*2 + 0.25*5 = 1.75
        # ATE = 0.5*2 + 0.5*5 = 3.5
        return _fixed_blocked_assignment(
            block=[0, 0, 0, 0, 1, 1, 1, 1],
            treatment=[1, 1, 0, 0, 1, 1, 0, 0],
            y=[3.0, 5.0, 1.0, 3.0, 10.0, 14.0, 6.0, 8.0],
        )

    def test_accepts_blocked_assignment(self) -> None:
        """fit() accepts a BlockedAssignment from BlockedCRD."""
        assignment = _make_blocked_assignment_with_ate(
            n_per_block=10, n_blocks=4, seed=0, true_ate=0.5
        )
        ci = NeymanCI(estimator=BlockedDifferenceInMeans(outcome_col="y"))
        ci.fit(assignment)
        assert isinstance(ci.variance_, float)
        assert ci.variance_ > 0.0

    def test_stratified_variance_hand_value(self) -> None:
        """The stratified variance matches the hand computation 1.75."""
        ci = NeymanCI(estimator=BlockedDifferenceInMeans(outcome_col="y"))
        ci.fit(self._hand_assignment())
        assert ci.variance_ == pytest.approx(1.75)

    def test_ate_hand_value(self) -> None:
        """The size-weighted ATE matches the hand computation 3.5."""
        ci = NeymanCI(estimator=BlockedDifferenceInMeans(outcome_col="y"))
        result = ci.fit(self._hand_assignment()).estimate()
        assert result.ate == pytest.approx(3.5)

    def test_variance_type_metadata(self) -> None:
        """variance_type is 'neyman_stratified' for blocked designs."""
        ci = NeymanCI(estimator=BlockedDifferenceInMeans(outcome_col="y"))
        result = ci.fit(self._hand_assignment()).estimate()
        assert result.extra["variance_type"] == "neyman_stratified"


# ---------------------------------------------------------------------------
# 6. Rerandomization
# ---------------------------------------------------------------------------


class TestNeymanCIRerandomization:
    """Tests for NeymanCI with ReRandomizedCRD assignments."""

    def test_accepts_rerandomized_crd(self) -> None:
        """fit() accepts a CRDAssignment from ReRandomizedCRD."""
        threshold = float(stats.chi2.ppf(0.05, df=2))
        assignment = _make_rerandomized_assignment_with_ate(
            n=80, seed=0, threshold=threshold, true_ate=0.5
        )
        assert assignment.rerandomization_metadata is not None

        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        result = ci.fit(assignment).estimate()
        assert np.isfinite(result.se)
        assert result.extra["variance_type"] == "neyman"


# ---------------------------------------------------------------------------
# 7. Immutability
# ---------------------------------------------------------------------------


class TestNeymanCIImmutability:
    """fit must not mutate the assignment's data."""

    def test_fit_does_not_mutate_assignment(self) -> None:
        """assignment.data_ is unchanged after fit/estimate."""
        assignment = _make_crd_assignment_with_ate(
            n=40, seed=0, true_ate=1.0
        )
        before = assignment.data_.copy()

        ci = NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))
        ci.fit(assignment).estimate()

        pd.testing.assert_frame_equal(assignment.data_, before)


# ---------------------------------------------------------------------------
# 8. Empirical coverage (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNeymanCICoverage:
    """Monte Carlo coverage tests for NeymanCI (slow).

    With a constant treatment effect the Neyman interval is exact, so
    coverage should be close to the nominal 1 - alpha. The assertion uses
    a generous lower bound because Neyman is conservative (coverage may
    exceed nominal) and to absorb Monte Carlo noise.
    """

    def test_crd_coverage_near_nominal(self) -> None:
        """A 95% NeymanCI covers the true ATE in ~95% of CRD replications."""
        n_reps = 400
        true_ate = 1.0
        covered = 0
        for i in range(n_reps):
            assignment = _make_crd_assignment_with_ate(
                n=100, seed=i, true_ate=true_ate
            )
            ci = NeymanCI(
                estimator=DifferenceInMeans(outcome_col="y"), alpha=0.05
            )
            result = ci.fit(assignment).estimate()
            if result.ci[0] <= true_ate <= result.ci[1]:
                covered += 1

        coverage = covered / n_reps
        assert coverage >= 0.90, (
            f"Empirical coverage {coverage:.3f} below tolerance for CRD."
        )

    def test_blocked_coverage_near_nominal(self) -> None:
        """A 95% NeymanCI covers the true ATE in ~95% of blocked replications."""
        n_reps = 300
        true_ate = 1.0
        covered = 0
        for i in range(n_reps):
            assignment = _make_blocked_assignment_with_ate(
                n_per_block=20, n_blocks=5, seed=i, true_ate=true_ate
            )
            ci = NeymanCI(
                estimator=BlockedDifferenceInMeans(outcome_col="y"),
                alpha=0.05,
            )
            result = ci.fit(assignment).estimate()
            if result.ci[0] <= true_ate <= result.ci[1]:
                covered += 1

        coverage = covered / n_reps
        assert coverage >= 0.90, (
            f"Empirical coverage {coverage:.3f} below tolerance for blocked."
        )
