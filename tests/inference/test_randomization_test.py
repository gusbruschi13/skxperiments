"""Tests for skxperiments.inference.randomization_test.RandomizationTest."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
    FactorialAssignment,
)
from skxperiments.core.base import BaseEstimator
from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InvalidDesignError,
    NotFittedError,
)
from skxperiments.core.results import Results
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.crd import CRD
from skxperiments.design.factorial import FactorialDesign
from skxperiments.design.rerandomized_crd import ReRandomizedCRD
from skxperiments.estimators.blocked_difference_in_means import (
    BlockedDifferenceInMeans,
)
from skxperiments.estimators.cuped import CUPED
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.estimators.lin_estimator import LinEstimator
from skxperiments.inference import RandomizationTest


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

    Steps:
    1. Generate covariate df = {x: ~N(0,1)}.
    2. Run CRD(p=p, seed=seed).randomize(df) to get the treatment vector.
    3. Build Y using the actual treatment vector from the randomized data.
    4. Return a new CRDAssignment with the augmented DataFrame (containing
       both x, treatment, and y) and the same design reference.

    The new Assignment is constructed via the public constructor; the
    DataFrame passed in already includes the treatment column produced
    by randomize(), so binary-validation succeeds.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(0.0, 1.0, n)})

    design = CRD(p=p, seed=seed)
    assignment = design.randomize(df)

    # Build outcome using the actual treatment vector.
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
    """Build a rerandomized CRDAssignment via ReRandomizedCRD.

    Two covariates x1, x2 are used both as rerandomization covariates
    and to construct the outcome Y = x1 + x2 + true_ate * T + noise.
    """
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


def _make_cuped_assignment(
    n: int,
    seed: int,
    true_ate: float = 0.0,
    correlation: float = 0.7,
) -> CRDAssignment:
    """Build a CRDAssignment with both y and y_pre columns.

    y_pre is correlated with the baseline of y (controlled by
    `correlation`), enabling a meaningful CUPED adjustment.
    """
    rng = np.random.default_rng(seed)
    y_pre = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)
    y_baseline = correlation * y_pre + np.sqrt(1.0 - correlation**2) * noise

    df = pd.DataFrame({"y_pre": y_pre, "_y_base": y_baseline})

    design = CRD(p=0.5, seed=seed)
    assignment = design.randomize(df)

    t = assignment.data_[assignment.treatment_col_].values.astype(float)
    df_with_y = assignment.data_.copy()
    df_with_y["y"] = df_with_y["_y_base"].values + true_ate * t
    df_with_y = df_with_y.drop(columns=["_y_base"])

    return CRDAssignment(
        data=df_with_y,
        treatment_col=assignment.treatment_col_,
        design=design,
        seed=seed,
    )


class _MultiEffectMockEstimator(BaseEstimator):
    """Mock estimator that returns Results in multi-effect mode.

    Used to test RandomizationTest's rejection of estimators that do
    not produce a scalar ATE. Decoupled from FactorialEstimator to
    isolate the validation of `Results.ate is None` from
    FactorialAssignment specifics.
    """

    def __init__(self, outcome_col: str = "y") -> None:
        self.outcome_col = outcome_col

    def fit(self, assignment) -> "_MultiEffectMockEstimator":
        self.assignment_ = assignment
        return self

    def estimate(self) -> Results:
        return Results(
            effects={("A",): 1.0, ("B",): 0.5},
            estimator_name="MockMultiEffect",
        )


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


class TestRandomizationTestCreation:
    """Tests for RandomizationTest.__init__ validations."""

    def test_basic_creation(self) -> None:
        """Defaults: n_permutations=10_000, alternative='two-sided', seed=None."""
        rt = RandomizationTest(estimator=DifferenceInMeans(outcome_col="y"))
        assert rt.n_permutations == 10_000
        assert rt.alternative == "two-sided"
        assert rt.seed is None

    def test_invalid_estimator_raises(self) -> None:
        """estimator must be a BaseEstimator instance."""
        with pytest.raises(InvalidDesignError, match="BaseEstimator"):
            RandomizationTest(estimator="not an estimator")  # type: ignore[arg-type]

    def test_invalid_n_permutations_negative_raises(self) -> None:
        """n_permutations < 0 is rejected."""
        with pytest.raises(InvalidDesignError):
            RandomizationTest(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_permutations=-1,
            )

    def test_invalid_n_permutations_zero_raises(self) -> None:
        """n_permutations == 0 is rejected."""
        with pytest.raises(InvalidDesignError):
            RandomizationTest(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_permutations=0,
            )

    def test_invalid_n_permutations_non_int_raises(self) -> None:
        """n_permutations must be int (float rejected)."""
        with pytest.raises(InvalidDesignError):
            RandomizationTest(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_permutations=100.5,  # type: ignore[arg-type]
            )

    def test_invalid_n_permutations_bool_raises(self) -> None:
        """n_permutations=True is rejected (bool is subclass of int)."""
        with pytest.raises(InvalidDesignError):
            RandomizationTest(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_permutations=True,  # type: ignore[arg-type]
            )

    def test_invalid_alternative_raises(self) -> None:
        """alternative must be one of the supported options."""
        with pytest.raises(InvalidDesignError, match="alternative"):
            RandomizationTest(
                estimator=DifferenceInMeans(outcome_col="y"),
                alternative="invalid",
            )


# ---------------------------------------------------------------------------
# 2. Validation in fit/estimate
# ---------------------------------------------------------------------------


class TestRandomizationTestValidation:
    """Tests for RandomizationTest.fit type checks and estimate state."""

    def test_rejects_factorial_assignment(self) -> None:
        """fit(FactorialAssignment) raises DesignEstimatorMismatch."""
        # Inline factorial setup: K=2 factors A, B; n_per_cell=10.
        rng = np.random.default_rng(seed=0)
        n = 40
        df = pd.DataFrame({"x": rng.normal(0.0, 1.0, n)})
        design = FactorialDesign(factor_cols=["A", "B"], n_per_cell=10, seed=0)
        factorial_assignment = design.randomize(df)
        assert isinstance(factorial_assignment, FactorialAssignment)

        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=100,
            seed=0,
        )
        with pytest.raises(DesignEstimatorMismatch):
            rt.fit(factorial_assignment)

    def test_rejects_multi_effect_estimator(self) -> None:
        """Estimator producing Results.effects (no ate) is rejected."""
        assignment = _make_crd_assignment_with_ate(n=20, seed=0, true_ate=0.0)
        rt = RandomizationTest(
            estimator=_MultiEffectMockEstimator(outcome_col="y"),
            n_permutations=100,
            seed=0,
        )
        with pytest.raises(InvalidDesignError, match="scalar|multi-effect|Results.ate"):
            rt.fit(assignment)

    def test_estimate_before_fit_raises_not_fitted(self) -> None:
        """estimate() before fit() raises NotFittedError."""
        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=100,
        )
        with pytest.raises(NotFittedError):
            rt.estimate()


# ---------------------------------------------------------------------------
# 3. Fit behavior
# ---------------------------------------------------------------------------


class TestRandomizationTestFit:
    """Tests for RandomizationTest.fit behavior and fitted attributes."""

    def test_fit_returns_self(self) -> None:
        """fit returns self."""
        assignment = _make_crd_assignment_with_ate(n=30, seed=0)
        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=100,
            seed=0,
        )
        assert rt.fit(assignment) is rt

    def test_fitted_attributes_exist(self) -> None:
        """After fit, all fitted attributes are populated with correct types."""
        assignment = _make_crd_assignment_with_ate(n=30, seed=0)
        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=100,
            seed=0,
        )
        rt.fit(assignment)

        assert isinstance(rt.assignment_, CRDAssignment)
        assert isinstance(rt.observed_statistic_, float)
        assert isinstance(rt.null_distribution_, np.ndarray)
        assert isinstance(rt.p_value_, float)

    def test_null_distribution_length(self) -> None:
        """null_distribution_ has length n_permutations."""
        assignment = _make_crd_assignment_with_ate(n=30, seed=0)
        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=250,
            seed=0,
        )
        rt.fit(assignment)
        assert len(rt.null_distribution_) == 250

    def test_observed_statistic_matches_dim(self) -> None:
        """observed_statistic_ equals DIM.fit(assignment).estimate().ate."""
        assignment = _make_crd_assignment_with_ate(
            n=50, seed=0, true_ate=1.5
        )
        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=100,
            seed=0,
        )
        rt.fit(assignment)

        dim = DifferenceInMeans(outcome_col="y")
        expected = dim.fit(assignment).estimate().ate

        assert rt.observed_statistic_ == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 4. Estimate behavior
# ---------------------------------------------------------------------------


class TestRandomizationTestEstimate:
    """Tests for RandomizationTest.estimate output."""

    def _fitted_rt(self) -> RandomizationTest:
        assignment = _make_crd_assignment_with_ate(
            n=50, seed=0, true_ate=1.0
        )
        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=200,
            seed=0,
        )
        rt.fit(assignment)
        return rt

    def test_estimate_returns_results(self) -> None:
        """estimate() returns a Results instance."""
        rt = self._fitted_rt()
        result = rt.estimate()
        assert isinstance(result, Results)

    def test_results_inference_name(self) -> None:
        """inference_name == 'RandomizationTest'."""
        rt = self._fitted_rt()
        result = rt.estimate()
        assert result.inference_name == "RandomizationTest"

    def test_results_p_value_in_range(self) -> None:
        """p_value is in (0, 1] (Phipson & Smyth guarantees > 0)."""
        rt = self._fitted_rt()
        result = rt.estimate()
        assert 0.0 < result.p_value <= 1.0

    def test_results_extra_keys(self) -> None:
        """extra contains n_permutations, null_distribution, alternative."""
        rt = self._fitted_rt()
        result = rt.estimate()
        assert result.extra is not None
        assert result.extra["n_permutations"] == 200
        assert isinstance(result.extra["null_distribution"], np.ndarray)
        assert len(result.extra["null_distribution"]) == 200
        assert result.extra["alternative"] == "two-sided"

    def test_results_se_and_ci_are_none(self) -> None:
        """se and ci are None — RandomizationTest produces only p-value."""
        rt = self._fitted_rt()
        result = rt.estimate()
        assert result.se is None
        assert result.ci is None

    def test_results_ate_equals_observed_statistic(self) -> None:
        """Results.ate equals observed_statistic_, not last permutation."""
        rt = self._fitted_rt()
        result = rt.estimate()
        assert result.ate == rt.observed_statistic_


# ---------------------------------------------------------------------------
# 5. Numerics (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRandomizationTestNumerics:
    """Statistical property tests for RandomizationTest (slow)."""

    def test_pvalue_uniform_under_true_null(self) -> None:
        """Under true null (true_ate=0), p-values are ~Uniform(0,1).

        200 reps, n_permutations=500, n=50. KS test against uniform
        with tolerance p > 0.01.
        """
        n_reps = 200
        p_values = np.empty(n_reps, dtype=float)
        for i in range(n_reps):
            assignment = _make_crd_assignment_with_ate(
                n=50, seed=i, true_ate=0.0
            )
            rt = RandomizationTest(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_permutations=500,
                seed=10_000 + i,
            )
            rt.fit(assignment)
            p_values[i] = rt.p_value_

        ks_stat, ks_pvalue = stats.kstest(p_values, "uniform")
        assert ks_pvalue > 0.01, (
            f"KS test against uniform rejected: ks_stat={ks_stat:.4f}, "
            f"ks_pvalue={ks_pvalue:.4f}"
        )

    def test_rejects_under_strong_alternative(self) -> None:
        """Under strong alternative (true_ate=2.0, n=200), all seeds reject.

        With effect of 2 SDs and n=200, theoretical power >99%.
        """
        rejections = 0
        for seed in range(5):
            assignment = _make_crd_assignment_with_ate(
                n=200, seed=seed, true_ate=2.0
            )
            rt = RandomizationTest(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_permutations=1000,
                seed=10_000 + seed,
            )
            rt.fit(assignment)
            if rt.p_value_ < 0.05:
                rejections += 1

        assert rejections == 5, (
            f"Expected all 5 seeds to reject under strong alternative, "
            f"got {rejections}/5."
        )


# ---------------------------------------------------------------------------
# 6. Reproducibility
# ---------------------------------------------------------------------------


class TestRandomizationTestReproducibility:
    """Tests for deterministic behavior given a fixed seed."""

    def test_same_seed_produces_same_null_distribution(self) -> None:
        """Two RandomizationTests with same seed produce identical null."""
        assignment = _make_crd_assignment_with_ate(n=40, seed=0)

        rt1 = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=200,
            seed=42,
        )
        rt1.fit(assignment)

        rt2 = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=200,
            seed=42,
        )
        rt2.fit(assignment)

        assert np.array_equal(rt1.null_distribution_, rt2.null_distribution_)
        assert rt1.p_value_ == rt2.p_value_


# ---------------------------------------------------------------------------
# 7. Rerandomization
# ---------------------------------------------------------------------------


class TestRandomizationTestRerandomization:
    """Tests for RandomizationTest with ReRandomizedCRD assignments."""

    def test_accepts_rerandomized_crd(self) -> None:
        """fit() accepts a CRDAssignment from ReRandomizedCRD."""
        threshold = float(stats.chi2.ppf(0.05, df=2))
        assignment = _make_rerandomized_assignment_with_ate(
            n=80, seed=0, threshold=threshold, true_ate=0.5
        )
        assert assignment.rerandomization_metadata is not None

        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=100,
            seed=0,
        )
        rt.fit(assignment)
        assert isinstance(rt.p_value_, float)

    def test_null_distribution_respects_mahalanobis(self) -> None:
        """Each draw under rerandomization respects Mahalanobis threshold.

        Reproduces the seed loop that RandomizationTest would use, calls
        assignment.draw(seed=...) manually, and verifies that each drawn
        Assignment has rerandomization_metadata populated and that the
        Mahalanobis distance computed from its means is <= threshold.

        Note: scaling_factor is fixed by design (n_treated is constant
        in CRD), so it is read once from rerandomization_metadata before
        the loop and reused for every draw.
        """
        threshold = float(stats.chi2.ppf(0.05, df=2))
        assignment = _make_rerandomized_assignment_with_ate(
            n=80, seed=0, threshold=threshold, true_ate=0.0
        )

        meta = assignment.rerandomization_metadata
        assert meta is not None
        cov_matrix = meta["cov_matrix"]
        scaling_factor = meta["scaling_factor"]
        cached_threshold = meta["threshold"]
        covariates = meta["covariates"]

        # Inverse of (scaling_factor * cov_matrix) — fixed across draws.
        inv_scaled_cov = np.linalg.inv(scaling_factor * cov_matrix)

        # Reproduce the seed pre-generation that RandomizationTest uses.
        rng = np.random.default_rng(seed=42)
        perm_seeds = rng.integers(0, 2**32, size=20)

        for ps in perm_seeds:
            drawn = assignment.draw(seed=int(ps))
            assert drawn.rerandomization_metadata is not None

            cov_values = drawn.data_[covariates].values
            t_mask = drawn.data_[drawn.treatment_col_].values == 1
            mean_t = cov_values[t_mask].mean(axis=0)
            mean_c = cov_values[~t_mask].mean(axis=0)
            d = mean_t - mean_c
            distance = float(d @ inv_scaled_cov @ d)

            assert distance <= cached_threshold + 1e-9, (
                f"Mahalanobis distance {distance:.6f} exceeded threshold "
                f"{cached_threshold:.6f} for seed {ps}."
            )

    @pytest.mark.slow
    def test_rerandomization_reduces_variance(self) -> None:
        """Mean variance of null_distribution under rerandomization is lower.

        Compare CRD vs ReRandomizedCRD with a tight threshold over 5 seeds;
        averaged variance under rerandomization should be smaller.

        Threshold: chi2.ppf(0.001, df=2) accepts ~0.1% of CRD
        randomizations, ensuring strong covariate balance and a clear
        variance reduction signal.
        """
        threshold = float(stats.chi2.ppf(0.001, df=2))
        n_seeds = 5
        var_crd = np.empty(n_seeds, dtype=float)
        var_rerand = np.empty(n_seeds, dtype=float)

        for i in range(n_seeds):
            # Plain CRD assignment with the same x1, x2 covariates.
            rng = np.random.default_rng(seed=i)
            n = 80
            df = pd.DataFrame(
                {
                    "x1": rng.normal(0.0, 1.0, n),
                    "x2": rng.normal(0.0, 1.0, n),
                }
            )
            crd_design = CRD(p=0.5, seed=i)
            crd_assignment = crd_design.randomize(df)
            t = crd_assignment.data_[
                crd_assignment.treatment_col_
            ].values.astype(float)
            x1 = crd_assignment.data_["x1"].values
            x2 = crd_assignment.data_["x2"].values
            df_with_y = crd_assignment.data_.copy()
            df_with_y["y"] = x1 + x2 + rng.normal(0.0, 1.0, n)
            crd_assignment_y = CRDAssignment(
                data=df_with_y,
                treatment_col=crd_assignment.treatment_col_,
                design=crd_design,
                seed=i,
            )

            rt_crd = RandomizationTest(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_permutations=2000,
                seed=10_000 + i,
            )
            rt_crd.fit(crd_assignment_y)
            var_crd[i] = float(np.var(rt_crd.null_distribution_))

            # Rerandomized assignment with tight threshold.
            rerand_assignment = _make_rerandomized_assignment_with_ate(
                n=n, seed=i, threshold=threshold, true_ate=0.0
            )
            rt_rerand = RandomizationTest(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_permutations=2000,
                seed=10_000 + i,
            )
            rt_rerand.fit(rerand_assignment)
            var_rerand[i] = float(np.var(rt_rerand.null_distribution_))

        assert var_rerand.mean() < var_crd.mean(), (
            f"Expected mean variance under rerandomization "
            f"({var_rerand.mean():.6f}) to be less than under CRD "
            f"({var_crd.mean():.6f})."
        )


# ---------------------------------------------------------------------------
# 8. Blocked
# ---------------------------------------------------------------------------


class TestRandomizationTestBlocked:
    """Tests for RandomizationTest with BlockedAssignment."""

    def test_accepts_blocked_assignment(self) -> None:
        """fit() accepts a BlockedAssignment."""
        assignment = _make_blocked_assignment_with_ate(
            n_per_block=10, n_blocks=4, seed=0, true_ate=0.5
        )
        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=100,
            seed=0,
        )
        rt.fit(assignment)
        assert isinstance(rt.p_value_, float)

    def test_blocked_permutations_preserve_block_proportion(self) -> None:
        """Each draw preserves the per-block treatment count.

        Reproduces 10 manual draws and verifies that within each block,
        the number of treated units equals that of the original assignment.
        """
        assignment = _make_blocked_assignment_with_ate(
            n_per_block=10, n_blocks=4, seed=0, true_ate=0.0
        )
        block_col = assignment.block_col_
        treat_col = assignment.treatment_col_

        original_treated_per_block = (
            assignment.data_.groupby(block_col)[treat_col].sum().to_dict()
        )

        rng = np.random.default_rng(seed=42)
        perm_seeds = rng.integers(0, 2**32, size=10)

        for ps in perm_seeds:
            drawn = assignment.draw(seed=int(ps))
            drawn_treated_per_block = (
                drawn.data_.groupby(block_col)[treat_col].sum().to_dict()
            )
            assert drawn_treated_per_block == original_treated_per_block, (
                f"Block proportions changed under draw with seed {ps}: "
                f"expected {original_treated_per_block}, got "
                f"{drawn_treated_per_block}."
            )

    def test_blocked_runs_with_blocked_dim(self) -> None:
        """Smoke: RandomizationTest with BlockedDIM and BlockedAssignment."""
        assignment = _make_blocked_assignment_with_ate(
            n_per_block=10, n_blocks=4, seed=0, true_ate=1.0
        )
        rt = RandomizationTest(
            estimator=BlockedDifferenceInMeans(outcome_col="y"),
            n_permutations=200,
            seed=0,
        )
        rt.fit(assignment)
        result = rt.estimate()
        assert np.isfinite(result.p_value)
        assert np.isfinite(result.ate)


# ---------------------------------------------------------------------------
# 9. Estimator integration (smoke)
# ---------------------------------------------------------------------------


class TestRandomizationTestEstimators:
    """Smoke tests for RandomizationTest with each Phase 3 estimator."""

    def test_works_with_difference_in_means(self) -> None:
        assignment = _make_crd_assignment_with_ate(
            n=50, seed=0, true_ate=1.0
        )
        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=200,
            seed=0,
        )
        result = rt.fit(assignment).estimate()
        assert 0.0 < result.p_value <= 1.0

    def test_works_with_blocked_dim(self) -> None:
        assignment = _make_blocked_assignment_with_ate(
            n_per_block=10, n_blocks=4, seed=0, true_ate=1.0
        )
        rt = RandomizationTest(
            estimator=BlockedDifferenceInMeans(outcome_col="y"),
            n_permutations=200,
            seed=0,
        )
        result = rt.fit(assignment).estimate()
        assert 0.0 < result.p_value <= 1.0

    def test_works_with_lin_estimator(self) -> None:
        assignment = _make_crd_assignment_with_ate(
            n=50, seed=0, true_ate=1.0
        )
        rt = RandomizationTest(
            estimator=LinEstimator(outcome_col="y", covariates=["x"]),
            n_permutations=200,
            seed=0,
        )
        result = rt.fit(assignment).estimate()
        assert 0.0 < result.p_value <= 1.0

    def test_works_with_cuped(self) -> None:
        assignment = _make_cuped_assignment(
            n=50, seed=0, true_ate=1.0, correlation=0.7
        )
        rt = RandomizationTest(
            estimator=CUPED(outcome_col="y", pre_experiment_col="y_pre"),
            n_permutations=200,
            seed=0,
        )
        result = rt.fit(assignment).estimate()
        assert 0.0 < result.p_value <= 1.0


# ---------------------------------------------------------------------------
# 10. Alternatives
# ---------------------------------------------------------------------------


class TestRandomizationTestAlternatives:
    """Tests for the three alternative hypothesis options."""

    def _assignment_with_positive_effect(self) -> CRDAssignment:
        return _make_crd_assignment_with_ate(
            n=200, seed=0, true_ate=1.0
        )

    def test_alternative_greater_pvalue_low_for_positive_effect(self) -> None:
        """For true_ate > 0, alternative='greater' yields a low p-value."""
        assignment = self._assignment_with_positive_effect()
        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=500,
            alternative="greater",
            seed=0,
        )
        rt.fit(assignment)
        assert rt.p_value_ < 0.1

    def test_alternative_less_pvalue_high_for_positive_effect(self) -> None:
        """For true_ate > 0, alternative='less' yields a high p-value."""
        assignment = self._assignment_with_positive_effect()
        rt = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=500,
            alternative="less",
            seed=0,
        )
        rt.fit(assignment)
        assert rt.p_value_ > 0.9

    def test_alternatives_ordering(self) -> None:
        """For true_ate > 0: p_less > p_two_sided > p_greater (same seed)."""
        assignment = self._assignment_with_positive_effect()

        def _run(alt: str) -> float:
            rt = RandomizationTest(
                estimator=DifferenceInMeans(outcome_col="y"),
                n_permutations=500,
                alternative=alt,
                seed=0,
            )
            rt.fit(assignment)
            return rt.p_value_

        p_less = _run("less")
        p_two_sided = _run("two-sided")
        p_greater = _run("greater")

        assert p_less > p_two_sided > p_greater, (
            f"Expected p_less > p_two_sided > p_greater, got "
            f"{p_less:.4f}, {p_two_sided:.4f}, {p_greater:.4f}."
        )