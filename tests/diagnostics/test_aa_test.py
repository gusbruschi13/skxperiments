"""Tests for skxperiments.diagnostics.aa_test."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.base import BaseInference, DiagnosticsReport
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results
from skxperiments.design.crd import CRD
from skxperiments.diagnostics import AAResult, AATest
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.inference import NeymanCI, RandomizationTest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aa_df(n: int, seed: int) -> pd.DataFrame:
    """A fixed dataset with a noise outcome and one covariate (no treatment).

    The outcome does not depend on treatment, and AATest re-randomizes the
    treatment each simulation, so the true effect is zero by construction.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"x": rng.normal(size=n), "y": rng.normal(size=n)}
    )


def _neyman() -> NeymanCI:
    return NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))


class _BrokenInference(BaseInference):
    """An anti-conservative inference that always rejects (p_value=0)."""

    def fit(self, assignment) -> "_BrokenInference":
        self._fitted_ = True
        return self

    def estimate(self) -> Results:
        return Results(ate=0.0, p_value=0.0)


class _NoPValueInference(BaseInference):
    """An inference that produces no p_value (CI-only)."""

    def fit(self, assignment) -> "_NoPValueInference":
        self._fitted_ = True
        return self

    def estimate(self) -> Results:
        return Results(ate=0.0)


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


class TestAATestCreation:
    """Tests for AATest.__init__ validations."""

    def test_defaults(self) -> None:
        """Defaults: n_simulations=1000, alpha=0.05, meta_threshold=0.001."""
        aa = AATest(CRD(p=0.5, seed=0), _neyman())
        assert aa.n_simulations == 1000
        assert aa.alpha == 0.05
        assert aa.meta_threshold == 0.001
        assert aa.seed is None

    def test_custom_parameters(self) -> None:
        """Custom config is stored."""
        aa = AATest(
            CRD(p=0.5, seed=0),
            _neyman(),
            n_simulations=200,
            alpha=0.10,
            meta_threshold=0.01,
            seed=7,
        )
        assert aa.n_simulations == 200
        assert aa.alpha == 0.10
        assert aa.meta_threshold == 0.01
        assert aa.seed == 7

    def test_rejects_non_design(self) -> None:
        """design must be a BaseDesign."""
        with pytest.raises(InvalidDesignError, match="design"):
            AATest("not a design", _neyman())  # type: ignore[arg-type]

    def test_rejects_non_inference(self) -> None:
        """inference must be a BaseInference."""
        with pytest.raises(InvalidDesignError, match="inference"):
            AATest(CRD(p=0.5, seed=0), "not an inference")  # type: ignore[arg-type]

    def test_rejects_non_positive_n_simulations(self) -> None:
        """n_simulations must be > 0."""
        with pytest.raises(InvalidDesignError):
            AATest(CRD(p=0.5, seed=0), _neyman(), n_simulations=0)

    def test_rejects_n_simulations_bool(self) -> None:
        """n_simulations=True is rejected."""
        with pytest.raises(InvalidDesignError):
            AATest(CRD(p=0.5, seed=0), _neyman(), n_simulations=True)  # type: ignore[arg-type]

    def test_rejects_alpha_out_of_range(self) -> None:
        """alpha must be in (0, 1)."""
        with pytest.raises(InvalidDesignError, match="alpha"):
            AATest(CRD(p=0.5, seed=0), _neyman(), alpha=1.0)

    def test_rejects_meta_threshold_out_of_range(self) -> None:
        """meta_threshold must be in (0, 1)."""
        with pytest.raises(InvalidDesignError, match="meta_threshold"):
            AATest(CRD(p=0.5, seed=0), _neyman(), meta_threshold=0.0)


# ---------------------------------------------------------------------------
# 2. Run behavior
# ---------------------------------------------------------------------------


class TestAATestRun:
    """Tests for AATest.run."""

    def test_calibrated_pipeline_not_flagged(self) -> None:
        """A correctly calibrated pipeline is not flagged."""
        aa = AATest(
            CRD(p=0.5, seed=0),
            _neyman(),
            n_simulations=300,
            seed=0,
        )
        result = aa.run(_aa_df(100, seed=1))
        assert result.flagged is False
        assert 0.0 <= result.false_positive_rate <= 0.2
        assert len(result.p_values) == 300
        assert np.all((result.p_values >= 0.0) & (result.p_values <= 1.0))

    def test_reproducible_with_seed(self) -> None:
        """Same seed yields identical p-values and false-positive rate."""
        df = _aa_df(60, seed=1)

        def _run() -> AAResult:
            return AATest(
                CRD(p=0.5, seed=0), _neyman(), n_simulations=100, seed=42
            ).run(df)

        a = _run()
        b = _run()
        assert np.array_equal(a.p_values, b.p_values)
        assert a.false_positive_rate == b.false_positive_rate

    def test_broken_inference_flagged(self) -> None:
        """An always-rejecting inference is flagged as miscalibrated."""
        aa = AATest(
            CRD(p=0.5, seed=0),
            _BrokenInference(),
            n_simulations=50,
            seed=0,
        )
        result = aa.run(_aa_df(40, seed=1))
        assert result.flagged is True
        assert result.false_positive_rate == pytest.approx(1.0)

    def test_no_pvalue_inference_raises(self) -> None:
        """An inference without a p_value raises InvalidDesignError."""
        aa = AATest(
            CRD(p=0.5, seed=0),
            _NoPValueInference(),
            n_simulations=10,
            seed=0,
        )
        with pytest.raises(InvalidDesignError, match="p_value"):
            aa.run(_aa_df(40, seed=1))


# ---------------------------------------------------------------------------
# 3. Result object
# ---------------------------------------------------------------------------


class TestAAResult:
    """Tests for the AAResult surface."""

    def _calibrated_result(self) -> AAResult:
        return AATest(
            CRD(p=0.5, seed=0), _neyman(), n_simulations=200, seed=0
        ).run(_aa_df(100, seed=1))

    def test_is_aa_result(self) -> None:
        """run() returns an AAResult."""
        assert isinstance(self._calibrated_result(), AAResult)

    def test_to_dict_keys(self) -> None:
        """to_dict exposes the scalar summary fields (no raw array)."""
        d = self._calibrated_result().to_dict()
        assert set(d) == {
            "n_simulations",
            "alpha",
            "meta_threshold",
            "false_positive_rate",
            "n_false_positives",
            "fp_test_pvalue",
            "ks_statistic",
            "ks_pvalue",
            "flagged",
        }

    def test_summary_returns_self(self, capsys) -> None:
        """summary() prints and returns self."""
        result = self._calibrated_result()
        assert result.summary() is result
        assert "A/A Test" in capsys.readouterr().out

    def test_report_flagged(self) -> None:
        """A flagged result yields a DiagnosticsReport with a flag."""
        result = AATest(
            CRD(p=0.5, seed=0), _BrokenInference(), n_simulations=50, seed=0
        ).run(_aa_df(40, seed=1))
        report = result.to_diagnostics_report()
        assert isinstance(report, DiagnosticsReport)
        assert len(report.flags) == 1
        assert "false-positive" in report.flags[0].lower()

    def test_report_clean(self) -> None:
        """A calibrated result yields no flags."""
        report = self._calibrated_result().to_diagnostics_report()
        assert report.flags == []


# ---------------------------------------------------------------------------
# 4. Numerics (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAATestNumerics:
    """Slow A/A calibration with a resampling inference."""

    def test_randomization_test_calibrated(self) -> None:
        """A/A with RandomizationTest is well-calibrated (nested loop)."""
        inference = RandomizationTest(
            estimator=DifferenceInMeans(outcome_col="y"),
            n_permutations=99,
            seed=0,
        )
        aa = AATest(
            CRD(p=0.5, seed=0),
            inference,
            n_simulations=100,
            seed=0,
        )
        result = aa.run(_aa_df(60, seed=1))
        assert result.flagged is False
        assert np.isfinite(result.ks_pvalue)
