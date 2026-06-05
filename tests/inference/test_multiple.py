"""Tests for skxperiments.inference.multiple.MultipleTestingCorrection."""

import numpy as np
import pytest

from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results
from skxperiments.inference import MultipleTestingCorrection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_multi_effect_results(
    p_values: dict,
    alpha: float = 0.05,
    extra: dict | None = None,
) -> Results:
    """Build a multi-effect Results with controlled p-values.

    Effect estimates are arbitrary nonzero floats; only p-values
    matter for the correction logic.
    """
    effects = {key: float(i + 1) for i, key in enumerate(p_values)}
    return Results(
        effects=effects,
        p_value=p_values,
        alpha=alpha,
        extra=extra,
    )


def _make_scalar_results_list(
    p_values: list,
    alpha: float = 0.05,
) -> list[Results]:
    """Build a list of scalar Results with controlled p-values."""
    return [
        Results(ate=float(i + 1), p_value=p, alpha=alpha)
        for i, p in enumerate(p_values)
    ]


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


class TestMultipleTestingCorrectionCreation:
    """Tests for MultipleTestingCorrection.__init__ validations."""

    def test_basic_creation_defaults(self) -> None:
        """Defaults: method='holm', alpha=0.05."""
        mtc = MultipleTestingCorrection()
        assert mtc.method == "holm"
        assert mtc.alpha == 0.05

    def test_invalid_method_raises(self) -> None:
        """Unknown method is rejected."""
        with pytest.raises(InvalidDesignError, match="method"):
            MultipleTestingCorrection(method="invalid")

    def test_invalid_alpha_raises_zero(self) -> None:
        """alpha=0.0 is rejected (must be in (0, 1) open interval)."""
        with pytest.raises(InvalidDesignError):
            MultipleTestingCorrection(alpha=0.0)

    def test_invalid_alpha_raises_one(self) -> None:
        """alpha=1.0 is rejected."""
        with pytest.raises(InvalidDesignError):
            MultipleTestingCorrection(alpha=1.0)


# ---------------------------------------------------------------------------
# 2. Validation in correct()
# ---------------------------------------------------------------------------


class TestMultipleTestingCorrectionValidation:
    """Tests for input validation in correct()."""

    def test_rejects_scalar_results_single(self) -> None:
        """A single scalar Results is rejected (must be list or multi-effect)."""
        r = Results(ate=0.5, p_value=0.03)
        mtc = MultipleTestingCorrection()
        with pytest.raises(InvalidDesignError, match="multi-effect|list"):
            mtc.correct(r)

    def test_rejects_multi_effect_without_p_value(self) -> None:
        """Multi-effect Results without p_value is rejected."""
        r = Results(effects={("A",): 0.5, ("B",): 0.3})
        mtc = MultipleTestingCorrection()
        with pytest.raises(InvalidDesignError, match="p_value"):
            mtc.correct(r)

    def test_rejects_empty_list(self) -> None:
        """Empty list is rejected."""
        mtc = MultipleTestingCorrection()
        with pytest.raises(InvalidDesignError, match="empty"):
            mtc.correct([])

    def test_rejects_list_with_multi_effect_element(self) -> None:
        """List containing a multi-effect Results is rejected."""
        r_multi = Results(
            effects={("A",): 0.5}, p_value={("A",): 0.03}
        )
        r_scalar = Results(ate=0.5, p_value=0.03)
        mtc = MultipleTestingCorrection()
        with pytest.raises(InvalidDesignError, match="scalar|index"):
            mtc.correct([r_scalar, r_multi])

    def test_rejects_double_correction_via_reserved_key(self) -> None:
        """Applying correct() twice on the same Results is rejected."""
        r = _make_multi_effect_results(
            {("A",): 0.01, ("B",): 0.04, ("C",): 0.10}
        )
        mtc = MultipleTestingCorrection(method="holm", alpha=0.05)
        first = mtc.correct(r)
        with pytest.raises(InvalidDesignError, match="reserved key"):
            mtc.correct(first)


# ---------------------------------------------------------------------------
# 3. Bonferroni
# ---------------------------------------------------------------------------


class TestBonferroni:
    """Tests for the Bonferroni correction."""

    def test_bonferroni_multiplies_by_m(self) -> None:
        """For p=[0.01, 0.02, 0.03], m=3, expect [0.03, 0.06, 0.09]."""
        results = _make_scalar_results_list([0.01, 0.02, 0.03])
        mtc = MultipleTestingCorrection(method="bonferroni", alpha=0.05)
        corrected = mtc.correct(results)

        expected = [0.03, 0.06, 0.09]
        for r, exp in zip(corrected, expected):
            assert r.p_value == pytest.approx(exp, abs=1e-12)

    def test_bonferroni_clipped_at_one(self) -> None:
        """Large p-values are clipped at 1.0 (not 1.8)."""
        results = _make_scalar_results_list([0.6, 0.7, 0.8])
        mtc = MultipleTestingCorrection(method="bonferroni")
        corrected = mtc.correct(results)

        for r in corrected:
            assert r.p_value == 1.0

    def test_bonferroni_preserves_ordering(self) -> None:
        """Relative ordering of p-values is preserved."""
        rng = np.random.default_rng(seed=0)
        raw = rng.uniform(0.001, 0.1, size=10).tolist()
        results = _make_scalar_results_list(raw)

        mtc = MultipleTestingCorrection(method="bonferroni")
        corrected = mtc.correct(results)
        corrected_p = [r.p_value for r in corrected]

        # Same argsort as input.
        assert np.argsort(corrected_p).tolist() == np.argsort(raw).tolist()


# ---------------------------------------------------------------------------
# 4. Holm
# ---------------------------------------------------------------------------


class TestHolm:
    """Tests for the Holm correction."""

    def test_holm_correction_known_values(self) -> None:
        """Holm correction matches manually computed values.

        Input: p = [0.01, 0.04, 0.03], m=3.
        Sorted: [0.01, 0.03, 0.04] (original positions [0, 2, 1]).
        Multipliers: [3, 2, 1].
        Sorted * multipliers: [0.03, 0.06, 0.04].
        Cumulative max: [0.03, 0.06, 0.06].
        Reordered to original positions: [0.03, 0.06, 0.06].
        """
        results = _make_scalar_results_list([0.01, 0.04, 0.03])
        mtc = MultipleTestingCorrection(method="holm", alpha=0.05)
        corrected = mtc.correct(results)

        expected = [0.03, 0.06, 0.06]
        for r, exp in zip(corrected, expected):
            assert r.p_value == pytest.approx(exp, abs=1e-12)

    def test_holm_monotonicity(self) -> None:
        """Holm-corrected p-values are non-decreasing in sorted-original order."""
        rng = np.random.default_rng(seed=42)
        raw = rng.uniform(0.001, 0.5, size=15).tolist()
        results = _make_scalar_results_list(raw)

        mtc = MultipleTestingCorrection(method="holm")
        corrected = mtc.correct(results)
        corrected_p = np.array([r.p_value for r in corrected])

        # Sort by original p-value ascending; corrected should be non-decreasing.
        order = np.argsort(raw)
        sorted_corrected = corrected_p[order]
        diffs = np.diff(sorted_corrected)
        assert np.all(diffs >= -1e-12), (
            f"Expected non-decreasing Holm-corrected p-values, got "
            f"diffs {diffs}"
        )

    def test_holm_dominates_bonferroni(self) -> None:
        """For any p-values, p_holm <= p_bonferroni elementwise."""
        rng = np.random.default_rng(seed=7)
        raw = rng.uniform(0.001, 0.2, size=10).tolist()

        results_holm = _make_scalar_results_list(raw)
        results_bonf = _make_scalar_results_list(raw)

        mtc_holm = MultipleTestingCorrection(method="holm")
        mtc_bonf = MultipleTestingCorrection(method="bonferroni")

        out_holm = mtc_holm.correct(results_holm)
        out_bonf = mtc_bonf.correct(results_bonf)

        for h, b in zip(out_holm, out_bonf):
            assert h.p_value <= b.p_value + 1e-12

    def test_holm_clipped_at_one(self) -> None:
        """Holm-corrected p-values are clipped at 1.0."""
        results = _make_scalar_results_list([0.5, 0.6, 0.7])
        mtc = MultipleTestingCorrection(method="holm")
        corrected = mtc.correct(results)
        for r in corrected:
            assert 0.0 <= r.p_value <= 1.0


# ---------------------------------------------------------------------------
# 5. Benjamini-Hochberg
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    """Tests for the Benjamini-Hochberg correction."""

    def test_bh_correction_known_values(self) -> None:
        """BH correction matches manually computed values.

        Input: p = [0.01, 0.04, 0.03], m=3.
        Sorted: [0.01, 0.03, 0.04] (original positions [0, 2, 1]).
        Ranks: [1, 2, 3]; BH = sorted * 3 / ranks = [0.03, 0.045, 0.04].
        Reverse cumulative min from right: [0.03, 0.04, 0.04].
        Reordered to original positions: [0.03, 0.04, 0.04].
        """
        results = _make_scalar_results_list([0.01, 0.04, 0.03])
        mtc = MultipleTestingCorrection(method="bh", alpha=0.05)
        corrected = mtc.correct(results)

        expected = [0.03, 0.04, 0.04]
        for r, exp in zip(corrected, expected):
            assert r.p_value == pytest.approx(exp, abs=1e-12)

    def test_bh_monotonicity_reversed(self) -> None:
        """BH-corrected p-values are non-decreasing in sorted-original order."""
        rng = np.random.default_rng(seed=123)
        raw = rng.uniform(0.001, 0.5, size=15).tolist()
        results = _make_scalar_results_list(raw)

        mtc = MultipleTestingCorrection(method="bh")
        corrected = mtc.correct(results)
        corrected_p = np.array([r.p_value for r in corrected])

        order = np.argsort(raw)
        sorted_corrected = corrected_p[order]
        diffs = np.diff(sorted_corrected)
        assert np.all(diffs >= -1e-12), (
            f"Expected non-decreasing BH-corrected p-values, got "
            f"diffs {diffs}"
        )

    @pytest.mark.slow
    def test_bh_controls_fdr(self) -> None:
        """BH controls FDR at the nominal level under independence.

        1000 reps. In each rep:
        - 10 p-values from the null (Uniform(0,1)).
        - 10 p-values from a strong alternative (z-score from N(3, 1) -> p).
        Apply BH at alpha=0.1; compute observed FDR (false discoveries
        / total discoveries, or 0 if no discoveries).
        Mean FDR over reps should be <= 0.15 (generous slack over 0.1).
        """
        rng = np.random.default_rng(seed=2024)
        n_reps = 1000
        n_null = 10
        n_alt = 10
        m = n_null + n_alt
        alpha = 0.10

        observed_fdrs = np.empty(n_reps, dtype=float)
        for rep in range(n_reps):
            # 10 nulls: uniform p-values.
            p_null = rng.uniform(0.0, 1.0, size=n_null)
            # 10 alternatives: |Z| under N(3,1) -> two-sided p.
            z = rng.normal(3.0, 1.0, size=n_alt)
            from scipy import stats as sstats
            p_alt = 2 * (1 - sstats.norm.cdf(np.abs(z)))

            # is_null mask: True for the first n_null entries.
            is_null = np.concatenate(
                [np.ones(n_null, dtype=bool), np.zeros(n_alt, dtype=bool)]
            )
            p_all = np.concatenate([p_null, p_alt])

            results = _make_scalar_results_list(p_all.tolist())
            mtc = MultipleTestingCorrection(method="bh", alpha=alpha)
            corrected = mtc.correct(results)
            corrected_p = np.array([r.p_value for r in corrected])

            rejected = corrected_p <= alpha
            n_rejected = int(rejected.sum())
            if n_rejected == 0:
                observed_fdrs[rep] = 0.0
            else:
                false_discoveries = int((rejected & is_null).sum())
                observed_fdrs[rep] = false_discoveries / n_rejected

        mean_fdr = float(observed_fdrs.mean())
        assert mean_fdr <= 0.15, (
            f"Expected mean FDR <= 0.15 under BH at alpha=0.10, got "
            f"{mean_fdr:.4f}."
        )


# ---------------------------------------------------------------------------
# 6. Multi-effect input
# ---------------------------------------------------------------------------


class TestMultiEffectInput:
    """Tests for multi-effect Results input handling."""

    def test_multi_effect_output_preserves_effects(self) -> None:
        """The effects dict is preserved identically in the output."""
        r = _make_multi_effect_results(
            {("A",): 0.01, ("B",): 0.04, ("A", "B"): 0.20}
        )
        mtc = MultipleTestingCorrection(method="holm")
        corrected = mtc.correct(r)

        assert corrected.effects == r.effects

    def test_multi_effect_output_preserves_metadata(self) -> None:
        """Metadata fields (n_obs etc.) are preserved."""
        r = Results(
            effects={("A",): 0.5, ("B",): 0.3},
            p_value={("A",): 0.01, ("B",): 0.04},
            n_obs=100,
            n_treated=None,
            n_control=None,
            estimator_name="FactorialEstimator",
            design_name="FactorialDesign",
            inference_name="RandomizationTest",
        )
        mtc = MultipleTestingCorrection(method="holm")
        corrected = mtc.correct(r)

        assert corrected.n_obs == 100
        assert corrected.estimator_name == "FactorialEstimator"
        assert corrected.design_name == "FactorialDesign"
        assert corrected.inference_name == "RandomizationTest"

    def test_multi_effect_alpha_overridden(self) -> None:
        """Output alpha equals self.alpha, overriding input alpha."""
        r = _make_multi_effect_results(
            {("A",): 0.01, ("B",): 0.04}, alpha=0.10
        )
        mtc = MultipleTestingCorrection(method="holm", alpha=0.025)
        corrected = mtc.correct(r)
        assert corrected.alpha == 0.025


# ---------------------------------------------------------------------------
# 7. List input
# ---------------------------------------------------------------------------


class TestListInput:
    """Tests for list-of-Results input handling."""

    def test_list_output_preserves_order(self) -> None:
        """The i-th corrected p-value corresponds to the i-th input."""
        raw = [0.05, 0.001, 0.30, 0.01, 0.20]
        results = _make_scalar_results_list(raw)
        mtc = MultipleTestingCorrection(method="holm")
        corrected = mtc.correct(results)

        # The output is in input order. Verify that the corrected p
        # at the position of the smallest original p is the smallest.
        original_argmin = int(np.argmin(raw))
        corrected_p = [r.p_value for r in corrected]
        assert int(np.argmin(corrected_p)) == original_argmin

    def test_list_output_preserves_per_results_metadata(self) -> None:
        """Each Results in output preserves its own ate, n_obs, etc."""
        results = [
            Results(
                ate=0.5,
                p_value=0.01,
                n_obs=100,
                estimator_name="DIM",
                design_name="CRD",
            ),
            Results(
                ate=0.3,
                p_value=0.04,
                n_obs=200,
                estimator_name="LinEstimator",
                design_name="CRD",
            ),
        ]
        mtc = MultipleTestingCorrection(method="holm")
        corrected = mtc.correct(results)

        assert corrected[0].ate == 0.5
        assert corrected[0].n_obs == 100
        assert corrected[0].estimator_name == "DIM"
        assert corrected[1].ate == 0.3
        assert corrected[1].n_obs == 200
        assert corrected[1].estimator_name == "LinEstimator"

    def test_list_extra_contains_family_metadata(self) -> None:
        """Each list element has the family-level reserved keys in extra."""
        raw = [0.01, 0.04, 0.20]
        results = _make_scalar_results_list(raw)
        mtc = MultipleTestingCorrection(method="bh", alpha=0.05)
        corrected = mtc.correct(results)

        for r in corrected:
            assert r.extra is not None
            assert r.extra["correction_method"] == "bh"
            assert r.extra["family_wise_alpha"] == 0.05
            assert r.extra["n_tests"] == 3
            assert r.extra["original_p_values"] == raw