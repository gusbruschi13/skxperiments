"""Tests for skxperiments.design.power."""

import math

import pytest

from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.design.power import PowerResult, power_analysis


class TestPowerAnalysisInputs:
    """Tests for input validation."""

    def test_two_targets_none_raises(self) -> None:
        """Should raise when more than one target is None."""
        with pytest.raises(InvalidDesignError, match="exactly one"):
            power_analysis(n=None, mde=None, power=0.8, std=1.0)

    def test_no_targets_none_raises(self) -> None:
        """Should raise when no target is None."""
        with pytest.raises(InvalidDesignError, match="exactly one"):
            power_analysis(n=100, mde=0.2, power=0.8, std=1.0)

    def test_alpha_zero_raises(self) -> None:
        """Should raise for alpha == 0."""
        with pytest.raises(InvalidDesignError, match="alpha"):
            power_analysis(mde=0.2, power=0.8, std=1.0, alpha=0.0)

    def test_alpha_one_raises(self) -> None:
        """Should raise for alpha == 1."""
        with pytest.raises(InvalidDesignError, match="alpha"):
            power_analysis(mde=0.2, power=0.8, std=1.0, alpha=1.0)

    def test_alpha_negative_raises(self) -> None:
        """Should raise for negative alpha."""
        with pytest.raises(InvalidDesignError, match="alpha"):
            power_analysis(mde=0.2, power=0.8, std=1.0, alpha=-0.05)

    def test_power_zero_raises(self) -> None:
        """Should raise for power == 0."""
        with pytest.raises(InvalidDesignError, match="power"):
            power_analysis(mde=0.2, power=0.0, std=1.0)

    def test_power_one_raises(self) -> None:
        """Should raise for power == 1."""
        with pytest.raises(InvalidDesignError, match="power"):
            power_analysis(mde=0.2, power=1.0, std=1.0)

    def test_std_zero_raises(self) -> None:
        """Should raise for std == 0."""
        with pytest.raises(InvalidDesignError, match="std"):
            power_analysis(mde=0.2, power=0.8, std=0.0)

    def test_std_negative_raises(self) -> None:
        """Should raise for negative std."""
        with pytest.raises(InvalidDesignError, match="std"):
            power_analysis(mde=0.2, power=0.8, std=-1.0)

    def test_mde_zero_raises(self) -> None:
        """Should raise for mde == 0."""
        with pytest.raises(InvalidDesignError, match="mde"):
            power_analysis(mde=0.0, power=0.8, std=1.0)

    def test_allocation_zero_raises(self) -> None:
        """Should raise for allocation == 0."""
        with pytest.raises(InvalidDesignError, match="allocation"):
            power_analysis(
                mde=0.2, power=0.8, std=1.0, allocation=0.0
            )

    def test_allocation_one_raises(self) -> None:
        """Should raise for allocation == 1."""
        with pytest.raises(InvalidDesignError, match="allocation"):
            power_analysis(
                mde=0.2, power=0.8, std=1.0, allocation=1.0
            )

    def test_n_zero_raises(self) -> None:
        """Should raise for n == 0."""
        with pytest.raises(InvalidDesignError, match="positive integer"):
            power_analysis(n=0, mde=0.2, std=1.0)

    def test_n_negative_raises(self) -> None:
        """Should raise for negative n."""
        with pytest.raises(InvalidDesignError, match="positive integer"):
            power_analysis(n=-100, mde=0.2, std=1.0)

    def test_n_float_raises(self) -> None:
        """Should raise for non-integer n."""
        with pytest.raises(InvalidDesignError, match="positive integer"):
            power_analysis(n=100.5, mde=0.2, std=1.0)  # type: ignore[arg-type]


class TestPowerAnalysisResolveN:
    """Tests for resolving n given mde and power."""

    def test_returns_power_result(self) -> None:
        """Should return a PowerResult instance."""
        result = power_analysis(mde=0.2, power=0.8, std=1.0)
        assert isinstance(result, PowerResult)

    def test_n_total_positive_integer(self) -> None:
        """n_total must be a positive integer."""
        result = power_analysis(mde=0.2, power=0.8, std=1.0)
        assert isinstance(result.n_total, int)
        assert result.n_total > 0

    def test_classical_reference_value(self) -> None:
        """Sanity check: std=1, mde=0.2, power=0.8, alpha=0.05,
        allocation=0.5 must yield n_total close to 784 (classical
        textbook value: ~392 per group, ~784 total).
        """
        result = power_analysis(
            mde=0.2, power=0.8, std=1.0, alpha=0.05, allocation=0.5
        )
        assert 782 <= result.n_total <= 786

    def test_allocation_counts_sum_exactly(self) -> None:
        """n_treated + n_control must equal n_total exactly."""
        result = power_analysis(mde=0.2, power=0.8, std=1.0)
        assert result.n_treated + result.n_control == result.n_total

    def test_inputs_echoed_in_result(self) -> None:
        """Input parameters must be reflected in the result."""
        result = power_analysis(
            mde=0.3, power=0.9, std=2.0, alpha=0.01, allocation=0.4
        )
        assert result.power == 0.9
        assert result.alpha == 0.01
        assert result.std == 2.0
        assert result.allocation == 0.4
        assert result.mde == 0.3
        assert result.two_sided is True


class TestPowerAnalysisResolveMDE:
    """Tests for resolving mde given n and power."""

    def test_returns_positive_mde(self) -> None:
        """Resolved mde must be positive."""
        result = power_analysis(n=400, power=0.8, std=1.0)
        assert result.mde > 0

    def test_mde_consistent_with_formula(self) -> None:
        """Resolved mde must match the closed-form expression."""
        from scipy.stats import norm

        n, power, std, alpha = 400, 0.8, 1.0, 0.05
        result = power_analysis(n=n, power=power, std=std, alpha=alpha)

        z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
        z_beta = float(norm.ppf(power))
        sigma_eff = std * math.sqrt(1.0 / 0.5 + 1.0 / 0.5)
        expected_mde = (z_alpha + z_beta) * sigma_eff / math.sqrt(n)

        assert result.mde == pytest.approx(expected_mde, rel=1e-9)

    def test_n_echoed(self) -> None:
        """n_total in result must equal input n."""
        result = power_analysis(n=500, power=0.8, std=1.0)
        assert result.n_total == 500


class TestPowerAnalysisResolvePower:
    """Tests for resolving power given n and mde."""

    def test_power_in_open_interval(self) -> None:
        """Resolved power must be in (0, 1)."""
        result = power_analysis(n=400, mde=0.2, std=1.0)
        assert 0.0 < result.power < 1.0

    def test_large_mde_yields_high_power(self) -> None:
        """A very large mde must produce power close to 1."""
        result = power_analysis(n=400, mde=10.0, std=1.0)
        assert result.power > 0.999

    def test_small_mde_yields_low_power(self) -> None:
        """A very small mde must produce power close to alpha/2 (two-sided)."""
        result = power_analysis(
            n=400, mde=1e-6, std=1.0, alpha=0.05, two_sided=True
        )
        # Under H0, two-sided rejection prob in one tail is alpha/2.
        # With negligible effect, power approaches alpha/2 from above.
        assert result.power == pytest.approx(0.025, abs=1e-3)

    def test_one_sided_returns_higher_power(self) -> None:
        """One-sided test must yield more power than two-sided for
        the same mde, n, alpha (effect aligned with the alternative).
        """
        r2 = power_analysis(n=200, mde=0.2, std=1.0, two_sided=True)
        r1 = power_analysis(n=200, mde=0.2, std=1.0, two_sided=False)
        assert r1.power > r2.power


class TestPowerAnalysisRoundtrip:
    """Critical: solving and re-solving must be self-consistent."""

    def test_n_to_mde_roundtrip(self) -> None:
        """Resolving n for a target mde, then resolving mde with that
        n, must recover the original mde within tolerance.
        """
        target_mde = 0.2
        r1 = power_analysis(
            mde=target_mde, power=0.8, std=1.0, alpha=0.05
        )
        r2 = power_analysis(
            n=r1.n_total, power=0.8, std=1.0, alpha=0.05
        )
        # Tolerance accommodates the ceil() in n resolution.
        assert r2.mde == pytest.approx(target_mde, abs=1e-2)

    def test_n_to_power_roundtrip(self) -> None:
        """Resolving n for a target power, then resolving power with
        that n, must recover the original power within tolerance.
        """
        target_power = 0.8
        r1 = power_analysis(
            mde=0.2, power=target_power, std=1.0, alpha=0.05
        )
        r2 = power_analysis(
            n=r1.n_total, mde=0.2, std=1.0, alpha=0.05
        )
        assert r2.power == pytest.approx(target_power, abs=1e-2)


class TestPowerAnalysisAllocation:
    """Tests for asymmetric allocation."""

    def test_unequal_allocation_requires_more_n(self) -> None:
        """Allocation 70/30 must require more n_total than 50/50,
        all else equal.
        """
        r_balanced = power_analysis(
            mde=0.2, power=0.8, std=1.0, alpha=0.05, allocation=0.5
        )
        r_unequal = power_analysis(
            mde=0.2, power=0.8, std=1.0, alpha=0.05, allocation=0.7
        )
        assert r_unequal.n_total > r_balanced.n_total

    def test_allocation_counts_sum_balanced(self) -> None:
        """n_treated + n_control == n_total under 50/50."""
        result = power_analysis(
            mde=0.2, power=0.8, std=1.0, allocation=0.5
        )
        assert result.n_treated + result.n_control == result.n_total

    def test_allocation_counts_sum_unequal(self) -> None:
        """n_treated + n_control == n_total under 70/30."""
        result = power_analysis(
            mde=0.2, power=0.8, std=1.0, allocation=0.7
        )
        assert result.n_treated + result.n_control == result.n_total

    def test_allocation_reflected_in_split(self) -> None:
        """n_treated must be approximately allocation * n_total."""
        result = power_analysis(
            mde=0.2, power=0.8, std=1.0, allocation=0.7
        )
        expected_treated = round(result.n_total * 0.7)
        assert result.n_treated == expected_treated