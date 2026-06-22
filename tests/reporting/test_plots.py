"""Tests for skxperiments.reporting.plots (diagnostic plots)."""

import sys

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from skxperiments.core.assignment import CRDAssignment  # noqa: E402
from skxperiments.core.exceptions import InvalidDesignError  # noqa: E402
from skxperiments.core.results import Results  # noqa: E402
from skxperiments.design.crd import CRD  # noqa: E402
from skxperiments.diagnostics import BalanceReport, SRMTest  # noqa: E402
from skxperiments.reporting.plots import (  # noqa: E402
    _require_matplotlib,
    plot_balance,
    plot_null_distribution,
    plot_srm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assignment() -> CRDAssignment:
    """A 20/20 CRD assignment with two covariates."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "treatment": [1] * 20 + [0] * 20,
        }
    )
    return CRDAssignment(
        data=df, treatment_col="treatment", design=CRD(p=0.5, seed=0)
    )


def _null_results() -> Results:
    rng = np.random.default_rng(1)
    return Results(
        ate=0.5,
        extra={"null_distribution": rng.normal(size=500)},
    )


# ---------------------------------------------------------------------------
# 1. plot_balance
# ---------------------------------------------------------------------------


class TestPlotBalance:
    def test_returns_axes_with_one_tick_per_covariate(self) -> None:
        result = BalanceReport().run(_assignment())
        ax = plot_balance(result)
        assert isinstance(ax, Axes)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["x1", "x2"]
        assert ax.get_xlabel() == "Standardized mean difference (SMD)"
        plt.close("all")

    def test_uses_provided_ax(self) -> None:
        result = BalanceReport().run(_assignment())
        _, ax = plt.subplots()
        returned = plot_balance(result, ax=ax)
        assert returned is ax
        plt.close("all")


# ---------------------------------------------------------------------------
# 2. plot_srm
# ---------------------------------------------------------------------------


class TestPlotSRM:
    def test_returns_axes_with_two_bar_groups(self) -> None:
        result = SRMTest().run(_assignment())
        ax = plot_srm(result)
        assert isinstance(ax, Axes)
        # One BarContainer for observed, one for expected.
        assert len(ax.containers) == 2
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["control", "treated"]
        assert ax.get_legend() is not None
        plt.close("all")

    def test_uses_provided_ax(self) -> None:
        result = SRMTest().run(_assignment())
        _, ax = plt.subplots()
        assert plot_srm(result, ax=ax) is ax
        plt.close("all")


# ---------------------------------------------------------------------------
# 3. plot_null_distribution
# ---------------------------------------------------------------------------


class TestPlotNullDistribution:
    def test_returns_axes_with_histogram_and_observed_line(self) -> None:
        ax = plot_null_distribution(_null_results())
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0  # histogram bars
        assert len(ax.lines) >= 1  # observed vline
        assert ax.get_title() == "Randomization null distribution"
        plt.close("all")

    def test_raises_without_null_distribution(self) -> None:
        # A Results with no extra (e.g., a point estimate) cannot be plotted.
        with pytest.raises(InvalidDesignError, match="null_distribution"):
            plot_null_distribution(Results(ate=0.5))

    def test_respects_bins(self) -> None:
        ax = plot_null_distribution(_null_results(), bins=10)
        assert len(ax.patches) == 10
        plt.close("all")


# ---------------------------------------------------------------------------
# 4. Optional-dependency guard
# ---------------------------------------------------------------------------


class TestMatplotlibGuard:
    def test_require_matplotlib_raises_when_absent(self, monkeypatch) -> None:
        """A clear ImportError is raised when matplotlib is unavailable."""
        # Setting the module to None makes `import matplotlib.pyplot` raise.
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        with pytest.raises(ImportError, match="skxperiments\\[viz\\]"):
            _require_matplotlib()
