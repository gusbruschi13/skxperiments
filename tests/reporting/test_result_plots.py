"""Tests for skxperiments.reporting.plots (result plots)."""

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from skxperiments.core.exceptions import InvalidDesignError  # noqa: E402
from skxperiments.core.results import Results  # noqa: E402
from skxperiments.pipeline import ExperimentComparison  # noqa: E402
from skxperiments.reporting.plots import (  # noqa: E402
    plot_effect,
    plot_forest,
    plot_interaction,
    plot_power_curve,
)


# ---------------------------------------------------------------------------
# plot_effect
# ---------------------------------------------------------------------------


class TestPlotEffect:
    def test_with_ci(self) -> None:
        ax = plot_effect(Results(ate=1.0, ci=(0.5, 1.5)))
        assert isinstance(ax, Axes)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["ATE"]
        plt.close("all")

    def test_uses_estimator_name_label(self) -> None:
        results = Results(ate=1.0, ci=(0.5, 1.5), estimator_name="NeymanCI")
        ax = plot_effect(results)
        assert [t.get_text() for t in ax.get_yticklabels()] == ["NeymanCI"]
        plt.close("all")

    def test_with_se_only(self) -> None:
        ax = plot_effect(Results(ate=1.0, se=0.5))
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_raises_without_ci_or_se(self) -> None:
        with pytest.raises(InvalidDesignError, match="confidence interval"):
            plot_effect(Results(ate=1.0))

    def test_raises_on_multi_effect(self) -> None:
        with pytest.raises(InvalidDesignError, match="multi-effect"):
            plot_effect(Results(effects={("A",): 1.0}))


# ---------------------------------------------------------------------------
# plot_forest
# ---------------------------------------------------------------------------


class TestPlotForest:
    def _comparison(self, with_missing_ci: bool = False):
        b = (
            Results(ate=0.3, p_value=0.20)
            if with_missing_ci
            else Results(ate=0.3, p_value=0.20, ci=(-0.1, 0.7))
        )
        results = {
            "a": Results(ate=1.0, p_value=0.01, ci=(0.5, 1.5)),
            "b": b,
        }
        return ExperimentComparison().run(results)

    def test_returns_axes_with_one_point_per_experiment(self) -> None:
        ax = plot_forest(self._comparison())
        assert isinstance(ax, Axes)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["a", "b"]
        assert len(ax.collections) == 2  # one scatter per experiment
        plt.close("all")

    def test_handles_missing_ci(self) -> None:
        ax = plot_forest(self._comparison(with_missing_ci=True))
        assert isinstance(ax, Axes)
        assert len(ax.collections) == 2
        plt.close("all")

    def test_uses_provided_ax(self) -> None:
        _, ax = plt.subplots()
        assert plot_forest(self._comparison(), ax=ax) is ax
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_interaction
# ---------------------------------------------------------------------------


class TestPlotInteraction:
    def test_returns_axes_with_one_bar_per_effect(self) -> None:
        results = Results(
            effects={("A",): 0.5, ("B",): 0.3, ("A", "B"): 0.1}
        )
        ax = plot_interaction(results)
        assert isinstance(ax, Axes)
        assert len(ax.patches) == 3  # one bar per effect
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "A:B" in labels
        plt.close("all")

    def test_with_se_dict(self) -> None:
        results = Results(
            effects={("A",): 0.5, ("B",): 0.3},
            se={("A",): 0.1, ("B",): 0.1},
        )
        ax = plot_interaction(results)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_raises_on_scalar(self) -> None:
        with pytest.raises(InvalidDesignError, match="multi-effect"):
            plot_interaction(Results(ate=1.0))


# ---------------------------------------------------------------------------
# plot_power_curve
# ---------------------------------------------------------------------------


class TestPlotPowerCurve:
    def test_power_increases_with_n(self) -> None:
        n_values = [50, 100, 200, 400]
        ax = plot_power_curve(n_values, mde=0.3, std=1.0)
        assert isinstance(ax, Axes)
        curve = ax.lines[0]
        ydata = list(curve.get_ydata())
        assert len(ydata) == len(n_values)
        assert all(ydata[i] <= ydata[i + 1] for i in range(len(ydata) - 1))
        assert ax.get_ylim() == (0.0, 1.0)
        plt.close("all")

    def test_target_line_optional(self) -> None:
        ax = plot_power_curve([100, 200], mde=0.3, std=1.0, target_power=None)
        # Only the power curve line, no target reference line.
        assert len(ax.lines) == 1
        plt.close("all")

    def test_raises_on_empty(self) -> None:
        with pytest.raises(InvalidDesignError, match="non-empty"):
            plot_power_curve([], mde=0.3, std=1.0)
