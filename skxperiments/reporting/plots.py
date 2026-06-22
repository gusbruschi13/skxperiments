"""Plotting utilities (optional ``matplotlib`` dependency).

These functions render diagnostic and result plots with matplotlib, which
is an **optional** dependency. Import the package without matplotlib and
nothing breaks; call a plotting function without it installed and you get
a clear ``ImportError`` pointing at the ``viz`` extra.

Every ``plot_*`` function accepts an optional ``ax`` (creating a new
figure/axes when ``None``) and returns the ``matplotlib.axes.Axes`` it
drew on, so callers can compose and further customize the figure.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results
from skxperiments.design.power import power_analysis
from skxperiments.diagnostics.balance_report import BalanceResult
from skxperiments.diagnostics.srm import SRMResult
from skxperiments.pipeline import ComparisonResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


def _require_matplotlib() -> Any:
    """Import and return ``matplotlib.pyplot`` or raise a clear error."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib, an optional dependency. Install "
            "it with: pip install 'skxperiments[viz]'."
        ) from exc
    return plt


def plot_balance(
    result: BalanceResult,
    ax: "Axes | None" = None,
) -> "Axes":
    """Love plot of standardized mean differences from a ``BalanceResult``.

    Parameters
    ----------
    result : BalanceResult
        Output of ``BalanceReport.run``.
    ax : matplotlib Axes or None, optional
        Axes to draw on; a new figure/axes is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    table = result.table
    covariates = list(table["covariate"])
    smds = [float(v) for v in table["smd"]]
    positions = np.arange(len(covariates))

    ax.scatter(smds, positions, zorder=3)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.axvline(result.threshold, color="red", linestyle="--", linewidth=1.0)
    ax.axvline(-result.threshold, color="red", linestyle="--", linewidth=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels(covariates)
    ax.set_xlabel("Standardized mean difference (SMD)")
    ax.set_title("Covariate balance")
    return ax


def plot_srm(
    result: SRMResult,
    ax: "Axes | None" = None,
) -> "Axes":
    """Grouped bar chart of observed vs. expected counts from an ``SRMResult``.

    Parameters
    ----------
    result : SRMResult
        Output of ``SRMTest.run``.
    ax : matplotlib Axes or None, optional
        Axes to draw on; a new figure/axes is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    groups = list(result.observed.keys())
    observed = [result.observed[g] for g in groups]
    expected = [result.expected[g] for g in groups]
    positions = np.arange(len(groups))
    width = 0.4

    ax.bar(positions - width / 2, observed, width, label="observed")
    ax.bar(positions + width / 2, expected, width, label="expected")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_ylabel("Count")
    ax.set_title("Sample ratio: observed vs. expected")
    ax.legend()
    return ax


def plot_null_distribution(
    results: Results,
    ax: "Axes | None" = None,
    bins: int = 30,
) -> "Axes":
    """Histogram of a randomization null distribution with the observed stat.

    Parameters
    ----------
    results : Results
        A ``Results`` carrying ``extra["null_distribution"]`` (e.g., from
        ``RandomizationTest``).
    ax : matplotlib Axes or None, optional
        Axes to draw on; a new figure/axes is created when None.
    bins : int, optional
        Number of histogram bins, by default 30.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    InvalidDesignError
        If ``results.extra`` does not contain ``"null_distribution"``.
    """
    if results.extra is None or "null_distribution" not in results.extra:
        raise InvalidDesignError(
            "plot_null_distribution requires a Results with "
            "extra['null_distribution'] (e.g., from RandomizationTest)."
        )

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    distribution = np.asarray(results.extra["null_distribution"], dtype=float)
    ax.hist(distribution, bins=bins, zorder=2)
    if results.ate is not None:
        ax.axvline(
            float(results.ate),
            color="red",
            linewidth=2.0,
            label="observed",
        )
        ax.legend()
    ax.set_xlabel("Statistic under the null")
    ax.set_ylabel("Frequency")
    ax.set_title("Randomization null distribution")
    return ax


def plot_effect(
    results: Results,
    ax: "Axes | None" = None,
) -> "Axes":
    """Point estimate with confidence interval for a scalar ``Results``.

    Parameters
    ----------
    results : Results
        A scalar ``Results`` (``ate`` set) carrying a confidence interval
        (``ci``) or a standard error (``se``).
    ax : matplotlib Axes or None, optional
        Axes to draw on; a new figure/axes is created when None.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    InvalidDesignError
        If ``results`` is multi-effect, or if it has neither a confidence
        interval nor a standard error.
    """
    if results.ate is None:
        raise InvalidDesignError(
            "plot_effect requires a scalar Results (ate set); for "
            "multi-effect results use plot_interaction."
        )

    ate = float(results.ate)
    if results.ci is not None:
        lower, upper = float(results.ci[0]), float(results.ci[1])
    elif results.se is not None:
        z = float(norm.ppf(1.0 - results.alpha / 2.0))
        half = z * float(results.se)
        lower, upper = ate - half, ate + half
    else:
        raise InvalidDesignError(
            "plot_effect requires a confidence interval (ci) or standard "
            "error (se) on the Results; got neither."
        )

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    ax.plot([lower, upper], [0, 0], color="C0", linewidth=2.0, zorder=2)
    ax.scatter([ate], [0], color="C0", zorder=3)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks([0])
    ax.set_yticklabels([results.estimator_name or "ATE"])
    ax.set_xlabel("Treatment effect")
    ax.set_title("Effect estimate")
    return ax


def plot_forest(
    comparison: ComparisonResult,
    ax: "Axes | None" = None,
) -> "Axes":
    """Forest plot of ATEs across experiments from a ``ComparisonResult``.

    Parameters
    ----------
    comparison : ComparisonResult
        Output of ``ExperimentComparison.run``.
    ax : matplotlib Axes or None, optional
        Axes to draw on; a new figure/axes is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    table = comparison.table
    positions = np.arange(len(table))
    for i, row in enumerate(table.itertuples()):
        if pd.notna(row.ci_lower) and pd.notna(row.ci_upper):
            ax.plot(
                [row.ci_lower, row.ci_upper],
                [i, i],
                color="C0",
                linewidth=2.0,
                zorder=2,
            )
        ax.scatter([row.ate], [i], color="C0", zorder=3)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels(list(table["experiment"]))
    ax.invert_yaxis()  # first experiment on top (forest convention)
    ax.set_xlabel("Average treatment effect (with CI)")
    ax.set_title("Experiment comparison")
    return ax


def plot_interaction(
    results: Results,
    ax: "Axes | None" = None,
) -> "Axes":
    """Bar chart of factorial main effects and interactions.

    Plots each estimate in a multi-effect ``Results`` (e.g., from
    ``FactorialEstimator``), labelling main effects (``"A"``) and
    interactions (``"A:B"``). Error bars are drawn when a matching ``se``
    dict is present. A true cell-mean interaction plot needs the raw
    outcomes, which are not carried by ``Results``; this is the
    effect-estimate view of the same information.

    Parameters
    ----------
    results : Results
        A multi-effect ``Results`` (``effects`` set).
    ax : matplotlib Axes or None, optional
        Axes to draw on; a new figure/axes is created when None.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    InvalidDesignError
        If ``results`` is not multi-effect.
    """
    if results.effects is None:
        raise InvalidDesignError(
            "plot_interaction requires a multi-effect Results (effects "
            "set), e.g. from FactorialEstimator; for a scalar ATE use "
            "plot_effect."
        )

    keys = list(results.effects.keys())
    labels = [":".join(key) for key in keys]
    values = [float(results.effects[key]) for key in keys]
    positions = np.arange(len(keys))

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    ax.bar(positions, values, zorder=2)
    if isinstance(results.se, dict):
        errors = [float(results.se.get(key, 0.0)) for key in keys]
        ax.errorbar(
            positions,
            values,
            yerr=errors,
            fmt="none",
            ecolor="black",
            capsize=4.0,
            zorder=3,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Effect")
    ax.set_title("Factorial effects")
    return ax


def plot_power_curve(
    n_values: "Sequence[int]",
    mde: float,
    std: float,
    alpha: float = 0.05,
    allocation: float = 0.5,
    two_sided: bool = True,
    target_power: float | None = 0.8,
    ax: "Axes | None" = None,
) -> "Axes":
    """Power as a function of total sample size for a fixed effect.

    Parameters
    ----------
    n_values : sequence of int
        Total sample sizes to evaluate.
    mde : float
        Minimum detectable effect (mean difference).
    std : float
        Outcome standard deviation (assumed equal across arms).
    alpha : float, optional
        Significance level, by default 0.05.
    allocation : float, optional
        Proportion allocated to treatment, by default 0.5.
    two_sided : bool, optional
        Whether the test is two-sided, by default True.
    target_power : float or None, optional
        If not None, draw a horizontal reference line at this power
        (by default 0.8).
    ax : matplotlib Axes or None, optional
        Axes to draw on; a new figure/axes is created when None.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    InvalidDesignError
        If ``n_values`` is empty. Per-point errors from ``power_analysis``
        (e.g., non-positive ``n``) propagate unchanged.
    """
    n_list = [int(n) for n in n_values]
    if len(n_list) == 0:
        raise InvalidDesignError("n_values must be a non-empty sequence.")

    powers = [
        power_analysis(
            n=n,
            mde=mde,
            std=std,
            alpha=alpha,
            allocation=allocation,
            two_sided=two_sided,
        ).power
        for n in n_list
    ]

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(n_list, powers, marker="o", zorder=2)
    if target_power is not None:
        ax.axhline(
            target_power,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"target = {target_power}",
        )
        ax.legend()
    ax.set_xlabel("Total sample size (n)")
    ax.set_ylabel("Power")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Power curve")
    return ax
