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

from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results
from skxperiments.diagnostics.balance_report import BalanceResult
from skxperiments.diagnostics.srm import SRMResult

if TYPE_CHECKING:
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
