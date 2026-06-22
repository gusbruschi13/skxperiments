"""Reporting module for generating experiment reports and visualizations.

This module contains tools for creating plots and formatted reports
summarizing experimental results. Plotting requires the optional
``matplotlib`` dependency (``pip install skxperiments[viz]``); importing
this module without it is fine — the error is raised only when a plotting
function is called.
"""

from skxperiments.reporting.plots import (
    plot_balance,
    plot_null_distribution,
    plot_srm,
)

__all__ = ["plot_balance", "plot_null_distribution", "plot_srm"]