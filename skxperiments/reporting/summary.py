"""Static HTML experiment report (optional ``matplotlib`` dependency).

``ExperimentReport`` turns a ``PipelineResult`` into a single, self-contained
HTML page: a results table, the diagnostics summary, and the relevant plots
embedded inline as base64 PNGs. No template engine and no external assets —
the page is one string with inline CSS.

Plots require matplotlib (the ``viz`` extra). Build a report with
``include_plots=False`` to produce the table/diagnostics page without it.
"""

import base64
import html
import io
from typing import Any

from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.pipeline import PipelineResult
from skxperiments.reporting.plots import (
    _require_matplotlib,
    plot_balance,
    plot_effect,
    plot_interaction,
    plot_null_distribution,
    plot_srm,
)

_STYLE = (
    "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;"
    "margin:2rem;color:#1a1a1a;}"
    "h1{font-size:1.6rem;}h2{font-size:1.2rem;margin-top:1.5rem;}"
    "table{border-collapse:collapse;}td,th{padding:4px 10px;"
    "border:1px solid #ddd;text-align:right;}th{background:#f5f5f5;}"
    ".flag{color:#b00020;}.warn{color:#9a6700;}"
    "img{max-width:100%;height:auto;margin:0.5rem 0;}"
)


class ExperimentReport:
    """Render a ``PipelineResult`` as a self-contained HTML report.

    Parameters
    ----------
    pipeline_result : PipelineResult
        Output of ``ExperimentPipeline.run``.
    title : str, optional
        Page title and top heading, by default ``"Experiment Report"``.
    include_plots : bool, optional
        Whether to embed plots (requires matplotlib). When False, the
        report contains only the table and diagnostics and needs no
        optional dependency. By default True.

    Notes
    -----
    The embedded plots are chosen from what the result carries: an effect
    plot for a scalar ATE with a CI/SE (or an interaction plot for a
    multi-effect result), a balance plot when a ``BalanceReport`` ran, an
    SRM plot when an ``SRMTest`` ran, and a null-distribution plot when the
    result carries ``extra["null_distribution"]``.
    """

    def __init__(
        self,
        pipeline_result: PipelineResult,
        title: str = "Experiment Report",
        include_plots: bool = True,
    ) -> None:
        if not isinstance(pipeline_result, PipelineResult):
            raise InvalidDesignError(
                f"pipeline_result must be a PipelineResult, got "
                f"{type(pipeline_result).__name__}."
            )
        if not isinstance(title, str):
            raise InvalidDesignError(
                f"title must be a string, got {type(title).__name__}."
            )
        if not isinstance(include_plots, bool):
            raise InvalidDesignError(
                f"include_plots must be a bool, got "
                f"{type(include_plots).__name__}."
            )

        self.pipeline_result = pipeline_result
        self.title = title
        self.include_plots = include_plots

    def to_html(self) -> str:
        """Render the report as an HTML string.

        Returns
        -------
        str

        Raises
        ------
        ImportError
            If ``include_plots`` is True and matplotlib is not installed.
        """
        results = self.pipeline_result.results
        parts: list[str] = [f"<h1>{html.escape(self.title)}</h1>"]

        parts.append("<h2>Summary</h2>")
        parts.append(results.to_dataframe().to_html(index=False, border=0))

        parts.append("<h2>Diagnostics</h2>")
        parts.append(self._diagnostics_html())

        if self.include_plots:
            images = self._plot_images()
            if images:
                parts.append("<h2>Plots</h2>")
                parts.extend(
                    f'<img alt="plot" src="data:image/png;base64,{img}" />'
                    for img in images
                )

        body = "".join(parts)
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(self.title)}</title>"
            f"<style>{_STYLE}</style></head><body>{body}</body></html>"
        )

    def save(self, path: str) -> None:
        """Write the rendered HTML to ``path`` (UTF-8)."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_html())

    def _diagnostics_html(self) -> str:
        """Render flags and warnings as an HTML list (or an all-clear)."""
        report = self.pipeline_result.diagnostics
        if not report.flags and not report.warnings:
            return "<p>No issues found.</p>"

        items = [
            f'<li class="flag">FLAG: {html.escape(flag)}</li>'
            for flag in report.flags
        ]
        items += [
            f'<li class="warn">WARNING: {html.escape(warning)}</li>'
            for warning in report.warnings
        ]
        return "<ul>" + "".join(items) + "</ul>"

    def _plot_images(self) -> list[str]:
        """Render the applicable plots to base64 PNG strings."""
        _require_matplotlib()  # fail fast with a clear error if absent

        results = self.pipeline_result.results
        diagnostic_results = self.pipeline_result.diagnostic_results
        images: list[str] = []

        if results.ate is not None and (
            results.ci is not None or results.se is not None
        ):
            images.append(self._render(plot_effect, results))
        elif results.effects is not None:
            images.append(self._render(plot_interaction, results))

        if "BalanceReport" in diagnostic_results:
            images.append(
                self._render(plot_balance, diagnostic_results["BalanceReport"])
            )
        if "SRMTest" in diagnostic_results:
            images.append(
                self._render(plot_srm, diagnostic_results["SRMTest"])
            )
        if results.extra is not None and "null_distribution" in results.extra:
            images.append(self._render(plot_null_distribution, results))

        return images

    @staticmethod
    def _render(plot_func: Any, obj: Any) -> str:
        """Draw ``plot_func(obj)`` on a fresh figure and return base64 PNG."""
        plt = _require_matplotlib()
        fig, ax = plt.subplots()
        try:
            plot_func(obj, ax=ax)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("ascii")
        finally:
            plt.close(fig)
