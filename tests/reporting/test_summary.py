"""Tests for skxperiments.reporting.summary.ExperimentReport."""

import sys

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from skxperiments.core.assignment import CRDAssignment  # noqa: E402
from skxperiments.core.base import DiagnosticsReport  # noqa: E402
from skxperiments.core.exceptions import InvalidDesignError  # noqa: E402
from skxperiments.core.results import Results  # noqa: E402
from skxperiments.design.crd import CRD  # noqa: E402
from skxperiments.estimators.difference_in_means import (  # noqa: E402
    DifferenceInMeans,
)
from skxperiments.inference import NeymanCI  # noqa: E402
from skxperiments.pipeline import ExperimentPipeline, PipelineResult  # noqa: E402
from skxperiments.reporting import ExperimentReport  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assignment(
    n_treated: int = 20, n_control: int = 20, p: float = 0.5, seed: int = 0
) -> CRDAssignment:
    rng = np.random.default_rng(seed)
    n = n_treated + n_control
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "treatment": [1] * n_treated + [0] * n_control,
        }
    )
    return CRDAssignment(
        data=df, treatment_col="treatment", design=CRD(p=p, seed=seed)
    )


def _pipeline_result(
    n_treated: int = 20, n_control: int = 20, p: float = 0.5
) -> PipelineResult:
    pipe = ExperimentPipeline(NeymanCI(DifferenceInMeans(outcome_col="y")))
    return pipe.run(_assignment(n_treated, n_control, p))


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


class TestExperimentReportCreation:
    def test_rejects_non_pipeline_result(self) -> None:
        with pytest.raises(InvalidDesignError, match="PipelineResult"):
            ExperimentReport("not a pipeline result")  # type: ignore[arg-type]

    def test_rejects_non_string_title(self) -> None:
        with pytest.raises(InvalidDesignError, match="title"):
            ExperimentReport(_pipeline_result(), title=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. to_html
# ---------------------------------------------------------------------------


class TestExperimentReportHtml:
    def test_basic_html_structure(self) -> None:
        html = ExperimentReport(_pipeline_result(), title="My A/B Test").to_html()
        assert "<!doctype html>" in html
        assert "My A/B Test" in html
        assert "Summary" in html and "Diagnostics" in html
        assert "NeymanCI" in html  # inference name in the results table
        # Embedded plots: effect + SRM.
        assert "<img" in html
        assert "data:image/png;base64," in html

    def test_include_plots_false_omits_images(self) -> None:
        html = ExperimentReport(
            _pipeline_result(), include_plots=False
        ).to_html()
        assert "<img" not in html
        assert "Summary" in html

    def test_flagged_diagnostics_rendered(self) -> None:
        # 90/10 split under p=0.5 trips the SRM flag.
        html = ExperimentReport(_pipeline_result(90, 10, p=0.5)).to_html()
        assert "FLAG" in html
        assert "Sample Ratio Mismatch" in html

    def test_multi_effect_uses_interaction_plot(self) -> None:
        pr = PipelineResult(
            results=Results(effects={("A",): 0.5, ("B",): 0.3}),
            diagnostics=DiagnosticsReport(),
            diagnostic_results={},
        )
        html = ExperimentReport(pr).to_html()
        assert "<img" in html

    def test_null_distribution_plot(self) -> None:
        rng = np.random.default_rng(0)
        pr = PipelineResult(
            results=Results(
                ate=0.5, extra={"null_distribution": rng.normal(size=200)}
            ),
            diagnostics=DiagnosticsReport(),
            diagnostic_results={},
        )
        html = ExperimentReport(pr).to_html()
        assert "<img" in html


# ---------------------------------------------------------------------------
# 3. save
# ---------------------------------------------------------------------------


class TestExperimentReportSave:
    def test_save_writes_html_file(self, tmp_path) -> None:
        path = tmp_path / "report.html"
        ExperimentReport(_pipeline_result()).save(str(path))
        content = path.read_text(encoding="utf-8")
        assert content.startswith("<!doctype html>")
        assert "</html>" in content


# ---------------------------------------------------------------------------
# 4. Optional-dependency guard
# ---------------------------------------------------------------------------


class TestExperimentReportGuard:
    def test_plots_require_matplotlib(self, monkeypatch) -> None:
        """With plots requested but matplotlib absent, to_html raises."""
        pr = _pipeline_result()
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        with pytest.raises(ImportError, match="skxperiments\\[viz\\]"):
            ExperimentReport(pr, include_plots=True).to_html()

    def test_no_plots_works_without_matplotlib(self, monkeypatch) -> None:
        """include_plots=False renders without touching matplotlib."""
        pr = _pipeline_result()
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        html = ExperimentReport(pr, include_plots=False).to_html()
        assert "Summary" in html
        assert "<img" not in html
