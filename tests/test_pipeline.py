"""Tests for skxperiments.pipeline.ExperimentPipeline."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import CRDAssignment
from skxperiments.core.base import DiagnosticsReport
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results
from skxperiments.design.crd import CRD
from skxperiments.diagnostics import BalanceReport, SRMTest
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.inference import NeymanCI
from skxperiments.pipeline import ExperimentPipeline, PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crd_assignment(
    n_treated: int,
    n_control: int,
    p: float | None = 0.5,
    x: list[float] | None = None,
    seed: int = 0,
) -> CRDAssignment:
    """Build a CRDAssignment with a noise outcome and a covariate.

    Treatment is hand-built so the observed split can differ from the
    design's ``p`` (to exercise SRM). ``p=None`` mimics an n_treated-based
    design where SRM cannot infer the expected allocation.
    """
    n = n_treated + n_control
    rng = np.random.default_rng(seed)
    treatment = [1] * n_treated + [0] * n_control
    cov = x if x is not None else list(rng.normal(size=n))
    df = pd.DataFrame(
        {"x": cov, "y": rng.normal(size=n), "treatment": treatment}
    )
    if p is None:
        design = CRD(n_treated=max(1, n_treated), seed=seed)
    else:
        design = CRD(p=p, seed=seed)
    return CRDAssignment(
        data=df, treatment_col="treatment", design=design, seed=seed
    )


def _neyman() -> NeymanCI:
    return NeymanCI(estimator=DifferenceInMeans(outcome_col="y"))


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


class TestExperimentPipelineCreation:
    """Tests for ExperimentPipeline.__init__ validations."""

    def test_default_diagnostics_is_srm(self) -> None:
        """Default diagnostics is a single SRMTest; raise_on_flag False."""
        pipe = ExperimentPipeline(_neyman())
        assert len(pipe.diagnostics) == 1
        assert isinstance(pipe.diagnostics[0], SRMTest)
        assert pipe.raise_on_flag is False

    def test_custom_diagnostics(self) -> None:
        """Custom diagnostics list and raise_on_flag are stored."""
        diags = [SRMTest(), BalanceReport(covariates=["x"])]
        pipe = ExperimentPipeline(_neyman(), diagnostics=diags, raise_on_flag=True)
        assert pipe.diagnostics is diags
        assert pipe.raise_on_flag is True

    def test_empty_diagnostics_allowed(self) -> None:
        """An empty diagnostics list is accepted."""
        pipe = ExperimentPipeline(_neyman(), diagnostics=[])
        assert pipe.diagnostics == []

    def test_rejects_non_inference(self) -> None:
        """inference must be a BaseInference."""
        with pytest.raises(InvalidDesignError, match="BaseInference"):
            ExperimentPipeline("not an inference")  # type: ignore[arg-type]

    def test_rejects_non_list_diagnostics(self) -> None:
        """diagnostics must be a list or None."""
        with pytest.raises(InvalidDesignError, match="diagnostics"):
            ExperimentPipeline(_neyman(), diagnostics=SRMTest())  # type: ignore[arg-type]

    def test_rejects_diagnostic_without_run(self) -> None:
        """Each diagnostic must expose a run() method."""
        with pytest.raises(InvalidDesignError, match="run"):
            ExperimentPipeline(_neyman(), diagnostics=["not a diagnostic"])

    def test_rejects_non_bool_raise_on_flag(self) -> None:
        """raise_on_flag must be a bool."""
        with pytest.raises(InvalidDesignError, match="raise_on_flag"):
            ExperimentPipeline(_neyman(), raise_on_flag="yes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Run behavior
# ---------------------------------------------------------------------------


class TestExperimentPipelineRun:
    """Tests for ExperimentPipeline.run."""

    def test_basic_run(self) -> None:
        """A clean experiment runs SRM (no flag) and produces results."""
        assignment = _crd_assignment(50, 50, p=0.5)
        result = ExperimentPipeline(_neyman()).run(assignment)

        assert isinstance(result, PipelineResult)
        assert isinstance(result.results, Results)
        assert isinstance(result.diagnostics, DiagnosticsReport)
        assert "SRMTest" in result.diagnostic_results
        assert result.flagged is False

    def test_results_match_direct_inference(self) -> None:
        """Pipeline results equal running the inference directly."""
        assignment = _crd_assignment(40, 40, p=0.5)
        pipe_result = ExperimentPipeline(_neyman()).run(assignment)
        direct = _neyman().fit(assignment).estimate()
        assert pipe_result.results.ate == pytest.approx(direct.ate)
        assert pipe_result.results.se == pytest.approx(direct.se)

    def test_srm_flag_annotated_not_raised(self) -> None:
        """A flagged SRM is surfaced but estimation still proceeds."""
        assignment = _crd_assignment(90, 10, p=0.5)
        result = ExperimentPipeline(_neyman()).run(assignment)
        assert result.flagged is True
        assert any("Sample Ratio Mismatch" in f for f in result.diagnostics.flags)
        assert result.results.ate is not None  # estimation still ran

    def test_raise_on_flag_aborts(self) -> None:
        """raise_on_flag=True aborts before inference when flagged."""
        assignment = _crd_assignment(90, 10, p=0.5)
        pipe = ExperimentPipeline(_neyman(), raise_on_flag=True)
        with pytest.raises(InvalidDesignError, match="halted"):
            pipe.run(assignment)

    def test_diagnostic_best_effort_on_error(self) -> None:
        """A diagnostic that cannot run is recorded as a warning, not fatal."""
        # n_treated-based design -> SRMTest cannot infer expected allocation.
        assignment = _crd_assignment(50, 50, p=None)
        result = ExperimentPipeline(_neyman()).run(assignment)
        assert result.flagged is False
        assert any("SRMTest could not run" in w for w in result.diagnostics.warnings)
        assert "SRMTest" not in result.diagnostic_results
        assert result.results.ate is not None

    def test_custom_balance_diagnostic_flags(self) -> None:
        """A BalanceReport on an imbalanced covariate adds a flag."""
        # Imbalanced x (SMD = 1.0), balanced 3/3 split (no SRM flag).
        assignment = _crd_assignment(
            3, 3, p=0.5, x=[1.0, 2.0, 3.0, 0.0, 1.0, 2.0]
        )
        diags = [SRMTest(), BalanceReport(covariates=["x"])]
        result = ExperimentPipeline(_neyman(), diagnostics=diags).run(assignment)
        assert "BalanceReport" in result.diagnostic_results
        assert any("imbalance" in f.lower() for f in result.diagnostics.flags)

    def test_empty_diagnostics_runs_inference_only(self) -> None:
        """With no diagnostics, only inference runs."""
        assignment = _crd_assignment(40, 40, p=0.5)
        result = ExperimentPipeline(_neyman(), diagnostics=[]).run(assignment)
        assert result.diagnostic_results == {}
        assert result.diagnostics.flags == []
        assert result.results.ate is not None


# ---------------------------------------------------------------------------
# 3. Result object
# ---------------------------------------------------------------------------


class TestPipelineResult:
    """Tests for the PipelineResult surface."""

    def test_summary_returns_self(self, capsys) -> None:
        """summary() prints diagnostics + results and returns self."""
        assignment = _crd_assignment(40, 40, p=0.5)
        result = ExperimentPipeline(_neyman()).run(assignment)
        assert result.summary() is result
        out = capsys.readouterr().out
        assert "Experiment Pipeline" in out

    def test_flagged_property(self) -> None:
        """flagged reflects the presence of diagnostic flags."""
        clean = ExperimentPipeline(_neyman()).run(_crd_assignment(50, 50, p=0.5))
        flagged = ExperimentPipeline(_neyman()).run(_crd_assignment(90, 10, p=0.5))
        assert clean.flagged is False
        assert flagged.flagged is True
