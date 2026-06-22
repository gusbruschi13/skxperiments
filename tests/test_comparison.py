"""Tests for skxperiments.pipeline.ExperimentComparison."""

import pytest

from skxperiments.core.base import DiagnosticsReport
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results
from skxperiments.pipeline import (
    ComparisonResult,
    ExperimentComparison,
    PipelineResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    ate: float,
    p_value: float,
    se: float | None = None,
    ci: tuple[float, float] | None = None,
) -> Results:
    return Results(ate=ate, p_value=p_value, se=se, ci=ci)


def _pipeline_result(ate: float, p_value: float) -> PipelineResult:
    return PipelineResult(
        results=_result(ate, p_value),
        diagnostics=DiagnosticsReport(),
        diagnostic_results={},
    )


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


class TestExperimentComparisonCreation:
    """Tests for ExperimentComparison.__init__ validations."""

    def test_defaults(self) -> None:
        """Default correction is holm; alpha 0.05."""
        comp = ExperimentComparison()
        assert comp.correction == "holm"
        assert comp.alpha == 0.05

    def test_custom_parameters(self) -> None:
        """Custom correction and alpha are stored."""
        comp = ExperimentComparison(correction="bonferroni", alpha=0.10)
        assert comp.correction == "bonferroni"
        assert comp.alpha == 0.10

    def test_rejects_invalid_method(self) -> None:
        """An invalid correction method is rejected (via MTC)."""
        with pytest.raises(InvalidDesignError, match="method"):
            ExperimentComparison(correction="invalid")


# ---------------------------------------------------------------------------
# 2. Run behavior
# ---------------------------------------------------------------------------


class TestExperimentComparisonRun:
    """Tests for ExperimentComparison.run."""

    def test_bonferroni_known_values(self) -> None:
        """Bonferroni multiplies p-values by the family size and clips."""
        results = {
            "a": _result(1.0, 0.01),
            "b": _result(0.5, 0.04),
            "c": _result(0.2, 0.50),
        }
        comp = ExperimentComparison(correction="bonferroni", alpha=0.05)
        result = comp.run(results)

        assert result.corrected_results["a"].p_value == pytest.approx(0.03)
        assert result.corrected_results["b"].p_value == pytest.approx(0.12)
        assert result.corrected_results["c"].p_value == pytest.approx(1.0)
        assert result.significant == ["a"]

    def test_accepts_pipeline_results_and_results(self) -> None:
        """A mix of PipelineResult and Results entries is accepted."""
        results = {
            "exp1": _pipeline_result(1.0, 0.01),
            "exp2": _result(0.5, 0.40),
        }
        result = ExperimentComparison().run(results)
        assert set(result.corrected_results) == {"exp1", "exp2"}

    def test_order_preserved(self) -> None:
        """The table preserves the dict insertion order."""
        results = {
            "z": _result(1.0, 0.2),
            "a": _result(1.0, 0.3),
            "m": _result(1.0, 0.4),
        }
        result = ExperimentComparison().run(results)
        assert list(result.table["experiment"]) == ["z", "a", "m"]

    def test_correction_does_not_decrease_pvalues(self) -> None:
        """Corrected p-values are >= original p-values."""
        results = {
            "a": _result(1.0, 0.01),
            "b": _result(0.5, 0.02),
            "c": _result(0.2, 0.03),
        }
        result = ExperimentComparison(correction="holm").run(results)
        for row in result.table.itertuples():
            assert row.p_value_corrected >= row.p_value

    def test_table_carries_ci_when_present(self) -> None:
        """CI columns are populated from the original Results."""
        results = {"a": _result(1.0, 0.01, se=0.5, ci=(0.5, 1.5))}
        result = ExperimentComparison().run(results)
        row = result.table.iloc[0]
        assert row["ci_lower"] == pytest.approx(0.5)
        assert row["ci_upper"] == pytest.approx(1.5)
        assert row["se"] == pytest.approx(0.5)

    def test_rejects_multi_effect(self) -> None:
        """A multi-effect Results is rejected, naming the experiment."""
        multi = Results(
            effects={("A",): 1.0}, p_value={("A",): 0.01}
        )
        with pytest.raises(InvalidDesignError, match="exp_x"):
            ExperimentComparison().run({"exp_x": multi})

    def test_rejects_missing_pvalue(self) -> None:
        """A Results without a p-value is rejected."""
        with pytest.raises(InvalidDesignError, match="p_value"):
            ExperimentComparison().run({"exp_x": Results(ate=1.0)})

    def test_rejects_empty_dict(self) -> None:
        """An empty dict is rejected."""
        with pytest.raises(InvalidDesignError, match="empty"):
            ExperimentComparison().run({})

    def test_rejects_non_dict(self) -> None:
        """A non-dict input is rejected."""
        with pytest.raises(InvalidDesignError, match="dict"):
            ExperimentComparison().run([_result(1.0, 0.01)])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Result object
# ---------------------------------------------------------------------------


class TestComparisonResult:
    """Tests for the ComparisonResult surface."""

    def _result_obj(self) -> ComparisonResult:
        results = {
            "a": _result(1.0, 0.01),
            "b": _result(0.5, 0.50),
        }
        return ExperimentComparison(correction="bonferroni").run(results)

    def test_is_comparison_result(self) -> None:
        """run() returns a ComparisonResult."""
        assert isinstance(self._result_obj(), ComparisonResult)

    def test_to_dataframe_is_copy(self) -> None:
        """to_dataframe returns a copy that does not alias the table."""
        result = self._result_obj()
        df = result.to_dataframe()
        df.loc[0, "ate"] = 999.0
        assert result.table.iloc[0]["ate"] == pytest.approx(1.0)

    def test_to_dict_keys(self) -> None:
        """to_dict exposes the expected summary fields."""
        d = self._result_obj().to_dict()
        assert set(d) == {"correction", "alpha", "n_experiments", "significant"}
        assert d["n_experiments"] == 2

    def test_summary_returns_self(self, capsys) -> None:
        """summary() prints and returns self."""
        result = self._result_obj()
        assert result.summary() is result
        assert "Experiment Comparison" in capsys.readouterr().out

    def test_significant_property(self) -> None:
        """significant lists experiments significant after correction."""
        result = self._result_obj()
        assert result.significant == ["a"]
