"""Tests for skxperiments.diagnostics.balance_report."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
    FactorialAssignment,
)
from skxperiments.core.base import DiagnosticsReport
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.diagnostics import BalanceReport, BalanceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crd(columns: dict, treatment: list[int]) -> CRDAssignment:
    """Build a CRDAssignment from explicit covariate columns and treatment."""
    df = pd.DataFrame({**columns, "treatment": treatment})
    return CRDAssignment(data=df, treatment_col="treatment", design=None)


def _blocked(
    columns: dict, treatment: list[int], block: list[int]
) -> BlockedAssignment:
    """Build a BlockedAssignment from explicit columns."""
    df = pd.DataFrame({**columns, "block": block, "treatment": treatment})
    block_sizes = df.groupby("block").size().to_dict()
    return BlockedAssignment(
        data=df,
        treatment_col="treatment",
        design=None,
        block_col="block",
        block_sizes=block_sizes,
    )


def _factorial() -> FactorialAssignment:
    """Build a minimal 2x2 FactorialAssignment."""
    df = pd.DataFrame(
        {
            "A": [0, 1, 0, 1],
            "B": [0, 0, 1, 1],
            "_cell": [0, 1, 2, 3],
            "x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    return FactorialAssignment(
        data=df,
        design=None,
        factor_cols=["A", "B"],
        cell_sizes={0: 1, 1: 1, 2: 1, 3: 1},
        seed=0,
    )


# An imbalanced assignment with a hand-computable SMD of exactly 1.0:
# treated x = [1, 2, 3] (mean 2, var 1); control x = [0, 1, 2] (mean 1, var 1);
# std_pooled = sqrt((1 + 1) / 2) = 1 -> SMD = (2 - 1) / 1 = 1.0.
_IMBALANCED_X = [1.0, 2.0, 3.0, 0.0, 1.0, 2.0]
# A balanced covariate: identical values across arms -> SMD = 0.
_BALANCED_X = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
_TREATMENT = [1, 1, 1, 0, 0, 0]


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


class TestBalanceReportCreation:
    """Tests for BalanceReport.__init__ validations."""

    def test_defaults(self) -> None:
        """Default threshold is 0.1 and covariates is None."""
        report = BalanceReport()
        assert report.threshold == 0.1
        assert report.covariates is None

    def test_custom_parameters(self) -> None:
        """Custom covariates and threshold are stored."""
        report = BalanceReport(covariates=["x1", "x2"], threshold=0.25)
        assert report.covariates == ["x1", "x2"]
        assert report.threshold == 0.25

    def test_rejects_non_positive_threshold(self) -> None:
        """threshold must be > 0."""
        with pytest.raises(InvalidDesignError, match="threshold"):
            BalanceReport(threshold=0.0)
        with pytest.raises(InvalidDesignError, match="threshold"):
            BalanceReport(threshold=-0.1)

    def test_rejects_threshold_bool(self) -> None:
        """threshold=True is rejected."""
        with pytest.raises(InvalidDesignError, match="threshold"):
            BalanceReport(threshold=True)  # type: ignore[arg-type]

    def test_rejects_bad_covariates_type(self) -> None:
        """covariates must be None or a list of strings."""
        with pytest.raises(InvalidDesignError, match="covariates"):
            BalanceReport(covariates="x")  # type: ignore[arg-type]
        with pytest.raises(InvalidDesignError, match="covariates"):
            BalanceReport(covariates=[1, 2])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 2. Run behavior
# ---------------------------------------------------------------------------


class TestBalanceReportRun:
    """Tests for BalanceReport.run."""

    def test_balanced_not_flagged(self) -> None:
        """A balanced covariate is not flagged."""
        assignment = _crd({"x": _BALANCED_X}, _TREATMENT)
        result = BalanceReport().run(assignment)
        assert result.flagged is False
        assert result.imbalanced == []
        assert result.max_abs_smd == pytest.approx(0.0)

    def test_imbalanced_flagged(self) -> None:
        """An imbalanced covariate (SMD = 1.0) is flagged."""
        assignment = _crd({"x": _IMBALANCED_X}, _TREATMENT)
        result = BalanceReport().run(assignment)
        assert result.flagged is True
        assert result.imbalanced == ["x"]
        assert result.smd["x"] == pytest.approx(1.0)
        assert result.max_abs_smd == pytest.approx(1.0)

    def test_threshold_controls_flag(self) -> None:
        """A high threshold suppresses the flag for the same SMD."""
        assignment = _crd({"x": _IMBALANCED_X}, _TREATMENT)
        result = BalanceReport(threshold=2.0).run(assignment)
        assert result.flagged is False
        assert result.imbalanced == []

    def test_covariate_subset(self) -> None:
        """Only the requested covariates appear in the table."""
        assignment = _crd(
            {"x1": _BALANCED_X, "x2": _IMBALANCED_X}, _TREATMENT
        )
        result = BalanceReport(covariates=["x2"]).run(assignment)
        assert list(result.table["covariate"]) == ["x2"]
        assert result.imbalanced == ["x2"]

    def test_multiple_covariates(self) -> None:
        """Only the imbalanced covariate is flagged among several."""
        assignment = _crd(
            {"x1": _BALANCED_X, "x2": _IMBALANCED_X}, _TREATMENT
        )
        result = BalanceReport().run(assignment)
        assert result.flagged is True
        assert result.imbalanced == ["x2"]

    def test_constant_covariate_is_nan_not_flagged(self) -> None:
        """A constant covariate has NaN SMD and is not flagged."""
        assignment = _crd({"x": [5.0] * 6}, _TREATMENT)
        result = BalanceReport().run(assignment)
        assert result.constant_covariates == ["x"]
        assert result.flagged is False
        assert np.isnan(result.smd["x"])

    def test_blocked_assignment(self) -> None:
        """BalanceReport accepts a BlockedAssignment."""
        assignment = _blocked(
            {"x": _IMBALANCED_X},
            _TREATMENT,
            block=[0, 0, 1, 0, 0, 1],
        )
        result = BalanceReport().run(assignment)
        assert isinstance(result, BalanceResult)
        assert "x" in result.smd

    def test_rejects_factorial(self) -> None:
        """FactorialAssignment is rejected."""
        with pytest.raises(InvalidDesignError, match="two-arm|factorial"):
            BalanceReport().run(_factorial())

    def test_propagates_nan_covariate_error(self) -> None:
        """A covariate with NaN propagates check_balance's error."""
        assignment = _crd(
            {"x": [1.0, np.nan, 3.0, 0.0, 1.0, 2.0]}, _TREATMENT
        )
        with pytest.raises(InvalidDesignError, match="NaN"):
            BalanceReport().run(assignment)


# ---------------------------------------------------------------------------
# 3. Result object
# ---------------------------------------------------------------------------


class TestBalanceResult:
    """Tests for the BalanceResult surface."""

    def _imbalanced_result(self) -> BalanceResult:
        return BalanceReport().run(_crd({"x": _IMBALANCED_X}, _TREATMENT))

    def test_to_dataframe_is_copy(self) -> None:
        """to_dataframe returns a copy that does not alias the table."""
        result = self._imbalanced_result()
        df = result.to_dataframe()
        df.loc[0, "smd"] = 999.0
        assert result.smd["x"] == pytest.approx(1.0)

    def test_to_dict_keys(self) -> None:
        """to_dict exposes the expected summary fields."""
        result = self._imbalanced_result()
        d = result.to_dict()
        assert set(d) == {
            "threshold",
            "flagged",
            "imbalanced",
            "constant_covariates",
            "max_abs_smd",
            "smd",
        }

    def test_summary_returns_self(self, capsys) -> None:
        """summary() prints and returns self."""
        result = self._imbalanced_result()
        assert result.summary() is result
        assert "Balance Report" in capsys.readouterr().out

    def test_to_diagnostics_report_flagged(self) -> None:
        """A flagged result yields a DiagnosticsReport with one flag."""
        report = self._imbalanced_result().to_diagnostics_report()
        assert isinstance(report, DiagnosticsReport)
        assert len(report.flags) == 1
        assert "imbalance" in report.flags[0].lower()

    def test_to_diagnostics_report_clean(self) -> None:
        """A balanced result yields no flags."""
        result = BalanceReport().run(_crd({"x": _BALANCED_X}, _TREATMENT))
        report = result.to_diagnostics_report()
        assert report.flags == []

    def test_to_diagnostics_report_constant_warns(self) -> None:
        """A constant covariate produces a warning, not a flag."""
        result = BalanceReport().run(_crd({"x": [5.0] * 6}, _TREATMENT))
        report = result.to_diagnostics_report()
        assert report.flags == []
        assert len(report.warnings) == 1
