"""Tests for skxperiments.diagnostics.srm.SRMTest and SRMResult."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.assignment import (
    CRDAssignment,
    FactorialAssignment,
)
from skxperiments.core.base import DiagnosticsReport
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.design.crd import CRD
from skxperiments.diagnostics import SRMResult, SRMTest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crd_with_counts(
    n_treated: int,
    n_control: int,
    p: float | None = 0.5,
    seed: int = 0,
) -> CRDAssignment:
    """Build a CRDAssignment with explicit arm counts.

    The treatment column is hand-built (not drawn) so the observed split
    can deviate from the design's intended ``p``, which is exactly what
    SRM must detect. ``design`` is a real ``CRD`` so ``design_.p`` is
    available; pass ``p=None`` to mimic an ``n_treated``-based design.
    """
    n = n_treated + n_control
    treatment = [1] * n_treated + [0] * n_control
    df = pd.DataFrame({"x": np.arange(n), "treatment": treatment})
    if p is None:
        design = CRD(n_treated=max(1, n_treated), seed=seed)
    else:
        design = CRD(p=p, seed=seed)
    return CRDAssignment(
        data=df, treatment_col="treatment", design=design, seed=seed
    )


def _factorial_with_counts(counts: list[int]) -> FactorialAssignment:
    """Build a 2x2 FactorialAssignment with the given per-cell counts."""
    rows = []
    for cell, count in enumerate(counts):
        a = cell % 2
        b = (cell // 2) % 2
        for _ in range(count):
            rows.append({"A": a, "B": b, "_cell": cell, "x": 0.0})
    df = pd.DataFrame(rows)
    cell_sizes = {cell: int(count) for cell, count in enumerate(counts)}
    return FactorialAssignment(
        data=df,
        design=None,
        factor_cols=["A", "B"],
        cell_sizes=cell_sizes,
        seed=0,
    )


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


class TestSRMTestCreation:
    """Tests for SRMTest.__init__ validations."""

    def test_default_threshold(self) -> None:
        """Default threshold is 0.001 and expected is None."""
        srm = SRMTest()
        assert srm.threshold == 0.001
        assert srm.expected is None

    def test_custom_threshold_and_expected(self) -> None:
        """Custom threshold and expected proportion are stored."""
        srm = SRMTest(threshold=0.05, expected=0.5)
        assert srm.threshold == 0.05
        assert srm.expected == 0.5

    def test_rejects_threshold_out_of_range(self) -> None:
        """threshold must be strictly in (0, 1)."""
        with pytest.raises(InvalidDesignError, match="threshold"):
            SRMTest(threshold=0.0)
        with pytest.raises(InvalidDesignError, match="threshold"):
            SRMTest(threshold=1.0)

    def test_rejects_threshold_bool(self) -> None:
        """threshold=True is rejected."""
        with pytest.raises(InvalidDesignError, match="threshold"):
            SRMTest(threshold=True)  # type: ignore[arg-type]

    def test_rejects_expected_wrong_type(self) -> None:
        """expected must be None, a number, or a dict."""
        with pytest.raises(InvalidDesignError, match="expected"):
            SRMTest(expected="0.5")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Two-arm: derive expected from the design
# ---------------------------------------------------------------------------


class TestSRMTestTwoArmFromDesign:
    """SRM on two-arm designs with expected inferred from design.p."""

    def test_balanced_not_flagged(self) -> None:
        """A 50/50 split under p=0.5 is not flagged and has high p-value."""
        assignment = _crd_with_counts(50, 50, p=0.5)
        result = SRMTest().run(assignment)
        assert result.flagged is False
        assert result.p_value > 0.5
        assert result.statistic == pytest.approx(0.0)
        assert result.dof == 1

    def test_imbalanced_flagged(self) -> None:
        """A 90/10 split under p=0.5 is flagged."""
        assignment = _crd_with_counts(90, 10, p=0.5)
        result = SRMTest().run(assignment)
        assert result.flagged is True
        assert result.p_value < 0.001

    def test_fresh_randomization_not_flagged(self) -> None:
        """A pristine assignment from randomize() never flags."""
        df = pd.DataFrame({"x": np.arange(200)})
        assignment = CRD(p=0.5, seed=42).randomize(df)
        result = SRMTest().run(assignment)
        assert result.flagged is False

    def test_observed_and_expected_counts(self) -> None:
        """observed/expected dicts carry the right counts."""
        assignment = _crd_with_counts(60, 40, p=0.5)
        result = SRMTest().run(assignment)
        assert result.observed == {"control": 40, "treated": 60}
        assert result.expected["control"] == pytest.approx(50.0)
        assert result.expected["treated"] == pytest.approx(50.0)

    def test_unequal_intended_proportion(self) -> None:
        """Expected counts follow a non-0.5 intended proportion."""
        # Intended 70% treated; observed 70/30 should match expectation.
        assignment = _crd_with_counts(70, 30, p=0.7)
        result = SRMTest().run(assignment)
        assert result.expected["treated"] == pytest.approx(70.0)
        assert result.expected["control"] == pytest.approx(30.0)
        assert result.flagged is False


# ---------------------------------------------------------------------------
# 3. Two-arm: explicit expected
# ---------------------------------------------------------------------------


class TestSRMTestExplicitExpected:
    """SRM with an explicitly supplied expected allocation."""

    def test_expected_float(self) -> None:
        """A float expected sets the treated proportion."""
        assignment = _crd_with_counts(90, 10, p=None)
        result = SRMTest(expected=0.5).run(assignment)
        assert result.expected["treated"] == pytest.approx(50.0)
        assert result.flagged is True

    def test_expected_dict_normalized(self) -> None:
        """A dict expected is normalized to proportions."""
        assignment = _crd_with_counts(50, 50, p=None)
        # Unnormalized weights {control: 1, treated: 1} -> 50/50.
        result = SRMTest(expected={"control": 1, "treated": 1}).run(assignment)
        assert result.expected["control"] == pytest.approx(50.0)
        assert result.flagged is False

    def test_n_treated_design_without_expected_raises(self) -> None:
        """An n_treated-based design with no expected raises."""
        df = pd.DataFrame({"x": np.arange(20)})
        assignment = CRD(n_treated=5, seed=0).randomize(df)
        with pytest.raises(InvalidDesignError, match="infer"):
            SRMTest().run(assignment)

    def test_expected_dict_wrong_keys_raises(self) -> None:
        """A dict whose keys don't match the groups raises."""
        assignment = _crd_with_counts(50, 50, p=None)
        with pytest.raises(InvalidDesignError, match="match"):
            SRMTest(expected={"a": 0.5, "b": 0.5}).run(assignment)

    def test_expected_proportion_out_of_range_raises(self) -> None:
        """A float expected outside (0, 1) raises."""
        assignment = _crd_with_counts(50, 50, p=None)
        with pytest.raises(InvalidDesignError):
            SRMTest(expected=1.5).run(assignment)


# ---------------------------------------------------------------------------
# 4. Factorial
# ---------------------------------------------------------------------------


class TestSRMTestFactorial:
    """SRM on factorial designs (uniform cells by default)."""

    def test_balanced_cells_not_flagged(self) -> None:
        """Equal cell sizes are not flagged; dof = n_cells - 1."""
        assignment = _factorial_with_counts([25, 25, 25, 25])
        result = SRMTest().run(assignment)
        assert result.flagged is False
        assert result.dof == 3
        assert result.statistic == pytest.approx(0.0)

    def test_imbalanced_cells_flagged(self) -> None:
        """A strongly skewed cell allocation is flagged."""
        assignment = _factorial_with_counts([70, 10, 10, 10])
        result = SRMTest().run(assignment)
        assert result.flagged is True

    def test_factorial_expected_counts_uniform(self) -> None:
        """Default expected is uniform across cells."""
        assignment = _factorial_with_counts([20, 30, 25, 25])
        result = SRMTest().run(assignment)
        for cell in range(4):
            assert result.expected[cell] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# 5. Result object
# ---------------------------------------------------------------------------


class TestSRMResult:
    """Tests for the SRMResult dataclass surface."""

    def test_summary_returns_self(self, capsys) -> None:
        """summary() prints and returns self."""
        result = SRMTest().run(_crd_with_counts(50, 50, p=0.5))
        returned = result.summary()
        assert returned is result
        captured = capsys.readouterr()
        assert "SRM Test" in captured.out

    def test_to_dict(self) -> None:
        """to_dict() exposes all fields."""
        result = SRMTest().run(_crd_with_counts(50, 50, p=0.5))
        assert isinstance(result, SRMResult)
        d = result.to_dict()
        assert set(d) == {
            "statistic",
            "p_value",
            "dof",
            "observed",
            "expected",
            "threshold",
            "flagged",
        }

    def test_to_diagnostics_report_flagged(self) -> None:
        """A flagged result yields a DiagnosticsReport with one flag."""
        result = SRMTest().run(_crd_with_counts(90, 10, p=0.5))
        report = result.to_diagnostics_report()
        assert isinstance(report, DiagnosticsReport)
        assert len(report.flags) == 1
        assert "Sample Ratio Mismatch" in report.flags[0]

    def test_to_diagnostics_report_not_flagged(self) -> None:
        """A clean result yields a DiagnosticsReport with no flags."""
        result = SRMTest().run(_crd_with_counts(50, 50, p=0.5))
        report = result.to_diagnostics_report()
        assert report.flags == []


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------


class TestSRMTestValidation:
    """Tests for run() input validation."""

    def test_rejects_unsupported_assignment(self) -> None:
        """A non-assignment object is rejected."""
        with pytest.raises(InvalidDesignError, match="supports"):
            SRMTest().run("not an assignment")  # type: ignore[arg-type]
