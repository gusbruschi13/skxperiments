"""Integration test: FactorialDesign -> FactorialAssignment ->
FactorialEstimator -> Results.
"""

import numpy as np
import pandas as pd

from skxperiments.core.results import Results
from skxperiments.design.factorial import FactorialDesign
from skxperiments.estimators.factorial_estimator import FactorialEstimator


class TestFactorialDesigntoFactorialEstimator:
    """End-to-end pipeline: FactorialDesign -> FactorialEstimator -> Results."""

    def test_full_pipeline_produces_multi_effect_results(self) -> None:
        """Full pipeline must produce a multi-effect Results with all
        expected metadata populated and recover injected effects within
        statistical tolerance.
        """
        n_per_cell = 200
        n_total = n_per_cell * 4  # K=2 -> 4 cells
        df = pd.DataFrame(
            {
                "x": np.zeros(n_total),
                "y": np.zeros(n_total),
            }
        )

        design = FactorialDesign(
            factors=["A", "B"], n_per_cell=n_per_cell, seed=42
        )
        assignment = design.randomize(df)

        # Inject true effects: tau_A = 1.0, tau_B = 0.5; no interaction.
        injected = {("A",): 1.0, ("B",): 0.5}
        factor_cols = ["A", "B"]
        for subset, magnitude in injected.items():
            subset_indices = [factor_cols.index(f) for f in subset]
            for unit_iloc in range(n_total):
                cell_idx = int(assignment.data_.iloc[unit_iloc]["_cell"])
                sign = 1
                for j in subset_indices:
                    x_j = (cell_idx >> j) & 1
                    sign *= 2 * x_j - 1
                assignment.data_.iat[
                    unit_iloc,
                    assignment.data_.columns.get_loc("y"),
                ] += sign * magnitude

        estimator = FactorialEstimator(outcome_col="y")
        result = estimator.fit(assignment).estimate()

        assert isinstance(result, Results)

        # Multi-effect mode: ate is None, effects populated.
        assert result.ate is None
        assert result.effects is not None

        # Exactly 2^K - 1 = 3 effects for K=2.
        assert set(result.effects.keys()) == {
            ("A",),
            ("B",),
            ("A", "B"),
        }

        # Auto-populated metadata.
        assert result.estimator_name == "FactorialEstimator"
        assert result.design_name == "FactorialDesign"
        assert result.n_obs == n_total
        assert result.n_treated is None
        assert result.n_control is None
        assert result.inference_name is None

        # Recovered effects within tolerance (no noise injected -> exact).
        assert result.effects[("A",)] == 1.0 + pytest.approx(0.0, abs=1e-6)
        assert result.effects[("B",)] == 0.5 + pytest.approx(0.0, abs=1e-6)
        assert result.effects[("A", "B")] == pytest.approx(0.0, abs=1e-6)


# pytest is imported lazily for the module-level approx helper.
import pytest  # noqa: E402