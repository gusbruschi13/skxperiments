"""Integration test: BlockedCRD -> BlockedAssignment ->
BlockedDifferenceInMeans -> Results.
"""

import numpy as np
import pandas as pd

from skxperiments.core.results import Results
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.estimators.blocked_difference_in_means import (
    BlockedDifferenceInMeans,
)


class TestBlockedCRDtoBlockedDifferenceInMeans:
    """End-to-end pipeline: BlockedCRD -> assignment -> estimator -> Results."""

    def test_full_pipeline_produces_results(self) -> None:
        """Full pipeline must produce a Results with all expected
        metadata populated and recover the injected ATE within
        statistical tolerance.
        """
        rng = np.random.default_rng(42)
        n_per_block = {"A": 250, "B": 250}
        block_labels: list[str] = []
        for b, n in n_per_block.items():
            block_labels.extend([b] * n)
        n_total = len(block_labels)

        df = pd.DataFrame(
            {
                "x": rng.normal(size=n_total),
                "block": block_labels,
                "y": rng.normal(size=n_total),
            }
        )

        design = BlockedCRD(block_col="block", p=0.5, seed=42)
        assignment = design.randomize(df)

        # Inject true ATE = 1.0 into the realized treated group.
        true_ate = 1.0
        treated = assignment.treated_ids()
        assignment.data_.iloc[
            treated, assignment.data_.columns.get_loc("y")
        ] += true_ate

        estimator = BlockedDifferenceInMeans(outcome_col="y")
        result = estimator.fit(assignment).estimate()

        assert isinstance(result, Results)

        # ate is a float, not None.
        assert result.ate is not None
        assert isinstance(result.ate, float)

        # Estimator and design names auto-populated.
        assert result.estimator_name == "BlockedDifferenceInMeans"
        assert result.design_name == "BlockedCRD"

        # Inference name is None at this stage.
        assert result.inference_name is None

        # Sample-size metadata correct and consistent.
        assert result.n_obs == n_total
        assert result.n_treated == assignment.n_treated_
        assert result.n_control == assignment.n_control_
        assert result.n_treated + result.n_control == result.n_obs

        # Recovered ATE within statistical tolerance.
        # SE under N(0,1) noise, n=500, p=0.5: ~ sqrt(2/250) ~ 0.089.
        # Tolerance 0.3 ~ 3.4 SE -> false-failure prob ~ 0.07%.
        assert abs(result.ate - true_ate) < 0.3