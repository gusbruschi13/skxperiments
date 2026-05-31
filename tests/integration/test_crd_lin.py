"""Integration test: CRD -> CRDAssignment -> LinEstimator -> Results."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.results import Results
from skxperiments.design.crd import CRD
from skxperiments.estimators.lin_estimator import LinEstimator


class TestCRDtoLinEstimator:
    """End-to-end pipeline: CRD -> assignment -> Lin -> Results."""

    def test_full_pipeline_produces_results(self) -> None:
        """Full pipeline must produce a Results with all expected
        metadata populated and recover the injected ATE within
        statistical tolerance tighter than DifferenceInMeans (Lin
        reduces variance with a strong covariate).
        """
        rng = np.random.default_rng(42)
        n = 500
        true_ate = 1.0
        cov_effect = 2.0
        x = rng.normal(size=n)
        epsilon = rng.normal(size=n)

        df = pd.DataFrame(
            {
                "x": x,
                "y": cov_effect * x + epsilon,
            }
        )

        design = CRD(p=0.5, seed=42)
        assignment = design.randomize(df)

        # Inject true ATE into the realized treated group.
        treated = assignment.treated_ids()
        assignment.data_.iloc[
            treated, assignment.data_.columns.get_loc("y")
        ] += true_ate

        estimator = LinEstimator(outcome_col="y", covariates=["x"])
        result = estimator.fit(assignment).estimate()

        assert isinstance(result, Results)

        # ate is float, not None.
        assert result.ate is not None
        assert isinstance(result.ate, float)

        # Estimator and design names auto-populated.
        assert result.estimator_name == "LinEstimator"
        assert result.design_name == "CRD"

        # inference_mode propagated to extra.
        assert result.extra is not None
        assert result.extra["inference_mode"] == "finite_population"

        # inference_name is None (no inference applied).
        assert result.inference_name is None

        # Sample-size metadata correct.
        assert result.n_obs == n
        assert result.n_treated == assignment.n_treated_
        assert result.n_control == assignment.n_control_
        assert result.n_treated + result.n_control == result.n_obs

        # Tighter tolerance than DIM: Lin with strong covariate
        # (cov_effect=2.0, theoretical R^2 ~ 0.8) reduces SE by
        # roughly sqrt(1 - R^2) ~ 0.45. With n=500 and N(0,1) noise,
        # SE_DIM ~ 0.089; SE_Lin ~ 0.040. Tolerance 0.2 is ~5 SE_Lin,
        # false-failure prob essentially zero.
        assert abs(result.ate - true_ate) < 0.2