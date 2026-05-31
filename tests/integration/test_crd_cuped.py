"""Integration test: CRD -> CRDAssignment -> CUPED -> Results."""

import numpy as np
import pandas as pd
import pytest

from skxperiments.core.results import Results
from skxperiments.design.crd import CRD
from skxperiments.estimators.cuped import CUPED


class TestCRDtoCUPED:
    """End-to-end pipeline: CRD -> assignment -> CUPED -> Results."""

    def test_full_pipeline_produces_results(self) -> None:
        """Full pipeline must produce a Results with all expected
        metadata populated and recover the injected ATE within tighter
        statistical tolerance than DifferenceInMeans (CUPED reduces
        variance with a correlated pre-experiment covariate).
        """
        rng = np.random.default_rng(42)
        n = 500
        true_ate = 1.0
        correlation = 0.7

        x_pre = rng.normal(size=n)
        epsilon = rng.normal(size=n)
        y_baseline = (
            correlation * x_pre
            + np.sqrt(max(1.0 - correlation**2, 0.0)) * epsilon
        )

        df = pd.DataFrame(
            {
                "y_pre": x_pre,
                "y": y_baseline,
            }
        )

        design = CRD(p=0.5, seed=42)
        assignment = design.randomize(df)

        # Inject true ATE into the realized treated group.
        treated = assignment.treated_ids()
        assignment.data_.iloc[
            treated, assignment.data_.columns.get_loc("y")
        ] += true_ate

        estimator = CUPED(outcome_col="y", pre_experiment_col="y_pre")
        result = estimator.fit(assignment).estimate()

        assert isinstance(result, Results)

        # ate is float, not None.
        assert result.ate is not None
        assert isinstance(result.ate, float)

        # Estimator and design names auto-populated.
        assert result.estimator_name == "CUPED"
        assert result.design_name == "CRD"

        # theta and correlation propagated to extra.
        assert result.extra is not None
        assert "theta" in result.extra
        assert "correlation" in result.extra
        assert result.extra["theta"] == pytest.approx(0.7, abs=0.15)
        assert result.extra["correlation"] == pytest.approx(0.7, abs=0.15)

        # inference_name is None.
        assert result.inference_name is None

        # Sample-size metadata correct.
        assert result.n_obs == n
        assert result.n_treated == assignment.n_treated_
        assert result.n_control == assignment.n_control_
        assert result.n_treated + result.n_control == result.n_obs

        # Tighter tolerance than DIM: CUPED with rho=0.7 reduces
        # variance ~ 1 - 0.49 = 51%. SE_DIM ~ sqrt(2/250) ~ 0.089;
        # SE_CUPED ~ sqrt(0.49) * 0.089 ~ 0.062. Tolerance 0.15 is
        # ~2.4 SE_CUPED, false-failure prob < 2%.
        assert abs(result.ate - true_ate) < 0.15