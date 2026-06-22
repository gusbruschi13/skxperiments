"""A/A test diagnostic: calibration of a design + inference pipeline.

An A/A test repeatedly re-randomizes a design on a *fixed* dataset and
runs an inference procedure each time. Because the treatment is re-drawn
while the outcome is held fixed, the true effect is zero by construction,
so a well-calibrated pipeline should reject the null at rate ``alpha`` and
produce uniformly distributed p-values. ``AATest`` measures both: the
false-positive rate (compared to ``alpha`` with an exact binomial test)
and the uniformity of the p-values (Kolmogorov-Smirnov).

Cost note
---------
Each simulation runs the wrapped inference once. With a resampling-based
inference (``RandomizationTest``, ``BootstrapCI``) this is a nested loop:
``O(n_simulations x n_resamples)`` estimator fits. For routine calibration
prefer the analytic ``NeymanCI``; reserve the resampling inferences for
small ``n_simulations`` (and expect a slow run).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kstest

from skxperiments.core.base import BaseDesign, BaseInference, DiagnosticsReport
from skxperiments.core.exceptions import InvalidDesignError


@dataclass(frozen=True, eq=False)
class AAResult:
    """Result of an A/A test.

    Attributes
    ----------
    n_simulations : int
        Number of re-randomizations performed.
    alpha : float
        Nominal significance level the pipeline was run at.
    meta_threshold : float
        Threshold for the binomial false-positive-rate test below which
        the pipeline is flagged as miscalibrated.
    false_positive_rate : float
        Fraction of simulations with ``p_value < alpha``.
    n_false_positives : int
        Number of simulations with ``p_value < alpha``.
    fp_test_pvalue : float
        Two-sided exact binomial-test p-value for ``n_false_positives``
        out of ``n_simulations`` under the null rate ``alpha``.
    ks_statistic : float
        Kolmogorov-Smirnov statistic of the p-values against Uniform(0, 1).
    ks_pvalue : float
        KS test p-value (secondary, full-distribution calibration signal).
    p_values : np.ndarray
        The ``n_simulations`` p-values, in simulation order.
    flagged : bool
        True if ``fp_test_pvalue < meta_threshold`` — the false-positive
        rate is incompatible with ``alpha``.
    """

    n_simulations: int
    alpha: float
    meta_threshold: float
    false_positive_rate: float
    n_false_positives: int
    fp_test_pvalue: float
    ks_statistic: float
    ks_pvalue: float
    p_values: np.ndarray
    flagged: bool

    def summary(self) -> "AAResult":
        """Print a formatted summary table and return self."""
        status = "FLAGGED — miscalibrated" if self.flagged else "OK"
        lines = ["A/A Test", "--------"]
        lines.append(f"simulations        {self.n_simulations}")
        lines.append(f"alpha              {self.alpha}")
        lines.append(
            f"false-positive     {self.false_positive_rate:.4f} "
            f"({self.n_false_positives}/{self.n_simulations})"
        )
        lines.append(f"FP binomial p      {self.fp_test_pvalue:.4f}")
        lines.append(f"KS uniform p       {self.ks_pvalue:.4f}")
        lines.append(f"status             {status}")
        print("\n".join(lines))
        return self

    def to_dict(self) -> dict:
        """Return the scalar summary fields (excludes the p_values array)."""
        return {
            "n_simulations": self.n_simulations,
            "alpha": self.alpha,
            "meta_threshold": self.meta_threshold,
            "false_positive_rate": self.false_positive_rate,
            "n_false_positives": self.n_false_positives,
            "fp_test_pvalue": self.fp_test_pvalue,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "flagged": self.flagged,
        }

    def to_diagnostics_report(self) -> DiagnosticsReport:
        """Convert to a ``DiagnosticsReport`` for pipeline aggregation.

        A miscalibrated false-positive rate is a flag; non-uniform
        p-values (KS below the meta-threshold) are a secondary warning.
        """
        report = DiagnosticsReport()
        if self.flagged:
            report.flags.append(
                f"A/A false-positive rate miscalibrated: "
                f"{self.false_positive_rate:.3f} vs alpha={self.alpha} "
                f"(binomial p={self.fp_test_pvalue:.2e} < "
                f"{self.meta_threshold})."
            )
        if self.ks_pvalue < self.meta_threshold:
            report.warnings.append(
                f"A/A p-values deviate from uniform "
                f"(KS p={self.ks_pvalue:.2e})."
            )
        return report


class AATest:
    """A/A test for a design + inference pipeline.

    Re-randomizes ``design`` on a fixed DataFrame ``n_simulations`` times
    and runs ``inference`` on each draw, collecting the p-values. Because
    the treatment is re-drawn while the outcome is fixed, the true effect
    is zero, so the false-positive rate should equal ``alpha`` and the
    p-values should be uniform.

    Parameters
    ----------
    design : BaseDesign
        The design whose randomization is being calibrated.
    inference : BaseInference
        A configured inference object (which already wraps an estimator),
        e.g. ``NeymanCI(DifferenceInMeans("y"))`` or
        ``RandomizationTest(...)``. Must produce a scalar ``p_value``.
    n_simulations : int, optional
        Number of re-randomizations, by default 1000.
    alpha : float, optional
        Nominal significance level, by default 0.05.
    meta_threshold : float, optional
        Threshold for the binomial false-positive-rate test, by default
        0.001. The pipeline is flagged when the binomial-test p-value
        falls below it.
    seed : int or None, optional
        Random seed for reproducibility, by default None.

    Notes
    -----
    The wrapped ``inference`` is refitted on each draw; its ``seed`` (if
    any) is varied per simulation and restored afterward. The outcome
    column is whatever the wrapped estimator resolves against the data;
    the supplied DataFrame must contain it (and any covariates) and must
    not contain the design's treatment column.
    """

    def __init__(
        self,
        design: BaseDesign,
        inference: BaseInference,
        n_simulations: int = 1000,
        alpha: float = 0.05,
        meta_threshold: float = 0.001,
        seed: int | None = None,
    ) -> None:
        if not isinstance(design, BaseDesign):
            raise InvalidDesignError(
                f"design must be an instance of BaseDesign, got "
                f"{type(design).__name__}."
            )
        if not isinstance(inference, BaseInference):
            raise InvalidDesignError(
                f"inference must be an instance of BaseInference, got "
                f"{type(inference).__name__}."
            )

        if not isinstance(n_simulations, int) or isinstance(
            n_simulations, bool
        ):
            raise InvalidDesignError(
                f"n_simulations must be an integer, got "
                f"{type(n_simulations).__name__}."
            )
        if n_simulations <= 0:
            raise InvalidDesignError(
                f"n_simulations must be > 0, got {n_simulations}."
            )

        for name, value in (("alpha", alpha), ("meta_threshold", meta_threshold)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidDesignError(
                    f"{name} must be a float in (0, 1), got "
                    f"{type(value).__name__}."
                )
            if not (0.0 < value < 1.0):
                raise InvalidDesignError(
                    f"{name} must be in (0, 1), got {value}."
                )

        self.design = design
        self.inference = inference
        self.n_simulations = n_simulations
        self.alpha = alpha
        self.meta_threshold = meta_threshold
        self.seed = seed

    def run(self, df: pd.DataFrame) -> AAResult:
        """Run the A/A test on a fixed DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Data with the outcome (and any covariates) but without the
            design's treatment column.

        Returns
        -------
        AAResult

        Raises
        ------
        InvalidDesignError
            If the wrapped inference does not produce a scalar p-value.
        """
        base = self.design.randomize(df)

        rng = np.random.default_rng(self.seed)
        sim_seeds = rng.integers(0, 2**32, size=self.n_simulations)
        inf_seeds = rng.integers(0, 2**32, size=self.n_simulations)

        has_seed = hasattr(self.inference, "seed")
        original_inf_seed = getattr(self.inference, "seed", None)

        p_values = np.empty(self.n_simulations, dtype=float)
        try:
            for i in range(self.n_simulations):
                assignment = base.draw(seed=int(sim_seeds[i]))
                if has_seed:
                    self.inference.seed = int(inf_seeds[i])
                self.inference.fit(assignment)
                result = self.inference.estimate()
                if result.p_value is None:
                    raise InvalidDesignError(
                        "AATest requires an inference that produces a scalar "
                        "p_value (e.g., RandomizationTest, NeymanCI, "
                        "BootstrapCI). The supplied "
                        f"{type(self.inference).__name__} returned "
                        "p_value=None."
                    )
                p_values[i] = float(result.p_value)
        finally:
            if has_seed:
                self.inference.seed = original_inf_seed

        n_fp = int(np.sum(p_values < self.alpha))
        fp_rate = n_fp / self.n_simulations
        fp_test_pvalue = float(
            binomtest(n_fp, self.n_simulations, self.alpha).pvalue
        )
        ks_statistic, ks_pvalue = kstest(p_values, "uniform")

        return AAResult(
            n_simulations=self.n_simulations,
            alpha=self.alpha,
            meta_threshold=self.meta_threshold,
            false_positive_rate=fp_rate,
            n_false_positives=n_fp,
            fp_test_pvalue=fp_test_pvalue,
            ks_statistic=float(ks_statistic),
            ks_pvalue=float(ks_pvalue),
            p_values=p_values,
            flagged=bool(fp_test_pvalue < self.meta_threshold),
        )
