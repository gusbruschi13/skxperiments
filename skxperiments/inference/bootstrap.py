"""Bootstrap confidence intervals for superpopulation inference.

Implements ``BootstrapCI``: wraps any scalar estimator and approximates
the sampling distribution of its ATE by resampling units **with
replacement within each treatment arm** (within each block-by-arm
stratum for blocked designs), then forms a percentile or BCa confidence
interval.

Unlike ``RandomizationTest`` (which permutes the treatment assignment
under Fisher's sharp null) and ``NeymanCI`` (finite-population variance),
the bootstrap treats the observed arms as samples drawn from
superpopulations and ignores the randomization mechanism. It is therefore
the library's explicit superpopulation method.

References
----------
Efron, B. (1987). Better bootstrap confidence intervals. Journal of the
    American Statistical Association, 82(397), 171-185.
Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
    Chapman & Hall.
"""

import numpy as np
from scipy.stats import norm

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
)
from skxperiments.core.base import BaseEstimator, BaseInference
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)
from skxperiments.core.results import Results


class BootstrapCI(BaseInference):
    """Bootstrap confidence intervals for superpopulation inference.

    Wraps a scalar estimator and approximates the sampling distribution of
    its ATE by resampling units with replacement, preserving the size of
    each treatment arm (and, for blocked designs, of each block-by-arm
    stratum). Two interval methods are supported:

    - ``"percentile"``: the ``alpha/2`` and ``1 - alpha/2`` empirical
      quantiles of the bootstrap distribution.
    - ``"bca"`` (default): bias-corrected and accelerated interval
      (Efron 1987), correcting for median bias (``z0``) and for a
      non-constant standard error (acceleration ``a``, estimated by the
      leave-one-out jackknife).

    The estimator is treated as a black box: each resample is turned back
    into an ``Assignment`` of the original type and the estimator is
    refitted on it. Any estimator producing a scalar ``Results.ate`` is
    therefore supported (``DifferenceInMeans``, ``BlockedDifferenceInMeans``,
    ``LinEstimator``, ``CUPED``).

    Parameters
    ----------
    estimator : BaseEstimator
        Causal estimator producing a scalar ATE (``Results.ate``). Need
        not be pre-fitted: ``fit`` refits it on the supplied assignment
        and on every resample. Estimators producing multi-effect
        ``Results`` (``Results.ate is None``) are rejected.
    method : {"bca", "percentile"}, optional
        Interval method, by default ``"bca"``.
    n_resamples : int, optional
        Number of bootstrap resamples, by default 10_000. Must be a
        positive integer.
    alpha : float, optional
        Significance level; the interval is two-sided at level
        ``1 - alpha``, by default 0.05.
    seed : int or None, optional
        Random seed for reproducibility. The same ``seed`` produces the
        same ``bootstrap_distribution_`` and interval. By default None.

    Attributes
    ----------
    assignment_ : CRDAssignment or BlockedAssignment
        The assignment passed to ``fit``.
    observed_statistic_ : float
        ATE on the original assignment, captured before resampling.
    bootstrap_distribution_ : np.ndarray
        Array of resampled ATEs. Length equals ``n_resamples``.
    se_ : float
        Bootstrap standard error: ``std(bootstrap_distribution_, ddof=1)``.
    ci_ : tuple of float
        The ``(1 - alpha)`` confidence interval bounds.
    p_value_ : float
        Approximate two-sided bootstrap p-value (achieved significance
        level) for ``H0: ATE = 0``.

    Notes
    -----
    **Superpopulation scope.** ``BootstrapCI`` always reports
    ``inference_mode="superpopulation"`` in ``Results.extra``, regardless
    of any ``inference_mode`` the wrapped estimator may have written: the
    bootstrap is a superpopulation procedure by construction.

    **Resampling scheme.** Resampling is stratified by arm (CRD) or by
    block-by-arm (blocked), preserving each stratum's size and so the
    fixed-margin structure of the design. Each block-by-arm stratum must
    contain at least 2 units; matched-pair blocked designs (one treated
    and one control per block) are not supported by the within-stratum
    bootstrap in v1 and raise ``InsufficientDataError``.

    **Rerandomization.** A ``CRDAssignment`` from ``ReRandomizedCRD`` is
    accepted, but the bootstrap ignores the Mahalanobis acceptance
    criterion (it does not re-randomize). The resulting interval may be
    conservative; for inference that respects rerandomization use
    ``RandomizationTest``.

    **BCa cost.** The acceleration estimate adds a leave-one-out jackknife
    of ``n_obs`` estimator refits on top of the ``n_resamples`` bootstrap
    refits. The bias-correction is undefined when no (or every) resampled
    estimate falls below the observed estimate; in that degenerate case
    ``fit`` raises ``InvalidDesignError`` suggesting ``method="percentile"``.

    Examples
    --------
    >>> from skxperiments.design.crd import CRD
    >>> from skxperiments.estimators.difference_in_means import (
    ...     DifferenceInMeans,
    ... )
    >>> from skxperiments.inference.bootstrap import BootstrapCI
    >>> design = CRD(p=0.5, seed=42)
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> dim = DifferenceInMeans(outcome_col="y")
    >>> ci = BootstrapCI(estimator=dim, method="bca", seed=0)
    >>> result = ci.fit(assignment).estimate()  # doctest: +SKIP
    >>> result.ci  # doctest: +SKIP
    """

    _VALID_METHODS = ("bca", "percentile")

    def __init__(
        self,
        estimator: BaseEstimator,
        method: str = "bca",
        n_resamples: int = 10_000,
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> None:
        if not isinstance(estimator, BaseEstimator):
            raise InvalidDesignError(
                f"estimator must be an instance of BaseEstimator, got "
                f"{type(estimator).__name__}."
            )

        if method not in self._VALID_METHODS:
            raise InvalidDesignError(
                f"method must be one of {self._VALID_METHODS}, got "
                f"{method!r}."
            )

        if not isinstance(n_resamples, int) or isinstance(n_resamples, bool):
            raise InvalidDesignError(
                f"n_resamples must be an integer, got "
                f"{type(n_resamples).__name__}."
            )
        if n_resamples <= 0:
            raise InvalidDesignError(
                f"n_resamples must be > 0, got {n_resamples}."
            )

        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise InvalidDesignError(
                f"alpha must be a float in (0, 1), got "
                f"{type(alpha).__name__}."
            )
        if not (0.0 < alpha < 1.0):
            raise InvalidDesignError(
                f"alpha must be in (0, 1), got {alpha}."
            )

        self.estimator = estimator
        self.method = method
        self.n_resamples = n_resamples
        self.alpha = alpha
        self.seed = seed

    def fit(
        self,
        assignment: CRDAssignment | BlockedAssignment,
    ) -> "BootstrapCI":
        """Run the bootstrap and compute the confidence interval.

        Parameters
        ----------
        assignment : CRDAssignment or BlockedAssignment
            Observed assignment. ``FactorialAssignment`` is rejected with
            ``DesignEstimatorMismatch``.

        Returns
        -------
        BootstrapCI
            Returns self.

        Raises
        ------
        DesignEstimatorMismatch
            If ``assignment`` is not a ``CRDAssignment`` or
            ``BlockedAssignment``.
        InvalidDesignError
            If the estimator returns a multi-effect ``Results``
            (``Results.ate is None``), or if the BCa bias-correction is
            undefined (use ``method="percentile"``).
        InsufficientDataError
            If any arm (CRD) or block-by-arm stratum (blocked) has fewer
            than 2 units.
        """
        self._validate_assignment_type(
            assignment, (CRDAssignment, BlockedAssignment)
        )

        # Refit on the original assignment for the observed statistic.
        self.estimator.fit(assignment)
        base_results = self.estimator.estimate()

        if base_results.ate is None:
            raise InvalidDesignError(
                "BootstrapCI supports only estimators producing a scalar "
                "ATE (Results.ate). The supplied estimator "
                f"({type(self.estimator).__name__}) produced a multi-effect "
                "Results (Results.effects). Multi-effect support is planned "
                "for v2."
            )

        observed = float(base_results.ate)

        # Capture metadata BEFORE the resampling loop, so estimate() does
        # not depend on the post-loop state of self.estimator.
        self._n_obs = base_results.n_obs
        self._n_treated = base_results.n_treated
        self._n_control = base_results.n_control
        self._estimator_name = base_results.estimator_name
        self._design_name = base_results.design_name

        strata = self._get_strata(assignment)

        rng = np.random.default_rng(self.seed)
        distribution = np.empty(self.n_resamples, dtype=float)
        for b in range(self.n_resamples):
            idx = np.concatenate(
                [rng.choice(s, size=len(s), replace=True) for s in strata]
            )
            distribution[b] = self._compute_statistic(assignment, idx)

        se = float(np.std(distribution, ddof=1))

        if self.method == "bca":
            ci_lower, ci_upper, z0, accel = self._bca_interval(
                assignment, distribution, observed
            )
            self._bias_correction = z0
            self._acceleration = accel
        else:
            lo, hi = np.quantile(
                distribution, [self.alpha / 2.0, 1.0 - self.alpha / 2.0]
            )
            ci_lower, ci_upper = float(lo), float(hi)
            self._bias_correction = None
            self._acceleration = None

        # Two-sided achieved significance level for H0: ATE = 0.
        p_less = float(np.mean(distribution <= 0.0))
        p_greater = float(np.mean(distribution >= 0.0))
        p_value = min(1.0, 2.0 * min(p_less, p_greater))

        self.assignment_: CRDAssignment | BlockedAssignment = assignment
        self.observed_statistic_: float = observed
        self.bootstrap_distribution_: np.ndarray = distribution
        self.se_: float = se
        self.ci_: tuple[float, float] = (ci_lower, ci_upper)
        self.p_value_: float = p_value

        return self

    def _get_strata(
        self,
        assignment: CRDAssignment | BlockedAssignment,
    ) -> list[np.ndarray]:
        """Return iloc positions for each resampling stratum.

        CRD: [treated, control]. Blocked: one stratum per block-by-arm.
        Each stratum must contain at least 2 units.
        """
        if isinstance(assignment, BlockedAssignment):
            data = assignment.data_
            treat = data[assignment.treatment_col_].values
            blocks = data[assignment.block_col_].values
            strata: list[np.ndarray] = []
            for block_val in assignment.block_sizes_:
                for arm in (1, 0):
                    idx = np.where((blocks == block_val) & (treat == arm))[0]
                    arm_name = "treated" if arm == 1 else "control"
                    if len(idx) < 2:
                        raise InsufficientDataError(
                            context=(
                                f"BootstrapCI resampling ({arm_name} arm in "
                                f"block '{block_val}')"
                            ),
                            minimum=2,
                            received=len(idx),
                        )
                    strata.append(idx)
            return strata

        treated = assignment.treated_ids()
        control = assignment.control_ids()
        if len(treated) < 2:
            raise InsufficientDataError(
                context="BootstrapCI resampling (treated arm)",
                minimum=2,
                received=len(treated),
            )
        if len(control) < 2:
            raise InsufficientDataError(
                context="BootstrapCI resampling (control arm)",
                minimum=2,
                received=len(control),
            )
        return [treated, control]

    def _compute_statistic(
        self,
        assignment: CRDAssignment | BlockedAssignment,
        idx: np.ndarray,
    ) -> float:
        """Refit the estimator on the units selected by ``idx`` (iloc)."""
        df_rs = assignment.data_.iloc[idx].reset_index(drop=True)

        if isinstance(assignment, BlockedAssignment):
            block_sizes = (
                df_rs.groupby(assignment.block_col_).size().to_dict()
            )
            resampled: CRDAssignment | BlockedAssignment = BlockedAssignment(
                data=df_rs,
                treatment_col=assignment.treatment_col_,
                design=None,
                block_col=assignment.block_col_,
                block_sizes=block_sizes,
            )
        else:
            resampled = CRDAssignment(
                data=df_rs,
                treatment_col=assignment.treatment_col_,
                design=None,
            )

        self.estimator.fit(resampled)
        return float(self.estimator.estimate().ate)

    def _bca_interval(
        self,
        assignment: CRDAssignment | BlockedAssignment,
        distribution: np.ndarray,
        observed: float,
    ) -> tuple[float, float, float, float]:
        """Compute the BCa interval bounds, bias-correction, and acceleration."""
        # Bias-correction z0 from the fraction of resamples below observed.
        prop_less = float(np.mean(distribution < observed))
        if prop_less <= 0.0 or prop_less >= 1.0:
            raise InvalidDesignError(
                "BootstrapCI: the BCa bias-correction is undefined because "
                "no (or every) bootstrap estimate falls below the observed "
                "estimate. This usually means a degenerate or tiny sample. "
                "Use method='percentile' instead."
            )
        z0 = float(norm.ppf(prop_less))

        # Acceleration via the leave-one-out jackknife.
        n = assignment.n_units_
        all_idx = np.arange(n)
        jack = np.empty(n, dtype=float)
        for i in range(n):
            jack[i] = self._compute_statistic(
                assignment, np.delete(all_idx, i)
            )
        jack_mean = jack.mean()
        diffs = jack_mean - jack
        denom = 6.0 * float(np.sum(diffs**2)) ** 1.5
        accel = 0.0 if denom == 0.0 else float(np.sum(diffs**3)) / denom

        z_lo = norm.ppf(self.alpha / 2.0)
        z_hi = norm.ppf(1.0 - self.alpha / 2.0)

        def _adjust(z: float) -> float:
            denom_z = 1.0 - accel * (z0 + z)
            return float(norm.cdf(z0 + (z0 + z) / denom_z))

        p_lo = _adjust(z_lo)
        p_hi = _adjust(z_hi)

        if not (np.isfinite(p_lo) and np.isfinite(p_hi)):
            raise InvalidDesignError(
                "BootstrapCI: the BCa adjustment produced non-finite "
                "percentiles (acceleration drove the denominator to zero). "
                "Use method='percentile' instead."
            )

        lo, hi = np.quantile(distribution, sorted([p_lo, p_hi]))
        return float(lo), float(hi), z0, accel

    def estimate(self) -> Results:
        """Return a Results object with the ATE, SE, CI, and p-value.

        Returns
        -------
        Results
            Results with ``ate`` (observed), ``se`` (bootstrap standard
            error), ``ci`` (percentile or BCa bounds), ``p_value``
            (two-sided achieved significance level), ``alpha``,
            ``inference_name="BootstrapCI"``, and ``extra`` containing
            ``method``, ``n_resamples``, ``bootstrap_distribution``,
            ``inference_mode="superpopulation"`` and, for BCa,
            ``bias_correction`` and ``acceleration``.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        self._check_is_fitted()

        extra: dict = {
            "method": self.method,
            "n_resamples": self.n_resamples,
            "bootstrap_distribution": self.bootstrap_distribution_,
            "inference_mode": "superpopulation",
        }
        if self.method == "bca":
            extra["bias_correction"] = self._bias_correction
            extra["acceleration"] = self._acceleration

        return Results(
            ate=self.observed_statistic_,
            se=self.se_,
            ci=self.ci_,
            p_value=self.p_value_,
            alpha=self.alpha,
            n_obs=self._n_obs,
            n_treated=self._n_treated,
            n_control=self._n_control,
            estimator_name=self._estimator_name,
            design_name=self._design_name,
            inference_name=type(self).__name__,
            extra=extra,
        )
