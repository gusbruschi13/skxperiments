"""Neyman variance-based confidence intervals for finite-population inference.

Implements ``NeymanCI``: wraps a fitted scalar estimator
(``DifferenceInMeans`` or ``BlockedDifferenceInMeans``), computes the
Neyman conservative variance for CRD or the stratified variance for
blocked designs, and constructs a two-sided Wald confidence interval and
p-value under the normal approximation.

References
----------
Neyman, J. (1923/1990). On the application of probability theory to
    agricultural experiments. Statistical Science, 5(4), 465-472.
Imbens, G. W., & Rubin, D. B. (2015). Causal Inference for Statistics,
    Social, and Biomedical Sciences. Cambridge University Press.
    Chapters 6 (CRD) and 9 (stratified/blocked).
"""

import numpy as np
from scipy.stats import norm

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
)
from skxperiments.core.base import BaseEstimator, BaseInference
from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InsufficientDataError,
    InvalidDesignError,
)
from skxperiments.core.results import Results
from skxperiments.estimators.blocked_difference_in_means import (
    BlockedDifferenceInMeans,
)
from skxperiments.estimators.difference_in_means import DifferenceInMeans

_ACCEPTED_ESTIMATORS = (DifferenceInMeans, BlockedDifferenceInMeans)


class NeymanCI(BaseInference):
    """Neyman conservative confidence intervals for finite-population inference.

    Wraps a scalar estimator and computes a two-sided Wald CI using
    Neyman's variance estimator, dispatched by the assignment type:

    **CRD** (Neyman 1923):

        V_hat = s_t^2 / n_t + s_c^2 / n_c

    where ``s_t^2`` and ``s_c^2`` are the sample variances (``ddof=1``) of
    the outcome in the treated and control arms.

    **Blocked** (stratified, consistent with the size-weighted ATE of
    ``BlockedDifferenceInMeans``):

        V_hat = sum_b (N_b / N)^2 * V_hat_b,
        V_hat_b = s_{t,b}^2 / n_{t,b} + s_{c,b}^2 / n_{c,b}

    The confidence interval is ``ATE_hat +/- z_{1 - alpha/2} * SE`` and the
    p-value is the two-sided Wald test ``z = ATE_hat / SE``,
    ``p = 2 * (1 - Phi(|z|))``, both under the normal approximation.

    Parameters
    ----------
    estimator : DifferenceInMeans or BlockedDifferenceInMeans
        Causal estimator producing a scalar ATE (``Results.ate``). Need
        not be pre-fitted: ``fit`` refits it on the supplied assignment.
        Any other type raises ``DesignEstimatorMismatch`` at construction.
        ``CUPED`` and ``LinEstimator`` support is planned for a future
        sub-phase (see ROADMAP).
    alpha : float, optional
        Significance level for the confidence interval, by default 0.05.
        The CI is two-sided: a ``(1 - alpha) * 100%`` interval.

    Attributes
    ----------
    assignment_ : CRDAssignment or BlockedAssignment
        The assignment passed to ``fit``.
    variance_ : float
        Estimated Neyman variance ``V_hat``.

    Notes
    -----
    **Finite-population scope.** ``NeymanCI`` v1 targets finite-population
    inference. The Neyman variance formula is numerically identical under
    the superpopulation interpretation, so the restriction is a scope
    choice rather than a mathematical limitation: for superpopulation
    inference use ``BootstrapCI`` (Phase 4.4). If a wrapped estimator
    reports ``inference_mode="superpopulation"`` in its ``Results.extra``,
    ``fit`` raises ``InvalidDesignError`` redirecting to ``BootstrapCI``.

    **Conservative variance.** Neyman's estimator is conservative
    (upward-biased) when individual treatment effects vary across units,
    and exact when the effect is constant. Empirical CI coverage is
    therefore ``>= (1 - alpha)``.

    **Rerandomization.** A ``CRDAssignment`` produced by ``ReRandomizedCRD``
    is accepted: the variance formula is the same as for plain CRD.
    Rerandomization improves covariate balance; Neyman's estimator remains
    valid (and conservative) under it.

    **Estimator compatibility.** ``NeymanCI`` accepts both ``CRDAssignment``
    and ``BlockedAssignment``, but each wrapped estimator enforces its own
    assignment contract in ``fit``: pairing ``DifferenceInMeans`` with a
    ``BlockedAssignment`` (or vice versa) raises ``DesignEstimatorMismatch``
    from the estimator. Only the matching pairs
    (``DifferenceInMeans`` + CRD, ``BlockedDifferenceInMeans`` + blocked)
    proceed to variance computation.

    Examples
    --------
    >>> from skxperiments.design.crd import CRD
    >>> from skxperiments.estimators.difference_in_means import (
    ...     DifferenceInMeans,
    ... )
    >>> from skxperiments.inference import NeymanCI
    >>> design = CRD(p=0.5, seed=42)
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> dim = DifferenceInMeans(outcome_col="y")
    >>> ci = NeymanCI(estimator=dim, alpha=0.05)
    >>> result = ci.fit(assignment).estimate()  # doctest: +SKIP
    >>> result.ci  # doctest: +SKIP
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        alpha: float = 0.05,
    ) -> None:
        if not isinstance(estimator, _ACCEPTED_ESTIMATORS):
            accepted_names = " or ".join(
                t.__name__ for t in _ACCEPTED_ESTIMATORS
            )
            raise DesignEstimatorMismatch(
                estimator_name=type(self).__name__,
                received_type=type(estimator).__name__,
                expected_type=accepted_names,
                suggestion=(
                    f"{accepted_names}. CUPED and LinEstimator support is "
                    f"planned for a future sub-phase (see ROADMAP)."
                ),
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
        self.alpha = alpha

    def fit(
        self,
        assignment: CRDAssignment | BlockedAssignment,
    ) -> "NeymanCI":
        """Refit the estimator and compute the Neyman variance.

        Parameters
        ----------
        assignment : CRDAssignment or BlockedAssignment
            Observed assignment. ``FactorialAssignment`` is rejected with
            ``DesignEstimatorMismatch``.

        Returns
        -------
        NeymanCI
            Returns self.

        Raises
        ------
        DesignEstimatorMismatch
            If ``assignment`` is not a ``CRDAssignment`` or
            ``BlockedAssignment``.
        InvalidDesignError
            If the estimator returns a multi-effect ``Results``
            (``Results.ate is None``; v1 supports only scalar estimands),
            or if it reports ``inference_mode="superpopulation"`` (use
            ``BootstrapCI`` instead).
        InsufficientDataError
            If any arm (CRD) or any arm within a block (blocked) has fewer
            than 2 observations, so the sample variance is undefined.
        """
        self._validate_assignment_type(
            assignment, (CRDAssignment, BlockedAssignment)
        )

        # Refit on the original assignment to obtain the point estimate.
        # Any prior fit state of self.estimator is discarded.
        self.estimator.fit(assignment)
        base_results = self.estimator.estimate()

        if base_results.ate is None:
            raise InvalidDesignError(
                "NeymanCI v1 supports only estimators producing a scalar "
                "ATE (Results.ate). The supplied estimator "
                f"({type(self.estimator).__name__}) produced a multi-effect "
                "Results (Results.effects). Multi-effect support is planned "
                "for v2."
            )

        # inference_mode defaults to finite_population; only LinEstimator
        # currently writes this key, but the guard is enforced for any
        # whitelisted estimator that may emit it.
        if base_results.extra is not None:
            inference_mode = base_results.extra.get(
                "inference_mode", "finite_population"
            )
        else:
            inference_mode = "finite_population"

        if inference_mode == "superpopulation":
            raise InvalidDesignError(
                "NeymanCI v1 targets finite-population inference. The "
                "supplied estimator reported "
                "inference_mode='superpopulation'. The Neyman variance "
                "formula is identical under both interpretations; this "
                "restriction is a scope choice, not a mathematical "
                "limitation. For superpopulation inference use BootstrapCI "
                "(Phase 4.4)."
            )

        # Capture metadata BEFORE variance computation, so estimate() does
        # not depend on any post-fit state of self.estimator.
        self._n_obs = base_results.n_obs
        self._n_treated = base_results.n_treated
        self._n_control = base_results.n_control
        self._estimator_name = base_results.estimator_name
        self._design_name = base_results.design_name
        self._ate = float(base_results.ate)
        self._inference_mode = inference_mode

        # Dispatch the variance computation by assignment type.
        if isinstance(assignment, BlockedAssignment):
            variance = self._neyman_variance_blocked(assignment)
            self._variance_type = "neyman_stratified"
        else:
            variance = self._neyman_variance_crd(assignment)
            self._variance_type = "neyman"

        self.assignment_: CRDAssignment | BlockedAssignment = assignment
        self.variance_: float = float(variance)

        return self

    def _neyman_variance_crd(self, assignment: CRDAssignment) -> float:
        """Compute the Neyman conservative variance for CRD."""
        y = assignment.data_[self.estimator.outcome_col].values
        y_t = y[assignment.treated_ids()]
        y_c = y[assignment.control_ids()]

        n_t = len(y_t)
        n_c = len(y_c)

        if n_t < 2:
            raise InsufficientDataError(
                context="NeymanCI variance (treated arm)",
                minimum=2,
                received=n_t,
            )
        if n_c < 2:
            raise InsufficientDataError(
                context="NeymanCI variance (control arm)",
                minimum=2,
                received=n_c,
            )

        s2_t = float(np.var(y_t, ddof=1))
        s2_c = float(np.var(y_c, ddof=1))

        return s2_t / n_t + s2_c / n_c

    def _neyman_variance_blocked(self, assignment: BlockedAssignment) -> float:
        """Compute the stratified Neyman variance for a blocked design."""
        data = assignment.data_
        y_col = self.estimator.outcome_col
        treat_col = assignment.treatment_col_
        block_col = assignment.block_col_
        n_total = assignment.n_units_

        variance_total = 0.0

        for block_val, n_b in assignment.block_sizes_.items():
            block_data = data.loc[data[block_col] == block_val]
            block_treat = block_data[treat_col].values
            block_y = block_data[y_col].values

            y_t = block_y[block_treat == 1]
            y_c = block_y[block_treat == 0]

            n_t_b = len(y_t)
            n_c_b = len(y_c)

            if n_t_b < 2:
                raise InsufficientDataError(
                    context=(
                        f"NeymanCI variance (treated arm in block "
                        f"'{block_val}')"
                    ),
                    minimum=2,
                    received=n_t_b,
                )
            if n_c_b < 2:
                raise InsufficientDataError(
                    context=(
                        f"NeymanCI variance (control arm in block "
                        f"'{block_val}')"
                    ),
                    minimum=2,
                    received=n_c_b,
                )

            s2_t_b = float(np.var(y_t, ddof=1))
            s2_c_b = float(np.var(y_c, ddof=1))

            v_b = s2_t_b / n_t_b + s2_c_b / n_c_b
            weight = n_b / n_total
            variance_total += weight**2 * v_b

        return variance_total

    def estimate(self) -> Results:
        """Return a Results object with the ATE, SE, CI, and p-value.

        Returns
        -------
        Results
            Results with:

            - ``ate`` set to the observed point estimate;
            - ``se`` set to ``sqrt(V_hat)``;
            - ``ci`` set to the two-sided ``(1 - alpha) * 100%`` Wald CI;
            - ``p_value`` set to the two-sided Wald p-value;
            - ``alpha`` set to ``self.alpha``;
            - ``inference_name="NeymanCI"``;
            - ``extra`` containing ``variance_type`` and ``inference_mode``.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        InvalidDesignError
            If the standard error is zero (degenerate case: constant
            outcomes within each arm).
        """
        self._check_is_fitted()

        se = float(np.sqrt(self.variance_))

        if se == 0.0:
            raise InvalidDesignError(
                "NeymanCI: the estimated standard error is zero, indicating "
                "a degenerate dataset (constant outcomes within each arm). "
                "Cannot compute a confidence interval or p-value."
            )

        z_crit = float(norm.ppf(1.0 - self.alpha / 2.0))
        ci_lower = self._ate - z_crit * se
        ci_upper = self._ate + z_crit * se

        z_stat = self._ate / se
        p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
        # Clamp to [0, 1] for numerical safety.
        p_value = max(0.0, min(1.0, p_value))

        return Results(
            ate=self._ate,
            se=se,
            ci=(ci_lower, ci_upper),
            p_value=p_value,
            alpha=self.alpha,
            n_obs=self._n_obs,
            n_treated=self._n_treated,
            n_control=self._n_control,
            estimator_name=self._estimator_name,
            design_name=self._design_name,
            inference_name=type(self).__name__,
            extra={
                "variance_type": self._variance_type,
                "inference_mode": self._inference_mode,
            },
        )
