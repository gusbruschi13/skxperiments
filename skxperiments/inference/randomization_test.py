"""Randomization-based inference via Fisher's sharp null hypothesis.

Implements ``RandomizationTest``: for each of ``n_permutations``
draws, generates a fresh ``Assignment`` via ``BaseAssignment.draw()``
(which respects the original randomization mechanism — including
rerandomization Mahalanobis criteria and within-block proportions —
because each Assignment subclass routes ``draw()`` through its
generating design), refits the estimator, and collects the resulting
ATE under the sharp null of no individual treatment effect.

The p-value is computed with the Phipson & Smyth (2010) continuity
correction, ``(1 + n_extreme) / (1 + n_permutations)``, which
guarantees a valid Monte Carlo p-value bounded away from zero.

References
----------
Fisher, R. A. (1935). The Design of Experiments. Oliver and Boyd.
Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never
    be zero: calculating exact P-values when permutations are randomly
    drawn. Statistical Applications in Genetics and Molecular Biology,
    9(1), Article 39.
"""

import numpy as np

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
)
from skxperiments.core.base import BaseEstimator, BaseInference
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results


class RandomizationTest(BaseInference):
    """Fisher randomization test via Monte Carlo permutations.

    Tests Fisher's sharp null hypothesis,
    ``H0: Y_i(1) = Y_i(0)`` for all i, by generating
    ``n_permutations`` fresh assignments from the same randomization
    mechanism that produced the observed assignment, refitting the
    estimator on each, and comparing the observed ATE to the resulting
    null distribution.

    The p-value uses the Phipson & Smyth (2010) continuity correction:

        p = (1 + n_extreme) / (1 + n_permutations)

    which guarantees a valid Monte Carlo p-value strictly greater than
    zero.

    Parameters
    ----------
    estimator : BaseEstimator
        Causal estimator producing a scalar ATE (``Results.ate``).
        Need not be pre-fitted: ``RandomizationTest.fit`` will refit
        it on the supplied assignment. Estimators producing
        multi-effect ``Results`` (e.g., ``FactorialEstimator``) are
        not supported in v1; ``fit`` will raise ``InvalidDesignError``
        if the estimator returns ``Results.ate is None``.
    n_permutations : int, optional
        Number of Monte Carlo permutations, by default 10_000. Must
        be a positive integer.
    alternative : {"two-sided", "greater", "less"}, optional
        Alternative hypothesis, by default ``"two-sided"``.

        - ``"two-sided"`` uses the criterion ``|T_perm| >= |T_obs|``.
          Valid under any null distribution shape but most natural
          when the null is approximately symmetric around zero (the
          typical case under CRD with Fisher's sharp null and
          balanced sample sizes). Under ``BlockedAssignment`` with
          highly unequal blocks or ``ReRandomizedCRD`` with a tight
          threshold, the null distribution may be slightly asymmetric;
          the absolute-value criterion remains valid and slightly
          conservative. For directional hypotheses with strong
          expected asymmetry, prefer ``"greater"`` or ``"less"``.
        - ``"greater"`` uses ``T_perm >= T_obs``.
        - ``"less"`` uses ``T_perm <= T_obs``.

    seed : int or None, optional
        Random seed for reproducibility. The same ``seed`` produces
        the same ``null_distribution_``. Internally, a single
        ``np.random.default_rng(seed)`` pre-generates one seed per
        permutation, which is then passed to ``Assignment.draw``.
        By default None.

    Attributes
    ----------
    assignment_ : CRDAssignment or BlockedAssignment
        The assignment passed to ``fit``.
    observed_statistic_ : float
        ATE estimated by the estimator on the original assignment,
        captured before the permutation loop runs.
    null_distribution_ : np.ndarray
        Array of permuted ATEs under the sharp null. Length equals
        ``n_permutations``.
    p_value_ : float
        Monte Carlo p-value computed with the Phipson & Smyth
        continuity correction.

    Notes
    -----
    **Sharp null vs. Neyman null.** This class tests Fisher's sharp
    null of no individual treatment effect, not Neyman's null of zero
    average treatment effect. ``BootstrapCI`` (Phase 4.4) will offer
    superpopulation inference.

    **Rerandomization.** When ``assignment`` is a ``CRDAssignment``
    produced by ``ReRandomizedCRD``, each permutation respects the
    Mahalanobis acceptance criterion automatically: ``CRDAssignment.draw``
    routes through ``ReRandomizedCRD._randomize_with_cached_cov``, which
    reuses the cached covariance matrix without recomputation.

    **Blocking.** When ``assignment`` is a ``BlockedAssignment``, each
    permutation rerandomizes within blocks, preserving the within-block
    treatment proportion. This is the correct null distribution for the
    blocked design.

    **Estimator state after `fit`.** The permutation loop refits
    ``self.estimator`` ``n_permutations`` times. After ``fit``
    completes, ``self.estimator`` is in the state of the *last*
    permutation, not the original assignment. To inspect the estimator
    on the original assignment, refit manually:
    ``rt.estimator.fit(rt.assignment_)``. The ``Results`` returned by
    ``estimate()`` is unaffected: it uses the observed statistic and
    metadata captured during ``fit`` before the loop runs.

    **Refit semantics.** Any prior fit state of ``estimator`` is
    discarded. Passing an estimator already fitted on a different
    dataset is allowed; it will be silently refitted on the assignment
    passed to ``fit``.

    **Future work (v2).** A ``"two-sided-conservative"`` alternative
    using ``2 * min(p_greater, p_less)`` may be added for cases with
    strong null asymmetry. Exact enumeration of all permutations for
    small N is also deferred to v2.

    Examples
    --------
    >>> from skxperiments.design.crd import CRD
    >>> from skxperiments.estimators.difference_in_means import (
    ...     DifferenceInMeans,
    ... )
    >>> from skxperiments.inference import RandomizationTest
    >>> design = CRD(p=0.5, seed=42)
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> dim = DifferenceInMeans(outcome_col="y")
    >>> rt = RandomizationTest(estimator=dim, n_permutations=10_000, seed=0)
    >>> result = rt.fit(assignment).estimate()  # doctest: +SKIP
    >>> result.p_value  # doctest: +SKIP
    """

    _VALID_ALTERNATIVES = ("two-sided", "greater", "less")

    def __init__(
        self,
        estimator: BaseEstimator,
        n_permutations: int = 10_000,
        alternative: str = "two-sided",
        seed: int | None = None,
    ) -> None:
        if not isinstance(estimator, BaseEstimator):
            raise InvalidDesignError(
                f"estimator must be an instance of BaseEstimator, got "
                f"{type(estimator).__name__}."
            )

        if not isinstance(n_permutations, int) or isinstance(
            n_permutations, bool
        ):
            raise InvalidDesignError(
                f"n_permutations must be an integer, got "
                f"{type(n_permutations).__name__}."
            )
        if n_permutations <= 0:
            raise InvalidDesignError(
                f"n_permutations must be > 0, got {n_permutations}."
            )

        if alternative not in self._VALID_ALTERNATIVES:
            raise InvalidDesignError(
                f"alternative must be one of {self._VALID_ALTERNATIVES}, "
                f"got {alternative!r}."
            )

        self.estimator = estimator
        self.n_permutations = n_permutations
        self.alternative = alternative
        self.seed = seed

    def fit(
        self,
        assignment: CRDAssignment | BlockedAssignment,
    ) -> "RandomizationTest":
        """Run the permutation loop and compute the p-value.

        Parameters
        ----------
        assignment : CRDAssignment or BlockedAssignment
            Observed assignment. ``FactorialAssignment`` is rejected
            with ``DesignEstimatorMismatch``.

        Returns
        -------
        RandomizationTest
            Returns self.

        Raises
        ------
        DesignEstimatorMismatch
            If ``assignment`` is not a ``CRDAssignment`` or
            ``BlockedAssignment``.
        InvalidDesignError
            If the estimator produces a multi-effect ``Results``
            (i.e., ``Results.ate is None``); v1 supports only scalar
            estimands.
        """
        self._validate_assignment_type(
            assignment, (CRDAssignment, BlockedAssignment)
        )

        # Refit on the original assignment to compute the observed
        # statistic. Any prior fit state of self.estimator is discarded.
        self.estimator.fit(assignment)
        base_results = self.estimator.estimate()

        if base_results.ate is None:
            raise InvalidDesignError(
                "RandomizationTest v1 supports only estimators producing "
                "a scalar ATE (Results.ate). The supplied estimator "
                f"({type(self.estimator).__name__}) produced a "
                "multi-effect Results (Results.effects). Multi-effect "
                "support is planned for v2."
            )

        observed_statistic = float(base_results.ate)

        # Capture metadata BEFORE the loop runs, so estimate() does
        # not depend on the post-loop state of self.estimator.
        self._n_obs = base_results.n_obs
        self._n_treated = base_results.n_treated
        self._n_control = base_results.n_control
        self._estimator_name = base_results.estimator_name
        self._design_name = base_results.design_name

        # Pre-generate one seed per permutation for reproducibility:
        # same self.seed -> same null_distribution_.
        rng = np.random.default_rng(self.seed)
        permutation_seeds = rng.integers(
            0, 2**32, size=self.n_permutations
        )

        null_distribution = np.empty(self.n_permutations, dtype=float)
        for i, perm_seed in enumerate(permutation_seeds):
            perm_assignment = assignment.draw(seed=int(perm_seed))
            self.estimator.fit(perm_assignment)
            null_distribution[i] = self.estimator.estimate().ate

        # Phipson & Smyth (2010) continuity correction.
        if self.alternative == "greater":
            n_extreme = int(np.sum(null_distribution >= observed_statistic))
        elif self.alternative == "less":
            n_extreme = int(np.sum(null_distribution <= observed_statistic))
        else:  # "two-sided"
            n_extreme = int(
                np.sum(np.abs(null_distribution) >= abs(observed_statistic))
            )

        p_value = (1 + n_extreme) / (1 + self.n_permutations)

        self.assignment_: CRDAssignment | BlockedAssignment = assignment
        self.observed_statistic_: float = observed_statistic
        self.null_distribution_: np.ndarray = null_distribution
        self.p_value_: float = float(p_value)

        return self

    def estimate(self) -> Results:
        """Return a Results object with the observed ATE and p-value.

        Reads metadata from attributes captured during ``fit`` (before
        the permutation loop ran), not from ``self.estimator``, which
        is in the state of the last permutation after ``fit`` completes.

        Returns
        -------
        Results
            Results with:

            - ``ate`` set to the observed statistic;
            - ``p_value`` set to the Monte Carlo p-value;
            - ``inference_name="RandomizationTest"``;
            - ``extra`` containing ``n_permutations``,
              ``null_distribution``, ``alternative``;
            - ``se`` and ``ci`` set to ``None`` (RandomizationTest
              produces only a p-value).

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        self._check_is_fitted()

        return Results(
            ate=self.observed_statistic_,
            p_value=self.p_value_,
            n_obs=self._n_obs,
            n_treated=self._n_treated,
            n_control=self._n_control,
            estimator_name=self._estimator_name,
            design_name=self._design_name,
            inference_name=type(self).__name__,
            extra={
                "n_permutations": self.n_permutations,
                "null_distribution": self.null_distribution_,
                "alternative": self.alternative,
            },
        )