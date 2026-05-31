"""Difference-in-means estimator for completely randomized designs.

Computes the simple ATE estimate for a CRDAssignment as the difference
between the treated-group mean and the control-group mean of the
outcome variable.
"""

import pandas as pd

from skxperiments.core.assignment import CRDAssignment
from skxperiments.core.base import BaseEstimator
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results


class DifferenceInMeans(BaseEstimator):
    """Difference-in-means estimator for the ATE under CRD.

    Estimates the average treatment effect as the difference between
    the sample mean of the outcome among treated units and the sample
    mean among control units:

        ATE_hat = mean(Y[treated]) - mean(Y[control])

    This estimator computes the point estimate only. Standard errors,
    confidence intervals, and p-values are produced by inference
    classes (Phase 4) such as ``RandomizationTest`` or ``NeymanCI``.
    The ``Results`` object returned by ``estimate()`` therefore has
    ``se``, ``ci``, and ``p_value`` set to ``None``.

    Parameters
    ----------
    outcome_col : str
        Name of the outcome column in ``assignment.data_``.

    Attributes
    ----------
    assignment_ : CRDAssignment
        The fitted assignment, stored for downstream use.
    ate_ : float
        Point estimate of the ATE.

    Notes
    -----
    Accepts ``CRDAssignment`` produced by either ``CRD`` or
    ``ReRandomizedCRD`` — the point estimator is the same regardless
    of the rerandomization criterion. Correct inference under
    rerandomization requires the corresponding inference class to
    consume ``rerandomization_metadata``, which is the responsibility
    of Phase 4. The estimator itself ignores the metadata.

    ``BlockedAssignment`` and ``FactorialAssignment`` are rejected
    via ``DesignEstimatorMismatch``: use ``BlockedDifferenceInMeans``
    or ``FactorialEstimator`` respectively.

    Examples
    --------
    >>> from skxperiments.design.crd import CRD
    >>> from skxperiments.estimators.difference_in_means import (
    ...     DifferenceInMeans,
    ... )
    >>> # df has columns "x" and "y" (the outcome)
    >>> design = CRD(p=0.5, seed=42)
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> estimator = DifferenceInMeans(outcome_col="y")
    >>> results = estimator.fit(assignment).estimate()  # doctest: +SKIP
    >>> results.ate  # doctest: +SKIP
    """

    def __init__(self, outcome_col: str) -> None:
        self.outcome_col = outcome_col

    def fit(self, assignment: CRDAssignment) -> "DifferenceInMeans":
        """Fit the estimator on a CRDAssignment.

        Parameters
        ----------
        assignment : CRDAssignment
            Assignment produced by ``CRD`` or ``ReRandomizedCRD``.

        Returns
        -------
        DifferenceInMeans
            Returns self.

        Raises
        ------
        DesignEstimatorMismatch
            If ``assignment`` is not a ``CRDAssignment`` (in particular,
            ``BlockedAssignment`` and ``FactorialAssignment`` are
            rejected).
        InvalidDesignError
            If ``outcome_col`` is missing from ``assignment.data_``,
            is not numeric, or contains NaN values.
        """
        self._validate_assignment_type(assignment, CRDAssignment)

        data = assignment.data_

        if self.outcome_col not in data.columns:
            raise InvalidDesignError(
                f"Outcome column '{self.outcome_col}' not found in "
                f"assignment.data_. Available columns: "
                f"{list(data.columns)}."
            )

        if not pd.api.types.is_numeric_dtype(data[self.outcome_col]):
            raise InvalidDesignError(
                f"Outcome column '{self.outcome_col}' must be numeric. "
                f"dtype found: {data[self.outcome_col].dtype}."
            )

        if data[self.outcome_col].isna().any():
            raise InvalidDesignError(
                f"Outcome column '{self.outcome_col}' contains NaN "
                f"values. Impute or drop NaN before fitting."
            )

        y = data[self.outcome_col].values
        treated_idx = assignment.treated_ids()
        control_idx = assignment.control_ids()

        self.assignment_: CRDAssignment = assignment
        self.ate_: float = float(
            y[treated_idx].mean() - y[control_idx].mean()
        )

        return self

    def estimate(self) -> Results:
        """Return a Results object with the point estimate and metadata.

        Returns
        -------
        Results
            Results with ``ate``, ``n_obs``, ``n_treated``, ``n_control``,
            ``estimator_name``, and ``design_name`` populated.
            ``se``, ``ci``, ``p_value`` are ``None`` — inference is
            Phase 4.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        self._check_is_fitted()

        design_name: str | None
        if self.assignment_.design_ is not None:
            design_name = type(self.assignment_.design_).__name__
        else:
            design_name = None

        return Results(
            ate=self.ate_,
            n_obs=self.assignment_.n_units_,
            n_treated=self.assignment_.n_treated_,
            n_control=self.assignment_.n_control_,
            estimator_name=type(self).__name__,
            design_name=design_name,
        )