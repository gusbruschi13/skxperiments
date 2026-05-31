"""Blocked difference-in-means estimator for blocked randomized designs.

Computes the SATE estimate as a size-weighted average of within-block
difference-in-means estimates, the canonical estimator under blocked
CRD (Imbens & Rubin 2015, Chapter 9).
"""

import pandas as pd

from skxperiments.core.assignment import BlockedAssignment
from skxperiments.core.base import BaseEstimator
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results


class BlockedDifferenceInMeans(BaseEstimator):
    """Size-weighted ATE estimator for BlockedAssignment.

    Estimates the SATE as a weighted average of within-block
    difference-in-means estimates, weighted by block size:

        ATE_hat = sum_b (n_b / N) * (mean(Y_treated_b) - mean(Y_control_b))

    This is the canonical estimator under blocked CRD (Imbens & Rubin
    2015, Chapter 9). It is unbiased for SATE without any assumption
    on within-block variance, and remains numerically stable even
    with very small blocks (n_b = 2 each).

    This estimator computes the point estimate only. Standard errors,
    confidence intervals, and p-values are produced by inference
    classes (Phase 4) such as ``RandomizationTest`` or ``NeymanCI``.
    The ``Results`` object returned by ``estimate()`` therefore has
    ``se``, ``ci``, and ``p_value`` set to ``None``.

    # TODO v2: adicionar parâmetro weighting: Literal["size", "precision"] = "size"
    # quando houver demanda concreta. Precision-weighting reduz variância
    # assintótica sob homocedasticidade dentro de bloco, mas é instável
    # com blocos pequenos e exige reformulação paralela do NeymanCI.

    Parameters
    ----------
    outcome_col : str
        Name of the outcome column in ``assignment.data_``.

    Attributes
    ----------
    assignment_ : BlockedAssignment
        The fitted assignment, stored for downstream use.
    ate_ : float
        Size-weighted point estimate of the ATE.
    block_ates_ : dict
        Mapping from block label to within-block ATE estimate.

    Notes
    -----
    Accepts only ``BlockedAssignment``. ``CRDAssignment`` and
    ``FactorialAssignment`` are rejected via ``DesignEstimatorMismatch``:
    use ``DifferenceInMeans`` or ``FactorialEstimator`` respectively.

    Every block must have at least one treated unit and one control
    unit; otherwise the within-block ATE is undefined and ``fit``
    raises ``InvalidDesignError`` identifying the offending block.

    Examples
    --------
    >>> from skxperiments.design.blocked_crd import BlockedCRD
    >>> from skxperiments.estimators.blocked_difference_in_means import (
    ...     BlockedDifferenceInMeans,
    ... )
    >>> # df has a "block" column, an outcome "y", and other covariates
    >>> design = BlockedCRD(block_col="block", p=0.5, seed=42)
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> estimator = BlockedDifferenceInMeans(outcome_col="y")
    >>> results = estimator.fit(assignment).estimate()  # doctest: +SKIP
    >>> results.ate  # doctest: +SKIP
    """

    def __init__(self, outcome_col: str) -> None:
        self.outcome_col = outcome_col

    def fit(
        self, assignment: BlockedAssignment
    ) -> "BlockedDifferenceInMeans":
        """Fit the estimator on a BlockedAssignment.

        Parameters
        ----------
        assignment : BlockedAssignment
            Assignment produced by ``BlockedCRD``.

        Returns
        -------
        BlockedDifferenceInMeans
            Returns self.

        Raises
        ------
        DesignEstimatorMismatch
            If ``assignment`` is not a ``BlockedAssignment``.
        InvalidDesignError
            If ``outcome_col`` is missing, non-numeric, or contains
            NaN; or if any block has zero treated or zero control
            units.
        """
        self._validate_assignment_type(assignment, BlockedAssignment)

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

        # Validate every block has at least 1 treated and 1 control,
        # before any computation, to fail fast with a clear message.
        for block_val in assignment.block_sizes_:
            mask = data[assignment.block_col_] == block_val
            block_treatment = data.loc[mask, assignment.treatment_col_]
            n_t = int((block_treatment == 1).sum())
            n_c = int((block_treatment == 0).sum())
            if n_t == 0 or n_c == 0:
                raise InvalidDesignError(
                    f"Block '{block_val}' has {n_t} treated and {n_c} "
                    f"control units; BlockedDifferenceInMeans requires "
                    f"at least 1 of each."
                )

        # Compute within-block ATEs and the size-weighted total in a
        # single pass.
        N = assignment.n_units_
        block_ates: dict = {}
        ate_total = 0.0

        for block_val, n_b in assignment.block_sizes_.items():
            mask = data[assignment.block_col_] == block_val
            block_outcome = data.loc[mask, self.outcome_col]
            block_treatment = data.loc[mask, assignment.treatment_col_]

            treated_mean = float(block_outcome[block_treatment == 1].mean())
            control_mean = float(block_outcome[block_treatment == 0].mean())

            ate_b = treated_mean - control_mean
            block_ates[block_val] = ate_b
            ate_total += (n_b / N) * ate_b

        self.assignment_: BlockedAssignment = assignment
        self.block_ates_: dict = block_ates
        self.ate_: float = float(ate_total)

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