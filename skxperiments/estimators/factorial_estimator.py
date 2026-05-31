"""Factorial estimator for 2^K factorial designs.

Computes all 2^K - 1 orthogonal contrasts (main effects and
interactions of all orders) from a FactorialAssignment. Each effect
is the standard factorial contrast: for a non-empty subset S of
factors, the effect is the average over cells of Y weighted by
prod_{j in S} (2 * x_j - 1), divided by 2^(K-1).

Reference: Box, Hunter & Hunter (2005), Statistics for Experimenters.
"""

import itertools

import pandas as pd

from skxperiments.core.assignment import FactorialAssignment
from skxperiments.core.base import BaseEstimator
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results


class FactorialEstimator(BaseEstimator):
    """Estimates main effects and all interactions in a 2^K design.

    For a FactorialAssignment with K factors, returns 2^K - 1 effects:
    K main effects, C(K,2) two-way interactions, ..., and one K-way
    interaction. Each effect is computed via the standard factorial
    contrast.

    Effect keys are tuples of factor names in **alphabetical order**,
    independent of the order in which factors were passed to
    ``FactorialDesign``. This guarantees reproducibility: the same
    DataFrame and design parameters produce the same key set, in the
    same order, regardless of factor ordering.

    Subset enumeration uses ``itertools.combinations`` over the sorted
    factor names, iterating r from 1 to K. For K=3 with factors
    ``["A", "B", "C"]``, the order is::

        ("A",), ("B",), ("C",),
        ("A", "B"), ("A", "C"), ("B", "C"),
        ("A", "B", "C")

    This estimator computes the point estimate only. Standard errors,
    confidence intervals, and p-values are produced by inference
    classes (Phase 4). The ``Results`` returned by ``estimate()`` has
    ``se``, ``ci``, ``p_value`` set to ``None``.

    Parameters
    ----------
    outcome_col : str
        Name of the outcome column in ``assignment.data_``.

    Attributes
    ----------
    assignment_ : FactorialAssignment
        The fitted assignment.
    effects_ : dict[tuple[str, ...], float]
        Mapping from effect-name tuple to point estimate. Has
        ``2 ** K - 1`` entries.

    Notes
    -----
    Accepts only ``FactorialAssignment``. ``CRDAssignment`` and
    ``BlockedAssignment`` are rejected via ``DesignEstimatorMismatch``;
    use ``DifferenceInMeans`` or ``BlockedDifferenceInMeans``
    respectively.

    Cell-level outcome means are computed via ``groupby`` on the
    synthetic ``"_cell"`` column of ``assignment.data_``, in a single
    pass over the data.

    Examples
    --------
    >>> from skxperiments.design.factorial import FactorialDesign
    >>> from skxperiments.estimators.factorial_estimator import (
    ...     FactorialEstimator,
    ... )
    >>> design = FactorialDesign(
    ...     factors=["A", "B"], n_per_cell=100, seed=42
    ... )
    >>> assignment = design.randomize(df)  # doctest: +SKIP
    >>> result = (
    ...     FactorialEstimator(outcome_col="y")
    ...     .fit(assignment)
    ...     .estimate()
    ... )
    >>> result.effects[("A",)]  # doctest: +SKIP
    """

    def __init__(self, outcome_col: str) -> None:
        self.outcome_col = outcome_col

    def fit(
        self, assignment: FactorialAssignment
    ) -> "FactorialEstimator":
        """Fit the estimator on a FactorialAssignment.

        Parameters
        ----------
        assignment : FactorialAssignment
            Assignment produced by ``FactorialDesign``.

        Returns
        -------
        FactorialEstimator
            Returns self.

        Raises
        ------
        DesignEstimatorMismatch
            If ``assignment`` is not a ``FactorialAssignment``.
        InvalidDesignError
            If ``outcome_col`` is missing, non-numeric, or contains
            NaN; or if any cell has zero units.
        """
        self._validate_assignment_type(assignment, FactorialAssignment)

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

        # Defensive: every cell must have at least 1 unit.
        for cell_idx, size in assignment.cell_sizes_.items():
            if size == 0:
                raise InvalidDesignError(
                    f"Cell {cell_idx} has 0 units; FactorialEstimator "
                    f"requires at least 1 unit per cell."
                )

        # Cell-level outcome means in a single pass over the data.
        cell_means_series = data.groupby("_cell")[self.outcome_col].mean()
        cell_means: dict[int, float] = {
            int(idx): float(val) for idx, val in cell_means_series.items()
        }

        # Compute all 2^K - 1 effects via factorial contrasts.
        factor_cols = assignment.factor_cols          # original design order
        factor_cols_sorted = sorted(factor_cols)      # alphabetical for keys
        K = len(factor_cols)

        effects: dict[tuple[str, ...], float] = {}

        for r in range(1, K + 1):
            for subset in itertools.combinations(factor_cols_sorted, r):
                # Indices in the original factor_cols determine which
                # bit of the cell index encodes each factor.
                subset_indices = [factor_cols.index(f) for f in subset]
                effect = 0.0
                for cell_idx, cell_mean in cell_means.items():
                    sign = 1
                    for j in subset_indices:
                        x_j = (cell_idx >> j) & 1
                        sign *= 2 * x_j - 1
                    effect += sign * cell_mean
                effects[subset] = effect / (2 ** (K - 1))

        self.assignment_: FactorialAssignment = assignment
        self.effects_: dict[tuple[str, ...], float] = effects

        return self

    def estimate(self) -> Results:
        """Return a Results object with all 2^K - 1 effects.

        Returns
        -------
        Results
            Multi-effect ``Results`` with ``effects`` populated,
            ``ate=None``, ``n_treated=None``, ``n_control=None``,
            ``estimator_name`` and ``design_name`` auto-populated.
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
            ate=None,
            effects=self.effects_,
            n_obs=self.assignment_.n_units_,
            n_treated=None,
            n_control=None,
            estimator_name=type(self).__name__,
            design_name=design_name,
        )