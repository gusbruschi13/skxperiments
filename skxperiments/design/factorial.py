"""2^K Factorial Design.

Randomly assigns units to one of 2^K cells defined by the values of
K binary factors, with all cells of equal size.
"""

import numpy as np
import pandas as pd

from skxperiments.core.assignment import FactorialAssignment
from skxperiments.core.base import BaseDesign
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)


class FactorialDesign(BaseDesign):
    """2^K Factorial Design with equal cell sizes.

    Randomly assigns units to one of 2^K cells defined by K binary
    factors. Each cell contains exactly ``n_per_cell`` units, so the
    DataFrame must have ``n_per_cell * 2^K`` rows.

    Parameters
    ----------
    factors : list of str
        Names of the K factors. These will be added as columns to the
        Assignment's data, alongside the synthetic ``"_cell"`` column.
        Must be non-empty and contain no duplicates.
    n_per_cell : int
        Number of units per cell. All cells have equal size in this
        version. Must be >= 1.
    seed : int or None, optional
        Random seed for reproducibility, by default None.

    Notes
    -----
    Cell encoding convention (little-endian):

        cell_index = sum(factor_value * 2**i
                         for i, factor_value in enumerate(factors))

    For K=2 with factors ``["A", "B"]``:
        A=0, B=0 -> cell 0
        A=1, B=0 -> cell 1
        A=0, B=1 -> cell 2
        A=1, B=1 -> cell 3

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": range(8)})
    >>> design = FactorialDesign(factors=["A", "B"], n_per_cell=2, seed=42)
    >>> assignment = design.randomize(df)
    >>> assignment.n_cells_
    4
    >>> assignment.cell_sizes_
    {0: 2, 1: 2, 2: 2, 3: 2}
    """

    def __init__(
        self,
        factors: list[str],
        n_per_cell: int,
        seed: int | None = None,
    ) -> None:
        if not isinstance(factors, list) or len(factors) == 0:
            raise InvalidDesignError(
                "FactorialDesign requires a non-empty list of factors."
            )

        if len(set(factors)) != len(factors):
            duplicates = [f for f in factors if factors.count(f) > 1]
            raise InvalidDesignError(
                f"FactorialDesign factors must be unique. "
                f"Duplicates found: {sorted(set(duplicates))}."
            )

        if not isinstance(n_per_cell, int) or n_per_cell < 1:
            raise InvalidDesignError(
                f"n_per_cell must be a positive integer, "
                f"received {n_per_cell!r}."
            )

        self.factors = factors
        self.n_per_cell = n_per_cell
        self.seed = seed

    def randomize(self, df: pd.DataFrame) -> FactorialAssignment:
        """Perform factorial randomization.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with experimental units. Must have exactly
            ``n_per_cell * 2^K`` rows and must not contain any column
            named in ``factors`` or named ``"_cell"``.

        Returns
        -------
        FactorialAssignment
            Assignment with factor columns and ``"_cell"`` added.

        Raises
        ------
        InvalidDesignError
            If column-name collisions exist or if any cell has size 0.
        InsufficientDataError
            If ``len(df) != n_per_cell * 2^K``.
        """
        k = len(self.factors)
        n_cells = 2**k
        n_required = self.n_per_cell * n_cells

        if len(df) != n_required:
            raise InsufficientDataError(
                context=(
                    f"FactorialDesign with K={k} factors and "
                    f"n_per_cell={self.n_per_cell} requires exactly "
                    f"{n_required} units"
                ),
                minimum=n_required,
                received=len(df),
            )

        # Detect column-name collisions
        forbidden = set(self.factors) | {"_cell"}
        collisions = sorted(forbidden & set(df.columns))
        if collisions:
            raise InvalidDesignError(
                f"DataFrame already contains columns reserved for "
                f"FactorialDesign output: {collisions}. Drop or rename "
                f"them before calling randomize()."
            )

        df_out = df.copy()
        rng = np.random.default_rng(self.seed)

        # Shuffle iloc positions and assign sequentially to cells.
        shuffled = rng.permutation(n_required)
        cell_assignment = np.empty(n_required, dtype=int)
        for cell_idx in range(n_cells):
            start = cell_idx * self.n_per_cell
            end = start + self.n_per_cell
            cell_assignment[shuffled[start:end]] = cell_idx

        df_out["_cell"] = cell_assignment

        # Decode cell index back into binary factor values
        # using little-endian convention.
        for i, factor_name in enumerate(self.factors):
            df_out[factor_name] = (cell_assignment >> i) & 1

        # Build cell_sizes and validate non-empty cells (defensive;
        # by construction every cell has n_per_cell >= 1).
        cell_sizes: dict = {}
        for cell_idx in range(n_cells):
            size = int((cell_assignment == cell_idx).sum())
            if size == 0:
                raise InvalidDesignError(
                    f"Cell {cell_idx} has zero units after randomization. "
                    f"This should not happen with n_per_cell={self.n_per_cell}; "
                    f"please report as a bug."
                )
            cell_sizes[cell_idx] = size

        return FactorialAssignment(
            data=df_out,
            design=self,
            factor_cols=self.factors,
            cell_sizes=cell_sizes,
            seed=self.seed,
        )