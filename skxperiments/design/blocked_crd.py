"""Blocked Completely Randomized Design.

Randomizes treatment independently within each block, preserving the
treatment proportion within every block. Useful when there are
pre-experiment covariates that define meaningful subgroups (e.g.,
geography, device type) where balance must be guaranteed.
"""

import numpy as np
import pandas as pd

from skxperiments.core.assignment import BlockedAssignment
from skxperiments.core.base import BaseDesign
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)


class BlockedCRD(BaseDesign):
    """Blocked Completely Randomized Design.

    Treatment is randomized independently within each block defined by
    ``block_col``. The treatment proportion ``p`` is applied uniformly
    to all blocks. Within each block, the number of treated units is
    ``round(p * n_block)``.

    Parameters
    ----------
    block_col : str
        Name of the column in the DataFrame that defines blocks.
    p : float
        Treatment proportion in (0, 1), applied uniformly across blocks.
    seed : int or None, optional
        Random seed for reproducibility, by default None.
    treatment_col : str, optional
        Name of the treatment column to be added to the output, by
        default ``"treatment"``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "x": range(8),
    ...     "region": ["A", "A", "A", "A", "B", "B", "B", "B"],
    ... })
    >>> design = BlockedCRD(block_col="region", p=0.5, seed=42)
    >>> assignment = design.randomize(df)
    >>> assignment.block_sizes_
    {'A': 4, 'B': 4}
    """

    def __init__(
        self,
        block_col: str,
        p: float | None = None,
        seed: int | None = None,
        treatment_col: str = "treatment",
    ) -> None:
        if p is None:
            raise InvalidDesignError(
                "BlockedCRD requires a treatment proportion p; "
                "received p=None."
            )
        if not (0.0 < p < 1.0):
            raise InvalidDesignError(
                f"Treatment proportion p must be in (0, 1), but received {p}."
            )

        self.block_col = block_col
        self.p = p
        self.seed = seed
        self.treatment_col = treatment_col

    def randomize(self, df: pd.DataFrame) -> BlockedAssignment:
        """Perform blocked randomization and return a BlockedAssignment.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the experimental units. Must contain
            ``block_col`` and must not contain ``treatment_col``.

        Returns
        -------
        BlockedAssignment
            Assignment with treatment column added. Original DataFrame
            is not modified.

        Raises
        ------
        InvalidDesignError
            If ``block_col`` is missing from ``df``, if ``treatment_col``
            already exists in ``df``, or if rounding results in 0 or n
            treated units in any block.
        InsufficientDataError
            If any block has fewer than 2 units.
        """
        if self.block_col not in df.columns:
            raise InvalidDesignError(
                f"Block column '{self.block_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}."
            )
        if self.treatment_col in df.columns:
            raise InvalidDesignError(
                f"Treatment column '{self.treatment_col}' already exists "
                f"in DataFrame. Drop or rename it before calling randomize()."
            )

        df_out = df.copy()
        treatment = np.zeros(len(df_out), dtype=int)

        rng = np.random.default_rng(self.seed)

        # Iterate blocks in stable order (sorted by label) for
        # reproducibility independent of pandas grouping internals.
        block_labels = sorted(df_out[self.block_col].unique(), key=lambda x: str(x))

        block_sizes: dict = {}

        for label in block_labels:
            block_iloc = np.where(df_out[self.block_col].values == label)[0]
            n_block = len(block_iloc)
            block_sizes[label] = n_block

            if n_block < 2:
                raise InsufficientDataError(
                    context=(
                        f"BlockedCRD randomization for block '{label}'"
                    ),
                    minimum=2,
                    received=n_block,
                )

            n_treated_block = int(round(self.p * n_block))

            if n_treated_block == 0 or n_treated_block == n_block:
                raise InvalidDesignError(
                    f"Block '{label}' has size {n_block}; with p={self.p}, "
                    f"rounding yields {n_treated_block} treated units. "
                    f"Each block must have at least 1 treated and 1 control "
                    f"unit. Increase block size or adjust p."
                )

            chosen = rng.choice(block_iloc, size=n_treated_block, replace=False)
            treatment[chosen] = 1

        df_out[self.treatment_col] = treatment

        return BlockedAssignment(
            data=df_out,
            treatment_col=self.treatment_col,
            design=self,
            block_col=self.block_col,
            block_sizes=block_sizes,
            seed=self.seed,
        )