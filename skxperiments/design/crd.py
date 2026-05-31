"""Completely Randomized Design (CRD).

Assigns units to treatment uniformly at random, with either a fixed
absolute count of treated units or a fixed treatment proportion.
"""

import numpy as np
import pandas as pd

from skxperiments.core.assignment import CRDAssignment
from skxperiments.core.base import BaseDesign
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)


class CRD(BaseDesign):
    """Completely Randomized Design.

    Treatment is assigned uniformly at random to a fixed number of
    units. The user provides exactly one of ``n_treated`` (absolute
    count) or ``p`` (proportion); rounding for ``p`` uses
    ``round(p * n)``.

    Parameters
    ----------
    n_treated : int or None, optional
        Absolute number of units to assign to treatment. Mutually
        exclusive with ``p``. By default None.
    p : float or None, optional
        Treatment proportion in (0, 1). Mutually exclusive with
        ``n_treated``. By default None.
    seed : int or None, optional
        Random seed for reproducibility, by default None.
    treatment_col : str, optional
        Name of the treatment column to be added to the output, by
        default ``"treatment"``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": range(100)})
    >>> design = CRD(p=0.5, seed=42)
    >>> assignment = design.randomize(df)
    >>> assignment.n_treated_
    50
    """

    def __init__(
        self,
        n_treated: int | None = None,
        p: float | None = None,
        seed: int | None = None,
        treatment_col: str = "treatment",
    ) -> None:
        # Mutual exclusivity: exactly one of n_treated or p.
        if n_treated is None and p is None:
            raise InvalidDesignError(
                "CRD requires exactly one of n_treated or p; both "
                "are None."
            )
        if n_treated is not None and p is not None:
            raise InvalidDesignError(
                "CRD requires exactly one of n_treated or p; both "
                "were provided."
            )

        if n_treated is not None:
            if not isinstance(n_treated, (int, np.integer)) or n_treated <= 0:
                raise InvalidDesignError(
                    f"n_treated must be a positive integer, but received "
                    f"{n_treated!r}."
                )

        if p is not None:
            if not isinstance(p, (int, float)) or not (0.0 < p < 1.0):
                raise InvalidDesignError(
                    f"p must be in (0, 1), but received {p!r}."
                )

        self.n_treated = n_treated
        self.p = p
        self.seed = seed
        self.treatment_col = treatment_col

    def randomize(self, df: pd.DataFrame) -> CRDAssignment:
        """Perform complete randomization and return a CRDAssignment.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with experimental units. Must not contain
            ``treatment_col``.

        Returns
        -------
        CRDAssignment
            Assignment with the treatment column added. The original
            DataFrame is not modified.

        Raises
        ------
        InvalidDesignError
            If ``treatment_col`` already exists in ``df``, or if the
            resolved number of treated is 0 or N (no treatment
            contrast possible).
        InsufficientDataError
            If ``len(df) < n_treated`` (when ``n_treated`` was given),
            or if ``len(df) < 2`` (no contrast possible).
        """
        n_total = len(df)

        if n_total < 2:
            raise InsufficientDataError(
                context="CRD randomization",
                minimum=2,
                received=n_total,
            )

        if self.treatment_col in df.columns:
            raise InvalidDesignError(
                f"Treatment column '{self.treatment_col}' already "
                f"exists in DataFrame. Drop or rename it before "
                f"calling randomize()."
            )

        # Resolve n_treated.
        if self.n_treated is not None:
            if n_total < self.n_treated:
                raise InsufficientDataError(
                    context="CRD randomization",
                    minimum=self.n_treated,
                    received=n_total,
                )
            n_treated_resolved = self.n_treated
        else:
            n_treated_resolved = int(round(self.p * n_total))

        if n_treated_resolved <= 0 or n_treated_resolved >= n_total:
            raise InvalidDesignError(
                f"Resolved n_treated={n_treated_resolved} for N={n_total}; "
                f"must be strictly between 0 and N. Adjust n_treated "
                f"or p."
            )

        # Defensive copy. Build treatment vector.
        df_out = df.copy()
        rng = np.random.default_rng(self.seed)

        treatment = np.zeros(n_total, dtype=int)
        chosen = rng.choice(n_total, size=n_treated_resolved, replace=False)
        treatment[chosen] = 1

        df_out[self.treatment_col] = treatment

        return CRDAssignment(
            data=df_out,
            treatment_col=self.treatment_col,
            design=self,
            seed=self.seed,
        )