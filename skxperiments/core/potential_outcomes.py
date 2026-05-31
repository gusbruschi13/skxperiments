"""Potential outcomes framework for unit-level causal quantities.

This module provides the PotentialOutcomes class, which represents
Y(0) and Y(1) for each unit. In real experiments, both potential
outcomes are never simultaneously observed; this class is used in
synthetic data generators and property-based tests.
"""

import numpy as np
import pandas as pd

from skxperiments.core.exceptions import InsufficientDataError


class PotentialOutcomes:
    """Represents unit-level potential outcomes Y(0) and Y(1).

    Parameters
    ----------
    y0 : array-like
        Potential outcomes under control Y(0).
    y1 : array-like
        Potential outcomes under treatment Y(1).
    unit_ids : array-like or None, optional
        Identifiers for each unit, by default None.

    Raises
    ------
    InsufficientDataError
        If y0 and y1 have different lengths, are empty, or unit_ids
        has a different length.

    Examples
    --------
    >>> import numpy as np
    >>> po = PotentialOutcomes(
    ...     y0=np.array([1.0, 2.0, 3.0]),
    ...     y1=np.array([2.0, 3.0, 5.0]),
    ... )
    >>> po.ate
    1.3333333333333333
    >>> po.ite
    array([1., 1., 2.])
    """

    def __init__(
        self,
        y0: np.ndarray | list,
        y1: np.ndarray | list,
        unit_ids: np.ndarray | list | None = None,
    ) -> None:
        self._y0 = np.asarray(y0, dtype=float)
        self._y1 = np.asarray(y1, dtype=float)

        if self._y0.size == 0:
            raise InsufficientDataError(
                context="PotentialOutcomes",
                minimum=1,
                received=0,
            )

        if self._y1.size == 0:
            raise InsufficientDataError(
                context="PotentialOutcomes",
                minimum=1,
                received=0,
            )

        if len(self._y0) != len(self._y1):
            raise InsufficientDataError(
                context="PotentialOutcomes (y0 and y1 must have the same length)",
                minimum=len(self._y0),
                received=len(self._y1),
            )

        if unit_ids is not None:
            self._unit_ids: np.ndarray | None = np.asarray(unit_ids)
            if len(self._unit_ids) != len(self._y0):
                raise InsufficientDataError(
                    context="PotentialOutcomes (unit_ids must match y0 length)",
                    minimum=len(self._y0),
                    received=len(self._unit_ids),
                )
        else:
            self._unit_ids = None

    @property
    def ite(self) -> np.ndarray:
        """Individual Treatment Effect for each unit.

        Returns
        -------
        np.ndarray
            Array of individual treatment effects (y1 - y0).
        """
        return self._y1 - self._y0

    @property
    def ate(self) -> float:
        """Average Treatment Effect across all units.

        Returns
        -------
        float
            Mean of individual treatment effects.
        """
        return float(np.mean(self.ite))

    @property
    def n(self) -> int:
        """Number of units.

        Returns
        -------
        int
            Total number of units.
        """
        return len(self._y0)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert potential outcomes to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns y0, y1, ite. If unit_ids were
            provided, includes unit_id as the first column.
        """
        data: dict[str, np.ndarray] = {}

        if self._unit_ids is not None:
            data["unit_id"] = self._unit_ids

        data["y0"] = self._y0
        data["y1"] = self._y1
        data["ite"] = self.ite

        return pd.DataFrame(data)

    def summary(self) -> str:
        """Generate a text summary of the potential outcomes.

        Returns
        -------
        str
            Formatted summary string with key statistics.
        """
        ite = self.ite
        lines = [
            "PotentialOutcomes Summary",
            "-------------------------",
            f"N units : {self.n}",
            f"ATE     : {self.ate:.4f}",
            f"ITE std : {float(np.std(ite, ddof=0)):.4f}",
            f"ITE min : {float(np.min(ite)):.4f}",
            f"ITE max : {float(np.max(ite)):.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            Compact representation with n and ate.
        """
        return f"PotentialOutcomes(n={self.n}, ate={self.ate:.4f})"