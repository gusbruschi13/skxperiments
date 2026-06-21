"""Covariate balance report diagnostic.

Wraps ``check_balance`` (Phase 2) to produce a covariate balance report:
the standardized mean difference (SMD) per covariate, plus a flag when
any covariate exceeds an absolute-SMD threshold. The conventional cutoff
for "meaningful" imbalance is ``|SMD| > 0.1`` (Austin 2009).

The Love plot is intentionally not produced here: rendering lives in the
Phase 7 reporting layer (``plot_balance``), which centralizes the
optional matplotlib dependency. ``BalanceResult.to_dataframe`` exposes
the table that such a plot would consume.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
)
from skxperiments.core.base import DiagnosticsReport
from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.design.balance import check_balance


@dataclass(frozen=True, eq=False)
class BalanceResult:
    """Result of a covariate balance check.

    Attributes
    ----------
    table : pd.DataFrame
        The full balance table from ``check_balance``: one row per
        covariate with columns ``covariate``, ``mean_treated``,
        ``mean_control``, ``std_pooled``, ``smd``.
    threshold : float
        Absolute-SMD threshold above which a covariate is considered
        imbalanced.

    Notes
    -----
    A covariate with no within-group variation has ``std_pooled == 0``
    and an undefined (NaN) SMD; such covariates are reported under
    ``constant_covariates`` and never counted as imbalanced.
    """

    table: pd.DataFrame
    threshold: float

    @property
    def smd(self) -> dict:
        """Mapping from covariate name to its SMD."""
        return {row.covariate: float(row.smd) for row in self.table.itertuples()}

    @property
    def imbalanced(self) -> list[str]:
        """Covariates whose absolute SMD exceeds the threshold."""
        return [
            row.covariate
            for row in self.table.itertuples()
            if not np.isnan(row.smd) and abs(row.smd) > self.threshold
        ]

    @property
    def constant_covariates(self) -> list[str]:
        """Covariates with an undefined (NaN) SMD (no within-group variance)."""
        return [
            row.covariate
            for row in self.table.itertuples()
            if np.isnan(row.smd)
        ]

    @property
    def flagged(self) -> bool:
        """True if any covariate is imbalanced."""
        return len(self.imbalanced) > 0

    @property
    def max_abs_smd(self) -> float:
        """Largest absolute SMD across covariates (NaN if all undefined)."""
        values = np.abs(self.table["smd"].to_numpy(dtype=float))
        if np.all(np.isnan(values)):
            return float("nan")
        return float(np.nanmax(values))

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the balance table."""
        return self.table.copy()

    def to_dict(self) -> dict:
        """Return the summary fields as a plain dictionary."""
        return {
            "threshold": self.threshold,
            "flagged": self.flagged,
            "imbalanced": self.imbalanced,
            "constant_covariates": self.constant_covariates,
            "max_abs_smd": self.max_abs_smd,
            "smd": self.smd,
        }

    def summary(self) -> "BalanceResult":
        """Print a formatted summary table and return self."""
        status = "FLAGGED — covariate imbalance" if self.flagged else "OK"
        lines = ["Balance Report", "--------------"]
        lines.append(f"threshold      |SMD| > {self.threshold}")
        lines.append(f"max |SMD|      {self.max_abs_smd:.4f}")
        lines.append(f"status         {status}")
        lines.append("covariate      SMD")
        for row in self.table.itertuples():
            if np.isnan(row.smd):
                smd_str = "nan (constant)"
                mark = ""
            else:
                smd_str = f"{row.smd:+.4f}"
                mark = " *" if abs(row.smd) > self.threshold else ""
            lines.append(f"  {row.covariate}: {smd_str}{mark}")
        print("\n".join(lines))
        return self

    def to_diagnostics_report(self) -> DiagnosticsReport:
        """Convert to a ``DiagnosticsReport`` for pipeline aggregation.

        Imbalanced covariates become flags; constant covariates become
        warnings.
        """
        report = DiagnosticsReport()
        if self.imbalanced:
            report.flags.append(
                f"Covariate imbalance (|SMD| > {self.threshold}): "
                f"{self.imbalanced}."
            )
        if self.constant_covariates:
            report.warnings.append(
                f"Constant covariates with undefined SMD: "
                f"{self.constant_covariates}."
            )
        return report


class BalanceReport:
    """Covariate balance diagnostic for two-arm designs.

    Computes the standardized mean difference (SMD) per covariate via
    ``check_balance`` and flags covariates whose absolute SMD exceeds
    ``threshold``.

    Parameters
    ----------
    covariates : list of str or None, optional
        Covariates to check. If None, all numeric columns except the
        treatment column are used (see ``check_balance``). By default None.
    threshold : float, optional
        Absolute-SMD threshold for flagging imbalance, by default 0.1
        (Austin 2009). Must be positive (SMDs are unbounded, so no upper
        limit applies).

    Notes
    -----
    Supports ``CRDAssignment`` (including rerandomized) and
    ``BlockedAssignment``. ``FactorialAssignment`` is rejected: a single
    treated-vs-control SMD is not defined for multi-cell designs.
    """

    def __init__(
        self,
        covariates: list[str] | None = None,
        threshold: float = 0.1,
    ) -> None:
        if not isinstance(threshold, (int, float)) or isinstance(
            threshold, bool
        ):
            raise InvalidDesignError(
                f"threshold must be a positive float, got "
                f"{type(threshold).__name__}."
            )
        if threshold <= 0.0:
            raise InvalidDesignError(
                f"threshold must be > 0, got {threshold}."
            )

        if covariates is not None:
            if not isinstance(covariates, list) or not all(
                isinstance(c, str) for c in covariates
            ):
                raise InvalidDesignError(
                    "covariates must be None or a list of column names."
                )

        self.covariates = covariates
        self.threshold = threshold

    def run(
        self,
        assignment: CRDAssignment | BlockedAssignment,
    ) -> BalanceResult:
        """Compute the balance report for an assignment.

        Parameters
        ----------
        assignment : CRDAssignment or BlockedAssignment
            Two-arm assignment to check.

        Returns
        -------
        BalanceResult

        Raises
        ------
        InvalidDesignError
            If the assignment is not two-arm, or if a covariate is missing
            or contains NaN (propagated from ``check_balance``).
        """
        if not isinstance(assignment, (CRDAssignment, BlockedAssignment)):
            raise InvalidDesignError(
                f"BalanceReport supports two-arm designs (CRDAssignment, "
                f"BlockedAssignment); received "
                f"{type(assignment).__name__}. Treated-vs-control SMD is "
                f"not defined for multi-cell factorial designs."
            )

        table = check_balance(assignment, self.covariates)
        return BalanceResult(table=table, threshold=self.threshold)
