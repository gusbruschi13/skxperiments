"""Sample Ratio Mismatch (SRM) diagnostic.

A Sample Ratio Mismatch occurs when the observed allocation of units to
treatment arms differs from the intended allocation by more than chance
would explain. It is a high-priority alarm for an *implementation* bug
(asymmetric logging, bot filtering, a broken assignment service), not a
scientific hypothesis test — hence the conventional decision threshold of
0.001 rather than 0.05 (Kohavi et al.).

``SRMTest`` compares the observed arm (or cell) counts to the counts
expected under the design's intended allocation using Pearson's
chi-squared goodness-of-fit test, and flags the experiment when the
p-value falls below the threshold.

References
----------
Kohavi, R., Tang, D., & Xu, Y. (2020). Trustworthy Online Controlled
    Experiments. Cambridge University Press (Sample Ratio Mismatch).
"""

from dataclasses import asdict, dataclass

from scipy.stats import chisquare

from skxperiments.core.assignment import (
    BlockedAssignment,
    CRDAssignment,
    FactorialAssignment,
)
from skxperiments.core.base import DiagnosticsReport
from skxperiments.core.exceptions import InvalidDesignError


@dataclass(frozen=True)
class SRMResult:
    """Result of a Sample Ratio Mismatch test.

    Attributes
    ----------
    statistic : float
        Pearson chi-squared statistic.
    p_value : float
        Chi-squared goodness-of-fit p-value.
    dof : int
        Degrees of freedom (number of groups minus one).
    observed : dict
        Mapping from group label to observed count. Groups are
        ``"control"``/``"treated"`` for two-arm designs and the integer
        cell index for factorial designs.
    expected : dict
        Mapping from group label to expected count under the intended
        allocation.
    threshold : float
        Decision threshold the p-value was compared against.
    flagged : bool
        True if ``p_value < threshold`` — an SRM is suspected.
    """

    statistic: float
    p_value: float
    dof: int
    observed: dict
    expected: dict
    threshold: float
    flagged: bool

    def summary(self) -> "SRMResult":
        """Print a formatted summary table and return self.

        Returns
        -------
        SRMResult
            Returns self for method chaining (mirrors ``Results.summary``).
        """
        status = "FLAGGED — possible SRM" if self.flagged else "OK"
        lines = ["SRM Test", "--------"]
        lines.append(f"chi-square     {self.statistic:.4f}")
        lines.append(f"dof            {self.dof}")
        lines.append(f"p-value        {self.p_value:.6f}")
        lines.append(f"threshold      {self.threshold}")
        lines.append(f"status         {status}")
        lines.append("group          observed / expected")
        for group in self.observed:
            lines.append(
                f"  {group}: {self.observed[group]} / "
                f"{self.expected[group]:.1f}"
            )
        print("\n".join(lines))
        return self

    def to_dict(self) -> dict:
        """Return the result as a plain dictionary."""
        return asdict(self)

    def to_diagnostics_report(self) -> DiagnosticsReport:
        """Convert to a ``DiagnosticsReport`` for pipeline aggregation.

        Returns
        -------
        DiagnosticsReport
            A report carrying a single flag when an SRM is suspected, and
            no flags otherwise.
        """
        report = DiagnosticsReport()
        if self.flagged:
            expected_rounded = {
                group: round(count, 1)
                for group, count in self.expected.items()
            }
            report.flags.append(
                f"Sample Ratio Mismatch (p={self.p_value:.2e} < "
                f"{self.threshold}): observed {self.observed} vs expected "
                f"{expected_rounded}."
            )
        return report


class SRMTest:
    """Sample Ratio Mismatch test via Pearson's chi-squared.

    Compares observed arm/cell counts to the counts expected under the
    design's intended allocation. Supports two-arm designs
    (``CRDAssignment``, including rerandomized, and ``BlockedAssignment``)
    and factorial designs (``FactorialAssignment``).

    Parameters
    ----------
    threshold : float, optional
        Decision threshold for the p-value, by default 0.001. An
        experiment is flagged when the chi-squared p-value is below it.
        Must be in (0, 1).
    expected : float, dict, or None, optional
        Intended allocation. By default None, in which case it is inferred
        from the design:

        - two-arm designs: the design's ``p`` (treatment proportion);
        - factorial designs: a uniform allocation across the ``2**K``
          cells.

        When the design has no intended proportion (e.g., ``CRD`` built
        with ``n_treated`` rather than ``p``, or ``design_`` is None),
        ``expected`` must be provided explicitly. For two-arm designs it
        may be a float (the treated proportion in (0, 1)); for any design
        it may be a dict mapping each group label to a positive expected
        proportion (normalized internally).

    Notes
    -----
    SRM is a check on the *observed* data, which in a pipeline may have
    been filtered or joined after randomization. Run directly on a fresh
    ``Assignment`` from ``randomize()`` it will not flag, because the
    library's designs fix the per-arm counts exactly.
    """

    def __init__(
        self,
        threshold: float = 0.001,
        expected: float | dict | None = None,
    ) -> None:
        if not isinstance(threshold, (int, float)) or isinstance(
            threshold, bool
        ):
            raise InvalidDesignError(
                f"threshold must be a float in (0, 1), got "
                f"{type(threshold).__name__}."
            )
        if not (0.0 < threshold < 1.0):
            raise InvalidDesignError(
                f"threshold must be in (0, 1), got {threshold}."
            )

        if expected is not None and not isinstance(
            expected, (int, float, dict)
        ):
            raise InvalidDesignError(
                f"expected must be None, a float, or a dict, got "
                f"{type(expected).__name__}."
            )
        if isinstance(expected, bool):
            raise InvalidDesignError("expected must not be a bool.")

        self.threshold = threshold
        self.expected = expected

    def run(
        self,
        assignment: CRDAssignment | BlockedAssignment | FactorialAssignment,
    ) -> SRMResult:
        """Run the SRM test on an assignment.

        Parameters
        ----------
        assignment : CRDAssignment, BlockedAssignment, or FactorialAssignment
            The assignment whose realized allocation is being checked.

        Returns
        -------
        SRMResult

        Raises
        ------
        InvalidDesignError
            If the assignment type is unsupported, if the expected
            allocation cannot be inferred and was not provided, or if
            ``expected`` is malformed.
        """
        observed, proportions = self._observed_and_proportions(assignment)

        total = sum(observed.values())
        if total <= 0:
            raise InvalidDesignError(
                "SRMTest requires at least one observed unit."
            )

        labels = list(observed.keys())
        f_obs = [observed[label] for label in labels]
        expected_counts = {
            label: proportions[label] * total for label in labels
        }
        f_exp = [expected_counts[label] for label in labels]

        statistic, p_value = chisquare(f_obs, f_exp)

        return SRMResult(
            statistic=float(statistic),
            p_value=float(p_value),
            dof=len(labels) - 1,
            observed=observed,
            expected=expected_counts,
            threshold=self.threshold,
            flagged=bool(p_value < self.threshold),
        )

    def _observed_and_proportions(
        self,
        assignment: CRDAssignment | BlockedAssignment | FactorialAssignment,
    ) -> tuple[dict, dict]:
        """Return observed counts and expected proportions per group."""
        if isinstance(assignment, FactorialAssignment):
            observed = {
                int(cell): int(count)
                for cell, count in sorted(assignment.cell_sizes_.items())
            }
            if self.expected is None:
                k = len(observed)
                proportions = {cell: 1.0 / k for cell in observed}
            else:
                proportions = self._proportions_from_dict(
                    self.expected, list(observed)
                )
            return observed, proportions

        if isinstance(assignment, (CRDAssignment, BlockedAssignment)):
            observed = {
                "control": int(assignment.n_control_),
                "treated": int(assignment.n_treated_),
            }
            if isinstance(self.expected, dict):
                proportions = self._proportions_from_dict(
                    self.expected, ["control", "treated"]
                )
            elif self.expected is not None:
                p_treated = self._validate_proportion(self.expected)
                proportions = {
                    "control": 1.0 - p_treated,
                    "treated": p_treated,
                }
            else:
                design = assignment.design_
                p = getattr(design, "p", None) if design is not None else None
                if p is None:
                    raise InvalidDesignError(
                        "SRMTest cannot infer the expected allocation: the "
                        "design has no intended proportion `p` (e.g., CRD "
                        "built with n_treated, or design_ is None). Pass "
                        "expected=<treated proportion> or a dict of expected "
                        "proportions."
                    )
                p_treated = float(p)
                proportions = {
                    "control": 1.0 - p_treated,
                    "treated": p_treated,
                }
            return observed, proportions

        raise InvalidDesignError(
            f"SRMTest supports CRDAssignment, BlockedAssignment, and "
            f"FactorialAssignment; received {type(assignment).__name__}."
        )

    @staticmethod
    def _validate_proportion(value: float) -> float:
        """Validate a scalar treated proportion in (0, 1)."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidDesignError(
                f"expected proportion must be a float in (0, 1), got "
                f"{type(value).__name__}."
            )
        if not (0.0 < value < 1.0):
            raise InvalidDesignError(
                f"expected proportion must be in (0, 1), got {value}."
            )
        return float(value)

    @staticmethod
    def _proportions_from_dict(expected: dict, labels: list) -> dict:
        """Normalize a dict of expected proportions over ``labels``."""
        if not isinstance(expected, dict):
            raise InvalidDesignError(
                "expected must be a dict mapping each group to a positive "
                "proportion."
            )
        if set(expected.keys()) != set(labels):
            raise InvalidDesignError(
                f"expected keys {sorted(map(str, expected.keys()))} must "
                f"match the assignment groups {sorted(map(str, labels))}."
            )
        values = list(expected.values())
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
            for v in values
        ):
            raise InvalidDesignError(
                "expected proportions must be positive numbers."
            )
        total = float(sum(values))
        return {label: float(expected[label]) / total for label in labels}
