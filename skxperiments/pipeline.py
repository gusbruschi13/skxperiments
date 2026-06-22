"""Experiment composition: pipeline and comparison.

``ExperimentPipeline`` ties together the analysis-time pieces of an
experiment — an inference procedure (which already wraps an estimator)
and a set of diagnostics — and runs them against a single ``Assignment``.
The design is not a separate argument: it travels with the assignment
(``assignment.design_``), and the estimator travels inside the inference.

Diagnostics run best-effort: a diagnostic that cannot run on the given
assignment (raising a library ``SkxperimentsError``) is recorded as a
warning rather than aborting the whole analysis. By default a flagged
diagnostic (e.g., a Sample Ratio Mismatch) is surfaced prominently but
does not stop estimation; pass ``raise_on_flag=True`` to abort instead.
"""

from dataclasses import dataclass

import pandas as pd

from skxperiments.core.assignment import BaseAssignment
from skxperiments.core.base import BaseInference, DiagnosticsReport
from skxperiments.core.exceptions import InvalidDesignError, SkxperimentsError
from skxperiments.core.results import Results
from skxperiments.diagnostics.srm import SRMTest
from skxperiments.inference.multiple import MultipleTestingCorrection


@dataclass(frozen=True)
class PipelineResult:
    """Bundle returned by ``ExperimentPipeline.run``.

    Attributes
    ----------
    results : Results
        The inference result (ATE, SE/CI/p-value as applicable).
    diagnostics : DiagnosticsReport
        Merged flags and warnings across all diagnostics that ran.
    diagnostic_results : dict
        Mapping from diagnostic class name to its full result object
        (e.g., ``{"SRMTest": SRMResult, "BalanceReport": BalanceResult}``).
        Diagnostics that could not run are absent here and appear as a
        warning in ``diagnostics`` instead.
    """

    results: Results
    diagnostics: DiagnosticsReport
    diagnostic_results: dict

    @property
    def flagged(self) -> bool:
        """True if any diagnostic raised a flag."""
        return len(self.diagnostics.flags) > 0

    def summary(self) -> "PipelineResult":
        """Print diagnostics then the inference result; return self."""
        print("Experiment Pipeline")
        print("===================")
        self.diagnostics.summary()
        self.results.summary()
        return self


class ExperimentPipeline:
    """Compose diagnostics and inference for a single experiment.

    Parameters
    ----------
    inference : BaseInference
        A configured inference object (which already wraps an estimator),
        e.g. ``NeymanCI(DifferenceInMeans("y"))`` or
        ``RandomizationTest(...)``.
    diagnostics : list or None, optional
        Diagnostics to run before inference. Each must expose a
        ``run(assignment)`` method returning an object with a
        ``to_diagnostics_report()`` method (e.g., ``SRMTest``,
        ``BalanceReport``). By default ``[SRMTest()]``. Pass ``[]`` to
        skip diagnostics. ``AATest`` is not a per-assignment diagnostic
        (it re-randomizes a design) and is not used here.
    raise_on_flag : bool, optional
        If True, a flagged diagnostic aborts the run with
        ``InvalidDesignError`` before inference. By default False — flags
        are surfaced in the result but estimation proceeds.

    Notes
    -----
    Diagnostics run best-effort: one that raises a ``SkxperimentsError``
    (e.g., ``SRMTest`` on a design with no intended proportion) is
    recorded as a warning and skipped. Errors from the inference itself
    are not caught.
    """

    def __init__(
        self,
        inference: BaseInference,
        diagnostics: list | None = None,
        raise_on_flag: bool = False,
    ) -> None:
        if not isinstance(inference, BaseInference):
            raise InvalidDesignError(
                f"inference must be an instance of BaseInference, got "
                f"{type(inference).__name__}."
            )

        if diagnostics is None:
            diagnostics = [SRMTest()]
        elif not isinstance(diagnostics, list):
            raise InvalidDesignError(
                f"diagnostics must be a list or None, got "
                f"{type(diagnostics).__name__}."
            )
        else:
            for diag in diagnostics:
                if not callable(getattr(diag, "run", None)):
                    raise InvalidDesignError(
                        "each diagnostic must expose a callable "
                        "run(assignment) method; got "
                        f"{type(diag).__name__}."
                    )

        if not isinstance(raise_on_flag, bool):
            raise InvalidDesignError(
                f"raise_on_flag must be a bool, got "
                f"{type(raise_on_flag).__name__}."
            )

        self.inference = inference
        self.diagnostics = diagnostics
        self.raise_on_flag = raise_on_flag

    def run(self, assignment: BaseAssignment) -> PipelineResult:
        """Run diagnostics then inference on an assignment.

        Parameters
        ----------
        assignment : BaseAssignment
            The (already randomized, outcome-bearing) assignment.

        Returns
        -------
        PipelineResult

        Raises
        ------
        InvalidDesignError
            If ``raise_on_flag`` is True and a diagnostic is flagged.
            Errors from the wrapped inference propagate unchanged.
        """
        combined = DiagnosticsReport()
        diagnostic_results: dict = {}

        for diag in self.diagnostics:
            name = type(diag).__name__
            try:
                diag_result = diag.run(assignment)
            except SkxperimentsError as exc:
                combined.warnings.append(f"{name} could not run: {exc}")
                continue
            diagnostic_results[name] = diag_result
            report = diag_result.to_diagnostics_report()
            combined.flags.extend(report.flags)
            combined.warnings.extend(report.warnings)

        if self.raise_on_flag and combined.flags:
            joined = "\n- ".join(combined.flags)
            raise InvalidDesignError(
                f"ExperimentPipeline halted by diagnostic flags:\n- {joined}"
            )

        self.inference.fit(assignment)
        results = self.inference.estimate()

        return PipelineResult(
            results=results,
            diagnostics=combined,
            diagnostic_results=diagnostic_results,
        )


@dataclass(frozen=True, eq=False)
class ComparisonResult:
    """Result of comparing several independent experiments.

    Attributes
    ----------
    corrected_results : dict
        Mapping from experiment name to its multiple-testing-corrected
        ``Results`` (corrected ``p_value``; original recorded in
        ``Results.extra["original_p_values"]``).
    correction : str
        The correction method applied (``"bonferroni"``, ``"holm"``,
        or ``"bh"``).
    alpha : float
        Family-wise significance level.
    table : pd.DataFrame
        One row per experiment with columns ``experiment``, ``ate``,
        ``se``, ``ci_lower``, ``ci_upper``, ``p_value`` (original),
        ``p_value_corrected``, and ``significant``.
    """

    corrected_results: dict
    correction: str
    alpha: float
    table: pd.DataFrame

    @property
    def significant(self) -> list[str]:
        """Names of experiments significant after correction."""
        return [
            row.experiment
            for row in self.table.itertuples()
            if row.significant
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the comparison table."""
        return self.table.copy()

    def to_dict(self) -> dict:
        """Return the summary fields as a plain dictionary."""
        return {
            "correction": self.correction,
            "alpha": self.alpha,
            "n_experiments": len(self.corrected_results),
            "significant": self.significant,
        }

    def summary(self) -> "ComparisonResult":
        """Print the comparison table and return self."""
        print(f"Experiment Comparison (correction={self.correction}, "
              f"alpha={self.alpha})")
        print(self.table.to_string(index=False))
        return self


class ExperimentComparison:
    """Compare several independent experiments with multiple-testing control.

    Collects the scalar ATE/p-value of each experiment and applies a
    multiple-testing correction across the family, so the family-wise
    error rate (Bonferroni/Holm) or false discovery rate (BH) is
    controlled across experiments.

    Parameters
    ----------
    correction : {"holm", "bonferroni", "bh"}, optional
        Correction method, by default ``"holm"``.
    alpha : float, optional
        Family-wise significance level, by default 0.05.

    Notes
    -----
    v1 compares **independent experiments** (one scalar ATE each).
    Subgroup analysis (multiple effects within a single experiment) is
    deferred to v2 (see `ROADMAP.md`); multi-effect ``Results`` are
    rejected here.
    """

    def __init__(
        self,
        correction: str = "holm",
        alpha: float = 0.05,
    ) -> None:
        # Validates the method and alpha eagerly.
        self._mtc = MultipleTestingCorrection(method=correction, alpha=alpha)
        self.correction = correction
        self.alpha = alpha

    def run(self, results: dict) -> ComparisonResult:
        """Compare a mapping of named experiment results.

        Parameters
        ----------
        results : dict
            Mapping from experiment name to a ``PipelineResult`` or a
            scalar ``Results``. ``PipelineResult`` is unwrapped to its
            ``results`` field.

        Returns
        -------
        ComparisonResult

        Raises
        ------
        InvalidDesignError
            If ``results`` is not a non-empty dict, if any entry is not a
            scalar ``Results``/``PipelineResult``, or if any lacks a
            scalar p-value.
        """
        if not isinstance(results, dict):
            raise InvalidDesignError(
                f"results must be a dict {{name: PipelineResult | Results}}, "
                f"got {type(results).__name__}."
            )
        if len(results) == 0:
            raise InvalidDesignError(
                "results is empty; nothing to compare."
            )

        names = list(results.keys())
        scalar: list[Results] = []
        for name, entry in results.items():
            res = entry.results if isinstance(entry, PipelineResult) else entry
            if not isinstance(res, Results):
                raise InvalidDesignError(
                    f"Experiment '{name}' must be a PipelineResult or "
                    f"Results, got {type(entry).__name__}."
                )
            if res.ate is None or res.effects is not None:
                raise InvalidDesignError(
                    f"Experiment '{name}' is not in scalar mode (ate must "
                    f"be set, effects must be None). Multi-effect/subgroup "
                    f"comparison is deferred to v2."
                )
            if res.p_value is None or not isinstance(res.p_value, (int, float)):
                raise InvalidDesignError(
                    f"Experiment '{name}' has no scalar p_value; run an "
                    f"inference that produces one before comparing."
                )
            scalar.append(res)

        corrected = self._mtc.correct(scalar)
        corrected_results = dict(zip(names, corrected))
        table = self._build_table(names, scalar, corrected)

        return ComparisonResult(
            corrected_results=corrected_results,
            correction=self.correction,
            alpha=self.alpha,
            table=table,
        )

    def _build_table(
        self,
        names: list[str],
        scalar: list[Results],
        corrected: list[Results],
    ) -> pd.DataFrame:
        """Assemble the per-experiment comparison table."""
        rows = []
        for name, orig, corr in zip(names, scalar, corrected):
            ci = orig.ci
            rows.append(
                {
                    "experiment": name,
                    "ate": orig.ate,
                    "se": orig.se,
                    "ci_lower": ci[0] if ci is not None else None,
                    "ci_upper": ci[1] if ci is not None else None,
                    "p_value": orig.p_value,
                    "p_value_corrected": corr.p_value,
                    "significant": bool(corr.p_value < self.alpha),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "experiment",
                "ate",
                "se",
                "ci_lower",
                "ci_upper",
                "p_value",
                "p_value_corrected",
                "significant",
            ],
        )
