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

from skxperiments.core.assignment import BaseAssignment
from skxperiments.core.base import BaseInference, DiagnosticsReport
from skxperiments.core.exceptions import InvalidDesignError, SkxperimentsError
from skxperiments.core.results import Results
from skxperiments.diagnostics.srm import SRMTest


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
