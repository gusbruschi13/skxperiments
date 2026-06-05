"""Abstract base classes for designs, estimators, and inference methods.

These classes define the API contract that all concrete implementations
must follow, ensuring consistency across the library.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InvalidDesignError,
    NotFittedError,
)

if TYPE_CHECKING:
    from skxperiments.core.assignment import BaseAssignment
    from skxperiments.core.results import Results


def _check_assignment_type(
    obj: Any,
    assignment: Any,
    expected_type: type | tuple[type, ...],
) -> None:
    """Validate that ``assignment`` is an instance of ``expected_type``.

    Module-level helper shared by ``BaseEstimator._validate_assignment_type``
    and ``BaseInference._validate_assignment_type``. Both ABCs expose
    thin wrappers that delegate here, so the validation logic and the
    error message format live in a single place.

    Parameters
    ----------
    obj : Any
        The estimator or inference instance calling this helper. Used
        to populate ``estimator_name`` in the raised exception.
    assignment : Any
        The assignment object to validate.
    expected_type : type or tuple of type
        Acceptable type(s) for ``assignment``.

    Raises
    ------
    DesignEstimatorMismatch
        If ``assignment`` is not an instance of any type in
        ``expected_type``.
    """
    if not isinstance(assignment, expected_type):
        if isinstance(expected_type, tuple):
            expected_name = " or ".join(t.__name__ for t in expected_type)
        else:
            expected_name = expected_type.__name__

        raise DesignEstimatorMismatch(
            estimator_name=type(obj).__name__,
            received_type=type(assignment).__name__,
            expected_type=expected_name,
        )


class BaseDesign(ABC):
    """Abstract base class for all experimental designs.

    Subclasses must implement the randomize() method, which takes a
    DataFrame and returns a BaseAssignment object.

    Examples
    --------
    Subclasses define their parameters in __init__:

    >>> class CRD(BaseDesign):
    ...     def __init__(self, n_treated=None, seed=None):
    ...         self.n_treated = n_treated
    ...         self.seed = seed
    ...     def randomize(self, df):
    ...         ...
    """

    @abstractmethod
    def randomize(self, df: pd.DataFrame) -> "BaseAssignment":
        """Perform randomization and return an Assignment object.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the experimental units.

        Returns
        -------
        BaseAssignment
            Assignment object with treatment assignments.
        """

    def get_params(self) -> dict[str, Any]:
        """Get parameters of this design.

        Uses inspect.signature to introspect __init__ parameters.
        Works in subclasses without override.

        Returns
        -------
        dict
            Parameter names mapped to their current values.
        """
        sig = inspect.signature(self.__init__)  # type: ignore[misc]
        params = {}
        for name in sig.parameters:
            if name == "self":
                continue
            params[name] = getattr(self, name, None)
        return params

    def set_params(self, **params: Any) -> "BaseDesign":
        """Set parameters of this design.

        Parameters
        ----------
        **params
            Keyword arguments with parameter names and new values.

        Returns
        -------
        BaseDesign
            Returns self.

        Raises
        ------
        InvalidDesignError
            If a parameter name does not exist.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise InvalidDesignError(
                    f"Invalid parameter '{key}' for {type(self).__name__}. "
                    f"Valid parameters: {list(valid_params.keys())}."
                )
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation with parameters.

        Returns
        -------
        str
            Format: ClassName(param1=val1, param2=val2)
        """
        params = self.get_params()
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{type(self).__name__}({params_str})"


class BaseEstimator(ABC):
    """Abstract base class for all causal estimators.

    Subclasses must implement fit() and estimate() methods.
    The fit() method receives a BaseAssignment object (not a DataFrame).

    Examples
    --------
    >>> class DifferenceInMeans(BaseEstimator):
    ...     def __init__(self, outcome_col="y"):
    ...         self.outcome_col = outcome_col
    ...     def fit(self, assignment):
    ...         ...
    ...     def estimate(self):
    ...         ...
    """

    @abstractmethod
    def fit(self, assignment: Any) -> "BaseEstimator":
        """Fit the estimator using an assignment object.

        Parameters
        ----------
        assignment : BaseAssignment
            Assignment object containing data and treatment assignments.

        Returns
        -------
        BaseEstimator
            Returns self.
        """

    @abstractmethod
    def estimate(self) -> "Results":
        """Compute the causal estimate.

        Returns
        -------
        Results
            Results object with the estimate and metadata.
        """

    def get_params(self) -> dict[str, Any]:
        """Get parameters of this estimator.

        Returns
        -------
        dict
            Parameter names mapped to their current values.
        """
        sig = inspect.signature(self.__init__)  # type: ignore[misc]
        params = {}
        for name in sig.parameters:
            if name == "self":
                continue
            params[name] = getattr(self, name, None)
        return params

    def set_params(self, **params: Any) -> "BaseEstimator":
        """Set parameters of this estimator.

        Parameters
        ----------
        **params
            Keyword arguments with parameter names and new values.

        Returns
        -------
        BaseEstimator
            Returns self.

        Raises
        ------
        InvalidDesignError
            If a parameter name does not exist.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise InvalidDesignError(
                    f"Invalid parameter '{key}' for {type(self).__name__}. "
                    f"Valid parameters: {list(valid_params.keys())}."
                )
            setattr(self, key, value)
        return self

    def _check_is_fitted(self) -> None:
        """Check if the estimator has been fitted.

        Raises
        ------
        NotFittedError
            If no attributes ending in underscore are found.
        """
        fitted_attrs = [
            attr for attr in self.__dict__ if attr.endswith("_")
        ]
        if not fitted_attrs:
            raise NotFittedError(
                class_name=type(self).__name__,
                required_methods=["fit"],
            )

    def _validate_assignment_type(
        self,
        assignment: Any,
        expected_type: type | tuple[type, ...],
    ) -> None:
        """Validate that the assignment is of the expected type(s).

        Thin wrapper that delegates to the module-level
        ``_check_assignment_type``. Kept as a method for API
        compatibility with concrete estimators that call
        ``self._validate_assignment_type(...)``.

        Parameters
        ----------
        assignment : Any
            The assignment object to validate.
        expected_type : type or tuple of type
            The expected type(s) of the assignment. A tuple may be passed
            when the estimator accepts multiple Assignment types (e.g.,
            LinEstimator accepts both CRDAssignment and BlockedAssignment).

        Raises
        ------
        DesignEstimatorMismatch
            If the assignment type does not match any of the expected types.
        """
        _check_assignment_type(self, assignment, expected_type)

    def __repr__(self) -> str:
        """Return string representation with parameters.

        Returns
        -------
        str
            Format: ClassName(param1=val1, param2=val2)
        """
        params = self.get_params()
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{type(self).__name__}({params_str})"


class BaseInference(ABC):
    """Abstract base class for all inference methods.

    Subclasses configure their dependencies (e.g., a wrapped estimator)
    in ``__init__``, receive a ``BaseAssignment`` in ``fit()``, and
    produce a new ``Results`` object via ``estimate()``.

    Subclasses **must** implement both ``fit()`` and ``estimate()``.

    Contract
    --------
    - ``fit(assignment)`` populates instance attributes ending in
      underscore (e.g., ``observed_statistic_``, ``p_value_``).
    - ``estimate()`` produces a **new** ``Results`` object. It must
      not mutate the estimator's ``Results`` or any other input.
    - Subclasses may accept arbitrary parameters in ``__init__`` (e.g.,
      a configured estimator, ``n_permutations``, ``alpha``).

    Examples
    --------
    >>> class RandomizationTest(BaseInference):
    ...     def __init__(self, estimator, n_permutations=1000):
    ...         self.estimator = estimator
    ...         self.n_permutations = n_permutations
    ...     def fit(self, assignment):
    ...         ...
    ...     def estimate(self):
    ...         ...
    """

    @abstractmethod
    def fit(self, assignment: Any) -> "BaseInference":
        """Fit the inference method using an assignment object.

        Parameters
        ----------
        assignment : BaseAssignment
            Assignment object containing data and treatment assignments.

        Returns
        -------
        BaseInference
            Returns self.
        """

    @abstractmethod
    def estimate(self) -> "Results":
        """Compute the inferential result.

        Returns
        -------
        Results
            Results object with point estimate (copied from the underlying
            estimator) and inferential quantities (p_value, ci, se as
            applicable) populated. Always returns a NEW Results object;
            never mutates the estimator's Results.
        """

    def get_params(self) -> dict[str, Any]:
        """Get parameters of this inference method.

        Returns
        -------
        dict
            Parameter names mapped to their current values.
        """
        sig = inspect.signature(self.__init__)  # type: ignore[misc]
        params = {}
        for name in sig.parameters:
            if name == "self":
                continue
            params[name] = getattr(self, name, None)
        return params

    def set_params(self, **params: Any) -> "BaseInference":
        """Set parameters of this inference method.

        Parameters
        ----------
        **params
            Keyword arguments with parameter names and new values.

        Returns
        -------
        BaseInference
            Returns self.

        Raises
        ------
        InvalidDesignError
            If a parameter name does not exist.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise InvalidDesignError(
                    f"Invalid parameter '{key}' for {type(self).__name__}. "
                    f"Valid parameters: {list(valid_params.keys())}."
                )
            setattr(self, key, value)
        return self

    def _check_is_fitted(self) -> None:
        """Check if the inference method has been fitted.

        Raises
        ------
        NotFittedError
            If no attributes ending in underscore are found.
        """
        fitted_attrs = [
            attr for attr in self.__dict__ if attr.endswith("_")
        ]
        if not fitted_attrs:
            raise NotFittedError(
                class_name=type(self).__name__,
                required_methods=["fit"],
            )

    def _validate_assignment_type(
        self,
        assignment: Any,
        expected_type: type | tuple[type, ...],
    ) -> None:
        """Validate that the assignment is of the expected type(s).

        Thin wrapper that delegates to the module-level
        ``_check_assignment_type``. Mirrors
        ``BaseEstimator._validate_assignment_type`` so inference
        classes have the same validation surface as estimators.

        Parameters
        ----------
        assignment : Any
            The assignment object to validate.
        expected_type : type or tuple of type
            The expected type(s) of the assignment. A tuple may be passed
            when the inference method accepts multiple Assignment types
            (e.g., RandomizationTest accepts both CRDAssignment and
            BlockedAssignment).

        Raises
        ------
        DesignEstimatorMismatch
            If the assignment type does not match any of the expected types.
        """
        _check_assignment_type(self, assignment, expected_type)

    def __repr__(self) -> str:
        """Return string representation with parameters.

        Returns
        -------
        str
            Format: ClassName(param1=val1, param2=val2)
        """
        params = self.get_params()
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{type(self).__name__}({params_str})"


@dataclass
class DiagnosticsReport:
    """Report containing diagnostic flags and warnings.

    Attributes
    ----------
    flags : list of str
        Critical issues that should be addressed.
    warnings : list of str
        Non-critical warnings for the user's attention.

    Examples
    --------
    >>> report = DiagnosticsReport()
    >>> report.summary()
    ✅ No issues found.
    """

    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> None:
        """Print diagnostic summary to stdout.

        Prints flags with ❌ prefix, warnings with ⚠️ prefix.
        If both are empty, prints ✅ No issues found.
        """
        if not self.flags and not self.warnings:
            print("✅ No issues found.")
            return

        for flag in self.flags:
            print(f"❌ {flag}")

        for warning in self.warnings:
            print(f"⚠️ {warning}")

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            Format: DiagnosticsReport(flags=N, warnings=N)
        """
        return (
            f"DiagnosticsReport(flags={len(self.flags)}, "
            f"warnings={len(self.warnings)})"
        )