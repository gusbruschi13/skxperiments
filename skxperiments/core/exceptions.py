"""Custom exceptions for the skxperiments library.

All exceptions inherit from SkxperimentsError, enabling users to catch
all library-specific errors with a single except clause.
"""


class SkxperimentsError(Exception):
    """Base exception for all skxperiments errors.

    Parameters
    ----------
    message : str
        Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class DesignEstimatorMismatch(SkxperimentsError):
    """Raised when an estimator receives an incompatible Assignment type.

    Parameters
    ----------
    estimator_name : str
        Name of the estimator that detected the mismatch.
    received_type : str
        Name of the Assignment type that was received.
    expected_type : str
        Name of the Assignment type that was expected.
    suggestion : str or None, optional
        Suggested alternative estimator or design, by default None.

    Examples
    --------
    >>> raise DesignEstimatorMismatch(
    ...     estimator_name="DifferenceInMeans",
    ...     received_type="BlockedAssignment",
    ...     expected_type="CRDAssignment",
    ...     suggestion="BlockedDifferenceInMeans",
    ... )
    """

    def __init__(
        self,
        estimator_name: str,
        received_type: str,
        expected_type: str,
        suggestion: str | None = None,
    ) -> None:
        self.estimator_name = estimator_name
        self.received_type = received_type
        self.expected_type = expected_type
        self.suggestion = suggestion

        message = (
            f"[{estimator_name}] expects {expected_type} "
            f"but received {received_type}."
        )
        if suggestion is not None:
            message += f" Suggestion: use {suggestion} instead."

        super().__init__(message)


class NotFittedError(SkxperimentsError):
    """Raised when methods dependent on fit() are called before fitting.

    Parameters
    ----------
    class_name : str
        Name of the class that is not yet fitted.
    required_methods : list of str
        Methods that must be called before using the object.

    Examples
    --------
    >>> raise NotFittedError(
    ...     class_name="DifferenceInMeans",
    ...     required_methods=["fit"],
    ... )
    """

    def __init__(self, class_name: str, required_methods: list[str]) -> None:
        self.class_name = class_name
        self.required_methods = required_methods

        methods_str = ", ".join(f"{m}()" for m in required_methods)
        message = (
            f"[{class_name}] is not fitted. "
            f"Call {methods_str} before using this object."
        )
        super().__init__(message)


class InsufficientDataError(SkxperimentsError):
    """Raised when the number of units or block size is insufficient.

    Parameters
    ----------
    context : str
        Description of the operation requiring more data.
    minimum : int
        Minimum number of units required.
    received : int
        Actual number of units received.

    Examples
    --------
    >>> raise InsufficientDataError(
    ...     context="CRD randomization",
    ...     minimum=2,
    ...     received=1,
    ... )
    """

    def __init__(self, context: str, minimum: int, received: int) -> None:
        self.context = context
        self.minimum = minimum
        self.received = received

        message = (
            f"{context} requires at least {minimum} units, "
            f"but received {received}."
        )
        super().__init__(message)


class InvalidDesignError(SkxperimentsError):
    """Raised when design parameters are inconsistent.

    Parameters
    ----------
    message : str
        Description of the inconsistency.

    Examples
    --------
    >>> raise InvalidDesignError("Treatment probability must be in (0, 1).")
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)