"""Core module providing base classes, exceptions, and fundamental data structures.

This module contains the foundational components of skxperiments:
- Custom exceptions for clear error reporting
- PotentialOutcomes for representing unit-level causal quantities
- Assignment classes as the contract between designs and estimators
- Results as the uniform output object
- Abstract base classes for designs, estimators, and inference methods
"""

from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InsufficientDataError,
    InvalidDesignError,
    NotFittedError,
    SkxperimentsError,
)
from skxperiments.core.potential_outcomes import PotentialOutcomes
from skxperiments.core.assignment import BaseAssignment, CRDAssignment
from skxperiments.core.results import Results
from skxperiments.core.base import (
    BaseDesign,
    BaseEstimator,
    BaseInference,
    DiagnosticsReport,
)

__all__ = [
    "SkxperimentsError",
    "DesignEstimatorMismatch",
    "NotFittedError",
    "InsufficientDataError",
    "InvalidDesignError",
    "PotentialOutcomes",
    "BaseAssignment",
    "CRDAssignment",
    "Results",
    "BaseDesign",
    "BaseEstimator",
    "BaseInference",
    "DiagnosticsReport",
]