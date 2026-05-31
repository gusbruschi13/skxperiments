"""Causal estimators.

Estimators consume Assignment objects produced by designs and return
Results objects. Inference (SE, CI, p-value) is the responsibility of
inference classes (Phase 4); estimators here compute point estimates
only.
"""

from skxperiments.estimators.blocked_difference_in_means import (
    BlockedDifferenceInMeans,
)
from skxperiments.estimators.cuped import CUPED
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.estimators.factorial_estimator import FactorialEstimator
from skxperiments.estimators.lin_estimator import LinEstimator

__all__ = [
    "BlockedDifferenceInMeans",
    "CUPED",
    "DifferenceInMeans",
    "FactorialEstimator",
    "LinEstimator",
]