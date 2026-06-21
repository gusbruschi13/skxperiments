"""Inference module for hypothesis testing and confidence intervals.

Phase 4: randomization-based inference, multiple testing correction,
Neyman variance, bootstrap.
"""

from skxperiments.inference.multiple import MultipleTestingCorrection
from skxperiments.inference.neyman import NeymanCI
from skxperiments.inference.randomization_test import RandomizationTest

__all__ = ["MultipleTestingCorrection", "NeymanCI", "RandomizationTest"]