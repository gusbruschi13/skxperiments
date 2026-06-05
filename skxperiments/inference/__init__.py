"""Inference module for hypothesis testing and confidence intervals.

Phase 4: randomization-based inference, Neyman variance, bootstrap,
multiple testing correction, sequential tests.
"""

from skxperiments.inference.randomization_test import RandomizationTest

__all__ = ["RandomizationTest"]