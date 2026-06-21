"""Diagnostics module for checking design and estimation assumptions.

This module contains tools for balance checks, covariate diagnostics,
and other pre- and post-estimation diagnostics.
"""

from skxperiments.diagnostics.srm import SRMResult, SRMTest

__all__ = ["SRMResult", "SRMTest"]