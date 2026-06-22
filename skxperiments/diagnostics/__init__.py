"""Diagnostics module for checking design and estimation assumptions.

This module contains tools for balance checks, covariate diagnostics,
and other pre- and post-estimation diagnostics.
"""

from skxperiments.diagnostics.aa_test import AAResult, AATest
from skxperiments.diagnostics.balance_report import (
    BalanceReport,
    BalanceResult,
)
from skxperiments.diagnostics.srm import SRMResult, SRMTest

__all__ = [
    "AAResult",
    "AATest",
    "BalanceReport",
    "BalanceResult",
    "SRMResult",
    "SRMTest",
]