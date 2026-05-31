"""Experimental design module.

Provides design objects (CRD, BlockedCRD, ReRandomizedCRD, FactorialDesign)
and standalone diagnostic functions (check_balance, power_analysis).
"""

from skxperiments.design.balance import check_balance
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.factorial import FactorialDesign
from skxperiments.design.power import PowerResult, power_analysis
from skxperiments.design.rerandomized_crd import ReRandomizedCRD

__all__ = [
    "BlockedCRD",
    "FactorialDesign",
    "PowerResult",
    "ReRandomizedCRD",
    "check_balance",
    "power_analysis",
]