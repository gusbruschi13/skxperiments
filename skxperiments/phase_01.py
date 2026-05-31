# Exceções — se comunicam bem
# Instancie cada exceção manualmente e leia as mensagens. 
# Esse é o momento de ajustar o tom e a clareza antes de elas aparecerem em produção.
from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    NotFittedError,
    InsufficientDataError,
    InvalidDesignError,
)

raise DesignEstimatorMismatch(
    estimator_name="BlockedDifferenceInMeans",
    received_type="CRDAssignment",
    expected_type="BlockedAssignment",
    suggestion="DifferenceInMeans"
)

# PotentialOutcomes — propriedades matemáticas
import numpy as np
from skxperiments.core import PotentialOutcomes

rng = np.random.default_rng(42)
y0 = rng.normal(40, 10, size=1000)
y1 = y0 + 2.5   # ATE verdadeiro = 2.5

po = PotentialOutcomes(y0=y0, y1=y1)

assert abs(po.ate - 2.5) < 1e-10   # deve ser exato
assert po.ite.shape == (1000,)
assert po.n == 1000

print(po.summary())
print(po.to_dataframe().head())
print(repr(po))

# CRDAssignment — imutabilidade do DataFrame original
import pandas as pd
import numpy as np
from skxperiments.core.assignment import CRDAssignment

df = pd.DataFrame({"user_id": range(100), "revenue": np.random.normal(50, 10, 100)})
cols_antes = set(df.columns)

treatment = np.array([1] * 50 + [0] * 50)
df_with_treatment = df.copy()
df_with_treatment["treatment"] = treatment

assignment = CRDAssignment(
    data=df_with_treatment,
    treatment_col="treatment",
    design=None,
    seed=42
)

# DataFrame original não foi tocado
assert set(df.columns) == cols_antes
assert assignment.n_treated_ + assignment.n_control_ == assignment.n_units_
assert len(set(assignment.treated_ids()) & set(assignment.control_ids())) == 0

# Results — contrato de output

from skxperiments.core.results import Results

r = Results(
    ate=0.142,
    se=0.056,
    ci=(0.031, 0.253),
    p_value=0.011,
    n_obs=1000,
    n_treated=500,
    n_control=500,
    estimator_name="DifferenceInMeans",
    design_name="CRD",
)

r.summary()                     # inspecionar formatação visual
assert r.is_significant()       # p=0.011 < alpha=0.05
assert "ate" in r.to_dict()
assert "se" in r.to_dict()
assert r.to_dataframe().shape == (1, len(r.to_dict()))

r2 = Results(ate=0.05, p_value=None)
assert not r2.is_significant()  # p_value None → False

# BaseEstimator — _check_is_fitted antes de implementar qualquer estimador

from skxperiments.core.base import BaseEstimator
from skxperiments.core.results import Results
from skxperiments.core.assignment import BaseAssignment

class DummyEstimator(BaseEstimator):
    def __init__(self, outcome: str):
        self.outcome = outcome
    def fit(self, assignment):
        self.fitted_ = True
        return self
    def estimate(self):
        return Results(ate=0.0)

est = DummyEstimator(outcome="revenue")

# Antes do fit: deve levantar NotFittedError
try:
    est._check_is_fitted()
except Exception as e:
    print(e)   # validar mensagem

# get_params e set_params
print(est.get_params())          # {"outcome": "revenue"}
est.set_params(outcome="conversion")
print(est.get_params())          # {"outcome": "conversion"}
print(repr(est))

# DiagnosticsReport — estados visuais

from skxperiments.core.base import DiagnosticsReport

r_clean = DiagnosticsReport()
r_clean.summary()    # deve imprimir ✅ No issues found.

r_issues = DiagnosticsReport(
    flags=["SRM detected: p=0.003"],
    warnings=["Small block size detected (min=3)"]
)
r_issues.summary()   # ❌ e ⚠️ formatados corretamente