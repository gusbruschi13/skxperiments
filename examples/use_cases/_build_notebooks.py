"""Build the bilingual use_cases notebooks (pt-br + en) from a single source.

Each notebook is defined once as an ordered list of blocks. A block is either
shared code (identical in both languages) or a bilingual markdown cell. This
guarantees the two language versions never drift structurally. Re-run after
editing content:

    python examples/use_cases/_build_notebooks.py

The generated .ipynb files are the committed artifacts; the datasets they read
come from ``_generate_data.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNEL_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

BOOT = '''from pathlib import Path

import numpy as np
import pandas as pd


def _find_data():
    """Locate examples/use_cases/data whether run from the notebook or the root."""
    for base in [Path.cwd(), *Path.cwd().parents]:
        for cand in (base / "data", base / "examples" / "use_cases" / "data"):
            if (cand / "ecommerce_checkout.csv").exists():
                return cand
    raise FileNotFoundError("Could not locate examples/use_cases/data")


DATA = _find_data()'''


def _src(text: str) -> list[str]:
    text = text.strip("\n")
    return (text + "\n").splitlines(keepends=True) if text else []


def md(cell_src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(cell_src)}


def code(cell_src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(cell_src),
    }


def build(blocks: list, lang: str) -> dict:
    cells = []
    for kind, payload in blocks:
        if kind == "code":
            cells.append(code(payload))
        else:
            cells.append(md(payload[lang]))
    return {"cells": cells, "metadata": KERNEL_META, "nbformat": 4, "nbformat_minor": 4}


def write(blocks: list, filename: str) -> None:
    for lang in ("pt-br", "en"):
        out = ROOT / lang / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        nb = build(blocks, lang)
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out}")


def B(pt: str, en: str) -> tuple:
    return ("md", {"pt-br": pt, "en": en})


def C(src: str) -> tuple:
    return ("code", src)


# =====================================================================
# 01 - e-commerce checkout (CRD + power + SRM + CUPED + Neyman/Bootstrap)
# =====================================================================
nb01 = [
    B(
        pt="""
# 01. Checkout de e-commerce: planejar, medir e reduzir variância

**Setor:** e-commerce de eletrônicos. **Decisão:** vale a pena lançar o novo
checkout de 1 clique? Queremos o efeito na **receita por usuário**.

Este caso percorre um fluxo completo de teste A/B online:

1. **Planejar** o tamanho da amostra (`power_analysis`) a partir da variância
   histórica da receita.
2. **Desenhar** com randomização completa (`CRD`) e **checar SRM** antes de
   olhar qualquer resultado.
3. **Estimar** o efeito com a variância de Neyman (população finita) e comparar
   com o bootstrap (superpopulação).
4. **Reduzir variância** com `CUPED`, usando o gasto pré-experimento como
   covariável.

Base teórica: [II. Fundamentos e desenhos](../../../docs/pt-br/teoria/01-fundamentos.md),
[IV. Inferência](../../../docs/pt-br/teoria/04-inferencia.md) (tópicos 14, 15 e 17)
e [III. Estimação](../../../docs/pt-br/teoria/03-estimacao.md) (tópico 11, CUPED).
""",
        en="""
# 01. E-commerce checkout: plan, measure, and reduce variance

**Sector:** e-commerce electronics. **Decision:** is the new one-click checkout
worth launching? We want the effect on **revenue per user**.

This case walks through a complete online A/B flow:

1. **Plan** the sample size (`power_analysis`) from the historical revenue
   variance.
2. **Design** with complete randomization (`CRD`) and **check SRM** before
   looking at any result.
3. **Estimate** the effect with the Neyman variance (finite population) and
   compare with the bootstrap (superpopulation).
4. **Reduce variance** with `CUPED`, using pre-experiment spend as the
   covariate.

Theory: [I. Foundations](../../../docs/en/theory/01-foundations.md),
[IV. Inference](../../../docs/en/theory/04-inference.md) (topics 14, 15 and 17)
and [III. Estimation](../../../docs/en/theory/03-estimation.md) (topic 11, CUPED).
""",
    ),
    C(
        BOOT
        + '''

df = pd.read_csv(DATA / "ecommerce_checkout.csv")
print(df.shape)
df.head()'''
    ),
    B(
        pt="""
## 1. Quantos usuários? (poder)

A receita por usuário tem uma cauda longa (poucos usuários gastam muito), então
sua variância é alta. Usamos o desvio padrão histórico (a coluna `revenue_y0`, o
comportamento sem tratamento) para dimensionar o teste para um efeito mínimo
detectável (MDE) de `+2,5` na receita, com poder 80% e `alpha = 0,05`.
""",
        en="""
## 1. How many users? (power)

Revenue per user has a long tail (a few users spend a lot), so its variance is
high. We use the historical standard deviation (the `revenue_y0` column, the
no-treatment behavior) to size the test for a minimum detectable effect (MDE) of
`+2.5` in revenue, with 80% power and `alpha = 0.05`.
""",
    ),
    C(
        '''from skxperiments.design.power import power_analysis

std_hist = float(df["revenue_y0"].std())
plan = power_analysis(mde=2.5, power=0.8, std=std_hist, alpha=0.05)
print(f"historical std: {std_hist:.1f}")
print(f"required n_total: {plan.n_total}  (~{plan.n_treated} per arm)")
print(f"we collected {len(df)} users, comfortably above the plan")'''
    ),
    C(
        '''from skxperiments.reporting import plot_power_curve

n_values = list(range(200, 4001, 200))
ax = plot_power_curve(n_values, mde=2.5, std=std_hist, alpha=0.05)
ax.set_title("Power vs. sample size (MDE = 2.5)")
ax.figure'''
    ),
    B(
        pt="""
## 2. Desenho e checagem de SRM

Randomizamos os usuários 50/50 com `CRD`. Depois construímos o resultado
observado: cada usuário revela `revenue_y1` se caiu no tratamento e `revenue_y0`
caso contrário (a "tabela da ciência" do dado sintético). Antes de estimar
qualquer coisa, rodamos o `SRMTest`: se a proporção observada não bate com a
planejada, algo quebrou e nada mais é confiável.
""",
        en="""
## 2. Design and SRM check

We randomize users 50/50 with `CRD`. Then we build the observed outcome: each
user reveals `revenue_y1` if assigned to treatment and `revenue_y0` otherwise
(the "science table" of the synthetic data). Before estimating anything, we run
the `SRMTest`: if the observed proportion does not match the planned one,
something broke and nothing else is trustworthy.
""",
    ),
    C(
        '''from skxperiments.core.assignment import CRDAssignment
from skxperiments.design.crd import CRD
from skxperiments.diagnostics import SRMTest

design = CRD(p=0.5, seed=101)
assignment = design.randomize(df[["device", "sessions_pre", "spend_pre"]].copy())

t = assignment.data_[assignment.treatment_col_].to_numpy()
data = assignment.data_.copy()
data["revenue"] = np.where(t == 1, df["revenue_y1"].to_numpy(), df["revenue_y0"].to_numpy())
assignment = CRDAssignment(
    data=data, treatment_col=assignment.treatment_col_, design=design, seed=101
)

srm = SRMTest().run(assignment)
print(f"SRM flagged: {srm.flagged}  (p={srm.p_value:.3f})")'''
    ),
    B(
        pt="""
## 3. Estimar o efeito e reduzir a variância

- `DifferenceInMeans` + `NeymanCI`: o ponto e o intervalo de confiança na visão
  de **população finita** (SATE).
- `BootstrapCI`: a visão de **superpopulação** (PATE), reamostrando dentro de
  cada braço.
- `CUPED`: usa o gasto pré-experimento (`spend_pre`), correlacionado com a
  receita, para remover ruído. O erro padrão cai sem introduzir viés.
""",
        en="""
## 3. Estimate the effect and reduce the variance

- `DifferenceInMeans` + `NeymanCI`: the point and confidence interval in the
  **finite-population** view (SATE).
- `BootstrapCI`: the **superpopulation** view (PATE), resampling within each arm.
- `CUPED`: uses pre-experiment spend (`spend_pre`), correlated with revenue, to
  remove noise. The standard error drops without introducing bias.
""",
    ),
    C(
        '''from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.estimators.cuped import CUPED
from skxperiments.inference import NeymanCI, BootstrapCI

neyman = NeymanCI(estimator=DifferenceInMeans(outcome_col="revenue")).fit(assignment).estimate()
boot = BootstrapCI(
    estimator=DifferenceInMeans(outcome_col="revenue"), n_resamples=1000, seed=0
).fit(assignment).estimate()
cuped = BootstrapCI(
    estimator=CUPED(outcome_col="revenue", pre_experiment_col="spend_pre"),
    n_resamples=1000, seed=0,
).fit(assignment).estimate()

pd.DataFrame([
    {"method": "DIM / Neyman (SATE)", "ATE": neyman.ate, "SE": neyman.se,
     "CI_low": neyman.ci[0], "CI_high": neyman.ci[1]},
    {"method": "DIM / Bootstrap (PATE)", "ATE": boot.ate, "SE": boot.se,
     "CI_low": boot.ci[0], "CI_high": boot.ci[1]},
    {"method": "CUPED / Bootstrap", "ATE": cuped.ate, "SE": cuped.se,
     "CI_low": cuped.ci[0], "CI_high": cuped.ci[1]},
]).round(3)'''
    ),
    C(
        '''from skxperiments.reporting import plot_effect

ax = plot_effect(neyman)
ax.set_title("Revenue lift (Neyman 95% CI)")
ax.figure'''
    ),
    B(
        pt="""
## Decisão

O efeito verdadeiro embutido no dado é `+2,5`. As três abordagens concordam no
ponto (perto de `+2,5`), e o intervalo exclui zero: o novo checkout **aumenta a
receita**. O `CUPED` entrega o **menor erro padrão** (o gasto pré-experimento
explica boa parte da variância da receita), então é a leitura mais precisa para
sustentar a decisão de lançar.

Próximo passo natural: [02. Lojas com blocagem](02_fashion_stores_blocking.ipynb).
""",
        en="""
## Decision

The true effect baked into the data is `+2.5`. All three approaches agree on the
point (close to `+2.5`), and the interval excludes zero: the new checkout
**increases revenue**. `CUPED` delivers the **smallest standard error**
(pre-experiment spend explains much of the revenue variance), so it is the most
precise reading to support the launch decision.

Natural next step: [02. Stores with blocking](02_fashion_stores_blocking.ipynb).
""",
    ),
]

# =====================================================================
# 02 - fashion stores (BlockedCRD + BlockedDIM + balance + naive contrast)
# =====================================================================
nb02 = [
    B(
        pt="""
# 02. Lojas de moda: blocagem quando as unidades são heterogêneas

**Setor:** varejo físico de moda. **Decisão:** um novo layout de vitrine
aumenta as vendas? As unidades são **120 lojas**, e elas são muito diferentes
entre si: uma loja `G` vende muito mais que uma `PP`, independentemente do
layout. Essa variação entre tamanhos, se não for controlada, "afoga" o efeito.

A resposta é **blocar por tamanho de loja** (`PP`, `P`, `M`, `G`): randomizar
dentro de cada bloco e combinar por média ponderada. Base teórica:
[II. Desenhos](../../../docs/pt-br/teoria/02-desenhos.md) (tópico 5) e
[III. Estimação](../../../docs/pt-br/teoria/03-estimacao.md) (tópico 9).
""",
        en="""
# 02. Fashion stores: blocking when the units are heterogeneous

**Sector:** fashion brick-and-mortar. **Decision:** does a new window layout
increase sales? The units are **120 stores**, and they differ a lot: a `G`
(large) store sells far more than a `PP` (tiny) one, regardless of layout. That
between-size variation, left uncontrolled, drowns the effect.

The answer is to **block by store size** (`PP`, `P`, `M`, `G`): randomize within
each block and combine by a weighted average. Theory:
[II. Designs](../../../docs/en/theory/02-designs.md) (topic 5) and
[III. Estimation](../../../docs/en/theory/03-estimation.md) (topic 9).
""",
    ),
    C(
        BOOT
        + '''

df = pd.read_csv(DATA / "fashion_stores.csv")
# Baseline sales differ strongly by size: this is the between-block variation.
print(df.groupby("store_size")["sales_y0"].mean().round(2))
df.head()'''
    ),
    B(
        pt="""
## 1. Desenho blocado e equilíbrio

Randomizamos dentro de cada tamanho com `BlockedCRD(block_col="store_size")`.
Logo após o sorteio (antes de anexar o resultado), checamos o **equilíbrio** das
covariáveis com `check_balance`: a randomização deve deixar tratamento e
controle parecidos em `foot_traffic_pre`. Avaliamos pela **magnitude da SMD**
(`|SMD| < 0,1` é bom), não por p-valor.
""",
        en="""
## 1. Blocked design and balance

We randomize within each size with `BlockedCRD(block_col="store_size")`. Right
after the draw (before attaching the outcome), we check covariate **balance**
with `check_balance`: randomization should leave treatment and control similar on
`foot_traffic_pre`. We assess it by the **magnitude of the SMD** (`|SMD| < 0.1`
is good), not by a p-value.
""",
    ),
    C(
        '''from skxperiments.core.assignment import BlockedAssignment
from skxperiments.design.blocked_crd import BlockedCRD
from skxperiments.design.balance import check_balance

design = BlockedCRD(block_col="store_size", p=0.5, seed=202)
assignment = design.randomize(df[["region", "store_size", "foot_traffic_pre"]].copy())

# Balance is checked on the pre-treatment covariates only.
check_balance(assignment)[["covariate", "smd"]].round(3)'''
    ),
    C(
        '''from skxperiments.diagnostics import BalanceReport
from skxperiments.reporting import plot_balance

report = BalanceReport().run(assignment)
ax = plot_balance(report)
ax.set_title("Covariate balance (Love plot)")
ax.figure'''
    ),
    B(
        pt="""
## 2. Estimativa blocada vs. ignorar os blocos

Anexamos o resultado observado e estimamos com `BlockedDifferenceInMeans` (média
dos efeitos por bloco, ponderada por tamanho) e `NeymanCI` (variância
estratificada). Para deixar o ganho explícito, comparamos com a análise
**ingênua** que trata tudo como um `CRD` único, ignorando o tamanho da loja: o
erro padrão fica bem maior, porque a variação entre tamanhos entra no ruído.
""",
        en="""
## 2. Blocked estimate vs. ignoring the blocks

We attach the observed outcome and estimate with `BlockedDifferenceInMeans` (the
size-weighted average of the per-block effects) and `NeymanCI` (stratified
variance). To make the gain explicit, we compare with the **naive** analysis that
treats everything as a single `CRD`, ignoring store size: the standard error is
much larger, because the between-size variation enters the noise.
""",
    ),
    C(
        '''from skxperiments.core.assignment import CRDAssignment
from skxperiments.design.crd import CRD
from skxperiments.estimators.blocked_difference_in_means import BlockedDifferenceInMeans
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.inference import NeymanCI

t = assignment.data_[assignment.treatment_col_].to_numpy()
data = assignment.data_.copy()
data["sales"] = np.where(t == 1, df["sales_y1"].to_numpy(), df["sales_y0"].to_numpy())

blocked_a = BlockedAssignment(
    data=data, treatment_col=assignment.treatment_col_, design=design,
    block_col=assignment.block_col_, block_sizes=assignment.block_sizes_, seed=202,
)
blocked_est = BlockedDifferenceInMeans(outcome_col="sales").fit(blocked_a)
blocked = NeymanCI(estimator=BlockedDifferenceInMeans(outcome_col="sales")).fit(blocked_a).estimate()

# Naive: same data, but analyzed as a single CRD (blocks ignored).
naive_a = CRDAssignment(data=data, treatment_col="treatment", design=CRD(p=0.5, seed=202), seed=202)
naive = NeymanCI(estimator=DifferenceInMeans(outcome_col="sales")).fit(naive_a).estimate()

print("per-block ATE:", {k: round(v, 3) for k, v in blocked_est.block_ates_.items()})
pd.DataFrame([
    {"analysis": "Blocked (correct)", "ATE": blocked.ate, "SE": blocked.se,
     "CI_low": blocked.ci[0], "CI_high": blocked.ci[1], "p": blocked.p_value},
    {"analysis": "Naive CRD (blocks ignored)", "ATE": naive.ate, "SE": naive.se,
     "CI_low": naive.ci[0], "CI_high": naive.ci[1], "p": naive.p_value},
]).round(3)'''
    ),
    C(
        '''from skxperiments.reporting import plot_effect

ax = plot_effect(blocked)
ax.set_title("Sales lift, blocked estimate (95% CI)")
ax.figure'''
    ),
    B(
        pt="""
## Decisão

O efeito verdadeiro é `+0,40` (mil em vendas diárias), constante entre tamanhos.
A estimativa blocada fica perto disso e o intervalo exclui zero. Compare os erros
padrão: a análise blocada é **bem mais precisa** que a ingênua, porque removeu do
erro a variação entre tamanhos de loja. Mesma coleta de dados, decisão mais
firme, só por respeitar a estrutura do desenho.

Próximo passo: [03. Campanha fatorial](03_fintech_factorial.ipynb).
""",
        en="""
## Decision

The true effect is `+0.40` (thousands in daily sales), constant across sizes.
The blocked estimate lands close to it and the interval excludes zero. Compare
the standard errors: the blocked analysis is **much more precise** than the naive
one, because it removed the between-size variation from the error. Same data
collection, firmer decision, just by respecting the structure of the design.

Next step: [03. Factorial campaign](03_fintech_factorial.ipynb).
""",
    ),
]

# =====================================================================
# 03 - fintech factorial (FactorialDesign 2^2 + FactorialEstimator)
# =====================================================================
nb03 = [
    B(
        pt="""
# 03. Campanha de fintech: dois fatores de uma vez (fatorial 2x2)

**Setor:** fintech (CRM). **Decisão:** como desenhar a campanha de ativação?
Dois fatores binários entram em jogo ao mesmo tempo:

- **cashback** (oferecer vs. não), fator `A`
- **horário de envio** (manhã vs. noite), fator `B`

Em vez de testar um por vez, um desenho **fatorial 2x2** mede, num único
experimento, os dois **efeitos principais** e a **interação** (será que cashback
rende mais quando enviado à noite?). Base teórica:
[II. Desenhos](../../../docs/pt-br/teoria/02-desenhos.md) (tópico 7) e
[III. Estimação](../../../docs/pt-br/teoria/03-estimacao.md) (tópico 12).
""",
        en="""
# 03. Fintech campaign: two factors at once (2x2 factorial)

**Sector:** fintech (CRM). **Decision:** how to design the activation campaign?
Two binary factors are in play at the same time:

- **cashback** (offer vs. not), factor `A`
- **send time** (morning vs. evening), factor `B`

Instead of testing one at a time, a **2x2 factorial** design measures, in a
single experiment, both **main effects** and the **interaction** (does cashback
pay off more when sent in the evening?). Theory:
[II. Designs](../../../docs/en/theory/02-designs.md) (topic 7) and
[III. Estimation](../../../docs/en/theory/03-estimation.md) (topic 12).
""",
    ),
    C(
        BOOT
        + '''

df = pd.read_csv(DATA / "fintech_crm.csv")
print(df.shape)
df.head()'''
    ),
    B(
        pt="""
## 1. Desenho fatorial e o resultado

`FactorialDesign(factors=["cashback", "send_time"], n_per_cell=1000)` sorteia as
`2^2 = 4` células com o mesmo tamanho. O resultado (ativações) é gerado a partir
dos fatores atribuídos, com coeficientes conhecidos numa codificação `{0,1}`:

```
ativacoes = 30 + 4*cashback + 2*send_time + 3*(cashback*send_time) + ruido
```

Repare que a codificação `{0,1}` da simulação **não** é a mesma escala dos
contrastes `±1` que o estimador usa. A teoria (tópico 12) mostra que o efeito
principal estimado é `b + metade da interação`. Então esperamos recuperar:

- cashback: `4 + 3/2 = 5,5`
- send_time: `2 + 3/2 = 3,5`
- interação: `3/2 = 1,5`
""",
        en="""
## 1. Factorial design and the outcome

`FactorialDesign(factors=["cashback", "send_time"], n_per_cell=1000)` draws the
`2^2 = 4` cells with equal size. The outcome (activations) is generated from the
assigned factors, with known coefficients in a `{0,1}` coding:

```
activations = 30 + 4*cashback + 2*send_time + 3*(cashback*send_time) + noise
```

Note that the simulation's `{0,1}` coding is **not** the same scale as the `±1`
contrasts the estimator uses. The theory (topic 12) shows the estimated main
effect is `b + half the interaction`. So we expect to recover:

- cashback: `4 + 3/2 = 5.5`
- send_time: `2 + 3/2 = 3.5`
- interaction: `3/2 = 1.5`
""",
    ),
    C(
        '''from skxperiments.core.assignment import FactorialAssignment
from skxperiments.design.factorial import FactorialDesign

design = FactorialDesign(factors=["cashback", "send_time"], n_per_cell=1000, seed=303)
assignment = design.randomize(df[["customer_id", "tenure_months"]].copy())

A = assignment.data_["cashback"].to_numpy()
B = assignment.data_["send_time"].to_numpy()
data = assignment.data_.copy()
data["activations"] = 30 + 4 * A + 2 * B + 3 * (A * B) + df["noise"].to_numpy()

assignment = FactorialAssignment(
    data=data, design=design, factor_cols=assignment.factor_cols,
    cell_sizes=assignment.cell_sizes_, seed=303,
)
assignment.data_.groupby(["cashback", "send_time"])["activations"].mean().round(2)'''
    ),
    C(
        '''from skxperiments.estimators.factorial_estimator import FactorialEstimator

result = FactorialEstimator(outcome_col="activations").fit(assignment).estimate()
expected = {"cashback": 5.5, "send_time": 3.5, "cashback:send_time": 1.5}
rows = []
for key, value in result.effects.items():
    name = ":".join(key)
    rows.append({"effect": name, "estimated": round(value, 3), "expected": expected[name]})
pd.DataFrame(rows)'''
    ),
    C(
        '''from skxperiments.reporting import plot_interaction

ax = plot_interaction(result)
ax.set_title("Main effects and interaction")
ax.figure'''
    ),
    B(
        pt="""
## Decisão

Os três efeitos são recuperados perto do esperado (`5,5`, `3,5`, `1,5`). A
**interação positiva** confirma que cashback e envio à noite se reforçam: juntos
rendem mais do que a soma dos efeitos isolados sugeriria. A recomendação de
campanha é combinar os dois fatores no nível alto. Um desenho "um fator por vez"
jamais revelaria essa sinergia.

Próximo passo: [04. Rerandomização na logística](04_logistics_rerandomization.ipynb).
""",
        en="""
## Decision

All three effects are recovered close to expectation (`5.5`, `3.5`, `1.5`). The
**positive interaction** confirms that cashback and evening send reinforce each
other: together they yield more than the sum of the isolated effects would
suggest. The campaign recommendation is to combine both factors at the high
level. A "one factor at a time" design would never reveal this synergy.

Next step: [04. Re-randomization in logistics](04_logistics_rerandomization.ipynb).
""",
    ),
]

# =====================================================================
# 04 - logistics DC (ReRandomizedCRD + RandomizationTest + Neyman)
# =====================================================================
nb04 = [
    B(
        pt="""
# 04. Centros de distribuição: rerandomização com poucas unidades

**Setor:** logística. **Decisão:** um novo roteirizador melhora a taxa de
entregas no prazo? As unidades são **apenas 24 centros de distribuição (CDs)**, e
eles diferem em volume (`throughput_pre`) e número de docas (`dock_count`). Com
tão poucas unidades, um único sorteio pode sair **desbalanceado por azar**,
concentrando os CDs grandes num braço.

A solução é **rerandomizar**: sortear repetidamente até que as covariáveis
fiquem equilibradas (critério de Mahalanobis), e depois usar uma **inferência de
randomização que respeita esse mesmo critério**. Base teórica:
[II. Desenhos](../../../docs/pt-br/teoria/02-desenhos.md) (tópico 6) e
[IV. Inferência](../../../docs/pt-br/teoria/04-inferencia.md) (tópico 13).
""",
        en="""
# 04. Distribution centers: re-randomization with few units

**Sector:** logistics. **Decision:** does a new routing engine improve the
on-time delivery rate? The units are **only 24 distribution centers (DCs)**, and
they differ in volume (`throughput_pre`) and number of docks (`dock_count`). With
so few units, a single draw can come out **imbalanced by bad luck**,
concentrating the large DCs in one arm.

The fix is to **re-randomize**: draw repeatedly until the covariates are balanced
(Mahalanobis criterion), then use a **randomization inference that respects that
same criterion**. Theory:
[II. Designs](../../../docs/en/theory/02-designs.md) (topic 6) and
[IV. Inference](../../../docs/en/theory/04-inference.md) (topic 13).
""",
    ),
    C(
        BOOT
        + '''

df = pd.read_csv(DATA / "logistics_dc.csv")
print(df.shape, "distribution centers")
df.head()'''
    ),
    B(
        pt="""
## 1. Um CRD simples pode desbalancear

Primeiro, um sorteio de `CRD` comum e o equilíbrio das covariáveis. Com 24
unidades, não é raro a SMD de `throughput_pre` ou `dock_count` passar de `0,1`.
""",
        en="""
## 1. A plain CRD can be imbalanced

First, an ordinary `CRD` draw and the covariate balance. With 24 units, it is not
unusual for the SMD of `throughput_pre` or `dock_count` to exceed `0.1`.
""",
    ),
    C(
        '''from skxperiments.design.crd import CRD
from skxperiments.design.balance import check_balance

crd = CRD(p=0.5, seed=404).randomize(df[["region", "throughput_pre", "dock_count"]].copy())
check_balance(crd)[["covariate", "smd"]].round(3)'''
    ),
    B(
        pt="""
## 2. Rerandomizar até equilibrar

`ReRandomizedCRD` aceita só os sorteios cuja distância de Mahalanobis fica abaixo
de um limiar. Usamos `chi2.ppf(0,10, df=2)` (aceita os ~10% sorteios mais
equilibrados; com `n` pequeno, um limiar não muito rígido evita rejeição
excessiva). O balanço depois deve estar visivelmente melhor.
""",
        en="""
## 2. Re-randomize until balanced

`ReRandomizedCRD` accepts only draws whose Mahalanobis distance falls below a
threshold. We use `chi2.ppf(0.10, df=2)` (accepts the ~10% most balanced draws;
with small `n`, a not-too-strict threshold avoids excessive rejection). Balance
afterward should be visibly better.
""",
    ),
    C(
        '''from scipy.stats import chi2

from skxperiments.design.rerandomized_crd import ReRandomizedCRD
from skxperiments.diagnostics import BalanceReport

threshold = float(chi2.ppf(0.10, df=2))
design = ReRandomizedCRD(
    covariates=["throughput_pre", "dock_count"], threshold=threshold,
    p=0.5, seed=404, max_attempts=20000,
)
assignment = design.randomize(df[["region", "throughput_pre", "dock_count"]].copy())
report = BalanceReport().run(assignment)
print("imbalanced covariates:", report.imbalanced)'''
    ),
    C(
        '''from skxperiments.reporting import plot_balance

ax = plot_balance(report)
ax.set_title("Balance after re-randomization")
ax.figure'''
    ),
    B(
        pt="""
## 3. Inferência que respeita o critério

Anexamos o resultado (taxa no prazo) e analisamos. O `RandomizationTest` re-sorteia
pelo **mesmo** mecanismo do desenho (via `draw()`), então as permutações também
obedecem ao critério de rerandomização, mantendo o teste válido (não
conservador). Com 24 unidades, `C(24,12)` é enorme, então o p-valor vem por Monte
Carlo. Comparamos com o intervalo de Neyman.
""",
        en="""
## 3. Inference that respects the criterion

We attach the outcome (on-time rate) and analyze. `RandomizationTest`
re-randomizes by the **same** mechanism as the design (via `draw()`), so the
permutations also obey the re-randomization criterion, keeping the test valid
(not conservative). With 24 units, `C(24,12)` is huge, so the p-value comes by
Monte Carlo. We compare with the Neyman interval.
""",
    ),
    C(
        '''from skxperiments.core.assignment import CRDAssignment
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.inference import RandomizationTest, NeymanCI

t = assignment.data_[assignment.treatment_col_].to_numpy()
data = assignment.data_.copy()
data["on_time"] = np.where(t == 1, df["on_time_y1"].to_numpy(), df["on_time_y0"].to_numpy())
assignment = CRDAssignment(
    data=data, treatment_col=assignment.treatment_col_, design=design, seed=404
)

rand = RandomizationTest(
    estimator=DifferenceInMeans(outcome_col="on_time"), n_permutations=2000, seed=0
).fit(assignment).estimate()
neyman = NeymanCI(estimator=DifferenceInMeans(outcome_col="on_time")).fit(assignment).estimate()

print(f"Randomization test: ATE={rand.ate:.3f} pp, p={rand.p_value:.4f}")
print(f"Neyman CI:          ATE={neyman.ate:.3f} pp, "
      f"CI=({neyman.ci[0]:.3f}, {neyman.ci[1]:.3f})")'''
    ),
    C(
        '''from skxperiments.reporting import plot_null_distribution

ax = plot_null_distribution(rand)
ax.set_title("Randomization null distribution")
ax.figure'''
    ),
    B(
        pt="""
## Decisão

O efeito verdadeiro é `+1,5` ponto percentual na taxa no prazo. A estimativa fica
perto disso e o teste de randomização rejeita a nula. A lição de desenho: com
poucas unidades grandes, a rerandomização protege contra um sorteio azarado, e a
inferência **precisa** honrar o critério para não ficar conservadora. Um teste t
comum aqui seria válido mas menos poderoso.

Próximo passo: [05. Muitas métricas no streaming](05_streaming_many_metrics.ipynb).
""",
        en="""
## Decision

The true effect is `+1.5` percentage points in the on-time rate. The estimate
lands close to it and the randomization test rejects the null. The design lesson:
with few large units, re-randomization protects against an unlucky draw, and the
inference **must** honor the criterion so it does not become conservative. A plain
t test here would be valid but less powerful.

Next step: [05. Many metrics in streaming](05_streaming_many_metrics.ipynb).
""",
    ),
]

# =====================================================================
# 05 - streaming many metrics (ExperimentComparison + pipeline + report)
# =====================================================================
nb05 = [
    B(
        pt="""
# 05. Streaming: muitas métricas, guardrails e correção múltipla

**Setor:** streaming de vídeo. **Decisão:** lançar o novo algoritmo de
recomendação? O problema não é uma métrica, são **várias**: tempo assistido
(primária), número de sessões, taxa de conclusão, e um **guardrail** de buffering
(não pode piorar). Testar muitas métricas infla a chance de um falso-positivo, então
precisamos **corrigir para múltiplos testes**.

Um único experimento, uma única alocação, várias métricas. Fechamos com um
`ExperimentPipeline` e um `ExperimentReport`. Base teórica:
[IV. Inferência](../../../docs/pt-br/teoria/04-inferencia.md) (tópico 16) e
[V. Diagnósticos](../../../docs/pt-br/teoria/05-diagnosticos.md) (tópico 21).
""",
        en="""
# 05. Streaming: many metrics, guardrails, and multiple-testing correction

**Sector:** video streaming. **Decision:** launch the new recommendation
algorithm? The problem is not one metric, it is **several**: watch time
(primary), number of sessions, completion rate, and a buffering **guardrail**
(must not get worse). Testing many metrics inflates the chance of a false
positive, so we must **correct for multiple testing**.

One experiment, one allocation, many metrics. We close with an
`ExperimentPipeline` and an `ExperimentReport`. Theory:
[IV. Inference](../../../docs/en/theory/04-inference.md) (topic 16) and
[V. Diagnostics](../../../docs/en/theory/05-diagnostics.md) (topic 21).
""",
    ),
    C(
        BOOT
        + '''

df = pd.read_csv(DATA / "streaming_metrics.csv")
print(df.shape)
df.head()'''
    ),
    B(
        pt="""
## 1. Uma alocação, várias métricas

Randomizamos os usuários **uma vez** e reusamos a **mesma** alocação para todas
as métricas (é o mesmo experimento). Para cada métrica, construímos o resultado
observado e estimamos o efeito com `NeymanCI`.
""",
        en="""
## 1. One allocation, several metrics

We randomize users **once** and reuse the **same** allocation for every metric
(it is the same experiment). For each metric, we build the observed outcome and
estimate the effect with `NeymanCI`.
""",
    ),
    C(
        '''from skxperiments.core.assignment import CRDAssignment
from skxperiments.design.crd import CRD
from skxperiments.estimators.difference_in_means import DifferenceInMeans
from skxperiments.inference import NeymanCI

design = CRD(p=0.5, seed=505)
base = design.randomize(df[["plan"]].copy())
t = base.data_[base.treatment_col_].to_numpy()

metrics = ["watch_time", "sessions", "completion", "buffering"]
results = {}
for m in metrics:
    data = base.data_.copy()
    data[m] = np.where(t == 1, df[f"{m}_y1"].to_numpy(), df[f"{m}_y0"].to_numpy())
    a = CRDAssignment(data=data, treatment_col=base.treatment_col_, design=design, seed=505)
    results[m] = NeymanCI(estimator=DifferenceInMeans(outcome_col=m)).fit(a).estimate()

{m: round(r.p_value, 4) for m, r in results.items()}'''
    ),
    B(
        pt="""
## 2. Correção para múltiplos testes (FWER, Holm)

Com quatro métricas, controlamos a taxa de erro por família com `Holm`. O
`ExperimentComparison` aplica a correção sobre a família e devolve uma tabela
pronta, além do forest plot.
""",
        en="""
## 2. Multiple-testing correction (FWER, Holm)

With four metrics, we control the family-wise error rate with `Holm`.
`ExperimentComparison` applies the correction over the family and returns a
ready-made table, plus the forest plot.
""",
    ),
    C(
        '''from skxperiments.pipeline import ExperimentComparison
from skxperiments.reporting import plot_forest

comparison = ExperimentComparison(correction="holm", alpha=0.05).run(results)
display_cols = ["experiment", "ate", "p_value", "p_value_corrected", "significant"]
print(comparison.table[display_cols].round(4).to_string(index=False))

ax = plot_forest(comparison)
ax.set_title("Effect by metric (Holm-corrected)")
ax.figure'''
    ),
    B(
        pt="""
## 3. Pipeline e relatório na métrica primária

Só o **tempo assistido** sobrevive à correção; sessões e conclusão não são
significativas, e o guardrail de **buffering não é sinalizado** (não houve
piora). Fechamos com o `ExperimentPipeline` na métrica primária, que roda o
`SRMTest` automaticamente, e geramos um `ExperimentReport` em HTML.
""",
        en="""
## 3. Pipeline and report on the primary metric

Only **watch time** survives the correction; sessions and completion are not
significant, and the buffering guardrail is **not flagged** (no harm). We close
with the `ExperimentPipeline` on the primary metric, which runs the `SRMTest`
automatically, and generate an HTML `ExperimentReport`.
""",
    ),
    C(
        '''from skxperiments.diagnostics import SRMTest
from skxperiments.pipeline import ExperimentPipeline

data = base.data_.copy()
data["watch_time"] = np.where(t == 1, df["watch_time_y1"].to_numpy(), df["watch_time_y0"].to_numpy())
primary = CRDAssignment(data=data, treatment_col=base.treatment_col_, design=design, seed=505)

pipeline_result = ExperimentPipeline(
    inference=NeymanCI(estimator=DifferenceInMeans(outcome_col="watch_time")),
    diagnostics=[SRMTest()],
).run(primary)
print(f"watch_time ATE={pipeline_result.results.ate:.3f}, "
      f"p={pipeline_result.results.p_value:.4f}, flagged={pipeline_result.flagged}")'''
    ),
    C(
        '''from IPython.display import HTML

from skxperiments.reporting import ExperimentReport

report = ExperimentReport(pipeline_result, title="Streaming recommender - watch time")
HTML(report.to_html())'''
    ),
    B(
        pt="""
## Decisão

O efeito verdadeiro em tempo assistido é `+3` minutos, e é o único real; as
demais métricas foram geradas sem efeito (ou efeito minúsculo). A correção de
Holm faz exatamente o trabalho certo: mantém a descoberta primária e não deixa o
ruído das outras métricas virar "vitória". Com o guardrail limpo, a recomendação
é **lançar**, monitorando buffering.

Isso fecha a trilha de use cases. Para os fundamentos por trás de cada peça, veja
a [série de teoria](../../../docs/pt-br/teoria/01-fundamentos.md).
""",
        en="""
## Decision

The true effect on watch time is `+3` minutes, and it is the only real one; the
other metrics were generated with no effect (or a tiny one). The Holm correction
does exactly the right job: it keeps the primary discovery and does not let the
noise of the other metrics turn into a "win". With the guardrail clean, the
recommendation is to **launch**, monitoring buffering.

This closes the use-cases track. For the fundamentals behind each piece, see the
[theory series](../../../docs/en/theory/01-foundations.md).
""",
    ),
]

write(nb01, "01_ecommerce_checkout.ipynb")
write(nb02, "02_fashion_stores_blocking.ipynb")
write(nb03, "03_fintech_factorial.ipynb")
write(nb04, "04_logistics_rerandomization.ipynb")
write(nb05, "05_streaming_many_metrics.ipynb")
print("done.")
