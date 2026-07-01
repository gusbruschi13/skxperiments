# Use cases / Estudos de caso

Applied, decision-oriented case studies. Unlike `for_starters` (one concept per
notebook), each use case is **end-to-end**: a business problem, the design
choice, diagnostics, inference, and a recommendation. Together the five cases
exercise the whole library.

Trilha aplicada e orientada a decisão. Diferente do `for_starters` (um conceito
por notebook), cada caso é **ponta a ponta**: problema de negócio, escolha do
desenho, diagnósticos, inferência e recomendação. Juntos, os cinco casos
exercitam a biblioteca inteira.

## The cases / Os casos

| # | Case / Caso | Sector / Setor | Library features |
|---|---|---|---|
| 01 | [E-commerce checkout](en/01_ecommerce_checkout.ipynb) | e-commerce | `power_analysis`, `CRD`, `SRMTest`, `CUPED`, `NeymanCI`, `BootstrapCI` |
| 02 | [Fashion stores](en/02_fashion_stores_blocking.ipynb) | retail | `BlockedCRD`, `BlockedDifferenceInMeans`, `check_balance` |
| 03 | [Fintech campaign](en/03_fintech_factorial.ipynb) | fintech | `FactorialDesign`, `FactorialEstimator` |
| 04 | [Distribution centers](en/04_logistics_rerandomization.ipynb) | logistics | `ReRandomizedCRD`, `RandomizationTest`, `NeymanCI` |
| 05 | [Streaming](en/05_streaming_many_metrics.ipynb) | streaming | `MultipleTestingCorrection`, `ExperimentComparison`, `ExperimentPipeline`, `ExperimentReport` |

Portuguese versions are in [`pt-br/`](pt-br/). Versões em português em
[`pt-br/`](pt-br/).

## Data / Dados

The datasets are **synthetic but realistic** and **versioned**: correlated
pre-period covariates, heterogeneous baselines, and a few outliers, but with a
**known ground truth** so each notebook can show the library recovering the true
effect. They live in [`data/`](data/) as small CSVs.

Os dados são **sintéticos, porém realistas** e **versionados**, com um efeito
verdadeiro conhecido, para que cada notebook mostre a biblioteca recuperando-o.

To regenerate them (deterministic, fixed seeds) / Para regerá-los:

```bash
python examples/use_cases/_generate_data.py
```

Following the "science table" idiom of `for_starters`, most datasets carry the
covariates plus both potential outcomes (`*_y0`, `*_y1`); the notebook randomizes
with the library and builds the observed outcome. The factorial case ships a
per-unit `noise` column instead, because the design draws the cells.

## Regenerating the notebooks / Regerando os notebooks

The two language versions are built from a single source, so they never drift.
Edit [`_build_notebooks.py`](_build_notebooks.py) and run:

```bash
python examples/use_cases/_build_notebooks.py
```

## How to run / Como rodar

```bash
pip install --pre "skxperiments[viz]"
pip install jupyter
```

The notebooks run in CI via `nbmake`. Each is seeded, so results are
reproducible. Os notebooks rodam no CI via `nbmake` e usam seed fixo.

See also the [theory series](../../docs/en/theory/01-foundations.md) /
[série de teoria](../../docs/pt-br/teoria/01-fundamentos.md) for the concepts and
math behind each case.
