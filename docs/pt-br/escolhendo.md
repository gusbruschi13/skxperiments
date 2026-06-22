# Como escolher

Guia rápido para montar um experimento: design, estimador, inferência e
diagnósticos. Na dúvida, comece pelo mais simples (`CRD` +
`DifferenceInMeans` + `NeymanCI`) e suba conforme a necessidade.

## 1. Qual design?

| Situação | Use |
|---|---|
| A/B simples, sem estrutura | `CRD` |
| Variável categórica conhecida que afeta o resultado (região, device) | `BlockedCRD` |
| Quer balanço garantido em covariáveis contínuas | `ReRandomizedCRD` |
| Testar vários fatores ao mesmo tempo | `FactorialDesign` |

## 2. Qual estimador?

| Situação | Use |
|---|---|
| CRD, sem covariáveis | `DifferenceInMeans` |
| CRD, com covariáveis medidas | `LinEstimator` |
| CRD, com métrica pré-experimento | `CUPED` |
| Blocado | `BlockedDifferenceInMeans` |
| Fatorial | `FactorialEstimator` |

## 3. Qual inferência?

| Quero... | Use |
|---|---|
| p-valor sem pressupostos, baseado no desenho | `RandomizationTest` |
| IC de população finita (sobre **estas** unidades) | `NeymanCI` |
| IC de superpopulação (população maior) | `BootstrapCI` |
| Comparar vários efeitos ou experimentos | `MultipleTestingCorrection`, `ExperimentComparison` |

Nota: na v1, o `NeymanCI` cobre `DifferenceInMeans` e
`BlockedDifferenceInMeans`. Para obter um intervalo com `LinEstimator` ou
`CUPED`, use o `BootstrapCI` (que aceita qualquer estimador escalar) ou o
`RandomizationTest`.

## 4. Sempre: diagnósticos

- **SRM** (`SRMTest`): rode **antes** de analisar; pega bug de coleta.
- **Balanço** (`check_balance`, `BalanceReport`): cheque as covariáveis.
- **A/A** (`AATest`): valide a calibração ao montar um pipeline novo.

## Fluxo recomendado

`power_analysis` (planejar o n) → `design.randomize` → coletar os outcomes →
`ExperimentPipeline` (diagnóstico automático + inferência) →
`ExperimentReport`.

Veja a trilha de notebooks em [`index.md`](index.md) para cada passo em
detalhe, e o [`glossario.md`](glossario.md) para os termos.
