# Documentação do skxperiments

Biblioteca de **desenho experimental e inferência causal** sob o framework de
*potential outcomes* (Rubin Causal Model). A ideia central: **o mecanismo de
atribuição do tratamento é o ponto de partida**, não o modelo estatístico.

Esta documentação ensina os conceitos junto com a API. É material didático
para quem está começando em experimentação, não só uma referência.

> **Status:** começamos com Markdown e notebooks. A intenção é evoluir para um
> site (Quarto ou mkdocs) depois; o conteúdo já é portável para lá.

---

## Duas tradições de "DoE"

O termo *Design of Experiments* carrega duas tradições que é fácil confundir:

1. **DoE clássico** (Fisher, Box, Montgomery): fatorial, superfície de
   resposta, otimização de processo, ANOVA. A pergunta é *qual combinação de
   fatores otimiza uma resposta*.
2. **Inferência causal e testes A/B** (Rubin, Imbens): potential outcomes,
   randomização, efeito médio de tratamento (ATE). A pergunta é *qual o efeito
   de um tratamento*.

`skxperiments` vive majoritariamente na **tradição 2**, com `FactorialDesign`
fazendo a ponte para a 1. Guarde essa fronteira: "interação fatorial" como
**efeito** (o que a lib estima) é coisa diferente do *interaction plot* de
médias da otimização de processo.

---

## Trilha de aprendizado (notebooks)

Cada notebook ensina um conceito de experimentação e a API correspondente.
Ficam em [`examples/pt-br/`](../../examples/pt-br/) e rodam no CI (nbmake).

| # | Notebook | Conceito |
|---|---|---|
| 01 | Seu primeiro experimento | potential outcomes; CRD, DifferenceInMeans, RandomizationTest |
| 02 | Inferência de três jeitos | randomização vs. Neyman (pop. finita) vs. bootstrap (superpop.) |
| 03 | Reduzir variância | covariáveis com Lin e CUPED |
| 04 | Equilíbrio e rerandomização | `check_balance`, `ReRandomizedCRD` |
| 05 | Blocagem | `BlockedCRD` quando há estratos |
| 06 | Fatorial | 2^K, efeitos principais e interações (ponte com o DoE clássico) |
| 07 | Muitos testes | correção múltipla (FWER vs. FDR), `ExperimentComparison` |
| 08 | Confie no experimento | diagnósticos: SRM, A/A, balanço |
| 09 | Juntando tudo | `power_analysis`, `ExperimentPipeline`, `ExperimentReport` |

*(Os notebooks 02 a 09 entram nos próximos passos.)*

---

## Glossário (em construção)

- **ATE** (*Average Treatment Effect*): efeito médio do tratamento na
  população de interesse.
- **Potential outcomes** (`Y(0)` e `Y(1)`): os dois resultados que uma unidade
  *teria* sob controle e sob tratamento; só um é observado.
- **População finita vs. superpopulação**: inferir sobre *estas* unidades
  (finita; Neyman) vs. sobre uma população maior da qual elas são amostra
  (superpopulação; bootstrap).
- **Sharp null**: hipótese nula forte de Fisher, efeito **zero em todas** as
  unidades (testada pelo `RandomizationTest`).
- **SMD** (*Standardized Mean Difference*): diferença de médias padronizada,
  usada para checar equilíbrio de covariáveis.
- **SRM** (*Sample Ratio Mismatch*): a alocação observada destoa da pretendida,
  um alarme de bug de implementação.
- **FWER vs. FDR**: controlar a chance de **qualquer** falso-positivo
  (family-wise) vs. a **proporção** de falsos-positivos entre as descobertas.

*(Será expandido no Passo 6, com guias de decisão: qual design, estimador e
inferência escolher.)*
