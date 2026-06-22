# Documentação do skxperiments

Biblioteca de **desenho experimental e inferência causal** sob o framework de
*potential outcomes* (Rubin Causal Model). A ideia central: **o mecanismo de
atribuição do tratamento é o ponto de partida**, não o modelo estatístico.

Esta documentação ensina os conceitos junto com a API. É material didático
para quem está começando em experimentação, não só uma referência.

> **Status:** começamos com Markdown e notebooks. A intenção é evoluir para um
> site (Quarto ou mkdocs) depois; o conteúdo já é portável para lá.

**Guias:** [como escolher](escolhendo.md) (design, estimador, inferência) e
[glossário](glossario.md).

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
Ficam na trilha **for_starters** (dados simulados, didática) em
[`examples/for_starters/pt-br/`](../../examples/for_starters/pt-br/) e rodam no
CI (nbmake). Uma trilha com **dados reais** virá depois.

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

## Glossário

Os termos centrais (potential outcomes, ATE, população finita vs.
superpopulação, sharp null, SMD, SRM, FWER/FDR) estão em
[`glossario.md`](glossario.md).
