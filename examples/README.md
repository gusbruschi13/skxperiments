# Examples / Exemplos

Notebooks that teach `skxperiments` by example. They are organized into
**tracks**, and each track is bilingual (`pt-br` and `en`).

Notebooks que ensinam o `skxperiments` na prática. Estão organizados em
**trilhas**, e cada trilha é bilíngue (`pt-br` e `en`).

## Tracks / Trilhas

- **`for_starters/`** — beginner track with simulated data, from "why
  randomize" to a full pipeline and report (notebooks 00 to 09). Didactic
  and self-contained. Trilha inicial com dados simulados.
- **`use_cases/`** — applied, decision-oriented case studies across sectors
  (e-commerce, fashion retail, fintech, logistics, streaming). Each notebook is
  end-to-end: a business problem, the design choice, diagnostics, inference, and
  a decision. Uses versioned, synthetic-but-realistic datasets with a known
  ground truth (see [`use_cases/README.md`](use_cases/README.md)). Trilha
  aplicada, orientada a decisão, com casos por setor.

## How to run / Como rodar

Install the library with the plotting extra and a notebook runner:

```bash
pip install --pre "skxperiments[viz]"
pip install jupyter
```

Then open any notebook in Jupyter or VSCode and run the cells top to bottom.
Each notebook is seeded, so results are reproducible.

Depois, abra qualquer notebook no Jupyter ou no VSCode e rode as células de
cima para baixo. Cada notebook usa seed fixo, então os resultados são
reproduzíveis.

> The notebooks run in CI via `nbmake`, so they stay in sync with the
> library. Os notebooks rodam no CI via `nbmake`, então acompanham a
> evolução da biblioteca.

## Suggested order / Ordem sugerida (for_starters)

`00` why randomize, `01` first experiment, `02` inference three ways,
`03` reducing variance, `04` balance and rerandomization, `05` blocking,
`06` factorial, `07` many tests, `08` diagnostics, `09` putting it together.

## Case studies / Estudos de caso (use_cases)

`01` e-commerce checkout (power, CRD, SRM, CUPED, Neyman vs. bootstrap),
`02` fashion stores (blocking), `03` fintech campaign (2x2 factorial),
`04` distribution centers (re-randomization), `05` streaming (many metrics and
multiple-testing correction).

See the conceptual docs in [`../docs/`](../docs/README.md) for the glossary,
the "how to choose" guide, and the theory series.
