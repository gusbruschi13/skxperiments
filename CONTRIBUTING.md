# Contributing to skxperiments

Thanks for considering a contribution. Please open an issue to discuss
substantial changes before sending a pull request.

## Development setup

```bash
git clone https://github.com/gusbruschi13/skxperiments
cd skxperiments
pip install -e ".[dev]"
```

The `dev` extra includes the test, lint, and notebook tooling
(`pytest`, `ruff`, `black`, `mypy`, `nbmake`, `matplotlib`).

## Running the checks

```bash
pytest                  # full library suite (includes slow tests)
pytest -m "not slow"    # skip the slow statistical tests
pytest --nbmake --no-cov examples/   # run the example notebooks
ruff check .            # lint
black --check .         # formatting
```

A change should keep the suite green and follow the existing test patterns
(grouping classes like `TestXxxCreation`, seeded randomness, real designs
over hand-built assignments).

## Architecture and conventions

The library has documented design decisions. Before adding code, read the
docstrings of the base classes (`BaseAssignment`, `BaseEstimator`,
`BaseInference`, `Results`) and the notes in `CHANGELOG.md` and `ROADMAP.md`.
Some load-bearing conventions:

- scikit-learn style: parameters in `__init__`, data in `fit()`, learned
  attributes end with `_`.
- `Assignment` is the contract between designs and estimators; estimators
  receive `Assignment` objects, not raw DataFrames.
- `fit()` and `randomize()` never mutate the input DataFrame.
- Deferred decisions and v2 items live in `ROADMAP.md`.

## Pull requests

- Branch off `main`, keep the change focused, and describe what and why.
- Update `CHANGELOG.md` under `[Unreleased]`.
- CI is path-filtered: changes under `skxperiments/` or `tests/` run the
  library suite; changes under `examples/` run the notebooks. A library
  change also revalidates the notebooks.

## Releasing

See [`RELEASING.md`](RELEASING.md).
