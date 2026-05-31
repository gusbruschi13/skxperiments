# skxperiments

> Randomization-based experimental design and causal inference, sklearn-style.

![CI](https://github.com/username/skxperiments/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-TBD-yellow)

## Installation

```bash
pip install skxperiments
```

### Quick Start
```python
import skxperiments

# Full usage examples will be added in upcoming phases.
print(skxperiments.__version__)
```

## Contributing
See CHANGELOG.md for the project history and release notes.
Contributions are welcome! Please open an issue or pull request on GitHub.

### `CHANGELOG.md`

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-dev] - 2024-01-01

### Added

- Project scaffold: pyproject.toml, README, CI workflow, pre-commit config
- Package structure with core, design, estimators, inference, diagnostics, reporting, testing submodules
- Custom exceptions: SkxperimentsError, DesignEstimatorMismatch, NotFittedError, InsufficientDataError, InvalidDesignError
- PotentialOutcomes class for representing unit-level potential outcomes
- BaseAssignment (ABC) and CRDAssignment for representing treatment assignments
- Results class as uniform output object for estimators and inference
- BaseDesign, BaseEstimator, BaseInference abstract base classes
- DiagnosticsReport dataclass
- Full test suite for core module