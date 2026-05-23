# rcdb_research

> Quantitative research framework for the RCDB multi-exchange automated trading platform. End-to-end pipeline from tick data to deployable strategy configs.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.18-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-1.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.23-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Numba](https://img.shields.io/badge/Numba-0.47-00A3E0)](https://numba.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Archived-lightgrey)](#lineage)

**Archived** - cloned from `hcmc-project/rcdb_research` for posterity. Part of the **RCDB** automated trading platform, later merged into [3Jane Technologies](https://github.com/3jane).

---

## Table of contents

- [What this was](#what-this-was)
- [Tech stack](#tech-stack)
- [Pipeline](#pipeline)
- [Module map](#module-map)
- [Techniques](#techniques)
- [Parameter-grid constraint DSL](#parameter-grid-constraint-dsl)
- [Installation](#installation)
- [Lineage](#lineage)
- [Sibling repos](#sibling-repos)

---

## What this was

`rcdb_research` was the **quantitative research engine** behind RCDB's multi-exchange systematic trading platform. It carried a signal from raw exchange ticks all the way to a backtested, sized, risk-managed strategy config ready for the live execution layer. Every stage - bar construction, feature generation, labeling, cross-validation, importance, backtest, sizing - lives in this repo as a composable Python module with a strong bias toward de Prado's *Advances in Financial Machine Learning* methodology.

The **bar layer** (`rcdb_research/bars`) turns tick streams into information-driven bars - time, fixed-threshold and adaptive tick/volume/quote-volume bars, percent-move bars, and hybrid sampling schemes. The **feature layer** (`rcdb_research/features`) produces hundreds of derived signals (rolling statistics, technical indicators via `tulipindicators`, microstructure features, Alpha101, entropy and fractal-dimension measures, stationarity tests) and is driven by a **combinatorial parameter grid with a sandboxed predicate DSL** (`p.a < p.b and p.c != -1`) that prunes the cross-product before computation.

The **modeling stack** ties it together: **triple-barrier labels** with dynamic widths (`rcdb_research/labeling`, with a C++ inner loop), **walk-forward and combinatorially-purged-with-embargo CV** (`rcdb_research/sampling/cv`), **ensemble feature importance** combining MDI, MDA (permutation), and mutual information (`rcdb_research/feature_importance`), sequential-bootstrap resampling (`rcdb_research/sampling/sequential_bootstrap`, also C++), a `backtrader`-backed simulator with commissions, slippage, leverage and pluggable risk management (`rcdb_research/simulation_bt`), and Kelly / fractional-Kelly / drawdown-bounded Monte-Carlo sizing (`rcdb_research/sizing`).

## Tech stack

| Layer | Tools |
|---|---|
| Core compute | Python 3, NumPy 1.18, pandas 1.0, SciPy 1.4, Numba 0.47, `numpy_ext` |
| ML | scikit-learn 0.23, XGBoost 0.90, LightGBM 2.2, statsmodels 0.11, hyperopt |
| Bootstrap & resampling | `recombinator`, custom C++ sequential bootstrap (`sampling/sequential_bootstrap/sb.cc`) |
| Backtesting | `backtrader` 1.9 |
| Indicators | `tulipindicators` |
| Storage / IO | `tables` (HDF5), `json-tricks`, `flock` |
| Parallelism | `joblib`, custom job manager (`rcdb_research/job_manager`) |
| Native extensions | Triple barrier (`labeling/triple_barrier/tb.cc`), sequential bootstrap, MC sizing - built via per-module `Makefile` to `.so` |
| Tooling | Jupyter, pytest, flake8, Docker (`.packaging/Dockerfile`, `run_tests_in_docker.sh`) |

## Pipeline

```mermaid
flowchart LR
    A[Tick data<br/>rcdb_data] --> B[Information-driven bars<br/>bars/]
    B --> C[Feature grid sweep<br/>features/ + job_manager/]
    C --> D[Triple-barrier labels<br/>labeling/]
    D --> E[Purged + embargoed CV<br/>sampling/cv/]
    E --> F[Model fit<br/>models/]
    F --> G[Ensemble feature importance<br/>feature_importance/]
    G --> H[Backtest<br/>simulation_bt/]
    H --> I[Sizing + risk<br/>sizing/]
    I --> J[Monte Carlo + scoring<br/>monte_carlo/ + scoring/]
    J --> K[Strategy config]
```

## Module map

- `rcdb_research/bars/` - time, fixed and adaptive tick / volume / quote-volume / percent-move bars, plus hybrid samplers (`facade.py`, `functions.py`); Numba-accelerated thresholding
- `rcdb_research/features/` - feature library (`alphas101`, `tulip`, `entropy`, `fracdim`, `highlow`, `stats`, `stattests`, `momentum`, `datetime`) with declarative configs under `features/configs/`
- `rcdb_research/job_manager/` - parallel feature-computation runner (`parallel_calc_all.py`), parameter-grid expansion, predicate-based constraint pruning (`utils.generate_constraints_function`)
- `rcdb_research/labeling/triple_barrier/` - triple-barrier method with C++ inner loop (`tb.cc`) and Python wrapper (`triple_barrier.py`)
- `rcdb_research/sampling/cv/` - `WalkForwardCV` (expanding or fixed, with gap), `CombinatorialCV` with **purging + embargo**
- `rcdb_research/sampling/bootstrap/` and `sampling/sequential_bootstrap/` - block bootstrap (via `recombinator`) and de Prado's sequential bootstrap (C++)
- `rcdb_research/feature_importance/` - `MDI`, `MDA` (permutation), `MutualInformation`, and `EFI` (ensemble feature importance with agglomerative clustering)
- `rcdb_research/feature_selection/` - `SelectKBest`-style and ensemble-driven selectors
- `rcdb_research/cross_validation/aggregated_learning.py` - aggregated learning across folds
- `rcdb_research/models/` - classifier wrappers (`classifiers.py`), no-skill baseline (`noskill.py`), sequentially bootstrapped bagging (`sbb.py`)
- `rcdb_research/simulation_bt/` - `backtrader` integration (`wrappers.py`, `glue.py`) with fractional commissions, configurable leverage, intrabar worst-case PnL, pre-trade risk hooks, and limit-order entry
- `rcdb_research/simulation/` - prediction / probability / trade simulators and voting ensembles
- `rcdb_research/sizing/` - Kelly, fractional Kelly, drawdown-bounded `RiskAdjustedKelly` backed by a Monte-Carlo sizing extension (`sizing/mc_sizing`)
- `rcdb_research/monte_carlo/coin_toss.py` - coin-toss Monte-Carlo equity-curve generator for sizing and robustness experiments
- `rcdb_research/metrics/`, `scoring/` - trading metrics (`trading.py`), prediction metrics, proximity, scoring helpers
- `rcdb_research/compute/`, `rcdb_research/cacher/`, `rcdb_research/plotter/` - compute orchestration, on-disk caching, plotting utilities
- `rcdb_research/datasets/`, `rcdb_research/rcdb_data.py` - dataset loaders for the RCDB tick warehouse
- `notebooks/`, `tests/` - research notebooks and a pytest suite (`tests/test_bars.py`, `test_cross_validators.py`, `test_triple_barrier.py`, `test_aggregated_learning.py`, etc.)

## Techniques

- **Information-driven bars** (`bars/`) - sample on transacted information, not the clock; supports fixed and adaptive thresholds on ticks / volume / quote-volume, plus percent-move and time bars and hybrids
- **Triple-barrier labeling** (`labeling/triple_barrier/`) - upper / lower / vertical barriers with dynamic widths, implemented in C++ (`tb.cc`) and exposed via `ctypes`
- **Purged + embargoed CV** (`sampling/cv/combinatorial.py`) - `CombinatorialCV` purges train samples that overlap test labels and applies a configurable embargo (bars or percent); `WalkForwardCV` provides expanding or fixed-window walk-forward splits with a gap
- **Sequential bootstrap** (`sampling/sequential_bootstrap/sb.cc`) - de Prado-style overlap-aware resampling, used to fit sequentially bootstrapped bagging classifiers (`models/sbb.py`)
- **Ensemble feature importance** (`feature_importance/ensemble_feature_importance.py`) - `EFI` aggregates MDI, MDA and MI across multiple estimators, with optional agglomerative clustering of correlated features
- **Backtesting** (`simulation_bt/`) - `backtrader`-based simulator with fractional commissions, leverage, intrabar worst-case PnL, pluggable pre-trade risk management, and limit-order entry
- **Sizing and robustness** (`sizing/`, `monte_carlo/`) - Kelly / fractional Kelly, drawdown-constrained `RiskAdjustedKelly` solved over Monte-Carlo equity paths

## Parameter-grid constraint DSL

`rcdb_research/job_manager` expands feature configs over a combinatorial parameter grid (`pg`) and prunes the cross-product with a string predicate (`cn`). The predicate is compiled with `generate_constraints_function` (`job_manager/utils.py`) into a sandboxed lambda evaluated against an `AttrDict` of the candidate parameter set, so authors can write natural Python expressions like `p.short_period < p.long_period`.

```python
test_config = dict(
    f=[
        dict(
            fn=inc_func,
            pg=km(a=[1, 2, 3], b=[1, 2, 3], c=[1, -1]),
            dm=km(x=['input']),
            cn='p.a < p.b and p.c != -1'
        )
    ]
)
```

Result parameter sets after pruning:

```python
[
    dict(a=1, b=2, c=1),
    dict(a=1, b=3, c=1),
    dict(a=2, b=3, c=1),
]
```

Real-world usage lives in `rcdb_research/features/configs/tulip.py`, e.g. `cn='p.short_period < p.medium_period < p.long_period'` for triple-period indicators.

## Installation

### From inside Jupyter (or anywhere with `pip`)

```bash
$ pip install -U \
    --extra-index-url https://pypi-private:***TOKEN***@pkgs.dev.azure.com/rcdb/_packaging/pypi-private/pypi/simple/ \
    git+https://github.com/tartakovsky-archive/rcdb_research
```

### For development

```bash
$ pip install --extra-index-url $(cat extra-index-url) .            # install from source
$ pip install --extra-index-url $(cat extra-index-url) -e .[dev]    # editable install with dev extras
$ pip install --extra-index-url $(cat extra-index-url) -e <git url> # install from git
$ jupyter notebook                                                  # start jupyter
```

The native extensions (`labeling/triple_barrier/tb.cc`, `sampling/sequential_bootstrap/sb.cc`, `sizing/mc_sizing`) are built by per-module `Makefile` during `egg_info` - a C++ toolchain is required.

## Lineage

- **Origin:** `hcmc-project/rcdb_research` (private)
- **Archive:** `tartakovsky-archive/rcdb_research` (this repo)
- **Successor:** [3Jane Technologies](https://github.com/3jane)

## Sibling repos

- [rcdb_commons](https://github.com/tartakovsky-archive/rcdb_commons) - shared client SDKs and schemas
- [rcdb_datastore](https://github.com/tartakovsky-archive/rcdb_datastore) - FastAPI time-series API
- [rcdb_dashboard](https://github.com/tartakovsky-archive/rcdb_dashboard) - Django operations console
