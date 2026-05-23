# rcdb_research

> Quant research stack for the RCDB trading platform. From raw ticks to a ready-to-ship strategy config.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.18-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-1.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.23-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Numba](https://img.shields.io/badge/Numba-0.47-00A3E0)](https://numba.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Archived-lightgrey)](#lineage)

**Archived.** Cloned from `hcmc-project/rcdb_research`. Part of the **RCDB** trading stack. Later folded into [3Jane](https://github.com/3jane).

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

This was RCDB's quant research stack. It took raw ticks. It shipped a sized, risk-aware config to live trading. Each stage is a Python module you can swap.

Stages: bars, features, labels, CV, importance, backtest, sizing. The design tracks de Prado.

See the [module map](#module-map) and [techniques](#techniques) tables for the full set.

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

| Path | Role |
|---|---|
| `bars/` | Time, tick, volume, quote-volume, percent-move bars. Fixed and live. Numba-tuned |
| `features/` | `alphas101`, `tulip`, `entropy`, `fracdim`, `highlow`, `stats`, `stattests`, `momentum`, `datetime` |
| `job_manager/` | Grid-sweep runner, grid expansion, predicate-based pruning |
| `labeling/triple_barrier/` | Triple-barrier method. C++ inner loop (`tb.cc`), Python wrapper |
| `sampling/cv/` | `WalkForwardCV` (expanding or fixed, with gap), `CombinatorialCV` (purge + embargo) |
| `sampling/bootstrap/` | Block bootstrap via `recombinator` |
| `sampling/sequential_bootstrap/` | de Prado sequential bootstrap (C++) |
| `feature_importance/` | `MDI`, `MDA`, `MutualInformation`, `EFI` (ensemble + clustering) |
| `feature_selection/` | `SelectKBest`-style and ensemble-driven picks |
| `cross_validation/aggregated_learning.py` | Folded fits with aggregated outputs |
| `models/` | Wrapped classifiers, no-skill base, seq-bootstrap bagging (`sbb.py`) |
| `simulation_bt/` | `backtrader` engine: fees, slippage, margin, risk hooks, limit entries |
| `simulation/` | Pred, proba, trade sims and voting ensembles |
| `sizing/` | Kelly, half-Kelly, drawdown-capped `RiskAdjustedKelly` over `mc_sizing` |
| `monte_carlo/coin_toss.py` | Coin-toss equity curves for sizing tests |
| `metrics/`, `scoring/` | Trading metrics, pred metrics, scoring helpers |
| `compute/`, `cacher/`, `plotter/` | Compute glue, on-disk cache, plots |
| `datasets/`, `rcdb_data.py` | Loaders for the RCDB tick store |
| `notebooks/`, `tests/` | Research notebooks plus a pytest suite |

## Techniques

| Block | Notes |
|---|---|
| Info-driven bars (`bars/`) | Sample on flow, not the clock. Fixed and live modes |
| Triple-barrier labels (`labeling/triple_barrier/`) | Up, down, time barriers. C++ (`tb.cc`), via `ctypes` |
| Purged + embargo CV (`sampling/cv/combinatorial.py`) | `CombinatorialCV` purges train rows that touch test labels |
| Walk-forward CV (`sampling/cv/`) | Expanding or fixed window, with a gap |
| Seq bootstrap (`sampling/sequential_bootstrap/sb.cc`) | de Prado overlap-aware resample. Feeds `models/sbb.py` |
| Ensemble importance (`feature_importance/`) | `EFI` blends MDI, MDA, MI. Optional clustering of correlated cols |
| Backtest (`simulation_bt/`) | `backtrader` with fees, slippage, margin, risk hooks, limit entries |
| Sizing (`sizing/`, `monte_carlo/`) | Kelly, half-Kelly, drawdown-capped Kelly over Monte-Carlo paths |

## Parameter-grid constraint DSL

`job_manager/` sweeps a param grid (`pg`). A string rule (`cn`) prunes it. The rule is built by `generate_constraints_function` in `job_manager/utils.py`. It runs as a safe lambda on the param set. You can write plain Python, like `p.short_period < p.long_period`.

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

Live usage is in `features/configs/tulip.py`, for example `cn='p.short_period < p.medium_period < p.long_period'`.

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

The C++ parts (`labeling/triple_barrier/tb.cc`, `sampling/sequential_bootstrap/sb.cc`, `sizing/mc_sizing`) build via a `Makefile` at `egg_info` time. You need a C++ toolchain.

## Lineage

- **Origin:** `hcmc-project/rcdb_research` (private)
- **Archive:** `tartakovsky-archive/rcdb_research` (this repo)
- **Successor:** [3Jane Technologies](https://github.com/3jane)

## Sibling repos

- [rcdb_commons](https://github.com/tartakovsky-archive/rcdb_commons) - shared client SDKs and schemas
- [rcdb_datastore](https://github.com/tartakovsky-archive/rcdb_datastore) - FastAPI time-series API
- [rcdb_dashboard](https://github.com/tartakovsky-archive/rcdb_dashboard) - Django operations console
