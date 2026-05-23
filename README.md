# rcdb_research

Cloned and published from hcmc-project/rcdb_research for archival purposes.

---

## Archival Notes

rcdb_research is a quantitative research framework for developing, backtesting, and evaluating systematic trading strategies. It provides an end-to-end pipeline from raw market data to production-ready strategy configs. The bar generation module converts tick data into information-driven bars (imbalance, tick, volume, dollar, and run bars) using the de Prado framework, with configurable thresholds and adaptive sampling. The feature engineering layer produces hundreds of derived signals — rolling statistics, technical indicators, market microstructure features — with a parameter grid system that supports combinatorial sweeps with pydantic-validated constraint expressions (e.g. `p.a < p.b and p.c != -1`).

The labeling module implements the triple barrier method with dynamic barrier widths, enabling supervised learning targets for directional prediction. Cross-validation uses purged and embargoed walk-forward splits to prevent data leakage across time. Feature importance is assessed through multiple lenses — mean decrease in impurity (MDI), mean decrease in accuracy (MDA via permutation), mutual information, and ensemble-aggregated importance — all designed to survive the high noise-to-signal ratio of financial data. The backtesting engine simulates trading with realistic constraints: commission models, slippage, borrow costs, position sizing algorithms, and multi-asset portfolio construction. Monte Carlo analysis validates strategy robustness against parameter and data perturbations. This code was developed as part of the RCDB team's work on a multi-exchange, multi-strategy automated trading platform, later merged into 3Jane Technologies (https://github.com/3jane).

---

## Installation/upgrade from inside of Jupyter

Run the following in the terminal, you'll be prompted for github credentials:

```bash
$ pip install -U --extra-index-url https://pypi-private:***TOKEN***@pkgs.dev.azure.com/rcdb/_packaging/pypi-private/pypi/simple/e/ git+https://github.com/tartakovsky-archive/rcdb_research
```

## Installation for development

`$ pip install --extra-index-url $(cat extra-index-url) .` - install requirements from source  
`$ pip install --extra-index-url $(cat extra-index-url) -e .[dev]` - install requirements from source for development  
`$ pip install --extra-index-url $(cat extra-index-url) -e <git url>` - install requirements from git  
`$ jupyter notebook` - start jupyter  

## Custom constraints for config parameters
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

Result parameters sets:
```python
[
    dict(a=1, b=2, c=1),
    dict(a=1, b=3, c=1),
    dict(a=2, b=3, c=1),
]
```