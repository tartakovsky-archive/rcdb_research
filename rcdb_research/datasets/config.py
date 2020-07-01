from typing import List

import pandas as pd
from joblib import Parallel, delayed
from rcdb_research import features as ft
from rcdb_research import bars as consolidators
from rcdb_research.bars.functions import DEFAULT_AGGREGATE_MAPPING


def add_basic_features(df):
    bars = df.copy()

    bars['timestamp'] = bars.index.values.astype("int64") / 1e9
    bars['timediff'] = ft.misc.diff(bars['timestamp'].values, fillna=1)
    bars['change'] = ft.misc.frac_change_open_to_close(o=bars['open'].values, c=bars['close'].values)

    return bars


def extract_subset(df, start=None, end=None):
    bars = df.copy()
    if start is not None:
        bars = bars[bars.index >= start]
    if end is not None:
        bars = bars[bars.index < end]

    return bars


def consolidate_datasets(datasets: List[dict], n_jobs=1) -> List[dict]:
    def config_to_bars(config: dict) -> pd.DataFrame:
        PREFIX_PATTERN = '__pref$'
        date_range = config.get('date_range', None)

        bars = config['bars']

        def new_column_name_func_factory(dataset_name):
            def fn(c):
                return f'{PREFIX_PATTERN}{dataset_name}{c}'
            return fn

        new_agg_mapping = {k: v for k, v in DEFAULT_AGGREGATE_MAPPING.items() if k in bars.columns}

        # Merge datasets
        for dataset_name, dataset_info in config.get('addtitonal_datasets', {}).items():
            new_column_name_func = new_column_name_func_factory(dataset_name)

            additional_bars = dataset_info['bars'].rename(new_column_name_func, axis=1)

            agg_mapping = {**DEFAULT_AGGREGATE_MAPPING, **dataset_info.get('agg_maping', {})}
            new_agg_mapping.update(
                {
                    new_column_name_func(c): v
                    for c, v in agg_mapping.items()
                    if c in dataset_info['bars'].columns
                }
            )

            bars = pd.concat([bars, additional_bars], axis=1)

        for cns in config['consolidators']:
            bars = getattr(consolidators, cns['type'])(bars, aggregate=new_agg_mapping, **cns['kwargs'])

        if date_range is not None:
            bars = extract_subset(bars, start=date_range.get('start'), end=date_range.get('end'))

        # unmerge datasets
        addtitonal_datasets = {}
        for dataset_name, dataset_info in config.get('addtitonal_datasets', {}).items():
            additional_columns_pattern = f'{PREFIX_PATTERN}{dataset_name}'
            additional_bars = bars[[c for c in bars.columns if c.startswith(additional_columns_pattern)]]
            addtitonal_datasets[dataset_name] = {
                **dataset_info,
                'bars': add_basic_features(
                    additional_bars.rename(lambda c: c.replace(additional_columns_pattern, ''), axis=1)
                )
            }

        res = {
            **config,
            'bars': add_basic_features(bars[[c for c in bars.columns if not c.startswith(PREFIX_PATTERN)]]),
            'addtitonal_datasets': addtitonal_datasets
        }

        return res

    def eval_config(config):
        return config_to_bars(config)

    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(delayed(eval_config)(dataset) for dataset in datasets)

    return results
