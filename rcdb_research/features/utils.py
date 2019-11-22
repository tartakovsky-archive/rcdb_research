import time

import numpy as np
import pandas as pd

from typing import Dict, List


def _rolling_window(a: np.array, window: int):
    res = np.empty((a.size, window))
    res.fill(None)

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    res[window - 1:] = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return res


def rolling_window(a: np.array, window: int, cache: dict = None) -> np.ndarray:
    if cache is None:
        return _rolling_window(a, window)

    cache_key = f"{hash(a.tostring())}_{window}"
    if cache_key in cache:
        return cache[cache_key]

    res = _rolling_window(a, window)

    cache[cache_key] = res
    return res


def measure_elapsed(f, *args, **kwargs):
    start = time.time()
    res = f(*args, **kwargs)
    end = time.time()
    return res, end - start


def generate_calc_all(prefix: str, feature_funcs: dict):
    def calc_all(
            data: pd.DataFrame,
            param_set: Dict[str, List[Dict]],
            window: int,
            column_names: dict = None,
            benchmark=False
    ) -> pd.DataFrame:
        """
        Calculate features from mne
        :param pd.DataFrame data: df with DatetimeIndex, with `column_names.values()` columns
        :param Dict[str, List[Dict]] param_set: keys - names of feature (f1, f2, etc.),
            set of parameters for feature calculation, required key "column"
        :param window: rolling window size
        :param dict column_names: mapping of required columns
        :return:
        """
        if column_names is None:
            column_names = dict(zip(data.columns, data.columns))

        vals = {name: data[column_names[orig_name]].values for orig_name, name in column_names.items()}

        df = pd.DataFrame(index=data.index)
        measurements = {}
        for feature, feature_params in param_set.items():
            for ps in feature_params:
                column = ps.pop("column", "__all__")
                inputs_series = vals if column == "__all__" else {column: vals[column]}

                for input_series_name, input_series in inputs_series.items():
                    res, elapsed = measure_elapsed(feature_funcs[feature], series=input_series, window=window, **ps)

                    postfix = "_".join(f"{k}{v}" for k, v in ps.items())
                    col_name = f"{prefix}_{feature}_{window}_{input_series_name}{'_' if postfix else ''}{postfix}"

                    expected_column_size = len(input_series)
                    if len(res) == expected_column_size and len(res.shape) == 1:
                        df[col_name] = res
                        measurements[col_name] = elapsed
                    else:
                        for i in range(len(res)):
                            df[f"{col_name}_col_{i}"] = res[i]
                            measurements[f"{col_name}_col_{i}"] = elapsed
        if not benchmark:
            return df
        else:
            return df, measurements

    return calc_all


def feature_registrator_factory(features_dict):

    def register_feature(features_dict):
        def inner(func):
            features_dict[func.__name__] = func
            return func

        return inner

    return register_feature(features_dict)
