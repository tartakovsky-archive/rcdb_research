import os
import sys
import inspect
from typing import Dict, List

import numpy as np
import pandas as pd


def get_inputs(features_list, exclude=[]):
    inputs = set()
    for f in features_list:
        for name, param in inspect.signature(f).parameters.items():
            if param.annotation == np.array and name not in exclude:
                inputs.add(name)
    return tuple(sorted(inputs))


def get_namespaces_around(file):
    """Returens subneamespaces list around file.

    Looking for python modules that locates around file. If the module name
    doesn't start with "_" then the returned list contains that name.
    You should to name your util modules beginning with "_" if you going to use
    this function.

    :param file: the file around which subnamespaces are located
    :return: list of namespaces
    """
    path = os.path.dirname(os.path.abspath(file))
    namespaces = [
        file.replace('.py', '') for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file[0] != '_'
    ]
    return namespaces


ROLLING_WINDOW_CACHE = {}


def rolling_window(a: np.array, window: int) -> np.ndarray:
    cache_key = f"{hash(a.tostring())}_{window}"
    if cache_key in ROLLING_WINDOW_CACHE:
        return ROLLING_WINDOW_CACHE[cache_key]

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    res = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    ROLLING_WINDOW_CACHE[cache_key] = res

    return res


def feature_filter(o):
    try:
        if inspect.isfunction(o) and o.__name__[0] == "f":
            int(o.__name__[1:])
            return True

    except ValueError:
        pass

    return False


def get_feature_funcs(module_name):
    return dict(inspect.getmembers(sys.modules[module_name], feature_filter))


def generate_calc_all(prefix: str, feature_funcs: dict):
    def calc_all(
            data: pd.DataFrame,
            param_set: Dict[str, List[Dict]],
            window: int,
            column_names: dict = None
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

        df = pd.DataFrame([])
        for feature, feature_params in param_set.items():
            for ps in feature_params:
                column = ps.pop("column", "__all__")
                inputs_series = vals if column == "__all__" else {column: vals[column]}

                for input_series_name, input_series in inputs_series.items():
                    res = feature_funcs[feature](series=input_series, window=window, **ps)

                    postfix = "_".join(f"{k}{v}" for k, v in ps.items())
                    col_name = f"{prefix}_{feature}_{window}_{input_series_name}{'_' if postfix else ''}{postfix}"

                    expected_column_size = len(input_series) - window + 1
                    if len(res) == expected_column_size:
                        df[col_name] = res
                    else:
                        cols = len(res) // expected_column_size
                        for i, res in zip(range(cols), res.reshape(-1, cols).transpose()):
                            df[f"{col_name}_col_{i}"] = res
        return df

    return calc_all
