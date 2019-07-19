import hashlib
import importlib
import inspect
import itertools
from functools import wraps
from typing import Dict, List

import pandas as pd

from commons.features.utils import get_namespaces_around

SERIES_MAPPING = ('open', 'high', 'low', 'close')
NAMESPACES = get_namespaces_around(__file__)

Params = List[Dict]


def _is_auto_mapping_feature(indicator, func_name):
    return indicator == 'bbands' and func_name in ['f7', 'f8', 'f9'] \
        or indicator == 'psar' and func_name in ['f2']


def _get_input_sets(indicator, func_name, func_args, data_mapping):
    input_sets = []
    if _is_auto_mapping_feature(indicator, func_name):
        for series in SERIES_MAPPING:
            input_sets.append([
                data_mapping[arg] if arg != 'series' else series
                for arg in func_args if arg in data_mapping
            ])
    else:
        input_sets.append(
            [data_mapping[arg] for arg in func_args if arg in data_mapping])
    return input_sets


def get_sub_inputs(namespaces):
    inputs = []
    for namespace in namespaces:
        module = importlib.import_module(name=f".{namespace}",
                                         package='commons.features.tulip')
        inputs += getattr(module, 'inputs', [])
    return tuple(set(inputs))


def cache(function):
    memo = {}

    @wraps(function)
    def wrapper(*args):
        indicator = function.__name__
        args_hashable_list = [
            hashlib.md5(arg.values.tobytes()).hexdigest()
            if type(arg) == pd.core.series.Series else str(arg) for arg in args
        ]
        args_hash = f"{indicator}_{'.'.join(args_hashable_list)}"
        if args_hash in memo:
            return memo[args_hash]
        else:
            rv = function(*args)
            memo[args_hash] = rv
            return rv

    return wrapper


def calc_all_helper(feature_functions, indicator):
    def _calc_all(data: pd.DataFrame,
                  data_mapping: Dict,
                  indicator_param_sets: Params = None,
                  feature_param_sets: Params = None,
                  inplace: bool = False) -> pd.DataFrame or None:
        res = data if inplace else {}
        if not indicator_param_sets:
            indicator_param_sets = [{}]
        if not feature_param_sets:
            feature_param_sets = [{}]

        for func in feature_functions:
            func_args = inspect.getfullargspec(func).args
            func_name = func.__name__

            input_sets = _get_input_sets(indicator, func_name, func_args,
                                         data_mapping)

            arg_sets = itertools.product(
                input_sets,
                indicator_param_sets,
                feature_param_sets,
            )

            for inputs, indicator_params, feature_params in arg_sets:
                feature = func(
                    *[data[i].values for i in inputs],
                    *indicator_params.values(),
                    *[v for k, v in feature_params.items() if k in func_args],
                )
                suffix = ''.join(
                    [f"_{i}" for i in inputs] +  # noqa
                    [f"_{p}" for p in indicator_params.values()] +  # noqa
                    [f"_{p}" for p in feature_params.values() if p in func_args]
                )  # yapf: disable
                res[f"tulip_{indicator}_{func_name}{suffix}"] = feature

        return pd.DataFrame(res, index=data.index) if not inplace else None

    return _calc_all


def calc_all(data: pd.DataFrame,
             data_mapping: Dict,
             indicators_param_sets: Params = None,
             features_param_sets: Params = None,
             inplace: bool = False) -> pd.DataFrame or None:
    def _call(func, indicator_param_sets, feature_param_sets):
        return func(data, data_mapping, indicator_param_sets,
                    feature_param_sets, inplace)

    results = []
    for namespace in NAMESPACES:
        module = importlib.import_module(name=f".{namespace}",
                                         package='commons.features.tulip')
        calc_all_func = getattr(module, 'calc_all')
        if inplace:
            _call(calc_all_func, indicators_param_sets.get(namespace),
                  features_param_sets.get(namespace))
        else:
            results.append(
                _call(calc_all_func, indicators_param_sets.get(namespace),
                      features_param_sets.get(namespace)))

    if results:
        df = pd.concat(results, axis='columns')
        df.index = data.index
        return df
