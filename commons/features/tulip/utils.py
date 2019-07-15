import hashlib
import inspect
from typing import List, Dict
from functools import wraps

import pandas as pd
import numpy as np

Params = List[Dict]


def cache(function):
    memo = {}

    @wraps(function)
    def wrapper(*args):
        prefix = function.__name__
        args_hashable_list = [
            hashlib.md5(arg.values.tobytes()).hexdigest()
            if type(arg) == pd.core.series.Series else str(arg)
            for arg in args
        ]
        args_hash = f"{prefix}_{'.'.join(args_hashable_list)}"
        if args_hash in memo:
            return memo[args_hash]
        else:
            rv = function(*args)
            memo[args_hash] = rv
            return rv

    return wrapper


def calc_all_helper(features_list, prefix):
    def _calc_all(data: pd.DataFrame,
                  data_mapping: Dict,
                  param_sets: Params = None,
                  inplace: bool = False) -> pd.DataFrame or None:
        res = data if inplace else dict()

        if not param_sets:
            param_sets = [{}]

        for f in features_list:
            for param_set in param_sets:
                f_args = inspect.getfullargspec(f).args
                feature = f(
                    *[data[data_mapping[name]]
                      if name in data_mapping else param_set[name]
                      for name in f_args])
                postfix = ''.join([f"_{param}" for param in param_set.values()])
                res[f"tulip_{prefix}_{f.__name__}{postfix}"] = feature

        return pd.DataFrame(res, index=data.index) if not inplace else None

    return _calc_all


def get_inputs(features_list):
    inputs = set()
    for f in features_list:
        for name, param in inspect.signature(f).parameters.items():
            if param.annotation in (np.array, pd.core.series.Series):
                inputs.add(name)
    return inputs
