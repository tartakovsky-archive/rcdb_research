import os
import pickle
import hashlib
from functools import wraps


def check_args(args):
    for arg in args:
        if not any([
            isinstance(arg, int),
            isinstance(arg, float),
            isinstance(arg, bool),
            isinstance(arg, str),
            isinstance(arg, list) and check_args(arg),
            isinstance(arg, tuple) and check_args(arg),
            isinstance(arg, dict) and check_args(arg.keys()) and check_args(arg.values())
        ]):
            raise ValueError(
                'please consider sticking to simpler datatypes for the function arguments: '
                'int, float, bool, str; list, tuple, or dict'
            )
    return True


def normalize(d):
    if isinstance(d, dict):
        d = {k: normalize(v) for k, v in sorted(d.items(), key=lambda x: x[0])}
    elif isinstance(d, list):
        d = [normalize(x) for x in d]
    elif isinstance(d, tuple):
        d = tuple(normalize(x) for x in d)
    return d


def cache(path):
    """
    Usage:
    ```
    @cache('./cache/ver1')
    def fun(a, b, c):
        sleep(1)
        return a + b + c
    ```

    Comparison with joblib.Memory.cache():
    - joblib can't handle functions defined in Jupyter and ran via joblib.Parallel(); this solution can
    - this solution provides only basic functionality:
        + no kwargs nor default args are supported
        + only the following datatypes are accepted as function arguments:
          int, float, bool, str; list, tuple, or dict
    - primary use case is grid search as part of experiments
    """

    def cache_inner(func):
        @wraps(func)
        def inner(*args):
            _ = check_args(args)
            args = normalize(args)
            cachepath = path
            os.makedirs(cachepath, exist_ok=True)
            encoded_args = hashlib.sha512(str(args).encode()).hexdigest()
            filepath = os.path.join(cachepath, encoded_args)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    result = pickle.load(f)
                    return result
            result = func(*args)
            with open(filepath, 'wb') as f:
                pickle.dump(result, f)
            return result

        return inner

    return cache_inner
