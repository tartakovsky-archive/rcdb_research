import os
import uuid
import json
import inspect
import importlib
from itertools import chain, repeat
from collections import defaultdict
from typing import Callable, Union, List, Dict

import pandas as pd
import numpy as np


def store_df_to_hdf_bytes(df: pd.DataFrame, key: str = "table") -> bytes:
    with pd.HDFStore(
            "hdfs.tmp",
            mode="w",
            driver_core_backing_store=0,
            driver="H5FD_CORE"
    ) as out:
        out[key] = df
        return out._handle.get_file_image()


def get_df_from_hdf_bytes(hdf_bytes: bytes, key: str = "table") -> pd.DataFrame:
    with pd.HDFStore(
            "hdfs.tmp",
            mode="r",
            driver_core_backing_store=0,
            driver_core_image=hdf_bytes,
            driver="H5FD_CORE",
    ) as storage:
        return storage[key]


def np_to_file(path_to_file: str, ndarray: np.ndarray) -> str:
    dtype = str(ndarray.dtype)
    fpath = f'{path_to_file}-npdata-.dtype{dtype}'
    shape = str(list(ndarray.shape))
    fpath = f'{fpath}.shape{shape}'
    # with open(fpath, "wb") as f:
    #     f.write(ndarray.tobytes())
    ndarray.tofile(fpath)
    return fpath


def np_from_file(path_to_file: str) -> np.array:
    path_to_dtype, shape = path_to_file.split(".shape")
    _, dtype = path_to_dtype.split(".dtype")
    shape = json.loads(shape)
    return np.fromfile(path_to_file, dtype=dtype).reshape(*shape)


def kwargs_to_str(kwargs, brackets=True):
    if kwargs:
        params = []
        for k, v in kwargs.items():
            if type(v) not in [np.ndarray, list]:
                params.append(f'{k}={v}')
        if params:
            res = ", ".join(params)
            return res if not brackets else f'({res})'

    return "()"


def json_to_folder(d, folder):
    fname = os.path.join(folder, str(uuid.uuid4()))
    json.dump(d, open(fname, "w"))
    return fname


def json_from_file(fname):
    return json.load(open(fname, "r"))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class FnSerializer:
    @staticmethod
    def get_full_name(fn: Union[Callable, str]) -> str:
        if type(fn) == str:
            return fn

        return f'{inspect.getmodule(fn).__name__}.{fn.__name__}'

    @staticmethod
    def get_by_name(name):
        module_name, fn_name = name.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), fn_name)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def generate_constraints_function(constraints_string):
    """
    Generates constraints function by eval
    :param constraints_string: string with python expression
    :return: lambda function
    """
    # Be careful with constraints string! it must be safe!
    return eval(
        f"lambda p: {constraints_string}", {"__builtins__": {"all": all, "any": any}}
    )


def split_dict_array_values(
    d: Dict[str, Union[List, np.ndarray]],
    splits: int
) -> List[Dict[str, Union[np.ndarray]]]:
    """
    Split dict with array values to list of dict

    Example:

    >>> split_dict_array_values(dict(a=[1,2,3,4], b=[-1, -2, -3, -4]), 2)
    [{'a': array([1, 2]), 'b': array([-1, -2])},
    {'a': array([3, 4]), 'b': array([-3, -4])}]

    :param d: input dict
    :param splits: number of splits
    :return:
    """
    return [
        dict(zip(*x))
        for x in zip(
            repeat(d.keys(), splits),
            zip(*list(map(lambda v: np.array_split(v, splits), d.values())))
        )
    ]


def merge_dicts_array_values(l: List[Dict[str, Union[np.ndarray]]]) -> Dict[str, np.ndarray]:
    """
    Merge dicts and them values

    Example:
    >>> a = [{'a': np.array([1, 2]), 'b': np.array([-1, -2])}, {'a': np.array([3, 4]), 'b': np.array([-3, -4])}]
    {'a': array([1, 2, 3, 4]), 'b': array([-1, -2, -3, -4])}

    :param l: list of dict
    :return:
    """
    d = defaultdict(list)
    for k, v in chain.from_iterable(map(lambda d: d.items(), l)):
        d[k] = d[k] + v.tolist()

    return dict(zip(d.keys(), map(lambda v: np.array(v), d.values())))


def probabilities_to_predictions(probabilities: np.ndarray, labels=(-1, 1)) -> np.ndarray:
    """

    :param probabilities: probabilities matrix, (n_samples, n_labels)
    :param labels: labels (n_labels,)
    :return:
    """
    if probabilities.shape[1] != len(labels):
        raise ValueError('Count of probas columns does not equals to labels')

    return np.choose(
        np.argmax(probabilities, axis=1),
        np.array(labels)
    )
