import os
import uuid
import json
import inspect
import importlib
from typing import Callable, Union

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
    fpath = f'{path_to_file}.dtype{dtype}'
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
    if dtype == "datetime64[ns]":
        return np.fromfile(path_to_file, dtype=np.int64).reshape(*shape).astype(dtype)
    else:
        return np.fromfile(path_to_file, dtype=getattr(np, dtype)).reshape(*shape)


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
