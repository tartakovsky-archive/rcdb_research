import os
import uuid
import json
import struct
import inspect
import importlib
from typing import Callable, Union, Tuple

import numpy as np


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


def np_to_file_custom_bytes(task_serialized: str, path_to_file: str, ndarray: np.ndarray):
    name_bytes = f'{task_serialized}__dtype__{ndarray.dtype}'.encode()
    shape_bytes = json.dumps(ndarray.shape).encode()

    meta = struct.pack('Q', len(name_bytes)) + name_bytes + struct.pack('Q', len(shape_bytes)) + shape_bytes
    with open(path_to_file, 'wb') as f:
        f.write(meta + ndarray.tobytes())


def np_from_file_custom_bytes(path_to_file: str) -> Tuple[str, np.ndarray]:
    with open(path_to_file, 'rb') as f:
        data = f.read()

    Q_BYTES_LEN = 8

    name_len = struct.unpack('Q', data[:Q_BYTES_LEN])[0]
    name_end = Q_BYTES_LEN + name_len
    name, dtype = data[Q_BYTES_LEN:name_end].decode().split('__dtype__')

    shape_len_end = name_end + Q_BYTES_LEN
    shape_len = struct.unpack('Q', data[name_end:shape_len_end])[0]
    shape_end = shape_len_end + shape_len

    shape = json.loads(data[shape_len_end:shape_end].decode())

    return name, np.frombuffer(data[shape_end:], dtype=dtype).reshape(shape)


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

        if inspect.getmodule(fn).__name__ == '__main__':
            raise ValueError('functions without import path are not allowed: {}'.format(fn.__name__))

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
