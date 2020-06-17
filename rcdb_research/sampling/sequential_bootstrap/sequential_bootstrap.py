# flake8: noqa


from ctypes import *
from ctypes import ARRAY

import numpy as np
import os
from numba import njit
import pandas as pd

dir_path = os.path.dirname(os.path.abspath(__file__))
lib = cdll.LoadLibrary(os.path.join(dir_path, 'lib.so'))

lib.sequential_sample.argtypes = (POINTER(ARRAY(c_int, 2)), c_int, c_int, c_int, c_int, POINTER(c_int))
lib.sequential_sample.restype = None

def sequential_sample_barebones(spans, sample_size, recalculate_every_n=1, seed=0):
    # spans = spans.astype(c_int)
    if spans.dtype != c_int:
        raise ValueError('spans must be of dtype `ctypes.c_int`, typically it is equal to np.int32')
    result = np.zeros(sample_size, dtype=c_int)
    lib.sequential_sample(
        spans.ctypes.data_as(POINTER(ARRAY(c_int, 2))),
        c_int(spans.shape[0]),
        c_int(sample_size),
        c_int(seed),
        c_int(recalculate_every_n),
        result.ctypes.data_as(POINTER(c_int))
    )
    return result


lib.sequential_sample_orig.argtypes = (POINTER(ARRAY(c_int, 2)), c_int, c_int, c_int, c_int, POINTER(c_int))
lib.sequential_sample_orig.restype = None


def sequential_sample_orig_barebones(spans, sample_size, recalculate_every_n=1, seed=0):
    # spans = spans.astype(c_int)
    if spans.dtype != c_int:
        raise ValueError('spans must be of dtype `ctypes.c_int`, typically it is equal to np.int32')
    result = np.zeros(sample_size, dtype=c_int)
    lib.sequential_sample_orig(
        spans.ctypes.data_as(POINTER(ARRAY(c_int, 2))),
        c_int(spans.shape[0]),
        c_int(sample_size),
        c_int(seed),
        c_int(recalculate_every_n),
        result.ctypes.data_as(POINTER(c_int))
    )
    return result


lib.sequential_sample_st.argtypes = (POINTER(ARRAY(c_int, 2)), c_int, c_int, c_int, c_int, POINTER(c_int))
lib.sequential_sample_st.restype = None


def sequential_sample_st_barebones(spans, sample_size, recalculate_every_n=1, seed=0):
    # spans = spans.astype(c_int)
    if spans.dtype != c_int:
        raise ValueError('spans must be of dtype `ctypes.c_int`, typically it is equal to np.int32')
    result = np.zeros(sample_size, dtype=c_int)
    lib.sequential_sample_st(
        spans.ctypes.data_as(POINTER(ARRAY(c_int, 2))),
        c_int(spans.shape[0]),
        c_int(sample_size),
        c_int(seed),
        c_int(recalculate_every_n),
        result.ctypes.data_as(POINTER(c_int))
    )
    return result


lib.sequential_sample_prefixsum.argtypes = (POINTER(ARRAY(c_int, 2)), c_int, c_int, c_int, c_int, POINTER(c_int))
lib.sequential_sample_prefixsum.restype = None


def sequential_sample_prefixsum_barebones(spans, sample_size, recalculate_every_n=1, seed=0):
    # spans = spans.astype(c_int)
    if spans.dtype != c_int:
        raise ValueError('spans must be of dtype `ctypes.c_int`, typically it is equal to np.int32')
    # result = np.zeros(sample_size, dtype=c_int)
    result = (c_int * sample_size)()
    lib.sequential_sample_prefixsum(
        spans.ctypes.data_as(POINTER(ARRAY(c_int, 2))),
        c_int(spans.shape[0]),
        c_int(sample_size),
        c_int(seed),
        c_int(recalculate_every_n),
        result
    )
    return np.ctypeslib.as_array(result)


lib.sequential_sample_prefixsum_optrng.argtypes = (POINTER(ARRAY(c_int, 2)), c_int, c_int, c_int, c_int, POINTER(c_int))
lib.sequential_sample_prefixsum_optrng.restype = None


def sequential_sample_prefixsum_optrng_barebones(spans, sample_size, recalculate_every_n=1, seed=0):
    # spans = spans.astype(c_int)
    if spans.dtype != c_int:
        raise ValueError('spans must be of dtype `ctypes.c_int`, typically it is equal to np.int32')
    result = np.zeros(sample_size, dtype=c_int)
    lib.sequential_sample_prefixsum_optrng(
        spans.ctypes.data_as(POINTER(ARRAY(c_int, 2))),
        c_int(spans.shape[0]),
        c_int(sample_size),
        c_int(seed),
        c_int(recalculate_every_n),
        result.ctypes.data_as(POINTER(c_int))
    )
    return result


@njit
def compute_gcd(a, b):
    if b == 0:
        return a
    else:
        return compute_gcd(b, a % b)


@njit
def get_their_gcd(timestamps):
    gcd = timestamps[0]
    for i in range(1, timestamps.size):
        gcd = compute_gcd(gcd, timestamps[i])
    return gcd


def encode(t1, bars_idx):
    spans = t1.reset_index().values
    m = pd.Series(
        np.arange(bars_idx.shape[0], dtype=c_int),
        index=bars_idx
    )
    spans = np.column_stack([m[spans[:, 0]], m[spans[:, 1]]])
    spans = spans - spans.min()
    # spans = spans // get_their_gcd(spans.flatten())
    return spans


def sequential_bootstrap(t1, bars_idx, sample_size=None, seed=0):
    spans = encode(t1, bars_idx)
    if isinstance(sample_size, float):
        sample_size = int(np.ceil(t1.shape[0] * sample_size))
    elif sample_size is None:
        sample_size = spans.shape[0]
    return sequential_sample_prefixsum_barebones(spans, sample_size, seed=seed)


def average_uniqueness(t1, bars_idx, sample):
    spans = encode(t1, bars_idx)
    s = spans[sample]
    active = np.ones(s.max() + 2)
    uniq = np.ones(s.shape[0])
    for a, b in s:
        active[a:b + 1] += 1
    for i, (a, b) in enumerate(s):
        uniq[i] = np.mean(1 / active[a:b + 1])
    avg_uniq = uniq.mean()
    return avg_uniq
