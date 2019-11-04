from typing import Callable, Any

import numpy as np


def drop_na(array: np.array) -> np.array:
    """
    Removes nans from array
    :param array: input array
    :return: prepared array
    """
    return array[~np.isnan(array)]


def fill_na(array: np.array, value: Any) -> np.array:
    """
    Replaces nan values to `value`
    :param array: input array
    :param value: replacement value
    :return: prepared array
    """
    ar = array.copy()
    ar[np.isnan(ar)] = value
    return ar


def fill_not_finite(array: np.array, value: Any = 0) -> np.array:
    """
    Replaces not finite values (inf, nan) to `value`
    :param array: input array
    :param value: replacement value
    :return: prepared array
    """
    ar = array.copy()
    ar[~np.isfinite(array)] = value
    return ar


def prepend_na(array: np.array, size: int) -> np.array:
    """
    Inserts nan items to array
    :param array: input array
    :param size: count of nans
    :return: prepared array
    """
    nans_array = np.empty(size)
    nans_array.fill(None)
    return np.hstack((nans_array, array))


def rolling(array: np.array, window: int) -> np.ndarray:
    """
    Creates rolling ndarray
    :param array: input array
    :param window: rolling window size
    :return: rolling ndarray
    """
    res = np.empty((array.size, window))
    res.fill(None)

    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    res[window - 1:] = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return res


def rolling_apply(func: Callable, window: int, *arrays: np.array, **kwargs) -> np.array:
    """
    Rolling apply for numpy arrays
    :param func: aggregation function
    :param window: rolling window size
    :param arrays: inputs list of 1-D arrays
    :param kwargs: input parameters (passed to func)
    :return:
    """
    if max(len(x.shape) for x in arrays) != 1:
        raise ValueError("Supported only 1-D arrays")

    if len({array.size for array in arrays}) != 1:
        raise ValueError("Arrays must be the same length")

    return prepend_na(
        array=np.array(
            [
                func(*[array[idxs] for array in arrays], **kwargs)
                for idxs in rolling(np.arange(len(arrays[0])), window)[window - 1:].astype(np.int)
            ]
        ),
        size=window - 1
    )


def nans_array(size: int) -> np.array:
    """
    Creates array with nans
    :param size: array size
    :return: np.array with nans
    """
    arr = np.empty(size)
    arr.fill(np.nan)
    return arr
