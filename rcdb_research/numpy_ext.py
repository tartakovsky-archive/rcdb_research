from typing import Callable, Any, Union, Generator

from joblib import Parallel, delayed
import numpy as np


Number = Union[int, float]


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
    if not isinstance(window, int):
        raise TypeError(f'window type is {type(window)}({window}). Required int')

    res = np.empty((array.size, window))
    res.fill(None)

    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    res[window - 1:] = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return res


def rolling_gen(array: np.array, window: int, skip_nans: bool = False) -> Generator[np.ndarray, None, None]:
    """
    Creates rolling ndarray generator
    :param array: input array
    :param window: rolling window size
    :param skip_nans: if True skip's first `window - 1` nans
    :return: rolling ndarray
    """
    if not isinstance(window, int):
        raise TypeError(f'window type is {type(window)}({window}). Required int')

    if array.size < window:
        raise ValueError('window > array.size!')

    if not skip_nans:
        yield from (nans_array(window) for _ in np.arange(window - 1))

    yield from (array[i:i + window] for i in np.arange(array.size - (window - 1)))


def rolling_apply(func: Callable, window: int, *arrays: np.array, n_jobs=1, **kwargs) -> np.array:
    """
    Rolling apply for numpy arrays
    :param func: aggregation function
    :param window: rolling window size
    :param arrays: inputs list of 1-D arrays
    :param kwargs: input parameters (passed to func)
    :return:
    """
    if not isinstance(window, int):
        raise TypeError(f'window type is {type(window)}({window}). Required int')

    if max(len(x.shape) for x in arrays) != 1:
        raise ValueError("Supported only 1-D arrays")

    if len({array.size for array in arrays}) != 1:
        raise ValueError("Arrays must be the same length")

    def apply(idxs):
        return func(*[array[idxs.astype(np.int)] for array in arrays], **kwargs)

    rolls = rolling_gen(np.arange(len(arrays[0])), window, skip_nans=True)
    arr = Parallel(n_jobs=n_jobs)(delayed(apply)(idxs) for idxs in rolls)

    return prepend_na(arr, size=window - 1)


def nans_array(size: int) -> np.array:
    """
    Creates array with nans
    :param size: array size
    :return: np.array with nans
    """
    arr = np.empty(size)
    arr.fill(np.nan)
    return arr


def pct_range(
    start: Number,
    end: Number,
    min_step: Number,
    mult_step: Number = 1,
    round_func: Callable = None
) -> np.array:
    """
    Generates range array with pct step
    :param start: start value
    :param end: end value (included)
    :param min_step: min step
    :param mult_step: step multiplier
    :param round_func: vectorized rounding function, e.g. np.ceil, np.floor etc.
    :return:
    """
    last = start
    values = []

    while last < end:
        values.append(last)
        step = abs(last * mult_step)
        last += max(step, min_step) * np.sign(mult_step)

    values = np.array(values)
    return np.unique(round_func(values)) if round_func else values
