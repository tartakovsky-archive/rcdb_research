import numba
import numpy as np


# @numba.jit
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


@numba.guvectorize(['void(float64[:], intp[:], float64[:])'], '(n),()->(n)')
@numba.jit(nopython=True, nogil=True, parallel=True)
def move_mean_numba(a, window_arr, out):
    window_width = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count


def move_mean(a, window_arr):
    out = np.zeros(a.size)
    window_width = window_arr
    asum = 0.0
    count = 0

    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count

    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count

    return out


def not_compiled(series: np.array, fast: int, slow: int) -> np.array:
    return move_mean(series, fast) - move_mean(series, slow)


def compiled(series: np.array, fast: int, slow: int) -> np.array:
    return move_mean_numba(series, fast) - move_mean_numba(series, slow)
