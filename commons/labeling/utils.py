from typing import Callable

import numpy as np

from ..features.utils import rolling_window


def _cond_after_n_bars(series: np.array, n: int, condition_func: Callable = lambda current, after_n_bars: True) -> np.array:
    r = np.delete(rolling_window(series, n + 2), [j for j in range(1, n + 1)], axis=1).transpose()
    return np.hstack(
            (
                condition_func(r[0], r[1]) * 1,
                [np.nan for _ in range(n + 1)]
            )
        )


def label_direction(series: np.array) -> np.array:
    shifted = np.hstack(([series[0]], series[:-1]))
    direction = np.zeros(series.size)

    is_up = (series > shifted)
    is_down = (series < shifted)

    direction[is_up] = 1
    direction[is_down] = -1
    return direction


def _n_consecutive(series: np.array, n: int, arr_operator: Callable) -> np.array:
    arr = np.zeros(series.size)
    arr[arr_operator(label_direction(series))] = 1

    n_after = np.delete(rolling_window(arr, n + 1), 0, axis=1)

    return np.hstack(
        (
            np.all(n_after, axis=1) * 1,
            [np.nan for _ in range(n)]
        )
    )
