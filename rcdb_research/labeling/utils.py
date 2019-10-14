from typing import Callable

import numba
import numpy as np
import pandas as pd

from ..features.utils import rolling_window


def _cond_after_n_bars(
    series: np.array,
    n: int,
    condition_func: Callable = lambda current, after_n_bars: True
) -> np.array:
    r = np.delete(rolling_window(series, n + 1)[n:], [j for j in range(1, n)], axis=1).transpose()
    return np.hstack(
        (
            condition_func(r[0], r[1]) * 1,
            [np.nan for _ in range(n)]
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

    n_after = np.delete(rolling_window(arr, n + 1)[n:], 0, axis=1)

    return np.hstack(
        (
            np.all(n_after, axis=1) * 1,
            [np.nan for _ in range(n)]
        )
    )


def calculate_daily_volatility(close: pd.Series, span0: int = 100) -> pd.DataFrame:
    """
    Daily volatility (from Lopez)
    :param close:
    :param span0:
    :return:
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1.0  # daily rets
    except Exception as e:
        print(f'error: {e}\nplease confirm no duplicate indices')
    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0


@numba.jit(nopython=True, parallel=True)
def triple_barrier(
    close: np.array,
    daily_volatility: np.array,
    pt_coef: float,
    sl_coef: float,
    window: int
) -> np.array:
    """
    Triple Barrier method
    :param close: close prices
    :param daily_volatility: daily volatility
    :param pt_coef: top border multiplier
    :param sl_coef: bottom border multiplier
    :param window: count of bars to vertical border
    :return:
    """
    result_set_size = daily_volatility.size - window + 1
    res = np.zeros(result_set_size, dtype=np.int8)

    sl_arr = daily_volatility * -sl_coef
    pt_arr = daily_volatility * pt_coef

    for start in numba.prange(result_set_size):
        end = start + window

        returns = close[start:end] / close[start] - 1.0

        pt = pt_arr[start]
        sl = sl_arr[start]

        for ret in returns:
            if ret < sl:
                res[start] = -1
                break

            elif ret > pt:
                res[start] = 1
                break
    return res
