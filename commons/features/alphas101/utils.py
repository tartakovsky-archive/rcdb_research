from typing import Any, List, Callable, Union

import numpy as np
import pandas as pd

from tulipindicators import ti
from scipy.stats import rankdata


def nan_to_value(ar: np.array, value: Any = 0) -> np.array:
    ar[~np.isfinite(ar)] = value


def nans_array(size: int) -> np.array:
    arr = np.empty(size)
    arr.fill(np.nan)
    return arr


def prepand_nans(arr: np.array, size: int) -> np.array:
    return np.hstack(
        (
            nans_array(size),
            arr
        )
    )


def rolling_window(a: np.array, window: int) -> List[np.array]:
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window_apply(a: np.array, window: int, func: Callable) -> np.array:
    return np.array([func(x) for x in rolling_window(a, window)])


def ts_sum(series: np.array, period: int = 10) -> np.array:
    return ti.sum(series, period=period)


def sma(series: np.array, period: int = 10) -> np.array:
    return np.hstack(
        (
            np.array([np.nan for _ in range(period - 1)]),
            np.convolve(series, np.ones(period), 'valid') / period
        )
    )


def stddev(series: np.array, period: int = 10) -> np.array:
    return pd.Series(series).rolling(period).std().values


def correlation(x: np.array, y: np.array, period: int = 10, nans: bool = False) -> np.array:
    """
    Wrapper function to estimate rolling correlations.
    :param x: corr param.
    :param y: corr param
    :param period: the rolling window period.
    :param nans: use version for inputs with NaN
    :return: a pandas DataFrame with the time-series min over
    the past 'window' days.
    """
    if not nans:
        rolling_idxs = rolling_window(np.arange(len(x)), period)
        return prepand_nans(
            [np.min(np.corrcoef(x[idxs], y[idxs])) for idxs in rolling_idxs],
            period - 1
        )

    return pd.Series(x).rolling(period).corr(pd.Series(y)).values


def covariance(x: np.array, y: np.array, period: int = 10) -> np.array:
    """
    Wrapper function to estimate rolling covariance.
    :param x: corr param 1
    :param y: corr param 2
    :param period: the rolling window period.
    :return: a pandas DataFrame with the time-series min over the past
    'window' days.
    """
    return pd.Series(x).rolling(period).cov(pd.Series(y)).values
    # rolling_idxs = rolling_window(np.arange(len(x)), period)
    # return prepand_nans(
    #     [np.min(np.cov(x[idxs], y[idxs])) for idxs in rolling_idxs],
    #     period - 1
    # )


def rolling_rank(na: np.array) -> Union[float, int]:
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]


def ts_rank(series: np.array, period: int = 10) -> np.array:
    return pd.Series(series).rolling(period).apply(rolling_rank, raw=True).values
    # return prepand_nans(
    #     rolling_window_apply(series, period, rolling_rank),
    #     period - 1
    # )


def rolling_prod(na: np.array) -> np.array:
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(series: np.array, period: int = 10) -> np.array:
    return np.hstack(
        (
            np.array([np.nan for _ in range(period - 1)]),
            rolling_window_apply(series, period, rolling_prod)
        )
    )


def ts_min(series: np.array, period: int = 10) -> np.array:
    return ti.min(series, period=period)


def ts_max(series: np.array, period: int = 10) -> np.array:
    return ti.max(series, period=period)


def delta(series: np.array, period: int = 1) -> np.array:
    return pd.core.algorithms.diff(series, period)


def delay(series: np.array, period: int = 1) -> np.array:
    return prepand_nans(
        series[:len(series) - period],
        period
    )


def rank(series: np.array) -> np.array:
    # r = rankdata(series)
    # return r / np.max(r)
    # return pd.Series(series).rank(pct=True).values

    return pd.core.algorithms.rank(series, pct=True)


def scale(series: np.array, k: int = 1) -> np.array:
    return series * k / np.sum(np.abs(np.nan_to_num(series)))


def ts_argmax(series: np.array, period: int = 10, nans: bool = False) -> np.array:
    if not nans:
        return np.hstack(
            (
                np.array([np.nan for _ in range(period - 1)]),
                np.argmax(rolling_window(series, period), axis=1)
                # rolling_window_apply(series, period, np.argmax)
            )
        )
    return pd.Series(series).rolling(period).apply(np.argmax, raw=True).values


def ts_argmin(series: np.array, period: int = 10, nans: bool = False) -> np.array:
    if not nans:
        return np.hstack(
            (
                np.array([np.nan for _ in range(period - 1)]),
                rolling_window_apply(series, period, np.argmin)
            )
        )
    return pd.Series(series).rolling(period).apply(np.argmin, raw=True).values


def decay_linear(series: np.array, period: int = 10) -> np.array:
    """
    Linear weighted moving average implementation.
    :param series: input series.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    df = pd.DataFrame(series)
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.values

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return np.hstack(na_lwma)


def indneutralize(df, ind):
    return df.groupby(ind).apply(lambda x: x - np.nanmean(x, axis=0))


def pct_change(a: np.array) -> np.array:
    return np.hstack(([np.nan], np.diff(a) / a[:-1]))
