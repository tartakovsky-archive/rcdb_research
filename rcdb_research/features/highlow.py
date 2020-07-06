from itertools import zip_longest

import numpy as np
import tulipindicators

##########
# Helpers
##########


def pct_change(a: np.ndarray) -> np.ndarray:
    return np.hstack(([np.nan], np.diff(a) / a[:-1]))


def is_extremum_bars_periods(series: np.ndarray, period: int, maximum: bool) -> np.ndarray:
    extremum_func = tulipindicators.ti.max if maximum else tulipindicators.ti.min
    return (extremum_func(series, period) == series) * 1


def bars_since_mark(series: np.ndarray):
    poses, = np.where(series == 1)
    if poses.size == 1:
        if not poses:
            return np.arange(series.size)

        return np.hstack(
            (
                np.zeros(poses[0], dtype=np.int8),
                np.arange(series.size - poses[0])
            )
        )

    res = np.zeros(series.size, dtype=np.int8)
    poses_end = np.append(poses[1:], res.size)

    sizes = poses_end - poses

    for start, end, size in zip(poses, poses_end, sizes):
        res[start:end] = np.arange(size)
    return res


def bars_in_marked(series: np.ndarray):
    res = np.zeros(series.size, dtype=np.uint64)
    res[0] = bool(series[0])
    for i in range(1, series.size):
        res[i] = (res[i - 1] + 1 if res[i - 1] else 1) if series[i] else 0

    return res


def change_since_mark(series: np.ndarray, marked: np.ndarray):
    poses, = np.where(marked == 1)
    change = np.zeros(marked.size)

    for start, end in zip_longest(poses, poses[1:], fillvalue=None):
        change[start:end] = (series[start:end] - series[start]) / series[start]

    return change

############
# Features
############


def is_local_high(high: np.ndarray, period: int) -> np.ndarray:
    """
    Check if the high is a highest in period

    Parameters
    ----------
    high : np.ndarray
        Input series
    period : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Array of 0 and 1 for each row

    Examples
    --------
    >>> is_local_high(np.array([0, 1, 2, 3, 5, 3, 2, 1, 4]), period=3)
    array([0, 0, 1, 1, 1, 0, 0, 0, 1])
    """
    return is_extremum_bars_periods(series=high, period=period, maximum=True)


def is_local_low(low: np.ndarray, period: int) -> np.ndarray:
    """
    Check if the low is a lowest in period

    Parameters
    ----------
    low : np.ndarray
        Input series
    period : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Array of 0 and 1 for each row

    Examples
    --------
    >>> is_local_low(np.array([0, 1, 2, 3, 5, 3, 2, 1, 4]), period=3)
    array([0, 0, 0, 0, 0, 1, 1, 1, 0])
    """
    return is_extremum_bars_periods(series=low, period=period, maximum=False)


def bars_since_local_high(high: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate numbers of bars since highest in period

    Parameters
    ----------
    high : np.ndarray
        Input series
    period : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Array with numbers of bars since highest in period

    Examples
    --------
    >>> bars_since_local_high(np.array([0, 1, 2, 3, 5, 3, 2, 1, 4]), period=3)
    array([0, 0, 0, 0, 0, 1, 2, 3, 0], dtype=int8)
    """
    return bars_since_mark(
        is_extremum_bars_periods(
            series=high,
            period=period,
            maximum=True
        )
    )


def bars_since_local_low(low: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate numbers of bars since lowest in period

    Parameters
    ----------
    low : np.ndarray
        Input series
    period : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Array with numbers of bars since lowest in period

    Examples
    --------
    >>> bars_since_local_low(np.array([0, 1, 2, 3, 5, 3, 2, 1, 4]), period=3)
    array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int8)
    """
    return bars_since_mark(
        is_extremum_bars_periods(
            series=low,
            period=period,
            maximum=False
        )
    )


def bars_in_drawdown(close: np.ndarray) -> np.ndarray:
    """
    Calculate numbers of bars in drawdown

    Parameters
    ----------
    close : np.ndarray
        Input series

    Returns
    -------
    np.ndarray
        Array with numbers of bars in drawdown

    Examples
    --------
    >>> bars_in_drawdown(np.array([0, 1, 2, 3, 5, 3, 2, 1, 4]))
    array([0, 0, 0, 0, 0, 1, 2, 3, 0], dtype=uint64)
    """
    return bars_in_marked(
        (pct_change(close) < 0) * 1
    )


def bars_in_runup(close: np.ndarray) -> np.ndarray:
    """
    Calculate numbers of bars in run up

    Parameters
    ----------
    close : np.ndarray
        Input series

    Returns
    -------
    np.ndarray
        Array with numbers of bars in run up

    Examples
    --------
    >>> bars_in_runup(np.array([0, 1, 2, 3, 5, 3, 2, 1, 4]))
    array([0, 1, 2, 3, 4, 0, 0, 0, 1], dtype=uint64)
    """
    return bars_in_marked(
        (pct_change(close) > 0) * 1
    )


def change_since_local_high(high: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate % change since highest in period

    Parameters
    ----------
    high : np.ndarray
        Input series
    period : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Array with % change since highest in period

    Examples
    --------
    >>> change_since_local_high(np.array([0, 1, 2, 3, 5, 3, 2, 1, 4]), period=3)
    array([ 0. ,  0. ,  0. ,  0. ,  0. , -0.4, -0.6, -0.8,  0. ])
    """
    return change_since_mark(
        series=high,
        marked=is_extremum_bars_periods(
            series=high,
            period=period,
            maximum=True
        )
    )


def change_since_local_low(low: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate % change since lowest in period

    Parameters
    ----------
    low : np.ndarray
        Input series
    period : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Array with % change since lowest in period

    Examples
    --------
    >>> change_since_local_low(np.array([0, 1, 2, 3, 5, 3, 2, 1, 4]), period=3)
    array([0., 0., 0., 0., 0., 0., 0., 0., 3.])
    """
    return change_since_mark(
        series=low,
        marked=is_extremum_bars_periods(
            series=low,
            period=period,
            maximum=False
        )
    )
