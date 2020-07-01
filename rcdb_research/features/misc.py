from typing import Union

import numpy as np
import tulipindicators as tind


def diff(series: np.ndarray, step: int = 1, fillna: Union[float, int] = np.nan) -> np.ndarray:
    """
    Analog of pandas.Series().diff(step).fillna(fillna)

    Parameters
    ----------
    series : np.ndarray
        Input series
    step : int, optional
        Periods to shift for calculating difference. Default is 1
    fillna : Union[float, int], optional
        Value to use to fill holes. Default is np.nan
    Returns
    -------
    np.ndarray
        Difference of the series

    Examples
    --------
    >>> diff(np.array([1, 2, 4, 5]))
    array([nan,  1.,  2.,  1.])
    >>> diff(np.array([1, 2, 4, 5]), step=2, fillna=0)
    array([0., 0., 3., 3.])
    """
    r = np.empty(series.size)
    r.fill(fillna)
    r[step:] = series[step:] - series[:-step]
    return r


def frac_change(series: np.ndarray, step: int = 1, fillna: Union[float, int] = np.nan) -> np.ndarray:
    """
    Analog of pandas.Series().pct_change(step).fillna(fillna)

    Parameters
    ----------
    series : np.ndarray
        Input series
    step : int, optional
        Periods to shift for forming percent change. Default is 1
    fillna : Union[float, int], optional
        Value to use to fill holes. Default is np.nan

    Returns
    -------
    np.ndarray
        Percentage change between the current and a prior element

    Examples
    --------
    >>> frac_change(np.array([1, 2, 4, 5]), step=2, fillna=0)
    array([0. , 0. , 3. , 1.5])
    >>> frac_change(np.array([1, 2, 4, 5]), step=2, fillna=0)
    array([0. , 0. , 3. , 1.5])
    """
    return np.hstack((
        [fillna for _ in range(step)],
        np.divide((series[step:] - series[:-step]), series[:-step])
    ))


def series_ma_frac_change(series: np.ndarray, window: int, minus: Union[float, int] = 1) -> np.ndarray:
    """
    MA frac change of series
    Parameters
    ----------
    series : np.ndarray
        Input series
    window : int
        Rolling window size of SMA
    minus : Union[float, int], optional
        Coefficient. Default is 1

    Returns
    -------
    np.ndarray
        Result series

    Examples
    --------
    >>> series_ma_frac_change(np.array([1, 2, 4, 5]), window=2)
    array([       nan, 0.33333333, 0.33333333, 0.11111111])
    >>> series_ma_frac_change(np.array([1, 2, 4, 5]), window=2, minus=0)
    array([       nan, 1.33333333, 1.33333333, 1.11111111])
    """
    return np.divide(series, tind.ti.sma(series, window)) - minus


def two_series_ma_frac_change(
    series1: np.ndarray,
    series2: np.ndarray,
    window: int,
    minus: Union[float, int] = 1
) -> np.ndarray:
    """
    MA frac change of two series

    Parameters
    ----------
    series1 : np.ndarray
        First input series.
    series2 : np.ndarray
        Second input series.
    window : int
        Rolling window size of SMA
    minus : Union[float, int], optional
        Coefficient. Default is 1

    Returns
    -------
    np.ndarray
        Result series

    Examples
    --------
    >>> two_series_ma_frac_change(np.array([1, 2, 4, 5]), np.array([3, 10, 2, 5]), 2)
    array([        nan, -0.76923077, -0.5       ,  0.28571429])
    """
    return np.divide(tind.ti.sma(series1, window), tind.ti.sma(series2, window)) - minus


def direction(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute bar direction

    Parameters
    ----------
    o : np.ndarray
        Open series
    c : np.ndarray
        Close series

    Returns
    -------
    np.ndarray
        Array of directions

    Examples
    --------
    >>> direction(np.array([1, 2, 4, 5]), np.array([3, 10, 2, 5]))
    array([1, 1, 0, 0], dtype=int8)
    """
    return (c > o).astype(np.int8)


def frac_change_open_to_close(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Percent change between open and close

    Parameters
    ----------
    o : np.ndarray
        Open series
    c : np.ndarray
        Close series

    Returns
    -------
    np.ndarray
        Array of percents

    Examples
    --------
    >>> frac_change_open_to_close(np.array([1, 2, 4, 5]), np.array([3, 10, 2, 5]))
    array([ 2. ,  4. , -0.5,  0. ])
    """
    return c / o - 1


def exposure(volume_buy: np.array, volume_sell: np.array) -> np.array:
    """
    Exposure

    Parameters
    ----------
    volume_sell : np.array
        Volume sell series
    volume_buy : np.array
        Volume buy series

    Returns
    -------
    np.array
        Feature series
    """
    return volume_buy - volume_sell
