"""
An module that implements bars consolidators.

Fixed threshold
---------------
- :func:`rcdb_research.bars.time`
- :func:`rcdb_research.bars.percent_o2c`
- :func:`rcdb_research.bars.percent`
- :func:`rcdb_research.bars.all_possible_percent_bars`
- :func:`rcdb_research.bars.fixed_volume`
- :func:`rcdb_research.bars.fixed_quote_volume`
- :func:`rcdb_research.bars.fixed_ticks`

Adaptive threshold
--------------------
- :func:`rcdb_research.bars.adaptive_volume`
- :func:`rcdb_research.bars.adaptive_quote_volume`
- :func:`rcdb_research.bars.adaptive_ticks`

Others
------
- :func:`rcdb_research.bars.fixed_percent_fixed_ticks`
- :func:`rcdb_research.bars.adaptive_percent`
- :func:`rcdb_research.bars.fixed_percent_fixed_time`


Consolidators
-------------
"""
from typing import Union, Generator, Optional, List

import pandas as pd

from . import facade


def time(df: pd.DataFrame, period: Union[str, pd.DateOffset, pd.Timedelta, int], **kwargs) -> pd.DataFrame:
    """
    Time Fixed Consolidator

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    period : Union[str, pd.DateOffset, pd.Timedelta, int]
        if type is int then used as seconds
        resample frequency https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    kwargs : dict
        Additional parameters for :func:`~rcdb_research.bars.functions.time_fixed`

    Returns
    -------
     pd.DataFrame
        Consolidated data
    """
    return facade.time(df, period, **kwargs)


def percent_o2c(df: pd.DataFrame, threshold: float, **kwargs) -> pd.DataFrame:
    """
    Price move (range) accumulation feature. Fixed % range.

    Parameters
    ----------
    df : pd.DataFrame
        input ohlcv dataframe
    threshold : float
        Event UP/DOWN is generated after price moves by more percent than this threshold
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.percent_o2c(df, threshold, **kwargs)


def percent(df: pd.DataFrame, threshold: float, **kwargs) -> pd.DataFrame:
    """
    Close-Close consolidator. Fixed % range.

    Parameters
    ----------
    df : pd.DataFrame
        input ohlcv dataframe
    threshold : float
        Event UP/DOWN is generated after price moves by more percent than this threshold
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.percent(df, threshold, **kwargs)


def all_possible_percent_bars(
    df: pd.DataFrame,
    threshold: float,
    n_bars: int,
    n_jobs: int = None,
    verbose: int = 0,
    **kwargs
) -> Union[Generator[Optional[pd.DataFrame], None, None], List[Optional[pd.DataFrame]]]:
    """
    Consolidates n percent bars back from each input bar.
    Results are returned in reverse order (from last input bar to first).

    Parameters
    ----------
    df : pd.DataFrame
        input ohlcv dataframe
    threshold : float
        Event UP/DOWN is generated after price moves by more percent than this threshold
    n_bars : int
        Count of percent bars
    n_jobs : int
        If None, then returns a generator, otherwise generates a list of data frames with bars using joblib
    verbose : int
        joblib verbose parameter. Default is 0
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    Union[Generator[Optional[pd.DataFrame], None, None], List[Optional[pd.DataFrame]]]
        Generator or list of bars

    """
    return facade.all_possible_percent_bars(
        df, threshold, n_bars, n_jobs, verbose, **kwargs
    )


def fixed_volume(df: pd.DataFrame, threshold: float, **kwargs) -> pd.DataFrame:
    """
    Volume Fixed Threshold
    Generates bar after cumulative volume reaches threshold

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    threshold : float
        Event is generated after cumulative volume reaches this threshold
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.fixed(df, threshold, column='volume', **kwargs)


def fixed_quote_volume(df: pd.DataFrame, threshold: float, **kwargs) -> pd.DataFrame:
    """
    Volume Quote Fixed Threshold
    Generates bar after cumulative volume quote reaches threshold

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    threshold : float
        Event is generated after cumulative volume reaches this threshold
    column : str
        The name of the aggregated column
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.fixed(df, threshold, column='volume_quote', **kwargs)


def fixed_ticks(df: pd.DataFrame, threshold: float, **kwargs) -> pd.DataFrame:
    """
    Ticks Fixed Threshold
    Generates bar after cumulative ticks reaches threshold

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    threshold : float
        Event is generated after cumulative ticks reaches this threshold
    column : str
        The name of the aggregated column
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.fixed(df, threshold, column='ticks', **kwargs)


def adaptive_volume(df: pd.DataFrame, avg_per: int, window: int, n: int = None, **kwargs) -> pd.DataFrame:
    """
    Volume Adaptive Threshold
    Create new bar when threshold reaches "weekly average for year".

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    avg_per : int
        Get rolling avg_per count series avg
    window : int
        Series should aggregate window amount of averaged (by avg_per) series
    n : int, optional
        Calculate threshold every n bars instead of each bar
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.adaptive(
        df, avg_per, window, n=n, column='volume', **kwargs
    )


def adaptive_quote_volume(df: pd.DataFrame, avg_per: int, window: int, n: int = None, **kwargs) -> pd.DataFrame:
    """
    Quote Volume Adaptive Threshold
    Create new bar when threshold reaches "weekly average for year".

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    avg_per : int
        Get rolling avg_per count series avg
    window : int
        Series should aggregate window amount of averaged (by avg_per) series
    n : int, optional
        Calculate threshold every n bars instead of each bar
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.adaptive(
        df, avg_per, window, n=n, column='volume_quote', **kwargs
    )


def adaptive_ticks(df: pd.DataFrame, avg_per: int, window: int, n: int = None, **kwargs) -> pd.DataFrame:
    """
    Ticks Adaptive Threshold
    Create new bar when threshold reaches "weekly average for year".

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    avg_per : int
        Get rolling avg_per count series avg
    window : int
        Series should aggregate window amount of averaged (by avg_per) series
    n : int, optional
        Calculate threshold every n bars instead of each bar
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.adaptive(
        df, avg_per, window, n=n, column='ticks', **kwargs
    )


def fixed_percent_fixed_ticks(
    df: pd.DataFrame,
    percent_threshold: float,
    series_threshold: float,
    **kwargs
) -> pd.DataFrame:
    """
    Percent price threshold combined with any fixed threshold series feature
    Price move (range) and ticks accumulation feature. Fixed % range, fixed n ticks.

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    percent_threshold : float
        Range condition satisfied is  after price moves by more percent than this threshold
    series_threshold : float
        Ticks condition is satisfied after cumulative number of ticks reaches this threshold
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.fixed_percent_fixed_series(
        df, percent_threshold, series_threshold, series_column='ticks', **kwargs
    )


def adaptive_percent(df: pd.DataFrame, avg_per: int, window: int, **kwargs) -> pd.DataFrame:
    """
    Adaptive percent bars

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    avg_per : int
        Average period
    window : int
        Rolling window size
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.adaptive_percent(df, avg_per, window, **kwargs)


def fixed_percent_fixed_time(
    df: pd.DataFrame,
    period: str,
    threshold: float,
    column: str = 'close',
    **kwargs
) -> pd.DataFrame:
    """
    Fixed percent fixed time

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    period : str
        Timeframe string, e.g. 1D, 1H, 3m, etc.
    threshold : float
        Threshold value
    column : str
        Column`s name
    kwargs : dict
        Additional data for :func:`~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    return facade.fixed_percent_fixed_time(
        df, period, threshold, column, **kwargs
    )
