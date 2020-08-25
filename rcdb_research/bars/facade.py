import uuid
import functools
from typing import Union, Generator, Optional, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .functions import consolidate, price_pct_threshold, \
    fixed_threshold, adaptive_threshold, price_pct__series_fixed, \
    time_fixed, adaptive_percent_dynamic_threshold, fixed_percent_fixed_time_feature, \
    DEFAULT_AGGREGATE_MAPPING


def check_type(func=None, type_patterns=('int', 'float')):
    """
    Decorator which check input dataframe columns types

    Parameters
    ----------
    func : Callable
        Function
    type_patterns : Iterable[str]
        List of type patterns

    Returns
    -------
    Callable
        Decorated function
    """
    if func is None:
        return functools.partial(check_type, type_patterns=type_patterns)

    @functools.wraps(func)
    def wrapper(df, *args, **kwargs):
        if not all(str(dtype).startswith(type_patterns) for dtype in df.dtypes):
            raise ValueError(f'Supported columns type patters: {type_patterns}')
        return func(df, *args, **kwargs)

    return wrapper


@check_type
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
        Additional parameters for `~rcdb_research.bars.functions.time_fixed`

    Returns
    -------
     pd.DataFrame
        Consolidated data
    """
    return time_fixed(df, period, **kwargs)


@check_type
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
        Additional data for `~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    df['f'] = price_pct_threshold(df.open.values, df.close.values, threshold, None, None)
    bars = consolidate(df, column_name="f", **kwargs).drop(columns=['f'])
    df.drop(columns=['f'], inplace=True)
    return bars


@check_type
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
        Additional data for `~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    prev_close = df.close.shift().copy()
    prev_close[0] = df.open[0]
    df['f'] = price_pct_threshold(prev_close.values, df.close.values, threshold, None, None)
    bars = consolidate(df, column_name="f", **kwargs).drop(columns=['f'])
    df.drop(columns=['f'], inplace=True)
    return bars


@check_type
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
        Additional data for `~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    Union[Generator[Optional[pd.DataFrame], None, None], List[Optional[pd.DataFrame]]]
        Generator or list of bars

    """
    idx_col = uuid.uuid4().hex
    agg = {**DEFAULT_AGGREGATE_MAPPING, **kwargs, 'open': 'last', 'close': 'first', idx_col: 'last'}

    def _func(i, df):
        df_reversed = df.iloc[:i + 1].iloc[::-1]
        f = price_pct_threshold(df_reversed.close.values, df_reversed.open.values, threshold, None, n_bars=n_bars + 1)

        df_reversed = df_reversed[:len(f)]
        df_reversed['f'] = f
        df_reversed[idx_col] = df_reversed.index

        res = consolidate(df_reversed, column_name='f', aggregate=agg)[:n_bars].drop('f', 1).iloc[::-1]

        res.set_index(idx_col, inplace=True)
        res.index.rename(df.index.name, inplace=True)
        return res

    idxs = list(reversed(range(1, len(df))))

    if n_jobs is None:
        def gen():
            for i in idxs:
                yield _func(i, df)
            yield None

        return gen()

    else:
        r = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_func)(i, df) for i in idxs)
        r.append(None)
        return r


@check_type
def fixed(df: pd.DataFrame, threshold: float, column: str, **kwargs) -> pd.DataFrame:
    """
    Fixed Threshold
    Fixed threshold accumulating feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    threshold : float
        Event is generated after cumulative volume reaches this threshold
    column : str
        The name of the aggregated column
    kwargs : dict
        Additional data for `~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    not_null_rows = ~pd.isnull(df[column])
    df['f'] = 0
    df.loc[not_null_rows, 'f'] = fixed_threshold(df[not_null_rows][column].values, threshold)
    bars = consolidate(df, column_name="f", **kwargs).drop(columns=['f'])
    df.drop(columns=['f'], inplace=True)
    return bars


@check_type
def adaptive(df: pd.DataFrame, avg_per: int, window: int, column: str, n: int = None, **kwargs):
    """
    Adaptive Threshold
    Adaptive accumulating feature. Create new bar when threshold reaches "weekly average for year".

    Parameters
    ----------
    df : pd.DataFrame
        Input ohlcv dataframe
    avg_per : int
        Get rolling avg_per count series avg
    window : int
        Series should aggregate window amount of averaged (by avg_per) series
    column : str
         The name of the aggregated column
    n : int, optional
        Calculate threshold every n bars instead of each bar
    kwargs : dict
        Additional data for `~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    not_null_rows = ~pd.isnull(df[column])
    df['f'] = 0
    df.loc[not_null_rows, 'f'] = adaptive_threshold(df[not_null_rows][column].values, avg_per, window, n)

    nan_end_index = df.f[pd.isnull(df.f)].index[-1]
    df.loc[:nan_end_index, 'f'] = np.nan

    bars = consolidate(df, column_name="f", **kwargs)
    bars.drop(columns=['f'], inplace=True)
    df.drop(columns=['f'], inplace=True)
    return bars


@check_type
def adaptive_percent(df: pd.DataFrame, avg_per: int, window: int, **kwargs):
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
        Additional data for `~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    df['f'] = adaptive_percent_dynamic_threshold(df.open, df.close, avg_per, window)
    bars = consolidate(df, column_name="f", **kwargs)
    bars.drop(columns=['f'], inplace=True)
    df.drop(columns=['f'], inplace=True)
    return bars


@check_type
def fixed_percent_fixed_series(df: pd.DataFrame, percent_threshold: float,
                               series_threshold: float, series_column: str, **kwargs):
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
    series_column : str
        The name of the aggregated column
    kwargs : dict
        Additional data for `~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    df['f'] = price_pct__series_fixed(
        open=df.open.values,
        close=df.close.values,
        price_threshold=percent_threshold,
        series=df[series_column],
        series_threshold=series_threshold
    )

    bars = consolidate(df, column_name="f", **kwargs).drop(columns=['f'])
    df.drop(columns=['f'], inplace=True)
    return bars


@check_type
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
        Additional data for `~rcdb_research.bars.functions.consolidate`

    Returns
    -------
    pd.DataFrame
        Consolidated data
    """
    df['f'] = fixed_percent_fixed_time_feature(
        indexes=df.index.values,
        values=df[column].values,
        period=pd.Timedelta(period).to_timedelta64(),
        threshold=threshold
    )
    bars = consolidate(df, column_name='f', **kwargs).drop(columns=['f'])
    df.drop(columns=['f'], inplace=True)
    return bars
