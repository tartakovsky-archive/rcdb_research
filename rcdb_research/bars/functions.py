import uuid
import logging
from typing import Union

import numpy as np
import pandas as pd
from numba import njit
from numpy_ext import rolling_apply, nans
from mlfinlab.util import get_bvc_buy_volume, ewma


DEFAULT_AGGREGATE_MAPPING = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume_buy': 'sum',
    'volume_sell': 'sum',
    'volume': 'sum',
    'volume_quote_buy': 'sum',
    'volume_quote_sell': 'sum',
    'volume_quote': 'sum',
    'ticks_buy': 'sum',
    'ticks_sell': 'sum',
    'ticks': 'sum'
}


def consolidate(
    df: pd.DataFrame,
    column_name: str,
    aggregate: dict = None,
    aggregate_default='first',
    verbose=False
):
    """
    Consolidate dataframe by `column_name` column values

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns from `aggregate`
    column_name : str
        The name of the aggregated column
    aggregate : dict, optional
        Mapping of column name and aggregate rule. Default is DEFAULT_AGGREGATE_MAPPING dict
    aggregate_default : str, optional
        Default aggregate rule. Default is 'first'
    verbose : bool, optional
        Show warnings. Default is False

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe

    Examples
    --------
    >>> df = pd.DataFrame({
    ... 'open': [7925.6, 7920.1, 7926.9, 7930.54873933, 7932.9, 7927.1, 7920.7, 7927.6, 7924.5, 7924.4],
    ... 'high': [7926.6, 7920.1, 7930.4, 7930.54873933, 7932.9, 7927.1, 7920.7, 7927.6, 7924.5, 7924.4],
    ... 'ticks_sell': [2, 0, 2, 0, 0, 0, 1, 0, 1, 1],
    ... 'f': [1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    ... 'custom': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    ... index=pd.DatetimeIndex(['2019-05-16 16:56:41+00:00', '2019-05-16 16:57:07+00:00',
    ...     '2019-05-16 16:57:26+00:00', '2019-05-16 16:57:37+00:00',
    ...     '2019-05-16 16:58:10+00:00', '2019-05-16 16:58:28+00:00',
    ...     '2019-05-16 16:59:11+00:00', '2019-05-16 16:59:19+00:00',
    ...     '2019-05-16 16:59:57+00:00', '2019-05-16 17:00:11+00:00'],
    ...     dtype='datetime64[ns, UTC]', freq=None))
    >>> consolidate(df, 'f')
                                      open    high  ticks_sell  f  custom
    2019-05-16 16:56:41+00:00  7925.600000  7926.6           2  0       1
    2019-05-16 16:57:26+00:00  7926.900000  7930.4           2  1       3
    2019-05-16 16:57:37+00:00  7930.548739  7932.9           0  1       4
    2019-05-16 16:59:11+00:00  7920.700000  7927.6           1  1       7
    2019-05-16 16:59:57+00:00  7924.500000  7924.5           2  1       9
    """
    if aggregate is None:
        aggregate = DEFAULT_AGGREGATE_MAPPING.copy()

    df = df.copy()

    if (df.isna().any(1) & ~df.isna().all(1)).any():
        df[column_name] = move_feature_by_nans(
            feature=df[column_name],
            series=df.open
        )

    if verbose:
        unexpected_columns = list(frozenset(df.columns) - frozenset(aggregate.keys()))
        logging.warning(
            f'WARNING: mapping rule has not been found for columns {unexpected_columns}. '
            f'Using the default rule: "{aggregate_default}".'
        )

    # save index
    index_tmp_name = str(uuid.uuid4())
    index_prev_name = df.index.name

    columns = list(df.columns) + [index_tmp_name]
    df[index_tmp_name] = df.index
    df.loc[df.open.isna(), index_tmp_name] = None
    df[index_tmp_name].fillna(method='bfill', inplace=True)

    # tmp column for aggregation
    agg_id_name = str(uuid.uuid4())
    df[column_name] = [0] + df[column_name].values.tolist()[:-1]
    df[agg_id_name] = df[column_name].cumsum()  # np.where(df[column_name] != df[column_name].shift(1), 1, 0).cumsum()
    tmp = df[agg_id_name].values
    tmp[0] = tmp[1]
    df[agg_id_name] = tmp

    # apply default aggregation
    cols_exists = []
    for col in df.columns:
        cols_exists.append(col)
        if col not in aggregate:
            aggregate[col] = aggregate_default

    for col in list(aggregate.keys()):
        if col not in cols_exists:
            del aggregate[col]

    # aggregate
    df_new = df.groupby([agg_id_name]).agg(aggregate)[columns]
    df.drop([agg_id_name, index_tmp_name], axis=1, inplace=True)

    # return original index
    df_new = df_new.set_index(index_tmp_name)
    df_new.index.rename(index_prev_name, inplace=True)

    # if drop_first_bar:
    #     df_new = df_new[1:]

    if pd.isnull(df_new.iloc[-1, :]).any():
        return df_new.iloc[:-1, :]
    return df_new


def move_feature_by_nans(feature: pd.Series, series: pd.Series) -> pd.Series:
    df = pd.DataFrame({'a': series, 'f': feature})
    df['_f'] = df.f
    df['x'] = df.f
    df.loc[df.a.isna(), 'x'] = None
    if np.isnan(df.x[0]):
        df.x[0] = 0

    df.x = df.x.fillna(method='ffill')
    df['r'] = (
        (df.x == 1) & (
            (((df.f == 1) | df.a.isna()) & ~(np.hstack((df.a.shift(-1).isna().values[:-1], [False]))))
        )
    ) * 1
    df.loc[df._f.isna(), 'r'] = None
    return df.r


@njit
def price_pct_threshold(open: np.ndarray, close: np.ndarray,
                        threshold_up: float, threshold_down: float = None,
                        n_bars: int = None) -> np.ndarray:
    """
    Fixed Range
    Price move (range) accumulation feature. Fixed % range.

    Parameters
    ----------
    open : np.ndarray
        Series of open prices
    close : np.ndarray
        Series of close prices
    threshold_up : float
        Event UP is generated after price moves by more percent than this threshold
    threshold_down : float
        (if None than equal to threshold_up) Event DOWN is generated after price
        moves by more percent than this threshold
    n_bars : int
        Return only first n bars, if parameter set. Default is None

    Returns
    -------
    np.ndarray
        Binary series. 1 signals firing of accumulation event.

    Examples
    --------
    >>> price_pct_threshold(np.array([1, 2, 3, 4, 5]), np.array([2, 1, 3, 4, 6]), 0.1)
    array([1, 1, 0, 1, 1])
    >>> price_pct_threshold(np.array([1, 2, 3, 4, 5]), np.array([2, 1, 3, 4, 6]), 0.1, 1)
    array([1, 0, 1, 0, 1])
    """

    if threshold_down is None:
        threshold_down = threshold_up

    bars = []
    bars_count = 0
    upper_limit, lower_limit = None, None
    for v_close, v_open in zip(close, open):
        if np.isnan(v_close) or np.isnan(v_open):
            bars.append(0)
            continue

        if upper_limit is None:
            upper_limit, lower_limit = (v_open * (1 + threshold_up), v_open * (1 - threshold_down))

        if v_close >= upper_limit or v_close <= lower_limit:
            upper_limit, lower_limit = None, None
            bars.append(1)
            if n_bars is not None:
                bars_count += 1
                if n_bars == bars_count:
                    return np.array(bars)
        else:
            bars.append(0)

    feature = np.array(bars)
    return feature


def fixed_threshold(series: np.ndarray, threshold: float) -> np.ndarray:
    """
    Fixed Threshold
    Fixed threshold accumulating feature.

    Parameters
    ----------
    series : np.ndarray
        Series of trading volume
    threshold : float
        Event is generated after cumulative volume reaches this threshold

    Returns
    -------
    np.ndarray
        Binary series. 1 signals firing of accumulation event.
    """

    bars = []
    agg_sum = 0
    for v in series:
        agg_sum += v
        if agg_sum >= threshold:
            bars.append(1)
            agg_sum = 0
        else:
            bars.append(0)

    feature = np.array(bars)
    assert feature.shape == series.shape
    return feature


def adaptive_threshold(series: np.ndarray, avg_per: int, window: int, n: int = None) -> np.ndarray:
    """
    Adaptive Threshold
    Adaptive accumulating feature. Create new bar when threshold reaches "weekly average for year".

    Parameters
    ----------
    series : np.ndarray
        Series of trading volume
    avg_per : int
        Get rolling avg_per count series avg
    window : int
        Series should aggregate window amount of averaged (by avg_per) series
    n : int, optional
        Calculate threshold every n bars instead of each bar

    Returns
    -------
    np.ndarray
        Binary series. 1 signals firing of accumulation event

    Examples
    --------
    >>> adaptive_threshold(np.array([100, 200, 300, 1000, 600, 200, 300, 1, 20]), 2, 5)
    array([nan, nan, nan, nan,  0.,  0.,  1.,  0.,  0.])
    """
    if n:
        series_threshold = calculate_adaptive_threshold_n_bars(series, avg_per, window, n)
    else:
        series_threshold = calculate_adaptive_threshold_rolling_window(series, avg_per, window)

    bars = []
    agg_sum = 0
    for [v_series, v_threshold] in np.column_stack([series, series_threshold]):
        if np.isnan(v_threshold):
            bars.append(v_threshold)
            continue

        agg_sum += v_series
        if agg_sum >= v_threshold:
            bars.append(1)
            agg_sum = 0
        else:
            bars.append(0)

    feature = np.array(bars)
    assert feature.shape == series.shape
    return feature


def calculate_adaptive_threshold_n_bars(series: np.ndarray, avg_per: int, window: int, n: int):
    size = len(series)
    end_range = np.arange(n, size, n)
    end_range = np.hstack((end_range, [size]))
    start_range = end_range - n

    range_filter = start_range >= window
    start_range = start_range[range_filter]
    end_range = end_range[range_filter]

    res = nans(size)

    for start, end in zip(start_range, end_range):
        s = series[start - window:start + 1]
        res[start:end] = np.sum(s) / (len(s) / avg_per)

    return res


def calculate_adaptive_threshold_rolling_window(series: np.ndarray, avg_per: int, window: int):
    return rolling_apply(
        (lambda s: np.sum(s) / (len(s) / avg_per)),
        window,
        series
    )


def price_pct__series_fixed(open: np.ndarray, close: np.ndarray, price_threshold: float,
                            series: np.ndarray, series_threshold: float) -> np.ndarray:
    """
    Percent price threshold combined with any fixed threshold series feature
    Price move (range) and ticks accumulation feature. Fixed % range, fixed n ticks.

    Parameters
    ----------
    open : np.ndarray
        Series of open prices
    close : np.ndarray
        Series of close prices
    price_threshold : float
        Range condition satisfied is  after price moves by more percent than this threshold
    series : np.ndarray
        Series of ticks
    series_threshold : float
        Ticks condition is satisfied after cumulative number of ticks reaches this threshold

    Returns
    -------
    np.array
        Binary series. 1 signals firing of accumulation event when both conditions are satisfied.

    Examples
    --------
    >>> open, close, series = np.array([1, 2, 3, 4, 5]), np.array([2, 1, 3, 4, 6]), np.array([1, 2, 3, 4, 2])
    >>> price_pct__series_fixed(open, close, 0.5, series, 0.9)
    array([1, 1, 0, 0, 1])
    """

    bars = []
    upper_limit, lower_limit = None, None
    series_sum = 0
    for [v_open, v_close, v_series] in np.column_stack([open, close, series]):
        if np.isnan([v_open, v_close, v_series]).any():
            bars.append(0)
            continue
        series_sum += v_series

        if upper_limit is None:
            upper_limit, lower_limit = (v_open * (1 + price_threshold), v_open * (1 - price_threshold))

        is_price_pct = v_close >= upper_limit or v_close <= lower_limit
        is_fixed = series_sum > series_threshold

        if is_price_pct and is_fixed:
            upper_limit, lower_limit = None, None
            series_sum = 0
            bars.append(1)
        else:
            bars.append(0)

    feature = np.array(bars)
    assert feature.shape == close.shape
    return feature


def time_fixed(
    df: pd.DataFrame,
    period: Union[str, pd.DateOffset, pd.Timedelta, int],
    aggregate: dict = None,
    aggregate_default: str = 'first',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Time Fixed Consolidator

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with ohlcv data
    period : Union[str, pd.DateOffset, pd.Timedelta, int]
        if type is int then used as seconds
        resample frequency https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    aggregate : dict, optional
        Mapping of column name and aggregate rule. Default is DEFAULT_AGGREGATE_MAPPING dict
    aggregate_default : str, optional
        Default aggregate rule. Default is 'first'
    verbose : bool, optional
        Show warnings. Default is False

    Returns
    -------
    pd.DataFrame
        Prepared bars

    Examples
    --------
    >>> df = pd.DataFrame({
    ... 'open': [7925.6, 7920.1, 7926.9, 7930.54873933, 7932.9, 7927.1, 7920.7, 7927.6, 7924.5, 7924.4],
    ... 'high': [7926.6, 7920.1, 7930.4, 7930.54873933, 7932.9, 7927.1, 7920.7, 7927.6, 7924.5, 7924.4],
    ... 'ticks_sell': [2, 0, 2, 0, 0, 0, 1, 0, 1, 1],
    ... 'f': [1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    ... 'custom': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    ... index=pd.DatetimeIndex(['2019-05-16 16:56:41+00:00', '2019-05-16 16:57:07+00:00',
    ...     '2019-05-16 16:57:26+00:00', '2019-05-16 16:57:37+00:00',
    ...     '2019-05-16 16:58:10+00:00', '2019-05-16 16:58:28+00:00',
    ...     '2019-05-16 16:59:11+00:00', '2019-05-16 16:59:19+00:00',
    ...     '2019-05-16 16:59:57+00:00', '2019-05-16 17:00:11+00:00'],
    ...     dtype='datetime64[ns, UTC]', freq=None))
    >>> time_fixed(df, '1min')
                                 open         high  ticks_sell  f  custom
    2019-05-16 16:56:00+00:00  7925.6  7926.600000           2  1       1
    2019-05-16 16:57:00+00:00  7920.1  7930.548739           2  1       2
    2019-05-16 16:58:00+00:00  7932.9  7932.900000           0  0       5
    2019-05-16 16:59:00+00:00  7920.7  7927.600000           2  0       7
    2019-05-16 17:00:00+00:00  7924.4  7924.400000           1  0      10
    >>> time_fixed(df, 60)
                                 open         high  ticks_sell  f  custom
    2019-05-16 16:56:00+00:00  7925.6  7926.600000           2  1       1
    2019-05-16 16:57:00+00:00  7920.1  7930.548739           2  1       2
    2019-05-16 16:58:00+00:00  7932.9  7932.900000           0  0       5
    2019-05-16 16:59:00+00:00  7920.7  7927.600000           2  0       7
    2019-05-16 17:00:00+00:00  7924.4  7924.400000           1  0      10
    """

    if isinstance(period, int):
        period = f'{period}S'

    if aggregate is None:
        aggregate = DEFAULT_AGGREGATE_MAPPING.copy()

    columns_set = frozenset(df.columns)
    aggregate_set = frozenset(aggregate)
    unexpected_columns = columns_set - aggregate_set

    for col in (aggregate_set - columns_set):
        del aggregate[col]

    if verbose:
        logging.warning(
            f'WARNING: mapping rule has not been found for columns {unexpected_columns}. '
            f'Using the default rule: "{aggregate_default}".'
        )

    columns = df.columns.to_list()
    aggregate.update({c: aggregate_default for c in sorted(unexpected_columns, key=lambda x: columns.index(x))})
    return df.resample(period).agg(aggregate).dropna()


@njit
def fixed_percent_fixed_time_feature(
    indexes: np.ndarray,
    values: np.ndarray,
    period: np.timedelta64,
    threshold: float
) -> np.ndarray:
    bars = np.zeros(values.shape[0], dtype=np.int8)

    # find first not nan
    for i in range(len(values)):
        start = i
        if not np.isnan(values[i]):
            break
    else:
        return bars

    last_index = indexes[start]
    last_value = values[start]
    for i in range(start, values.shape[0]):
        if not np.isnan(values[i]) \
           and (indexes[i] - last_index) >= period \
           and (np.abs(last_value - values[i]) / last_value) >= threshold:

            last_index = indexes[i]
            last_value = values[i]
            bars[i] = 1
        else:
            bars[i] = 0
    return bars


def adaptive_percent_dynamic_threshold(open: pd.Series, close: pd.Series, avg_per: int, window: int) -> np.ndarray:
    change_per_period = close.pct_change().rolling(avg_per).sum().abs()
    dynamic_threshold = change_per_period.rolling(window).mean()
    bars = [
        v_close >= (v_open * (1 + threshold)) or v_close <= (v_open * (1 - threshold))
        for v_close, v_open, threshold in np.c_[close, open, dynamic_threshold]
    ]

    feature = np.array(bars) * 1
    assert feature.shape == close.shape
    return feature


def calculate_expected_imbalance(imbalances, window):
    if len(imbalances) < window:
        return np.nan

    expected_imbalance = ewma(imbalances[-window:], window=window)[-1]
    return expected_imbalance


def bvc_imbalance(volume: np.ndarray, close: np.ndarray):
    bvc_buy_volume = get_bvc_buy_volume(pd.Series(close), pd.Series(volume)).to_numpy()
    bvc_sell_volume = volume - bvc_buy_volume
    imbalance = bvc_buy_volume - bvc_sell_volume
    return imbalance


def get_nan_offset(series: np.nan) -> int:
    nan_offset = 0
    for elm in series:
        if np.isnan(elm):
            nan_offset += 1
        else:
            break
    return nan_offset


def imbalance_feature(volume: np.ndarray, close: np.ndarray, ema_window: int):
    imbalance = bvc_imbalance(volume, close)
    theta = 0.
    expected_imbalance = np.nan
    bars = np.zeros(len(close))

    nan_offset = get_nan_offset(imbalance)
    bars[:nan_offset] = np.nan

    for i in range(nan_offset, len(bars)):
        if np.isnan(imbalance[i]):
            continue

        theta += imbalance[i]

        if np.isnan(expected_imbalance):
            expected_imbalance = calculate_expected_imbalance(imbalance[:i + 1], window=ema_window)

        if np.abs(theta) >= np.abs(expected_imbalance):
            bars[i] = 1
            theta = 0
            expected_imbalance = calculate_expected_imbalance(imbalance[:i + 1], window=ema_window)

    return bars


def imbalance_feature_const(volume: np.ndarray, close: np.ndarray, threshold: float):
    imbalance = bvc_imbalance(volume, close)
    bars = np.zeros(len(close))
    theta = 0.

    nan_offset = get_nan_offset(imbalance)
    bars[:nan_offset] = np.nan

    for i in range(nan_offset, len(bars)):
        theta += imbalance[i]

        if np.abs(theta) >= threshold:
            bars[i] = 1
            theta = 0.

    return bars
