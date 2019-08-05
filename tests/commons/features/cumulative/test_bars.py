import os
import pytest
import pandas as pd
import numpy as np
from commons import bars, features

ERROR = 1e-9
DATASET = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../../../datasets/bitfinex__BTC_USD.hdf"))

AGGREGATE = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume_buy': 'sum',
    'volume_sell': 'sum',
    'volume_quote_sell': 'sum',
    'volume_quote_buy': 'sum',
    'ticks_sell': 'sum',
    'ticks_buy': 'sum',
}

DATA_MAPPING = dict(
    series='close',
    open='open',
    high='high',
    low='low',
    close='close',
    volume='volume',
)


@pytest.fixture
def df():
    yield pd.read_hdf(DATASET, key='table')[bars.base.BaseConsolidator.COLUMNS[2:]]


def open_close_diff(open, close):
    return np.where(open < close, close / open - 1, open / close - 1)


def test_fr(df):
    threshold = 0.01
    df['f'] = features.cumulative.bars.price_pct_threshold(df.open.values, df.close.values, threshold)

    # we need at least two unique elements
    assert np.unique(df['f'].values).size > 1

    agg_df = bars.feature(df, 'f')
    # drop last column, often not fully aggregated
    agg_df = agg_df[:-1]

    # check percent change between open/close (don't use build in `pct_change()` for that case)
    agg_df['diff'] = open_close_diff(agg_df.open, agg_df.close)

    # there shouldn't be any bars less then threshold
    assert agg_df[np.abs(agg_df['diff']) < threshold].empty


def test_fr_asymmetric(df):
    threshold_up = 0.02
    threshold_down = 0.01
    df['f'] = features.cumulative.bars.price_pct_threshold(df.open.values, df.close.values,
                                                           threshold_up, threshold_down)

    # we need at least two unique elements
    assert np.unique(df['f'].values).size > 1

    agg_df = bars.feature(df, 'f')
    # drop last column, often not fully aggregated
    agg_df = agg_df[:-1]

    # check percent change between open/close (don't use build in `pct_change()` for that case)
    agg_df['diff'] = open_close_diff(agg_df.open, agg_df.close)

    # there shouldn't be any bars less then threshold
    assert agg_df[np.where(agg_df.open > agg_df.close,
                           abs(agg_df['diff']) < threshold_down,
                           abs(agg_df['diff']) < threshold_up,
                           )].empty


def test_adaptive(df):
    avg_per, window = 10, 100

    # df = pd.DataFrame(dict(
    #     volume=np.array([i for i in range(0, 100)])
    # ))
    # vals = df.volume.values # (df.volume_quote_buy + df.volume_quote_sell).values
    vals = (df.volume_quote_buy + df.volume_quote_sell).values

    df['f'] = features.cumulative.bars.adaptive_threshold(vals, avg_per, window)

    # we need at least two unique elements
    assert np.unique(df['f'].values).size > 1

    agg_df = bars.feature(df, 'f')

    # drop last column, often not fully aggregated
    agg_df = agg_df[:-1]

    # Compares with commons.bars
    consolidated_bars = bars.volume.adaptive(df, avg_per, window)

    agg_df['volume'] = agg_df.volume_quote_buy + agg_df.volume_quote_sell
    consolidated_bars['volume'] = consolidated_bars.volume_quote_buy + consolidated_bars.volume_quote_sell


def test_ft(df):
    threshold = 500
    df['f'] = features.cumulative.bars.fixed_threshold(df.ticks_buy.values + df.ticks_sell.values, threshold)
    assert np.unique(df['f'].values).size > 1

    agg_df = bars.feature(df, 'f')
    agg_df = agg_df[:-1]
    assert agg_df[agg_df.ticks_sell + agg_df.ticks_buy < threshold].empty

    agg_df.drop(['f'], axis='columns', inplace=True)

    # Compares with commons.bars
    consolidated_bars = bars.tick.fixed(df, threshold, timestamp_close=True)
    consolidated_bars.drop(['timestamp_close'], axis='columns', inplace=True)

    # print(consolidated_bars.columns)
    assert agg_df.equals(consolidated_bars)


def test_fv(df):
    threshold = 500000

    df['f'] = features.cumulative.bars.fixed_threshold(df.volume_quote_sell.values + df.volume_quote_buy.values,
                                                       threshold)
    assert np.unique(df['f'].values).size > 1

    agg_df = bars.feature(df, 'f')
    agg_df = agg_df[:-1]
    assert agg_df[agg_df.volume_quote_sell + agg_df.volume_quote_buy < threshold].empty

    agg_df.drop(['f'], axis='columns', inplace=True)

    # Compares with commons.bars
    consolidated_bars = bars.volume.fixed(df, threshold, timestamp_close=True)
    consolidated_bars.drop(['timestamp_close'], axis='columns', inplace=True)

    agg_df['volume'] = agg_df.volume_quote_sell + agg_df.volume_quote_buy
    consolidated_bars['volume'] = consolidated_bars.volume_quote_sell + consolidated_bars.volume_quote_buy

    agg_df.reset_index(inplace=True)
    consolidated_bars.reset_index(inplace=True)

    assert agg_df.equals(consolidated_bars)


def test_frft(df):
    pct_threshold = 0.01
    ticks_threshold = 1000

    df['f'] = features.cumulative.bars.price_pct__series_fixed(
        df.open.values,
        df.close.values,
        pct_threshold,
        df.ticks_buy.values + df.ticks_sell.values,
        ticks_threshold
    )

    assert np.unique(df['f'].values).size > 1

    agg_df = bars.feature(df, 'f')
    agg_df = agg_df[:-1]
    agg_df['ticks'] = agg_df.ticks_sell + agg_df.ticks_buy
    agg_df['diff'] = open_close_diff(agg_df.open, agg_df.close)

    print(agg_df[['diff', 'ticks']])

    assert agg_df[agg_df.ticks_sell + agg_df.ticks_buy < ticks_threshold].empty
    assert agg_df[agg_df['diff'] < pct_threshold].empty

# def test__f_combine(df):
#     pct_threshold = 0.01
#     ticks_threshold = 500
#
#     df['f1'] = features.cumulative.bars.price_pct__series_fixed(
#         df.open.values,
#         df.close.values,
#         pct_threshold,
#         df.ticks_buy.values + df.ticks_sell.values,
#         ticks_threshold
#     )
#
#     f_range = features.cumulative.bars.price_pct_threshold(df.open, df.close, pct_threshold)
#     f_ticks = features.cumulative.bars.fixed_threshold(df.ticks_buy.values + df.ticks_sell.values, ticks_threshold)
#
#     df['f2'] = f_ticks # features.cumulative.bars.f_combine([f_range], [f_ticks])
#
#     agg_df = bars.feature(df, 'f2')
#     agg_df = agg_df[:-1]
#
#     agg_df['ticks'] = agg_df.ticks_sell + agg_df.ticks_buy
#     agg_df['diff'] = open_close_diff(agg_df.open, agg_df.close)
#     print(agg_df[['ticks', 'diff']])
#     assert False
#
#     assert (df['f1'] == df['f2']).all()
