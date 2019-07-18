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


@pytest.fixture
def df():
    yield pd.read_hdf(DATASET,
                      key='table')[bars.base.BaseConsolidator.COLUMNS[2:]]


def test_f1(df):
    threshold = 0.001
    df['f1'] = features.cumulative.bars.f1(df.close.values, threshold)
    assert not df[df['f1'] != 0].empty

    # df.loc[:, 'bar_no'] = df['f1'].cumsum().shift().fillna(0)
    # agg_df = df.groupby(['bar_no']).agg(AGGREGATE).iloc[:-1]
    # agg_df.loc[:, 'log_close_diff'] = np.log(agg_df.close.diff().fillna(0))
    # agg_df.loc[:, 's_pos'] = agg_df.log_close_diff.apply(lambda x: max(0.0, x))
    # agg_df.loc[:, 's_neg'] = agg_df.log_close_diff.apply(lambda x: min(0.0, x))
    # n = agg_df[(agg_df.s_pos < threshold) & (agg_df.s_neg > -threshold)]
    # assert n.empty

    # Compares with commons.bars
    feature_index = df[df['f1'] == 1].index.values
    consolidated_bars = bars.cusum.fixed(df, threshold, timestamp_close=True)
    consolidated_timestamp_close = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in consolidated_timestamp_close - feature_index
    ]
    # The timedelta always equals one, because timestamp_close ==
    # timestamp of row that satisfied the condition + 1s.
    assert all(timedeltas)


def test_f2(df):
    threshold = 100
    df['f2'] = features.cumulative.bars.f2(df.ticks_buy.values,
                                           df.ticks_sell.values, threshold)
    assert not df[df['f2'] != 0].empty

    df.loc[:, 'bar_no'] = df['f2'].cumsum().shift().fillna(0)
    agg_df = df.groupby(['bar_no']).agg(AGGREGATE).iloc[:-1]
    assert agg_df[agg_df.ticks_sell + agg_df.ticks_buy < threshold].empty

    # Compares with commons.bars
    feature_index = df[df['f2'] == 1].index.values
    consolidated_bars = bars.tick.fixed(df, threshold, timestamp_close=True)
    consolidated_timestamp_close = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in consolidated_timestamp_close - feature_index
    ]
    # The timedelta always equals one, because timestamp_close ==
    # timestamp of row that satisfied the condition + 1s.
    assert all(timedeltas)


def test_f3(df):
    threshold = 500
    df['f3'] = features.cumulative.bars.f3(
        df.volume_sell.values + df.volume_buy.values, threshold)
    assert not df[df['f3'] != 0].empty

    df.loc[:, 'bar_no'] = df['f3'].cumsum().shift().fillna(0)
    agg_df = df.groupby(['bar_no']).agg(AGGREGATE).iloc[:-1]
    assert agg_df[agg_df.volume_sell + agg_df.volume_buy < threshold].empty

    # Compares with commons.bars
    feature_index = df[df['f3'] == 1].index.values
    consolidated_bars = bars.volume.fixed(df,
                                          threshold,
                                          by_quote=False,
                                          timestamp_close=True)
    consolidated_timestamp_close = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in consolidated_timestamp_close - feature_index
    ]
    # The timedelta always equals one, because timestamp_close ==
    # timestamp of row that satisfied the condition + 1s.
    assert all(timedeltas)


def test_f4(df):
    threshold = 10**6
    df['f4'] = features.cumulative.bars.f4(
        df.volume_quote_buy.values + df.volume_quote_sell.values, threshold)
    assert not df[df['f4'] != 0].empty

    df.loc[:, 'bar_no'] = df['f4'].cumsum().shift().fillna(0)
    agg_df = df.groupby(['bar_no']).agg(AGGREGATE).iloc[:-1]
    assert agg_df[agg_df.volume_quote_buy +  # noqa
                  agg_df.volume_quote_sell < threshold].empty

    # Compares with commons.bars
    feature_index = df[df['f4'] == 1].index.values
    consolidated_bars = bars.volume.fixed(df, threshold, timestamp_close=True)
    consolidated_timestamp_close = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in consolidated_timestamp_close - feature_index
    ]
    # The timedelta always equals one, because timestamp_close ==
    # timestamp of row that satisfied the condition + 1s.
    assert all(timedeltas)


def test_f5(df):
    threshold = 0.01
    df['f5'] = features.cumulative.bars.f5(df.open.values, df.close.values,
                                           threshold)
    assert not df[df['f5'] != 0].empty

    df.loc[:, 'bar_no'] = df['f5'].cumsum().shift().fillna(0)
    agg_df = df.groupby(['bar_no']).agg(AGGREGATE).iloc[:-1]
    assert agg_df[abs(agg_df.close.pct_change()) < threshold].empty

    # Compares with commons.bars
    feature_index = df[df['f5'] == 1].index.values
    consolidated_bars = bars.range.fixed(df, threshold, timestamp_close=True)
    consolidated_timestamp_close = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in consolidated_timestamp_close - feature_index
    ]
    # The timedelta always equals one, because timestamp_close ==
    # timestamp of row that satisfied the condition + 1s.
    assert all(timedeltas)


def test_f6(df):
    threshold = 100
    df['f6'] = features.cumulative.bars.f6(df.open.values, df.close.values,
                                           threshold)
    assert not df[df['f6'] != 0].empty

    df.loc[:, 'bar_no'] = df['f6'].cumsum().shift().fillna(0)
    new_df = df.groupby(['bar_no']).agg(AGGREGATE).iloc[:-1]
    assert new_df[abs(new_df.close.diff()) < threshold].iloc[:-1].empty

    # Compares with commons.bars
    feature_index = df[df['f6'] == 1].index.values
    consolidated_bars = bars.range.fixed(df,
                                         threshold,
                                         threshold_is_absolute=True,
                                         timestamp_close=True)
    consolidated_timestamp_close = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in consolidated_timestamp_close - feature_index
    ]
    # The timedelta always equals one, because timestamp_close ==
    # timestamp of row that satisfied the condition + 1s.
    assert all(timedeltas)


def test_f8(df):
    pct_threshold = 0.01
    ticks_threshold = 5000
    df['f8'] = features.cumulative.bars.f8(
        df.open.values,
        df.close.values,
        df.ticks_buy.values,
        df.ticks_sell.values,
        pct_threshold,
        ticks_threshold,
    )
    assert not df[df['f8'] != 0].empty

    df.loc[:, 'bar_no'] = df['f8'].cumsum().shift().fillna(0)
    agg_df = df.groupby(['bar_no']).agg(AGGREGATE).iloc[:-1]
    assert agg_df[agg_df.ticks_sell + agg_df.ticks_buy < ticks_threshold].empty
    assert agg_df[abs(agg_df.close.pct_change()) < pct_threshold].empty

    # Compares with commons.bars
    feature_index = df[df['f8'] == 1].index.values
    consolidated_bars = bars.hybrid.range_fixed_ticks_fixed(
        df, pct_threshold, ticks_threshold, timestamp_close=True)
    consolidated_timestamp_close = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in consolidated_timestamp_close - feature_index
    ]
    # The timedelta always equals one, because timestamp_close ==
    # timestamp of row that satisfied the condition + 1s.
    assert all(timedeltas)
