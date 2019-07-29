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
    yield pd.read_hdf(DATASET,
                      key='table')[bars.base.BaseConsolidator.COLUMNS[2:]]


def test_fr(df):
    threshold = 0.01
    df['f5'] = features.cumulative.bars.fr(df.open.values, df.close.values,
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


def test_ft(df):
    threshold = 100
    df['f2'] = features.cumulative.bars.ft(df.ticks_buy.values + df.ticks_sell.values, threshold)
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


def test_fv(df):
    threshold = 500
    df['f3'] = features.cumulative.bars.fv(
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


def test_frft(df):
    pct_threshold = 0.01
    ticks_threshold = 5000
    df['f8'] = features.cumulative.bars.frft(
        df.open.values,
        df.close.values,
        df.ticks_buy.values + df.ticks_sell.values,
        pct_threshold,
        ticks_threshold
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
