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
    period = 5
    feature_bars = features.cumulative.bars.f1(df.index.values, period)
    feature_index = feature_bars[feature_bars == 1].index.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 5 for td in np.diff(feature_index)
    ]
    assert all(timedeltas)


def test_f2(df):
    threshold = 100
    df['f2'] = features.cumulative.bars.f2(df.ticks_sell + df.ticks_buy,
                                           threshold)
    assert not df[df.f2 != 0].empty

    new_df = df.groupby(
        ['f2']).agg(AGGREGATE).loc[:, ['ticks_buy', 'ticks_sell']]  # yapf: disable
    assert new_df[new_df.ticks_sell + new_df.ticks_buy < threshold].empty

    # Compares with commons.bars
    feature_index = df[df.f2 == 1].index.values
    consolidated_bars = bars.tick.fixed(df, threshold, timestamp_close=True)
    consolidated_timestamp_close = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in (consolidated_timestamp_close - feature_index)
    ]
    # The timedelta always equals one, because timestamp_close ==
    # timestamp of row that satisfied the condition + 1s.
    assert all(timedeltas)


def test_f3(df):
    threshold = 500
    df['f3'] = features.cumulative.bars.f3(df.volume_sell + df.volume_buy,
                                           threshold)
    assert not df[df.f3 != 0].empty

    new_df = df.groupby(
        ['f3']).agg(AGGREGATE).loc[:, ['volume_buy', 'volume_sell']]  # yapf: disable
    assert new_df[new_df.volume_sell +  # noqa
                  new_df.volume_buy < threshold].iloc[:-1].empty

    # Compares with commons.bars
    feature_index = df[df.f3 == 1].index.values
    consolidated_bars = bars.volume.fixed(df,
                                          threshold,
                                          by_quote=False,
                                          timestamp_close=True)
    consolidated_timestamp_close = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in (consolidated_timestamp_close - feature_index)
    ]
    # The timedelta always equals one, because timestamp_close ==
    # timestamp of row that satisfied the condition + 1s.
    assert all(timedeltas)


def test_f4(df):
    threshold = 10**6
    df['f4'] = features.cumulative.bars.f4(
        df.volume_quote_buy + df.volume_quote_sell, threshold)
    assert not df[df.f4 != 0].empty

    new_df = df.groupby(
        ['f4']).agg(AGGREGATE).loc[:, ['volume_quote_buy', 'volume_quote_sell']]  # yapf: disable
    assert new_df[new_df.volume_quote_buy +  # noqa
                  new_df.volume_quote_sell < threshold].iloc[:-1].empty

    # Compares with commons.bars
    feature_index = df[df.f4 == 1].index.values
    consolidated_bars = bars.volume.fixed(df, threshold, timestamp_close=True)
    consolidated_timestamp_close = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in (consolidated_timestamp_close - feature_index)
    ]
    # The timedelta always equals one, because timestamp_close ==
    # timestamp of row that satisfied the condition + 1s.
    assert all(timedeltas)
