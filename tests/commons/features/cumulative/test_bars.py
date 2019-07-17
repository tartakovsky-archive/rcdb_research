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
def test_dataset():
    yield pd.read_hdf(DATASET,
                      key='table')[bars.base.BaseConsolidator.COLUMNS[2:]]


def test_f1(test_dataset):
    period = 5
    feature_bars = features.cumulative.bars.f1(test_dataset.index.values,
                                               period)
    feature_index = feature_bars[feature_bars == 1].index.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 5 for td in np.diff(feature_index)
    ]
    assert all(timedeltas)


def test_f2(test_dataset):
    threshold = 100
    df = test_dataset
    df['f2'] = features.cumulative.bars.f2(df.ticks_sell + df.ticks_buy,
                                           threshold)
    assert not df[df.f2 != 0].empty

    new_df = df.groupby(
        ['f2']).agg(AGGREGATE).loc[:, ['ticks_buy', 'ticks_sell']]  # yapf: disable
    new_df.ticks = new_df.ticks_sell + new_df.ticks_buy
    assert new_df[new_df.ticks < threshold].empty

    # Compares with commons.bars
    feature_index = df[df.f2 == 1].index.values
    consolidated_bars = bars.tick.fixed(df, threshold, timestamp_close=True)
    consolidated_index = consolidated_bars.timestamp_close.values
    timedeltas = [
        td / np.timedelta64(1, 's') == 1
        for td in (consolidated_index - feature_index)
    ]
    # Because timestamp_close == timestamp of row that satisfied condition + 1s
    # the timedelta always equals one.
    assert all(timedeltas)
