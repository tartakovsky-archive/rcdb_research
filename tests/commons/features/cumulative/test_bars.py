import os
import pytest
import pandas as pd
import numpy as np

from commons import bars, features

ERROR = 1e-9
DATASET = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../../../datasets/bitfinex__BTC_USD.hdf"))


def _volume_sum(df):
    return df.volume_buy.sum() + df.volume_sell.sum()


def _ticks_sum(df):
    return df.ticks_buy.sum() + df.ticks_sell.sum()


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
