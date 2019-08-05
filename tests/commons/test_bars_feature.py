import os
import pytest
import pandas as pd
import numpy as np

from commons.bars import feature

ERROR = 1e-9
DATASET = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../datasets/bitfinex__BTC_USD.hdf"))


def _volume_sum(df):
    return df.volume_buy.sum() + df.volume_sell.sum()


def _ticks_sum(df):
    return df.ticks_buy.sum() + df.ticks_sell.sum()


@pytest.fixture
def test_dataset():
    yield pd.read_hdf(DATASET, key='table')


class TestSmaCrossConsolidate:
    @staticmethod
    def calc_sma_cross(df, fast, slow):
        sma_fast = df.close.rolling(fast).sum() / fast
        sma_slow = df.close.rolling(slow).sum() / slow
        df['sma_cross'] = np.where(sma_fast > sma_slow, 1, -1)
        print(df['sma_cross'])
        df['sma_cross'] = np.where(df['sma_cross'] != df['sma_cross'].shift(1), 1, 0)
        print(df['sma_cross'])

    def test_all_groups_is_consolidated(self, test_dataset):
        df = test_dataset
        self.calc_sma_cross(df, 10, 20)
        cross_count = df['sma_cross'].sum()
        consolidated = feature(test_dataset, column_name="sma_cross")
        assert not consolidated.empty
        assert consolidated.shape[0] == cross_count
