import os

import pytest
import pandas as pd

from commons.consolidators import min_pct_bars

DATASET = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../datasets/data_notebook_test.hdf"))


@pytest.fixture
def test_dataset():
    dataset = pd.read_hdf(DATASET, key='table')
    yield dataset[['open', 'high', 'low', 'close', 'volume',
                   'volume_sell', 'ticks', 'ticks_sell']]

@pytest.fixture
def synthetic_ohlc():
    yield pd.DataFrame({
        'open': [100, 150, 200],
        'high': [200, 300, 250],
        'low': [100, 20, 3],
        'close': [150, 200, 250],
        'volume': [1000, 2123, 12343],
        'volume_sell': [465, 1323, 5234],
        'ticks': [1566, 6546, 3434],
        'ticks_sell': [234, 2345, 2355]
    })


class TestMinPctBars:
    def test_missing_lesser_pct(self, test_dataset):
        df = min_pct_bars(test_dataset, 0.1)
        df["price_change"] = df.close.pct_change()
        lesser_pct = df[abs(df.price_change) < 0.1]
        assert lesser_pct.empty

    def test_bar_calculation(self, synthetic_ohlc):
        df = min_pct_bars(synthetic_ohlc, 1.5)
        bar = df.iloc[0]
        assert bar['open'] == df.iloc[0]['open']
        assert bar['high'] == 300
        assert bar['low'] == 3
        assert bar['close'] == df.iloc[-1]['close']
        assert bar['volume'] == df['volume'].sum()
        assert bar['ticks'] == df['ticks'].sum()
