import os
import pytest
import pandas as pd

from commons import bars

DATASET = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../datasets/bitfinex__BTC_USD.hdf"))


@pytest.fixture
def test_dataset():
    yield pd.read_hdf(DATASET, key='table')[bars.COLUMNS[1:]]


@pytest.fixture
def synthetic_ohlc():
    yield pd.DataFrame({
        'open': [100, 150, 200],
        'high': [200, 300, 250],
        'low': [100, 20, 3],
        'close': [150, 200, 250],
        'volume_buy': [1000, 2000, 1500],
        'volume_sell': [500, 200, 300],
        'volume_quote_buy': [50000, 20000, 15000],
        'volume_quote_sell': [50000, 20000, 15000],
        'ticks_buy': [3000, 5000, 5000],
        'ticks_sell': [1000, 3000, 2000]
    })


@pytest.mark.skip("failed")
@pytest.mark.parametrize(
    "args", [
        (bars.range.fixed, 1.5),
        (bars.range.fixed, 150, True),
        (bars.volume.fixed, 5500, False),
        (bars.ticks.fixed, 19000)]
)
def test_bar_calculation(args, synthetic_ohlc):
    func, *params = args
    df = func(synthetic_ohlc, *params)
    bar = df.iloc[0]
    assert bar['open'] == df.iloc[0]['open']
    assert bar['high'] == 300
    assert bar['low'] == 3
    assert bar['close'] == df.iloc[-1]['close']
    assert bar['volume_buy'] == df['volume_buy'].sum()
    assert bar['volume_sell'] == df['volume_sell'].sum()
    assert bar['volume_quote_buy'] == df['volume_quote_buy'].sum()
    assert bar['volume_quote_sell'] == df['volume_quote_sell'].sum()
    assert bar['ticks_buy'] == df['ticks_buy'].sum()
    assert bar['ticks_sell'] == df['ticks_sell'].sum()


@pytest.mark.skip("failed")
class TestRangeFixedBars:
    def test_missing_lesser_pct_threshold(self, test_dataset):
        df = bars.range.fixed(test_dataset, 0.05)
        df["price_change"] = df.close.pct_change()
        lesser_pct = df[abs(df.price_change) < 0.05]
        assert not df.empty
        assert lesser_pct.empty

    def test_missing_lesser_abs_threshold(self, test_dataset):
        df = bars.range.fixed(test_dataset, 100, absolute=True)
        df["price_diff"] = df.close.diff()
        lesser_abs = df[abs(df.price_diff) < 100]
        assert not df.empty
        assert lesser_abs.empty


class TestVolumeFixedBars:
    def test_missing_lesser_threshold(self, test_dataset):
        df = bars.volume.fixed(test_dataset, 500, by_quote=False)
        assert not df.empty
        assert df[df.volume_buy + df.volume_sell < 500].empty

    def test_missing_lesser_quote_threshold(self, test_dataset):
        df = bars.volume.fixed(test_dataset, 10**7)
        assert not df.empty
        assert df[df.volume_quote_buy + df.volume_quote_sell < 10**7].empty


class TestTicksFixedBars:
    def test_missing_lesser_threshold(self, test_dataset):
        df = bars.ticks.fixed(test_dataset, 5000)
        assert not df.empty
        assert df[df.ticks_buy + df.ticks_sell < 5000].empty


@pytest.mark.skip("failed")
class TestHybridFixedRangeAdaptiveVolumeBars:
    def test_missing_lesser_pct_threshold_and_avg(self, test_dataset):
        df = bars.hybrid.range_fixed_volume_adaptive(
            ohlc=test_dataset,
            pct_threshold=0.05,
            avg_per=2,
            window=30
        )
        df['price_change'] = df.close.pct_change()
        assert not df.empty
        assert df[abs(df.price_change) < 0.05].empty


@pytest.mark.skip("failed")
class TestHybridFixedRangeFixedTicks:
    def test_missing_lesser_thresholds(self, test_dataset):
        df = bars.hybrid.range_fixed_ticks_fixed(
            ohlc=test_dataset,
            pct_threshold=0.05,
            ticks_threshold=5000
        )
        df['price_change'] = df.close.pct_change()
        assert not df.empty
        assert df[abs(df.price_change) < 0.01].empty
        assert df[df.ticks_buy + df.ticks_sell < 50].empty
