import os
import pytest
import pandas as pd
import numpy as np

from rcdb_research import bars

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
    yield pd.read_hdf(
        DATASET, key='table')[bars.base.BaseConsolidator.COLUMNS[2:]]


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
    }, index=[pd.Timestamp(100 + i, unit='s') for i in range(3)])


@pytest.mark.parametrize(
    "args", [
        (bars.volume.fixed, 5500, False, True),
        (bars.time.fixed, 4, True),
        (bars.tick.fixed, 19000, True),
        (bars.range.fixed, 1.5, False, True),
        (bars.range.fixed, 150, True, True),
    ]
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
    if func == bars.time.fixed:
        assert bar['timestamp_close'] == pd.Timestamp(104, unit='s')
    else:
        assert bar['timestamp_close'] == pd.Timestamp(103, unit='s')


@pytest.mark.parametrize(
    "args", [
        (bars.volume.fixed, 100, False, True),
        (bars.volume.adaptive, 2, 30, True),
        (bars.time.fixed, 3, True),
        (bars.tick.fixed, 1000, True),
        (bars.range.fixed, 0.01, False, True),
        (bars.range.fixed, 50, True, True),
        (bars.cusum.fixed, 0.001, True),
    ]
)
def test_missing_leaks(args, test_dataset):
    source = test_dataset
    func, *params = args
    df = func(source, *params)
    assert not df.empty
    first_open = df.index[0]
    last_close = df.iloc[-1].timestamp_close
    reached_source = source.loc[first_open: last_close]
    assert abs(_volume_sum(df) - _volume_sum(reached_source)) < ERROR
    assert _ticks_sum(df) == _ticks_sum(reached_source)


def test_source_data_does_not_mutate(test_dataset):
    source = test_dataset
    copy = source.copy()
    bars.volume.fixed(source, 5500, False)
    assert source.equals(copy)


class TestRangeFixedBars:
    def test_missing_lesser_pct_threshold(self, test_dataset):
        df = bars.range.fixed(test_dataset, 0.01)
        df["price_change"] = df.close.pct_change()
        lesser_pct = df[abs(df.price_change) < 0.01]
        assert not df.empty
        assert lesser_pct.empty

    def test_missing_lesser_abs_threshold(self, test_dataset):
        df = bars.range.fixed(test_dataset, 100, threshold_is_absolute=True)
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
        df = bars.volume.fixed(test_dataset, 10**6)
        assert not df.empty
        assert df[df.volume_quote_buy + df.volume_quote_sell < 10**6].empty


class TestTicksFixedBars:
    def test_missing_lesser_threshold(self, test_dataset):
        df = bars.tick.fixed(test_dataset, 5000)
        assert not df.empty
        assert df[df.ticks_buy + df.ticks_sell < 5000].empty


class TestTimeFixedBars:
    def test_bar_periods(self, test_dataset):
        period = 5
        df = bars.time.fixed(test_dataset, period, timestamp_close=True)
        assert not df.empty
        for bar in df.itertuples():
            ts_open = bar.timestamp_close - pd.Timedelta(seconds=period)
            assert ts_open == bar.Index


class TestVolumeAdaptiveBars:
    def test_volume_bars(self, test_dataset):
        window_size = 60 * 60 * 2
        avg_per = 60
        df = bars.volume.adaptive(
            ohlc=test_dataset,
            avg_per=avg_per,
            window=window_size,
            timestamp_close=True
        )
        assert not df.empty
        for bar in df.itertuples():
            window_start = bar.timestamp_close - pd.Timedelta(
                seconds=window_size)
            window_end = bar.timestamp_close
            avg_volume_source = test_dataset.loc[window_start: window_end]
            volume_sum = _volume_sum(avg_volume_source)
            avg_volume = volume_sum / window_size / avg_per
            assert bar.volume_sell + bar.volume_buy >= avg_volume


class TestHybridFixedRangeAdaptiveVolumeBars:
    def test_missing_lesser_pct_threshold_and_avg(self, test_dataset):
        window_size = 60 * 60 * 2
        avg_per = 60
        df = bars.hybrid.range_fixed_volume_adaptive(
            ohlc=test_dataset,
            range_threshold=0.01,
            avg_per=avg_per,
            window=window_size,
            timestamp_close=True
        )
        df['price_change'] = df.close.pct_change()
        assert not df.empty
        assert df[abs(df.price_change) < 0.01].empty
        for bar in df.itertuples():
            window_start = bar.timestamp_close - pd.Timedelta(
                seconds=window_size)
            window_end = bar.timestamp_close
            avg_volume_source = test_dataset.loc[window_start: window_end]
            volume_sum = _volume_sum(avg_volume_source)
            avg_volume = volume_sum / window_size / avg_per
            assert bar.volume_sell + bar.volume_buy >= avg_volume


class TestHybridFixedRangeFixedTicksBars:
    def test_missing_lesser_thresholds(self, test_dataset):
        df = bars.hybrid.range_fixed_ticks_fixed(
            ohlc=test_dataset,
            range_threshold=0.01,
            ticks_threshold=5000
        )
        df['price_change'] = df.close.pct_change()
        assert not df.empty
        assert df[abs(df.price_change) < 0.01].empty
        assert df[df.ticks_buy + df.ticks_sell < 50].empty


class TestCusumFixedBars:
    def test_missing_lesser_thresholds(self, test_dataset):
        source = test_dataset
        threshold = 0.001
        df = bars.cusum.fixed(
            ohlc=test_dataset,
            threshold=threshold,
            timestamp_close=True
        )
        assert not df.empty
        source.loc[:, 'log_close_diff'] = np.log(
            source.loc[:, 'close']).diff().fillna(0)
        for bar in df.itertuples():
            bar_source = test_dataset.loc[bar.Index:bar.timestamp_close]
            s_pos, s_neg = 0, 0
            for i in bar_source.index:
                pos = float(s_pos + source.loc[i, 'log_close_diff'])
                neg = float(s_neg + source.loc[i, 'log_close_diff'])
                s_pos = max(0.0, pos)
                s_neg = min(0.0, neg)

            s_neg_diff = abs(s_neg) - threshold
            s_pos_diff = s_pos - threshold
            assert s_neg_diff < ERROR or s_pos_diff < ERROR
