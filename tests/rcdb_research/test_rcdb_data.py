import re
import os
from importlib import resources

import pytest
import numpy as np
import pandas as pd

from rcdb_research.rcdb_data import RcdbData

TEST_OHLCV_API_URL = "https://storage.com"


@pytest.fixture
def local_cache_path(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    yield str(path.resolve())


@pytest.fixture
def clean_ohlcv():
    yield
    RcdbData.clean_up()


@pytest.fixture
def mock_storage_url(requests_mock):
    with resources.open_binary("tests.datasets", "bitfinex_btcusd.csv.gz") as f:
        yield requests_mock.get(
            re.compile(TEST_OHLCV_API_URL), content=f.read()
        )


def test_init(test_url="test_url", test_cache_path="test_cache_path"):
    ohlcv = RcdbData(ohlcv_api_url=test_url, local_cache_path=test_cache_path)
    assert ohlcv.ohlcv_api_url == test_url
    assert ohlcv.local_cache_path == test_cache_path


def test_get_ohlcv_url_method(test_url="test_url", test_cache_path="test_cache_path"):
    ohlcv = RcdbData(ohlcv_api_url=test_url, local_cache_path=test_cache_path)
    config = RcdbData.OHLCVConfig(
        exchange="bf",
        base="usd",
        quote="btc",
        timeframe="3s",
        start=None,
        end=None,
        is_whole_period=True
    )
    assert ohlcv.get_ohlcv_url(config) == "test_url?exchange=bf&symbol=usdbtc&timeframe=3s"


@pytest.mark.parametrize(
    "args",
    [(None, "param"), ("param", None), (None, None)]
)
def test_fail_init(args):
    with pytest.raises(ValueError):
        RcdbData(*args)


@pytest.fixture
def fetch_params(local_cache_path):
    return {
        "base": "BTC",
        "quote": "USD",
        "timeframe": "1m",
        "exchange": "bitfinex",
        "ohlcv_api_url": TEST_OHLCV_API_URL,
        "local_cache_path": local_cache_path
    }


@pytest.mark.skip('write tests for shards fetch')
class TestFetch:
    def test_fetch_remote(self, fetch_params, mock_storage_url):
        res = RcdbData.fetch(**fetch_params)

        assert not res.empty
        assert mock_storage_url.called
        assert len(os.listdir(RcdbData._instance.local_cache_path)) == 1

    def test_fetch_local_cache_file(self, fetch_params, mock_storage_url, local_cache_path):
        with resources.open_binary("tests.datasets", "bitfinex__BTC_USD.hdf") as f:
            with open(os.path.join(local_cache_path, "bitfinex__BTC_USD.hdf"), "wb") as dest:
                dest.write(
                    f.read()
                )

            res = RcdbData.fetch(**fetch_params)
            assert not res.empty
            assert not mock_storage_url.called


@pytest.mark.parametrize(
    "input_df, test_result",
    [
        (
            pd.DataFrame(columns=[
                'open', 'high', 'low', 'close', 'volume', 'volume_buy', 'volume_sell',
                'volume_quote', 'volume_quote_buy', 'volume_quote_sell', 'ticks', 'ticks_buy', 'ticks_sell'
            ]),
            []
        ),

        (
            pd.DataFrame(columns=[
                'open', 'high', 'low', 'close', 'volume', 'volume_buy', 'volume_sell',
                'volume_quote', 'volume_quote_buy', 'volume_quote_sell', 'ticks', 'ticks_buy', 'ticks_sell',
                'some-columns'
            ]),
            []
        ),
        (
            pd.DataFrame(columns=[
                'open', 'high', 'low', 'close', 'volume', 'volume_buy', 'volume_sell',
                'volume_quote', 'volume_quote_buy', 'volume_quote_sell', 'ticks'
            ]),
            ['ticks_buy', 'ticks_sell']
        ),

    ],
    ids=[
        'test_method__missed_columns[same columns]',
        'test_method__missed_columns[additional columns]',
        'test_method__missed_columns[missed columns]',
    ]
)
def test_method__missed_columns(input_df, test_result):
    assert set(RcdbData.missing_columns(input_df)) == set(test_result)


@pytest.mark.parametrize(
    'input_df, test_result',
    [
        (pd.DataFrame(dict(a=[1, 2, 3])), True),
        (pd.DataFrame(dict(a=[1., 2, 3])), True),
        (pd.DataFrame(dict(a=np.array([1, 2, 3]))), True),
        (pd.DataFrame(dict(a=np.array([1., 2, 3]))), True),
        (pd.DataFrame(dict(a=[1, 2, 3], b=['a', 'b', 'c'])), False),
        (pd.DataFrame(dict(a=[1, None, 3])), False),
        (pd.DataFrame(dict(a=[1, np.nan, 3])), False),
        (pd.DataFrame(dict(a=[pd.datetime.now()])), False),

    ]
)
def test_method__check_consistency(input_df, test_result):
    assert RcdbData.check_consistency(input_df) == test_result
