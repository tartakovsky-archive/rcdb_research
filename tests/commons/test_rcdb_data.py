import re
import os
from importlib import resources

import pytest


from commons.rcdb_data import RcdbData

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
    with resources.open_binary("tests.datasets", "bitfinex__BTC_USD.hdf") as f:
        yield requests_mock.get(
            re.compile(TEST_OHLCV_API_URL), content=f.read()
        )


def test_init(test_url="test_url", test_cache_path="test_cache_path"):
    ohlcv = RcdbData(ohlcv_api_url=test_url, local_cache_path=test_cache_path)
    assert ohlcv.ohlcv_api_url == test_url
    assert ohlcv.local_cache_path == test_cache_path


@pytest.mark.parametrize(
    "args",
    [(None, "param"), ("param", None), (None, None)]
)
def test_fail_init(args):
    with pytest.raises(AssertionError):
        RcdbData(*args)


@pytest.fixture
def fetch_params(local_cache_path):
    return {
        "base": "BTC",
        "quote": "USD",
        # "timeframe": "1m",
        "exchange": "bitfinex",
        "ohlcv_api_url": TEST_OHLCV_API_URL,
        "local_cache_path": local_cache_path
    }


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
