from importlib import resources

import pytest
from talib import get_function_groups

from rcdb_research.utils import get_df_from_hdf_bytes
from rcdb_research.features import candlestick_patterns


@pytest.fixture(scope="module")
def data():
    with resources.open_binary("tests.datasets", "bitfinex__BTC_USD.hdf") as f:
        df = get_df_from_hdf_bytes(f.read())
    return df


def test_calc_all(data):
    res = candlestick_patterns.calc_all(data)
    assert len(res) == len(data)
    assert set(res.columns) == set(get_function_groups()["Pattern Recognition"])
