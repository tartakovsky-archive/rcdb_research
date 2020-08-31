from importlib import resources

import pytest
import pandas as pd


@pytest.fixture(scope="session")
def ohlcv_df():
    with resources.path("tests.datasets", "bitfinex__BTC_USD.hdf") as f:
        df = pd.read_hdf(f, "table")
    return df


@pytest.fixture
def Xy(ohlcv_df):
    return ohlcv_df.copy(), (ohlcv_df.close.diff() > 0) * 1
