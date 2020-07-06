import sys
import inspect
import numpy as np

import pytest

from rcdb_research.features import alphas101


@pytest.fixture(scope="module")
def data(ohlcv_df):
    ohlcv_df = ohlcv_df[:1100]
    input_data = {}
    input_data["open"] = ohlcv_df["open"].values + np.random.uniform(20, 100, len(ohlcv_df))
    input_data["high"] = ohlcv_df["high"].values
    input_data["low"] = ohlcv_df["low"].values
    input_data["close"] = ohlcv_df["close"].values
    input_data["volume"] = (ohlcv_df["volume_buy"] + ohlcv_df["volume_sell"]).values
    input_data["returns"] = ohlcv_df["close"].pct_change().values
    input_data["vwap"] = (
        (ohlcv_df.volume_quote_buy + ohlcv_df.volume_quote_sell) / (ohlcv_df.volume_buy + ohlcv_df.volume_sell)
    ).values

    return input_data


def feature_filter(o):
    try:
        if inspect.isfunction(o) and o.__name__[0] == "f":
            int(o.__name__[1:])
            return True

    except ValueError:
        pass

    return False


@pytest.mark.parametrize(
    "feature",
    dict(inspect.getmembers(sys.modules[alphas101.__name__], feature_filter)).values()
)
def test_alphas(feature, data: dict):
    inputs = {k: data[k] for k in inspect.getfullargspec(feature).args if k in data}
    res = feature(**inputs)
    assert len(res) == len(data[next(iter(data))])
    assert not np.isnan(res).all()
