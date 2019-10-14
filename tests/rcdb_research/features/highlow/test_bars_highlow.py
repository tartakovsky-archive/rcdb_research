import pytest
import numpy as np

from rcdb_research.features import highlow


TEST_SERIES = np.array([0, 1, 2, 3, 5, 3, 2, 1, 4])
TEST_PERIOD = 3


@pytest.fixture(scope="module")
def data(ohlcv_df):
    return ohlcv_df[["high", "low", "close"]][:len(ohlcv_df) // 2]


@pytest.mark.parametrize(
    "feature, test_res",
    [
        (highlow.bars_in_drawdown, [0, 0, 0, 0, 0, 1, 2, 3, 0]),
        (highlow.bars_in_runup, [0, 1, 2, 3, 4, 0, 0, 0, 1]),
    ]
)
def test_features(feature, test_res):
    assert np.array_equal(feature(TEST_SERIES), test_res)


@pytest.mark.parametrize(
    "feature, test_res",
    [
        (highlow.is_local_high, [0, 0, 1, 1, 1, 0, 0, 0, 1]),
        (highlow.is_local_low, [0, 0, 0, 0, 0, 1, 1, 1, 0]),
        (highlow.bars_since_local_high, [0, 0, 0, 0, 0, 1, 2, 3, 0]),
        (highlow.bars_since_local_low, [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        (highlow.change_since_local_high, [0., 0., 0., 0., 0., -0.4, -0.6, -0.8, 0.]),
        (highlow.change_since_local_low, [0, 0, 0, 0, 0, 0, 0, 0, 3.])
    ]
)
def test_period_features(feature, test_res):
    assert np.array_equal(
        feature(TEST_SERIES, TEST_PERIOD), test_res
    )
