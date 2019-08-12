import pytest
import numpy as np

from commons.features import highlow


TEST_SERIES = np.array([0, 1, 2, 3, 5, 3, 2, 1, 4])
TEST_PERIOD = 3


@pytest.fixture(scope="module")
def data(ohlcv_df):
    return ohlcv_df[["high", "low", "close"]][:len(ohlcv_df) // 2]


@pytest.fixture(
    params=[
        [15],
        [10, 20]
    ]
)
def params_set(request):
    return [{"period": p} for p in request.param]


def test_calc_all(params_set, data):
    res = highlow.calc_all(data, params_set)
    assert len(data) == len(res)

    test_cols = {
        f"{highlow.PREFIX}_f1", f"{highlow.PREFIX}_f4",
        f"{highlow.PREFIX}_f7", f"{highlow.PREFIX}_f8", f"{highlow.PREFIX}_f9",
        *[f"{highlow.PREFIX}_f{f}{p['period']}" for f in [2, 3, 5, 6, 10, 11] for p in params_set]
    }
    assert set(res.columns) == test_cols


@pytest.mark.parametrize(
    "column_names",
    [
        {},
        dict(high="_high", low="_low", close="_close")
    ]
)
def test_wrong_cols_mapping(column_names, data):
    with pytest.raises(KeyError):
        highlow.calc_all(data, [{"period": 10}], column_names)


@pytest.mark.parametrize(
    "feature, test_res",
    [
        (highlow.f1, [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        (highlow.f4, [0, 0, 0, 0, 0, 1, 2, 3, 4]),
        (highlow.f7, [0, 0, 0, 0, 0, 1, 2, 3, 0]),
        (highlow.f8, [0, 1, 2, 3, 4, 0, 0, 0, 1]),
        (highlow.f9, [0., 0., 0., 0., 0., -0.4, -0.6, -0.8, -0.2])
    ]
)
def test_features(feature, test_res):
    assert np.array_equal(feature(TEST_SERIES), test_res)


@pytest.mark.parametrize(
    "feature, test_res",
    [
        (highlow.f2, [0, 0, 1, 1, 1, 0, 0, 0, 1]),
        (highlow.f3, [0, 0, 0, 0, 0, 1, 1, 1, 0]),
        (highlow.f5, [0, 0, 0, 0, 0, 1, 2, 3, 0]),
        (highlow.f6, [0, 0, 0, 0, 0, 0, 0, 0, 1]),
        (highlow.f10, [0., 0., 0., 0., 0., -0.4, -0.6, -0.8, 0.]),
        (highlow.f11, [0, 0, 0, 0, 0, 0, 0, 0, 3.])
    ]
)
def test_period_features(feature, test_res):
    assert np.array_equal(
        feature(TEST_SERIES, TEST_PERIOD), test_res
    )
