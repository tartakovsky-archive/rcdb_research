from datetime import datetime, timezone

import pytest
import numpy as np
import pandas as pd

from commons.features.highlow import time_periods


TEST_SERIES = pd.Series(
    [1, 2, 3,
     3, 2, 1,
     5, 7, 4,
     1, 2, 1,
     5, 3, 5,
     1, 2, 1],
    index=pd.DatetimeIndex(
        [
            datetime(2018, 12, 29, 12), datetime(2018, 12, 29, 13), datetime(2018, 12, 29, 23),
            datetime(2018, 12, 30, 12), datetime(2018, 12, 30, 13), datetime(2018, 12, 30, 23),
            datetime(2019, 1, 1, 12), datetime(2019, 1, 1, 13), datetime(2019, 1, 1, 23),
            datetime(2019, 1, 2, 12), datetime(2019, 1, 2, 13), datetime(2019, 1, 2, 23),
            datetime(2019, 10, 1, 12), datetime(2019, 10, 1, 13), datetime(2019, 10, 1, 23),
            datetime(2019, 10, 2, 12), datetime(2019, 10, 2, 13), datetime(2019, 10, 2, 23),
        ],
        tz=timezone.utc
    )
)


@pytest.fixture(scope="module")
def data(ohlcv_df):
    return ohlcv_df[["high", "low", "close"]][:len(ohlcv_df) // 2]


@pytest.fixture(
    params=[
        ["30m"],
        ["3h"],
        ["1D"],
        ["1W", "1M", "1Q", "1Y"]
    ]
)
def params_set(request):
    return [{"period": p} for p in request.param]


def test_calc_all(params_set, data):
    res = time_periods.calc_all(data, params_set)
    assert len(data) == len(res)

    test_cols = {
        f"{time_periods.PREFIX}_f1", f"{time_periods.PREFIX}_f4",
        f"{time_periods.PREFIX}_f7", f"{time_periods.PREFIX}_f8", f"{time_periods.PREFIX}_f9",
        *[f"{time_periods.PREFIX}_f{f}{p['period']}" for f in [2, 3, 5, 6, 10, 11] for p in params_set]
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
        time_periods.calc_all(data, [{"period": "1D"}], column_names)


@pytest.mark.parametrize(
    "feature, test_res",
    [
        (
            time_periods.f1,
            np.concatenate(
                (np.zeros(7), np.array([1]), np.zeros(10))
            )
        ),
        (
            time_periods.f4,
            [0, 0, 0, 0, 0, 0, 0, 0,
             36000, 82800, 86400, 122400,
             23583600, 23587200, 23623200,
             23670000, 23673600, 23709600]
        ),
        (
            time_periods.f7,
            [0, 0, 0, 0, 3600, 39600, 0, 0, 36000, 82800, 0, 36000, 0, 3600, 0, 46800, 0, 0]
        ),
        (
            time_periods.f8,
            [0, 3600, 39600, 0, 0, 0, 133200, 136800, 0, 0, 3600, 0, 23461200, 0, 36000, 0, 3600, 0]
        ),
        (
            time_periods.f9,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             -0.42857142857142855, -0.8571428571428571,
             -0.7142857142857143, -0.8571428571428571,
             -0.2857142857142857, -0.5714285714285714,
             -0.2857142857142857, -0.8571428571428571,
             -0.7142857142857143, -0.8571428571428571]
        )
    ]
)
def test_features(feature, test_res):
    assert np.array_equal(feature(TEST_SERIES), test_res)


@pytest.mark.parametrize(
    "feature, period, test_res",
    [
        (time_periods.f2, "1D", [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        (time_periods.f2, "1M", [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]),
        (time_periods.f2, "1Y", [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (time_periods.f3, "1D", [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        (time_periods.f3, "1M", [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]),
        (time_periods.f3, "1Y", [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]),

        (time_periods.f5, "1D", [0, 0, 0, 0, 3600, 39600, 172800, 0, 36000, 82800, 0,
                                 36000, 0, 3600, 0, 46800, 0, 36000]),
        (time_periods.f5, "1M", [0, 0, 0, 0, 3600, 39600, 172800, 0, 36000, 82800, 86400,
                                 122400, 0, 3600, 0, 46800, 50400, 86400]),
        (time_periods.f5, "1Y", [0, 0, 0, 0, 3600, 39600, 172800, 0, 36000, 82800, 86400,
                                 122400, 23583600, 23587200, 23623200, 23670000, 23673600, 23709600]),
        (time_periods.f6, "1D", [0, 3600, 39600, 86400, 90000, 0, 133200, 136800, 0, 0, 3600, 0,
                                 23461200, 0, 36000, 0, 3600, 0]),
        (time_periods.f6, "1M", [0, 3600, 39600, 86400, 90000, 0, 133200, 136800, 172800,
                                 0, 3600, 0, 23461200, 23464800, 23500800, 0, 3600, 0]),
        (time_periods.f6, "1Y", [0, 3600, 39600, 86400, 90000, 0, 133200, 136800, 172800,
                                 0, 3600, 0, 23461200, 23464800, 23500800, 0, 3600, 0]),

        (
            time_periods.f10, "1D",
            [0.0, 0.0, 0.0, 0.0, -0.3333333333333333,
             -0.6666666666666666, 0.6666666666666666,
             0.0, -0.42857142857142855, -0.8571428571428571,
             0.0, -0.5, 0.0, -0.4, 0.0, -0.8, 0.0, -0.5],
        ),
        (
            time_periods.f10, "1M",
            [0.0, 0.0, 0.0, 0.0, -0.3333333333333333,
             -0.6666666666666666, 0.6666666666666666, 0.0,
             -0.42857142857142855, -0.8571428571428571,
             -0.7142857142857143, -0.8571428571428571,
             0.0, -0.4, 0.0, -0.8, -0.6, -0.8],
        ),
        (
            time_periods.f10, "1Y",
            [0.0, 0.0, 0.0, 0.0, -0.3333333333333333,
             -0.6666666666666666, 0.6666666666666666, 0.0,
             -0.42857142857142855, -0.8571428571428571,
             -0.7142857142857143, -0.8571428571428571,
             -0.2857142857142857, -0.5714285714285714,
             -0.2857142857142857, -0.8571428571428571,
             -0.7142857142857143, -0.8571428571428571],
        ),

        (
            time_periods.f11, "1D",
            [0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 4.0, 6.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.6666666666666666, 0.0, 1.0, 0.0],
        ),
        (
            time_periods.f11, "1M",
            [0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 4.0, 6.0, 3.0, 0.0, 1.0, 0.0, 4.0, 2.0, 4.0, 0.0, 1.0, 0.0],
        ),
        (
            time_periods.f11, "1Y",
            [0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 4.0, 6.0, 3.0, 0.0, 1.0, 0.0, 4.0, 2.0, 4.0, 0.0, 1.0, 0.0],
        )
    ]
)
def test_period_features(feature, period, test_res):
    assert np.array_equal(
        feature(TEST_SERIES, period), test_res
    )
