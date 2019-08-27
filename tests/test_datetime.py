from datetime import datetime as dt

import pytest
import numpy as np

from commons.features import datetime


TEST_ARRAY = np.array([
    dt(2019, 8, 27, 11, 44, 11),
    dt(2019, 8, 28, 12, 34, 21),
    dt(2019, 12, 1, 1, 2, 1),
])


@pytest.mark.parametrize(
    "feature, result",
    [
        (datetime.sec_of_min, np.array([11, 21, 1])),
        (datetime.min_of_hour, np.array([44, 34, 2])),
        (datetime.hour_of_day, np.array([11, 12, 1])),
        (datetime.day_of_month, np.array([27, 28, 1])),
        (datetime.day_of_week, np.array([1, 2, 6])),
        (datetime.day_of_year, np.array([239, 240, 335])),
        (datetime.week_of_month, np.array([5, 5, 1])),
        (datetime.week_of_year, np.array([35, 35, 48])),
        (datetime.month_of_year, np.array([8, 8, 12])),
    ]
)
def test_datetime(feature, result):
    assert np.array_equal(feature(TEST_ARRAY), result)