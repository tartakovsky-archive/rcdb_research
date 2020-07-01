from datetime import datetime as dt

import pytest
import numpy as np

from rcdb_research.features import datetimes


TEST_ARRAY = np.array([
    dt(2019, 8, 27, 11, 44, 11),
    dt(2019, 8, 28, 12, 34, 21),
    dt(2019, 12, 1, 1, 2, 1),
])


@pytest.mark.parametrize(
    "feature, result",
    [
        (datetimes.sec_of_min, np.array([11, 21, 1])),
        (datetimes.min_of_hour, np.array([44, 34, 2])),
        (datetimes.hour_of_day, np.array([11, 12, 1])),
        (datetimes.day_of_month, np.array([27, 28, 1])),
        (datetimes.day_of_week, np.array([1, 2, 6])),
        (datetimes.day_of_year, np.array([239, 240, 335])),
        (datetimes.week_of_month, np.array([5, 5, 1])),
        (datetimes.week_of_year, np.array([35, 35, 48])),
        (datetimes.month_of_year, np.array([8, 8, 12])),
    ]
)
def test_datetime(feature, result):
    assert np.array_equal(feature(TEST_ARRAY), result)
