from datetime import datetime, timezone

import pytest
import numpy as np
from workalendar.registry import registry

from commons.features.datetime import holidays

COUNTRY_NAME = "US"


@pytest.fixture(scope="module")
def calendar():
    return registry.get_calendar_class("US")()


@pytest.mark.parametrize(
    "timestamp, is_holidays",
    [
        (
            np.array([datetime(2012, 1, 1), datetime(2019, 1, 21), datetime(2019, 2, 21)]),
            [1, 1, 0],
        )
    ]
)
def test_is_holidays(timestamp, is_holidays):
    assert np.array_equal(
        holidays.f1(timestamp, COUNTRY_NAME),
        is_holidays
    )


@pytest.mark.parametrize(
    "timestamp, is_working",
    [
        (
            np.array([datetime(2019, 2, 15), datetime(2019, 2, 16), datetime(2019, 2, 17), datetime(2019, 2, 18)]),
            [1, 0, 0, 0]
        )
    ]
)
def test_is_working_day(timestamp, is_working):
    assert np.array_equal(
        holidays.f2(timestamp, COUNTRY_NAME),
        is_working
    )


@pytest.mark.parametrize(
    "timestamp, holidays_array, distances",
    [
        (
            np.array(
                [
                    datetime(2019, 1, 2, 23, tzinfo=timezone.utc),
                    datetime(2019, 1, 2, 23, 30, tzinfo=timezone.utc),
                    datetime(2019, 1, 1, 0, tzinfo=timezone.utc)
                ]
            ),
            np.array([datetime(2019, 1, 3, 0).date()]),
            [3600, 1800, 172800]
        )
    ]
)
def test_distance_to_holidays(timestamp, holidays_array, distances, mocker):
    mocker.patch("commons.features.datetime.utils.get_holidays", return_value=holidays_array)

    assert np.array_equal(
        holidays.f3(timestamp, COUNTRY_NAME),
        distances
    )


@pytest.mark.parametrize(
    "timestamp, holidays_array, check_arr",
    [
        (
            np.array(
                [
                    datetime(2019, 1, 2, 23, tzinfo=timezone.utc),
                    datetime(2019, 1, 2, 23, 30, tzinfo=timezone.utc),
                    datetime(2019, 1, 1, 0, tzinfo=timezone.utc)
                ]
            ),
            np.array([datetime(2019, 1, 3, 0).date()]),
            [1, 1, 0]
        )
    ]
)
def test_is_day_before_holidays(timestamp, holidays_array, check_arr, mocker):
    mocker.patch("commons.features.datetime.utils.get_holidays", return_value=holidays_array)

    assert np.array_equal(
        holidays.f4(timestamp, COUNTRY_NAME),
        check_arr
    )


@pytest.mark.parametrize(
    "timestamp, holidays_array, check_arr",
    [
        (
            np.array(
                [
                    datetime(2019, 1, 4, 23, tzinfo=timezone.utc),
                    datetime(2019, 1, 4, 23, 30, tzinfo=timezone.utc),
                    datetime(2019, 1, 5, 0, tzinfo=timezone.utc)
                ]
            ),
            np.array([datetime(2019, 1, 3, 0).date()]),
            [1, 1, 0]
        )
    ]
)
def test_is_day_after_holidays(timestamp, holidays_array, check_arr, mocker):
    mocker.patch("commons.features.datetime.utils.get_holidays", return_value=holidays_array)
    assert np.array_equal(
        holidays.f5(timestamp, COUNTRY_NAME),
        check_arr
    )


@pytest.mark.parametrize(
    "timestamp, is_working",
    [
        (
            np.array([datetime(2019, 2, 15), datetime(2019, 2, 16), datetime(2019, 2, 17), datetime(2019, 2, 18)]),
            [0, 1, 1, 0]
        )
    ]
)
def test_is_weekend(timestamp, is_working):
    assert np.array_equal(
        holidays.f6(timestamp),
        is_working
    )
