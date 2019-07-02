from datetime import datetime, timezone

import pytest
import numpy as np
import pandas_market_calendars as mcal

from commons.features.datetime import markets


@pytest.fixture(scope="module")
def nyse_market_calendar():
    return mcal.get_calendar("NYSE")


@pytest.fixture
def nyse_schedule(nyse_market_calendar):
    return nyse_market_calendar.schedule(
        start_date=datetime(2019, 7, 2).date(),
        end_date=datetime(2019, 7, 5).date()
    )


@pytest.mark.parametrize(
    "timestamp, check_val",
    [
        (np.array([datetime(2019, 7, 2, 14, tzinfo=timezone.utc)]), 1),
        (np.array([datetime(2019, 7, 2, 21, tzinfo=timezone.utc)]), 0),
        (np.array([datetime(2019, 7, 3, 14, tzinfo=timezone.utc)]), 1),
        (np.array([datetime(2019, 7, 4, 14, tzinfo=timezone.utc)]), 0),

    ]
)
def test_is_market_open(timestamp, check_val, nyse_schedule, nyse_market_calendar):
    assert markets.f1(timestamp, nyse_market_calendar, nyse_schedule)[0] == check_val


@pytest.mark.parametrize(
    "feature, timestamp, market_timestamp, check_val",
    [
        # Check if the timestamps are 1h before the market open
        (
            markets.f2,
            np.array([datetime(2019, 7, 2, 14, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 15, tzinfo=timezone.utc)]),
            1
        ),
        (
            markets.f2,
            np.array([datetime(2019, 7, 2, 14, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 16, tzinfo=timezone.utc)]),
            0
        ),

        # Check if the timestamps are 1h after the market open
        (
            markets.f3,
            np.array([datetime(2019, 7, 2, 15, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 15, tzinfo=timezone.utc)]),
            1
        ),
        (
            markets.f3,
            np.array([datetime(2019, 7, 2, 17, 1, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 16, tzinfo=timezone.utc)]),
            0
        ),

        # Check if the timestamps are 1h before the market close
        (
            markets.f4,
            np.array([datetime(2019, 7, 2, 14, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 15, tzinfo=timezone.utc)]),
            1
        ),
        (
            markets.f4,
            np.array([datetime(2019, 7, 2, 14, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 16, tzinfo=timezone.utc)]),
            0
        ),

        # Check if the timestamps are 1h after the market close
        (
            markets.f5,
            np.array([datetime(2019, 7, 2, 15, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 15, tzinfo=timezone.utc)]),
            1
        ),
        (
            markets.f5,
            np.array([datetime(2019, 7, 2, 17, 1, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 16, tzinfo=timezone.utc)]),
            0
        ),
    ],
    ids=[
        "1h before the market open", "not 1h before the market open",
        "1h after the market open", "not 1h after the market open",

        "1h before the market close", "not 1h before the market close",
        "1h after the market close", "not 1h after the market close"
    ]
)
def test_is_hour_away_features(feature, timestamp, market_timestamp, check_val):
    assert feature(timestamp, market_timestamp)[0] == check_val
