from datetime import datetime, timezone

import pytest
import numpy as np

from commons.features.datetime import markets


MARKET_NAME = "NYSE"


@pytest.mark.parametrize(
    "timestamp, check_val",
    [
        (np.array([datetime(2019, 7, 2, 14, tzinfo=timezone.utc)]), 1),
        (np.array([datetime(2019, 7, 2, 21, tzinfo=timezone.utc)]), 0),
        (np.array([datetime(2019, 7, 3, 14, tzinfo=timezone.utc)]), 1),
        (np.array([datetime(2019, 7, 4, 14, tzinfo=timezone.utc)]), 0),

    ]
)
def test_is_market_open(timestamp, check_val):
    assert markets.f1(timestamp, MARKET_NAME)[0] == check_val


@pytest.mark.parametrize(
    "feature, timestamp, market_timestamp, check_val, is_open",
    [
        # Check if the timestamps are 1h before the market open
        (
            markets.f2,
            np.array([datetime(2019, 7, 2, 14, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 15, tzinfo=timezone.utc)]),
            1,
            True
        ),
        (
            markets.f2,
            np.array([datetime(2019, 7, 2, 14, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 16, tzinfo=timezone.utc)]),
            0,
            True
        ),

        # Check if the timestamps are 1h after the market open
        (
            markets.f3,
            np.array([datetime(2019, 7, 2, 15, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 15, tzinfo=timezone.utc)]),
            1,
            True
        ),
        (
            markets.f3,
            np.array([datetime(2019, 7, 2, 17, 1, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 16, tzinfo=timezone.utc)]),
            0,
            True
        ),

        # Check if the timestamps are 1h before the market close
        (
            markets.f4,
            np.array([datetime(2019, 7, 2, 14, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 15, tzinfo=timezone.utc)]),
            1,
            False
        ),
        (
            markets.f4,
            np.array([datetime(2019, 7, 2, 14, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 16, tzinfo=timezone.utc)]),
            0,
            False
        ),

        # Check if the timestamps are 1h after the market close
        (
            markets.f5,
            np.array([datetime(2019, 7, 2, 15, 30, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 15, tzinfo=timezone.utc)]),
            1,
            False
        ),
        (
            markets.f5,
            np.array([datetime(2019, 7, 2, 17, 1, tzinfo=timezone.utc)]),
            np.array([datetime(2019, 7, 2, 16, tzinfo=timezone.utc)]),
            0,
            False
        ),
    ],
    ids=[
        "1h before the market open", "not 1h before the market open",
        "1h after the market open", "not 1h after the market open",

        "1h before the market close", "not 1h before the market close",
        "1h after the market close", "not 1h after the market close"
    ]
)
def test_is_hour_away_features(feature, timestamp, market_timestamp, check_val, is_open, mocker):
    ret_value = (market_timestamp, None) if is_open else (None, market_timestamp)
    mocker.patch("commons.features.datetime.utils.get_market_open_close", return_value=ret_value)
    assert feature(timestamp, MARKET_NAME)[0] == check_val
