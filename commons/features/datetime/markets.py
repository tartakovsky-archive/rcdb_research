from datetime import timedelta
from typing import List, Dict

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

from . import utils


def calc_all(data: pd.DataFrame, param_sets: List[Dict] = None, column_names=None):
    """
    Calculate features based on the markets open/close time

    Supported markets:
        [
            'NYSE', 'LSE', 'CME', 'ICE',
            'CFE', 'BMF', 'TSX', 'EUREX',
            'JPX', 'SIX', 'OSE', 'SSE', 'HKEX'
        ]

    :param data: df with DatetimeIndex. No required columns
    :param List[Dict] param_sets: list of dict with market name. Example [{"market": "NYSE"}, ...]
    :param column_names: unused template parameter
    :return: dataframe with calculated features
    """
    timestamp = data.index.to_pydatetime()
    df = data.copy()

    markets = {
        'NYSE', 'LSE', 'CME', 'ICE',
        'CFE', 'BMF', 'TSX', 'EUREX',
        'JPX', 'SIX', 'OSE', 'SSE', 'HKEX'
    }

    for ps in param_sets:
        market_name = ps["market"]
        if market_name not in markets:
            raise ValueError("Unsupported market {market_name}. Choose from supported: {markets}")

    for ps in param_sets:
        market_name = ps["market"]
        calendar = mcal.get_calendar(market_name)
        calendar_schedule = calendar.schedule(
            start_date=data.index[0].date(),
            end_date=data.index[-1].date() + timedelta(days=2)
        )

        market_open = pd.DatetimeIndex(calendar_schedule.market_open.values).tz_localize(tz='UTC').to_pydatetime()
        market_close = pd.DatetimeIndex(calendar_schedule.market_open.values).tz_localize(tz='UTC').to_pydatetime()

        df[f"f1{market_name}"] = f1(timestamp, calendar, calendar_schedule)
        df[f"f2{market_name}"] = f2(timestamp, market_open)
        df[f"f3{market_name}"] = f3(timestamp, market_open)
        df[f"f4{market_name}"] = f4(timestamp, market_close)
        df[f"f5{market_name}"] = f5(timestamp, market_close)

    return df


def f1(timestamp: np.array, calendar: mcal.MarketCalendar, calendar_schedule: pd.DataFrame) -> np.array:
    """
    Check if the timestamps at the open market
    :param np.array timestamp: array of datetime`s
    :param mcal.MarketCalendar calendar: instance of pandas_market_calendars MarketCalendar
    :param pd.DataFrame calendar_schedule:
        df[date index, market_open[datetime64], market_close[datetime64]] getted from mcal
    :return: array of 0 and 1 for each timestamp
    """
    is_open = np.vectorize(
        lambda dt: calendar.open_at_time(calendar_schedule, dt) * 1
    )
    return is_open(timestamp)


def f2(timestamp: np.array, market_open: np.array) -> np.array:
    """
    Check if the timestamps are 1h before the market open
    :param np.array timestamp: array of datetime`s
    :param np.array market_open: array of datetime`s
    :return: array of 0 and 1 for each timestamp
    """
    return utils.is_hour_away(timestamp, market_open)


def f3(timestamp: np.array, market_open: np.array) -> np.array:
    """
    Check if the timestamps are 1h after the market open
    :param np.array timestamp: array of datetime`s
    :param np.array market_open: array of datetime`s
    :return: array of 0 and 1 for each timestamp
    """
    return utils.is_hour_away(timestamp, market_open + timedelta(hours=1))


def f4(timestamp: np.array, market_close: np.array) -> np.array:
    """
    Check if the timestamps are 1h before the market close
    :param np.array timestamp: array of datetime`s
    :param np.array market_close: array of datetime`s
    :return: array of 0 and 1 for each timestamp
    """
    return utils.is_hour_away(timestamp, market_close)


def f5(timestamp: np.array, market_close: np.array) -> np.array:
    """
    Check if the timestamps are 1h after the market close
    :param np.array timestamp: array of datetime`s
    :param np.array market_close: array of datetime`s
    :return: array of 0 and 1 for each timestamp
    """
    return utils.is_hour_away(timestamp, market_close + timedelta(hours=1))
