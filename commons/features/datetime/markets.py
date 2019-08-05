from datetime import timedelta
from typing import List, Dict

import pandas as pd
import numpy as np

from . import utils

PREFIX = "dt_markets"


def calc_all(data: pd.DatetimeIndex, param_sets: List[Dict] = None, column_names=None):
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
    if type(data) != pd.DatetimeIndex:
        raise ValueError("calc_all `data` arg expected to be `pd.DateTimeIndex`")

    ts_col = "ts"
    df = pd.DataFrame(dict(data=data.values))
    df[ts_col] = df['data']
    df.set_index('data', inplace=True)

    timestamp = data.to_pydatetime()

    markets = {
        'NYSE', 'LSE', 'CME', 'ICE',
        'CFE', 'BMF', 'TSX', 'EUREX',
        'JPX', 'SIX', 'OSE', 'SSE', 'HKEX'
    }

    for ps in param_sets:
        market_name = ps["market"]
        if market_name not in markets:
            raise ValueError(f"Unsupported market {market_name}. Choose from supported: {markets}")

    for ps in param_sets:
        market_name = ps["market"]

        df[f"{PREFIX}_f1{market_name}"] = f1(timestamp, market_name)
        df[f"{PREFIX}_f2{market_name}"] = f2(timestamp, market_name)
        df[f"{PREFIX}_f3{market_name}"] = f3(timestamp, market_name)
        df[f"{PREFIX}_f4{market_name}"] = f4(timestamp, market_name)
        df[f"{PREFIX}_f5{market_name}"] = f5(timestamp, market_name)

    return df


def f1(timestamps: np.array, market_name: str) -> np.array:
    """
    Check if the timestamps at the open market
    :param np.array timestamps: array of datetime`s
    :param str market_name: name of the market
    :return: array of 0 and 1 for each timestamp
    """
    calendar = utils.get_market_calendar(market_name)
    calendar_schedule = utils.get_market_calendar_schedule(market_name, timestamps)

    is_open = np.vectorize(
        lambda dt: calendar.open_at_time(calendar_schedule, dt)
    )
    return is_open(timestamps) * 1


def f2(timestamps: np.array, market_name: str) -> np.array:
    """
    Check if the timestamps are 1h before the market open
    :param np.array timestamps: array of datetime`s
    :param str market_name: name of the market
    :return: array of 0 and 1 for each timestamp
    """
    market_open, _ = utils.get_market_open_close(market_name, timestamps)
    return utils.is_hour_away(timestamps, market_open)


def f3(timestamps: np.array, market_name: str) -> np.array:
    """
    Check if the timestamps are 1h after the market open
    :param np.array timestamps: array of datetime`s
    :param str market_name: name of the market
    :return: array of 0 and 1 for each timestamp
    """
    market_open, _ = utils.get_market_open_close(market_name, timestamps)
    return utils.is_hour_away(timestamps, market_open + timedelta(hours=1))


def f4(timestamps: np.array, market_name: str) -> np.array:
    """
    Check if the timestamps are 1h before the market close
    :param np.array timestamps: array of datetime`s
    :param str market_name: name of the market
    :return: array of 0 and 1 for each timestamp
    """
    _, market_close = utils.get_market_open_close(market_name, timestamps)
    return utils.is_hour_away(timestamps, market_close)


def f5(timestamps: np.array, market_name: str) -> np.array:
    """
    Check if the timestamps are 1h after the market close
    :param np.array timestamps: array of datetime`s
    :param str market_name: name of the market
    :return: array of 0 and 1 for each timestamp
    """
    _, market_close = utils.get_market_open_close(market_name, timestamps)
    return utils.is_hour_away(timestamps, market_close + timedelta(hours=1))
