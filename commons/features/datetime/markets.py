import numpy as np

from . import utils
from datetime import timedelta


PREFIX = "dt_markets"


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
