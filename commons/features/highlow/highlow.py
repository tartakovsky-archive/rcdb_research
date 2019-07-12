from typing import List, Dict

import numpy as np
import pandas as pd

from . import utils

PREFIX = "highlow"

def calc_all(
    data: pd.DataFrame,
    param_set: List[Dict],
    column_names: dict = dict(high = "high", low = "low", close = "close")
) -> pd.DataFrame:
    """
    Calculate high/low features
    :param pd.DataFrame data: df with DatetimeIndex, with `column_names.values()` columns
    :param List[Dict] param_set: set of parameters for feature calculation, required key "period"
    :param dict column_names: mapping of required columns
    :return:
    """

    high = data[column_names["high"]]
    low = data[column_names["low"]]
    close = data[column_names["close"]]

    df = pd.DataFrame(
        {
            f"{PREFIX}_f1": f1(high),
            f"{PREFIX}_f4": f4(high),
            f"{PREFIX}_f7": f7(close),
            f"{PREFIX}_f8": f8(close),
            f"{PREFIX}_f9": f9(high),
        },
        index=data.index
    )

    for ps in param_set:
        period = ps["period"]

        df[f"{PREFIX}_f2{period}"] = f2(high, period)
        df[f"{PREFIX}_f3{period}"] = f3(low, period)
        df[f"{PREFIX}_f5{period}"] = f5(high, period)
        df[f"{PREFIX}_f6{period}"] = f6(low, period)
        df[f"{PREFIX}_f10{period}"] = f10(high, period)
        df[f"{PREFIX}_f11{period}"] = f11(low, period)

    return df


def f1(high: pd.Series) -> np.array:
    """
    Check if the high is ath
    :param pd.Series high: series with DatetimeIndex
    :return: array of 0 and 1 for each row
    """
    return utils.is_extremum(series=high, period="ALL", maximum=True)


def f2(high: pd.Series, period: str) -> np.array:
    """
    Check if the high is a highest in period
    :param pd.Series high: series with DatetimeIndex
    :param str period: tf in format `<int>(s|m|h|D|W|M|Q|Y)` Example: 35s, 1h, 3D, 15m, etc.
    :return: array of 0 and 1 for each row
    """
    return utils.is_extremum(series=high, period=period, maximum=True)


def f3(low: pd.Series, period: str) -> np.array:
    """
    Check if the low is a lowest in period
    :param pd.Series low: series with DatetimeIndex
    :param str period: tf in format `<int>(s|m|h|D|W|M|Q|Y)` Example: 35s, 1h, 3D, 15m, etc.
    :return: array of 0 and 1 for each row
    """
    return utils.is_extremum(series=low, period=period, maximum=False)


def f4(high: pd.Series) -> np.array:
    """
    Calculate time since ath
    :param pd.Series high: series with DatetimeIndex
    :return: array of seconds
    """
    return utils.time_since_extremum(series=high, period="ALL", maximum=True)


def f5(high: pd.Series, period: str) -> np.array:
    """
    Calculate time since highest in period
    :param pd.Series high: series with DatetimeIndex
    :param str period: tf in format `<int>(s|m|h|D|W|M|Q|Y)` Example: 35s, 1h, 3D, 15m, etc.
    :return: array of seconds
    """
    return utils.time_since_extremum(series=high, period=period, maximum=True)


def f6(low: pd.Series, period: str) -> np.array:
    """
    Calculate time since lowest in period
    :param pd.Series low: series with DatetimeIndex
    :param str period: tf in format `<int>(s|m|h|D|W|M|Q|Y)` Example: 35s, 1h, 3D, 15m, etc.
    :return: array of seconds
    """
    return utils.time_since_extremum(series=low, period=period, maximum=False)


def f7(close: pd.Series) -> np.array:
    """
    Calculate time in drawdown
    :param pd.Series close: series with DatetimeIndex
    :return: array of seconds
    """
    return utils.time_in(series=close, drawdown=True)


def f8(close: pd.Series):
    """
    Calculate time in run up
    :param pd.Series close: series with DatetimeIndex
    :return: array of seconds
    """
    return utils.time_in(series=close, drawdown=False)


def f9(high: pd.Series) -> np.array:
    """
    Calculate % change since ath
    :param pd.Series high: series with DatetimeIndex
    :return: array of float
    """
    return utils.change_since_period(series=high, period="ALL", maximum=True)


def f10(high: pd.Series, period: str) -> np.array:
    """
    Calculate % change since highest in period
    :param pd.Series high: series with DatetimeIndex
    :param str period: tf in format `<int>(s|m|h|D|W|M|Q|Y)` Example: 35s, 1h, 3D, 15m, etc.
    :return: array of float
    """
    return utils.change_since_period(series=high, period=period, maximum=True)


def f11(low: pd.Series, period: str) -> np.array:
    """
    Calculate % change since lowest in period
    :param pd.Series low: series with DatetimeIndex
    :param str period: tf in format `<int>(s|m|h|D|W|M|Q|Y)` Example: 35s, 1h, 3D, 15m, etc.
    :return: array of float
    """
    return utils.change_since_period(series=low, period=period, maximum=False)
