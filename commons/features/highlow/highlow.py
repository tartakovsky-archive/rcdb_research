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
    :param pd.DataFrame data: df with `column_names.values()` columns
    :param List[Dict] param_set: set of parameters for feature calculation, required key "period"
    :param dict column_names: mapping of required columns
    :return:
    """

    high = data[column_names["high"]].values
    low = data[column_names["low"]].values
    close = data[column_names["close"]].values

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

        df[f"{PREFIX}_f2_{period}"] = f2(high, period)
        df[f"{PREFIX}_f3_{period}"] = f3(low, period)
        df[f"{PREFIX}_f5_{period}"] = f5(high, period)
        df[f"{PREFIX}_f6_{period}"] = f6(low, period)
        df[f"{PREFIX}_f10_{period}"] = f10(high, period)
        df[f"{PREFIX}_f11_{period}"] = f11(low, period)

    return df


def f1(high: np.array) -> np.array:
    """
    Check if the high is ath
    :param np.array high: input series
    :return: array of 0 and 1 for each row
    """
    return utils.is_ath(high)


def f2(high: np.array, period: int) -> np.array:
    """
    Check if the high is a highest in period
    :param np.array high: input series
    :param int period: rolling window size
    :return: array of 0 and 1 for each row
    """
    return utils.is_extremum_bars_periods(series=high, period=period, maximum=True)


def f3(low: np.array, period: int) -> np.array:
    """
    Check if the low is a lowest in period
    :param np.array low: input series
    :param int period: window size
    :return: array of 0 and 1 for each row
    """
    return utils.is_extremum_bars_periods(series=low, period=period, maximum=False)


def f4(high: np.array) -> np.array:
    """
    Calculate bars since ath
    :param np.array high: input series
    :return: array of seconds
    """
    return utils.bars_since_mark(
        utils.is_ath(high)
    )


def f5(high: np.array, period: int) -> np.array:
    """
    Calculate numbers of bars since highest in period
    :param np.array high: input series
    :param int period: rolling window size
    :return: array of seconds
    """
    return utils.bars_since_mark(
        utils.is_extremum_bars_periods(
            series=high,
            period=period,
            maximum=True
        )
    )


def f6(low: np.array, period: int) -> np.array:
    """
    Calculate numbers of bars since lowest in period
    :param np.array low: input series
    :param int period: rolling window size
    :return: array of seconds
    """
    return utils.bars_since_mark(
        utils.is_extremum_bars_periods(
            series=low,
            period=period,
            maximum=False
        )
    )


def f7(close: np.array) -> np.array:
    """
    Calculate numbers of bars in drawdown
    :param np.array close: input series
    :return: array of seconds
    """
    return utils.bars_in_marked(
        (utils.pct_change(series=close) < 0) * 1
    )


def f8(close: np.array) -> np.array:
    """
    Calculate numbers of bars in run up
    :param np.array close: input series
    :return: array of seconds
    """
    return utils.bars_in_marked(
        (utils.pct_change(series=close) > 0) * 1
    )


def f9(high: np.array) -> np.array:
    """
    Calculate % change since ath
    :param np.array high: input series
    :return: array of float
    """
    return utils.change_since_mark(
        series=high,
        marked=utils.is_ath(high)
    )


def f10(high: np.array, period: int) -> np.array:
    """
    Calculate % change since highest in period
    :param np.array high: input series
    :param int period: rolling window size
    :return: array of float
    """
    return utils.change_since_mark(
        series=high,
        marked=utils.is_extremum_bars_periods(
            series=high,
            period=period,
            maximum=True
        )
    )


def f11(low: np.array, period: int) -> np.array:
    """
    Calculate % change since lowest in period
    :param np.array low: input series
    :param int period: rolling window size
    :return: array of float
    """
    return utils.change_since_mark(
        series=low,
        marked=utils.is_extremum_bars_periods(
            series=low,
            period=period,
            maximum=False
        )
    )


__all__ = (
    "PREFIX",
    "calc_all",
    *[key for key in locals().keys() if key[1:].isdigit()]
)
