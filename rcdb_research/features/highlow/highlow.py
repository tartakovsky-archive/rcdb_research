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
            f"{PREFIX}_bars_in_drawdown": bars_in_drawdown(close),
            f"{PREFIX}_bars_in_runup": bars_in_runup(close),
        },
        index=data.index
    )

    for ps in param_set:
        period = ps["period"]

        df[f"{PREFIX}_is_local_high_{period}"] = is_local_high(high, period)
        df[f"{PREFIX}_is_local_low_{period}"] = is_local_low(low, period)
        df[f"{PREFIX}_bars_since_local_high_{period}"] = bars_since_local_high(high, period)
        df[f"{PREFIX}_bars_since_local_low_{period}"] = bars_since_local_low(low, period)
        df[f"{PREFIX}_change_since_local_high_{period}"] = change_since_local_high(high, period)
        df[f"{PREFIX}_change_since_local_low_{period}"] = change_since_local_low(low, period)

    return df


def is_local_high(high: np.array, period: int) -> np.array:
    """
    Check if the high is a highest in period
    :param np.array high: input series
    :param int period: rolling window size
    :return: array of 0 and 1 for each row
    """
    return utils.is_extremum_bars_periods(series=high, period=period, maximum=True)


def is_local_low(low: np.array, period: int) -> np.array:
    """
    Check if the low is a lowest in period
    :param np.array low: input series
    :param int period: window size
    :return: array of 0 and 1 for each row
    """
    return utils.is_extremum_bars_periods(series=low, period=period, maximum=False)


def bars_since_local_high(high: np.array, period: int) -> np.array:
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


def bars_since_local_low(low: np.array, period: int) -> np.array:
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


def bars_in_drawdown(close: np.array) -> np.array:
    """
    Calculate numbers of bars in drawdown
    :param np.array close: input series
    :return: array of seconds
    """
    return utils.bars_in_marked(
        (utils.pct_change(series=close) < 0) * 1
    )


def bars_in_runup(close: np.array) -> np.array:
    """
    Calculate numbers of bars in run up
    :param np.array close: input series
    :return: array of seconds
    """
    return utils.bars_in_marked(
        (utils.pct_change(series=close) > 0) * 1
    )


def change_since_local_high(high: np.array, period: int) -> np.array:
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


def change_since_local_low(low: np.array, period: int) -> np.array:
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
