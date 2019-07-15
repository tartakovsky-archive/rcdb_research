from typing import NamedTuple

from tulipindicators import ti
import pandas as pd
import numpy as np

from .utils import cache, calc_all_helper, get_inputs


@cache
def macd(series: pd.Series, short_period: int, long_period: int, signal_period: int) -> NamedTuple:
    """Calculates Moving Average Convergence/Divergence indicator

    :param series: series of real
    :param short_period: short period
    :param long_period: long period
    :param signal_period: signal period
    :return: BBANDS outputs: bbands_upper, bbands_middle, bbands_lower
    """
    return ti.macd(series, short_period, long_period, signal_period)


# Feature functions region:


def f1(series: pd.Series, short_period: int, long_period: int, signal_period: int) -> np.array:
    """Extracts series of MACD indicator values

    :param series: series of real
    :param short_period: MACD short period
    :param long_period: MACD long period
    :param signal_period: MACD signal period
    :return: series of MACD values
    """
    return macd(series, short_period, long_period, signal_period).macd


def f2(series: pd.Series, short_period: int, long_period: int, signal_period: int) -> np.array:
    """Extracts series of Signal EMA by MACD values

    :param series: series of real
    :param short_period: MACD short period
    :param long_period: MACD long period
    :param signal_period: MACD signal period
    :return: series of Signal EMA by MACD values
    """
    return macd(series, short_period, long_period, signal_period).macd_signal


def f3(series: pd.Series, short_period: int, long_period: int, signal_period: int) -> np.array:
    """Extracts series of MACD Histogram values

    :param series: series of real
    :param short_period: MACD short period
    :param long_period: MACD long period
    :param signal_period: MACD signal period
    :return: series of MACD Histogram values
    """
    return macd(series, short_period, long_period, signal_period).macd_histogram


def f4(series: pd.Series, short_period: int, long_period: int, signal_period: int) -> np.array:
    """Extracts difference between MACD values and Signal EMA by MACD values

    :param series: series of real
    :param short_period: MACD short period
    :param long_period: MACD long period
    :param signal_period: MACD signal period
    :return: series of difference between MACD values and Signal EMA by MACD
    values
    """
    outputs = macd(series, short_period, long_period, signal_period)
    return abs(outputs.macd - outputs.macd_signal)


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, prefix='macd')
inputs = get_inputs(features_list)
