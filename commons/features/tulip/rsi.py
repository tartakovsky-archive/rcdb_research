from tulipindicators import ti
import pandas as pd
import numpy as np


from .utils import cache, calc_all_helper


@cache
def rsi(series: pd.Series, period: int) -> np.array:
    """Calculates Relative Strength Index indicator

    :param series: series of real
    :param period: RSI period
    :return: series of RSI values
    """
    return ti.rsi(series, period)


# Feature functions region:


def f1(series: pd.Series, period: int) -> np.array:
    """Extracts series of RSI indicator values

    :param series: series of real
    :param period: RSI period
    :return: series of RSI values
    """
    return rsi(series, period)


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, prefix='rsi')
