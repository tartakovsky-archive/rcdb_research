import numpy as np
from scipy.ndimage.interpolation import shift
from tulipindicators import ti

from rcdb_research.features.tulip._utils import cache, calc_all_helper
from rcdb_research.features.utils import get_inputs


@cache
def rsi(series: np.array, period: int) -> np.array:
    """Calculates Relative Strength Index indicator

    :param series: series of real
    :param period: RSI period
    :return: series of RSI values
    """
    return ti.rsi(series, period)


# Feature functions region


def f1(series: np.array, period: int) -> np.array:
    """Extracts series of RSI indicator values

    :param series: series of real
    :param period: RSI period
    :return: series of RSI values
    """
    return rsi(series, period)


def f2(series: np.array, period: int, n: int = 1) -> np.array:
    """Extracts series of RSI changes.

    :param series: series of real
    :param period: RSI period
    :param n: no for shift
    :return: series of RSI changes
    """
    output = rsi(series, period)
    return output - shift(output, n, cval=np.nan)


# Helpers region


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, indicator='rsi')
inputs = get_inputs(features_list)
