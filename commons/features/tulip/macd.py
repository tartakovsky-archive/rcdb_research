from typing import NamedTuple

import numpy as np
from scipy.ndimage.interpolation import shift
from tulipindicators import ti

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def macd(series: np.array, short_period: int, long_period: int,
         signal_period: int) -> NamedTuple:
    """Calculates Moving Average Convergence/Divergence indicator

    :param series: series of real
    :param short_period: short period
    :param long_period: long period
    :param signal_period: signal period
    :return: MACD outputs: macd, macd_signal, macd_histogram
    """
    return ti.macd(series, short_period, long_period, signal_period)


# Feature functions region


def f1(series: np.array, short_period: int, long_period: int,
       signal_period: int) -> np.array:
    """Extracts series of MACD indicator values

    :param series: series of real
    :param short_period: MACD short period
    :param long_period: MACD long period
    :param signal_period: MACD signal period
    :return: series of MACD values
    """
    return macd(series, short_period, long_period, signal_period).macd


def f2(series: np.array, short_period: int, long_period: int,
       signal_period: int) -> np.array:
    """Extracts series of Signal EMA by MACD values

    :param series: series of real
    :param short_period: MACD short period
    :param long_period: MACD long period
    :param signal_period: MACD signal period
    :return: series of Signal EMA by MACD values
    """
    return macd(series, short_period, long_period, signal_period).macd_signal


def f3(series: np.array, short_period: int, long_period: int,
       signal_period: int) -> np.array:
    """Extracts series of MACD Histogram values

    :param series: series of real
    :param short_period: MACD short period
    :param long_period: MACD long period
    :param signal_period: MACD signal period
    :return: series of MACD Histogram values
    """
    return macd(series, short_period, long_period,
                signal_period).macd_histogram


def f4(series: np.array, short_period: int, long_period: int,
       signal_period: int) -> np.array:
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


def f5(series: np.array,
       short_period: int,
       long_period: int,
       signal_period: int,
       n: int = 1) -> np.array:
    """Extracts series of MACD changes.

    :param series: series of real
    :param short_period: MACD short period
    :param long_period: MACD long period
    :param signal_period: MACD signal period
    :param n: no for shift
    :return: series of MACD changes.
    """
    output = macd(series, short_period, long_period, signal_period).macd
    return output - shift(output, n, cval=np.nan)


def f6(series: np.array,
       short_period: int,
       long_period: int,
       signal_period: int,
       n: int = 1) -> np.array:
    """Extracts series of MACD Signal changes.

    :param series: series of real
    :param short_period: MACD short period
    :param long_period: MACD long period
    :param signal_period: MACD signal period
    :param n: no for shift
    :return: series of MACD Signal changes.
    """
    output = macd(series, short_period, long_period, signal_period).macd_signal
    return output - shift(output, n, cval=np.nan)


def f7(series: np.array,
       short_period: int,
       long_period: int,
       signal_period: int,
       n: int = 1) -> np.array:
    """Extracts series of MACD Histogram changes.

    :param series: series of real
    :param short_period: MACD short period
    :param long_period: MACD long period
    :param signal_period: MACD signal period
    :param n: no for shift
    :return: series of MACD Histogram changes.
    """
    output = macd(series, short_period, long_period,
                  signal_period).macd_histogram
    return output - shift(output, n, cval=np.nan)


# Helpers region


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, indicator='macd')
inputs = get_inputs(features_list)
