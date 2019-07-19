from typing import NamedTuple

import numpy as np
from scipy.ndimage.interpolation import shift
from tulipindicators import ti

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def stoch(high: np.array, low: np.array, close: np.array, k_period: int,
          k_slowing_period: int, d_period: int) -> NamedTuple:
    """Calculates Stochastic indicator

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param k_period: STOCH k period
    :param k_slowing_period: STOCH k slowing period
    :param d_period: STOCH d period
    :return: STOCH outputs: stoch_k, stoch_d
    """
    return ti.stoch(high, low, close, k_period, k_slowing_period, d_period)


# Feature functions region:


def f1(high: np.array, low: np.array, close: np.array, k_period: int,
       k_slowing_period: int, d_period: int) -> np.array:
    """Extracts K value series from STOCH

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param k_period: STOCH k period
    :param k_slowing_period: STOCH k slowing period
    :param d_period: STOCH d period
    :return: series of K values
    """
    return stoch(high, low, close, k_period, k_slowing_period,
                 d_period).stoch_k


def f2(high: np.array, low: np.array, close: np.array, k_period: int,
       k_slowing_period: int, d_period: int) -> np.array:
    """Extracts D value series from STOCH

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param k_period: STOCH k period
    :param k_slowing_period: STOCH k slowing period
    :param d_period: STOCH d period
    :return: series of D values
    """
    return stoch(high, low, close, k_period, k_slowing_period,
                 d_period).stoch_d


def f3(high: np.array, low: np.array, close: np.array, k_period: int,
       k_slowing_period: int, d_period: int) -> np.array:
    """Extracts difference between K and D

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param k_period: STOCH k period
    :param k_slowing_period: STOCH k slowing period
    :param d_period: STOCH d period
    :return: series of difference between K and D
    """
    outputs = stoch(high, low, close, k_period, k_slowing_period, d_period)
    return outputs.stoch_k - outputs.stoch_d


def f4(high: np.array,
       low: np.array,
       close: np.array,
       k_period: int,
       k_slowing_period: int,
       d_period: int,
       n: int = 1) -> np.array:
    """Extracts STOCH K changes.

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param k_period: STOCH k period
    :param k_slowing_period: STOCH k slowing period
    :param d_period: STOCH d period
    :param n: no for shift
    :return: series of STOCH K changes
    """
    output = stoch(high, low, close, k_period, k_slowing_period,
                   d_period).stoch_k
    return output - shift(output, n, cval=np.nan)


def f5(high: np.array,
       low: np.array,
       close: np.array,
       k_period: int,
       k_slowing_period: int,
       d_period: int,
       n: int = 1) -> np.array:
    """Extracts STOCH D changes.

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param k_period: STOCH k period
    :param k_slowing_period: STOCH k slowing period
    :param d_period: STOCH d period
    :param n: no for shift
    :return: series of STOCH D changes
    """
    output = stoch(high, low, close, k_period, k_slowing_period,
                   d_period).stoch_d
    return output - shift(output, n, cval=np.nan)


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, indicator='stoch')
inputs = get_inputs(features_list)
