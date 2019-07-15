from typing import NamedTuple

import numpy as np
import pandas as pd
from tulipindicators import ti

from commons.features.tulip.utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def stoch(high: pd.Series,
          low: pd.Series,
          close: pd.Series,
          k_period: int,
          k_slowing_period: int,
          d_period: int) -> NamedTuple:
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


def f1(high: pd.Series,
       low: pd.Series,
       close: pd.Series,
       k_period: int,
       k_slowing_period: int,
       d_period: int) -> np.array:
    """Extracts K value series from STOCH

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param k_period: STOCH k period
    :param k_slowing_period: STOCH k slowing period
    :param d_period: STOCH d period
    :return: series of K values
    """
    return stoch(high, low, close, k_period, k_slowing_period, d_period).stoch_k


def f2(high: pd.Series,
       low: pd.Series,
       close: pd.Series,
       k_period: int,
       k_slowing_period: int,
       d_period: int) -> np.array:
    """Extracts D value series from STOCH

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param k_period: STOCH k period
    :param k_slowing_period: STOCH k slowing period
    :param d_period: STOCH d period
    :return: series of D values
    """
    return stoch(high, low, close, k_period, k_slowing_period, d_period).stoch_d


def f3(high: pd.Series,
       low: pd.Series,
       close: pd.Series,
       k_period: int,
       k_slowing_period: int,
       d_period: int) -> np.array:
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


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, prefix='stoch')
inputs = get_inputs(features_list)
