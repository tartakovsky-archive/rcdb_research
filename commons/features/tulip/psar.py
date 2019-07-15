import numpy as np
import pandas as pd
from tulipindicators import ti

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def psar(high: pd.Series,
         low: pd.Series,
         acceleration_factor_step: float,
         acceleration_factor_maximum: int) -> np.array:
    """Calculates Parabolic SAR indicator

    :param high: series of bar high
    :param low: series of bar low
    :param acceleration_factor_step: PSAR acceleration factor step
    :param acceleration_factor_maximum: PSAR acceleration factor maximum
    :return: series of PASR values
    """
    return ti.psar(
        high, low, acceleration_factor_step, acceleration_factor_maximum)


# Feature functions region:


def f1(high: pd.Series,
       low: pd.Series,
       acceleration_factor_step: float,
       acceleration_factor_maximum: int) -> np.array:
    """Extracts series of PSAR indicator values

    :param high: series of bar high
    :param low: series of bar low
    :param acceleration_factor_step: PSAR acceleration factor step
    :param acceleration_factor_maximum: PSAR acceleration factor maximum
    :return: series of PASR values
    """
    return psar(
        high, low, acceleration_factor_step, acceleration_factor_maximum)


def f2(open: pd.Series,
       high: pd.Series,
       low: pd.Series,
       acceleration_factor_step: float,
       acceleration_factor_maximum: int) -> np.array:
    """Extracts difference between PSAR values and open prices

    :param open: series of bar open
    :param high: series of bar high
    :param low: series of bar low
    :param acceleration_factor_step: PSAR acceleration factor step
    :param acceleration_factor_maximum: PSAR acceleration factor maximum
    :return: series difference between PSAR values and open prices
    """
    output = psar(
        high, low, acceleration_factor_step, acceleration_factor_maximum)
    return output - open


def f3(high: pd.Series,
       low: pd.Series,
       acceleration_factor_step: float,
       acceleration_factor_maximum: int) -> np.array:
    """Extracts difference between PSAR values and high prices

    :param high: series of bar high
    :param low: series of bar low
    :param acceleration_factor_step: PSAR acceleration factor step
    :param acceleration_factor_maximum: PSAR acceleration factor maximum
    :return: series difference between PSAR values and high prices
    """
    output = psar(
        high, low, acceleration_factor_step, acceleration_factor_maximum)
    return output - high


def f4(high: pd.Series,
       low: pd.Series,
       acceleration_factor_step: float,
       acceleration_factor_maximum: int) -> np.array:
    """Extracts difference between PSAR values and open prices

    :param high: series of bar high
    :param low: series of bar low
    :param acceleration_factor_step: PSAR acceleration factor step
    :param acceleration_factor_maximum: PSAR acceleration factor maximum
    :return: series difference between PSAR values and low prices
    """
    output = psar(
        high, low, acceleration_factor_step, acceleration_factor_maximum)
    return output - low


def f5(high: pd.Series,
       low: pd.Series,
       close: pd.Series,
       acceleration_factor_step: float,
       acceleration_factor_maximum: int) -> np.array:
    """Extracts difference between PSAR values and open prices

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param acceleration_factor_step: PSAR acceleration factor step
    :param acceleration_factor_maximum: PSAR acceleration factor maximum
    :return: series difference between PSAR values and close prices
    """
    output = psar(
        high, low, acceleration_factor_step, acceleration_factor_maximum)
    return output - close


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, prefix='psar')
inputs = get_inputs(features_list)
