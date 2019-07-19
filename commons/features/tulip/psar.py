import numpy as np
from scipy.ndimage.interpolation import shift
from tulipindicators import ti

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def psar(high: np.array,
         low: np.array,
         acceleration_factor_step: float,
         acceleration_factor_maximum: float) -> np.array:
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


def f1(high: np.array,
       low: np.array,
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


def f2(series: np.array,
       high: np.array,
       low: np.array,
       acceleration_factor_step: float,
       acceleration_factor_maximum: int) -> np.array:
    """Extracts difference between PSAR values and series

    :param series: series of real
    :param high: series of bar high
    :param low: series of bar low
    :param acceleration_factor_step: PSAR acceleration factor step
    :param acceleration_factor_maximum: PSAR acceleration factor maximum
    :return: series difference between PSAR values and series
    """
    output = psar(
        high, low, acceleration_factor_step, acceleration_factor_maximum)
    return output - series


def f3(series: np.array,
       high: np.array,
       low: np.array,
       acceleration_factor_step: float,
       acceleration_factor_maximum: int,
       n: int = 1) -> np.array:
    """Extracts series of PSAR changes.

    :param series: series of real
    :param high: series of bar high
    :param low: series of bar low
    :param acceleration_factor_step: PSAR acceleration factor step
    :param acceleration_factor_maximum: PSAR acceleration factor maximum
    :param n: no for shift
    :return: series of PSAR changes.
    """
    output = psar(
        high, low, acceleration_factor_step, acceleration_factor_maximum)
    return output - shift(output, n, cval=np.nan)


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, indicator='psar')
inputs = get_inputs(features_list)
