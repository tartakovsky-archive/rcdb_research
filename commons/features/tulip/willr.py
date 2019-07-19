from tulipindicators import ti
from scipy.ndimage.interpolation import shift
import numpy as np

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def willr(high: np.array, low: np.array, close: np.array,
          period: int) -> np.array:
    """Сalculates Williams %R indicator

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param period: WILLR period
    :return: series of WILLR values
    """
    return ti.willr(high, low, close, period)


# Feature functions region:


def f1(high: np.array, low: np.array, close: np.array,
       period: int) -> np.array:
    """Extracts series of WILLR indicator values

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param period: WILLR period
    :return: series of WILLR values
    """
    return willr(high, low, close, period)


def f2(high: np.array, low: np.array, close: np.array, period: int,
       n: int = 1) -> np.array:
    """Extracts series of WILLR changes.

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param period: WILLR period
    :param n: no for shift
    :return: series of WILLR changes.
    """
    output = willr(high, low, close, period)
    return output - shift(output, n, cval=np.nan)


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, indicator='willr')
inputs = get_inputs(features_list)
