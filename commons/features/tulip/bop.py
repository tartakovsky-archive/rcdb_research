import numpy as np
from scipy.ndimage.interpolation import shift
from tulipindicators import ti

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def bop(open: np.array, high: np.array, low: np.array,
        close: np.array) -> np.array:
    """Сalculates Balance Of Power indicator

    :param open: series of bar open
    :param high: series of bar high
    :param low: series of bar low
    :param close: bar close series
    :return: series of BOP values
    """
    return ti.bop(open, high, low, close)


# Feature functions region


def f1(open: np.array, high: np.array, low: np.array,
       close: np.array) -> np.array:
    """Extracts series of BOP indicator values

    :param open: series of bar open
    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :return: series of BOP values
    """
    return bop(open, high, low, close)


def f2(open: np.array,
       high: np.array,
       low: np.array,
       close: np.array,
       n: int = 1) -> np.array:
    """Extracts series of BOP changes.

    :param open: series of bar open
    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param n: no for shift
    :return: series of BOP changes
    """
    output = bop(open, high, low, close)
    return output - shift(output, n, cval=np.nan)


# Helpers region


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, indicator='bop')
inputs = get_inputs(features_list)
