import numpy as np
from scipy.ndimage.interpolation import shift
from tulipindicators import ti

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def obv(series: np.array, volume: np.array) -> np.array:
    """Calculates On Balance Volume indicator

    :param series: series of real
    :param volume: series of bar volume
    :return: series of OBV values
    """
    return ti.obv(series, volume)


# Feature functions region:


def f1(series: np.array, volume: np.array) -> np.array:
    """Extracts series of OBV indicator values

    :param series: series of real
    :param volume: series of bar volume
    :return: series of OBV values
    """
    return obv(series, volume)


def f2(series: np.array, volume: np.array, n: int = 1) -> np.array:
    """Extracts series of OBV changes.

    :param series: series of real
    :param volume: series of bar volume
    :param n: no for shift
    :return: series of OBV changes
    """
    output = obv(series, volume)
    return output - shift(output, n, cval=np.nan)


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, indicator='obv')
inputs = get_inputs(features_list)
