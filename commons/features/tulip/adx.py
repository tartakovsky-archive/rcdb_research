import numpy as np
from tulipindicators import ti

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def adx(high: np.array, low: np.array, period: int) -> np.array:
    """Сalculates Average Directional Movement Index indicator

    :param high: series of bar high
    :param low: series of bar low
    :param period: ADX period
    :return: series of ADX values
    """
    return ti.adx(high, low, period)


# Feature functions region:


def f1(high: np.array, low: np.array, period: int) -> np.array:
    """Extracts series of ADX indicator values

    :param high: series of bar high
    :param low: series of bar low
    :param period: ADX period
    :return: series of ADX values
    """
    return adx(high, low, period)


# Calc all region:

features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, prefix='adx')
inputs = get_inputs(features_list)
