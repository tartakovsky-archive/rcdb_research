import numpy as np
from scipy.ndimage.interpolation import shift
from tulipindicators import ti

from rcdb_research.features.tulip._utils import cache, calc_all_helper
from rcdb_research.features.utils import get_inputs


@cache
def cci(high: np.array, low: np.array, close: np.array,
        period: int) -> np.array:
    """Сalculates Commodity Channel Index indicator

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param period: CCI period
    :return: series of CCI values
    """
    return ti.cci(high, low, close, period)


# Feature functions region


def f1(high: np.array, low: np.array, close: np.array,
       period: int) -> np.array:
    """Extracts series of CCI indicator values

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param period: CCI period
    :return: series of CCI values
    """
    return cci(high, low, close, period)


def f2(high: np.array, low: np.array, close: np.array, period: int,
       n: int = 1) -> np.array:
    """Extracts series of CCI changes.

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param period: CCI period
    :param n: no for shift
    :return: series of CCI changes
    """
    output = cci(high, low, close, period)
    return output - shift(output, n, cval=np.nan)


# Helpers region


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, indicator='cci')
inputs = get_inputs(features_list)
