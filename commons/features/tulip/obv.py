import numpy as np
import pandas as pd
from tulipindicators import ti

from commons.features.tulip.utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def obv(series: pd.Series, volume: pd.Series) -> np.array:
    """Calculates On Balance Volume indicator

    :param series: series of real
    :param volume: series of bar volume
    :return: series of OBV values
    """
    return ti.obv(series, volume)


# Feature functions region:


def f1(series: pd.Series, volume: pd.Series) -> np.array:
    """Extracts series of OBV indicator values

    :param series: series of real
    :param volume: series of bar volume
    :return: series of OBV values
    """
    return obv(series, volume)


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, prefix='obv')
inputs = get_inputs(features_list)
