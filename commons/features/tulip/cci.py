import numpy as np
import pandas as pd
from tulipindicators import ti

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> np.array:
    """Сalculates Commodity Channel Index indicator

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param period: CCI period
    :return: series of CCI values
    """
    return ti.cci(high, low, close, period)


# Feature functions region:


def f1(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> np.array:
    """Extracts series of CCI indicator values

    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :param period: CCI period
    :return: series of CCI values
    """
    return cci(high, low, close, period)


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, prefix='cci')
inputs = get_inputs(features_list)
