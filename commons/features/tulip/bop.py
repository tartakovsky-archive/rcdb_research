from tulipindicators import ti
import pandas as pd
import numpy as np

from .utils import cache, calc_all_helper, get_inputs


@cache
def bop(open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series) -> np.array:
    """Сalculates Balance Of Power indicator

    :param open: series of bar open
    :param high: series of bar high
    :param low: series of bar low
    :param close: bar close series
    :return: series of BOP values
    """
    return ti.bop(open, high, low, close)


# Feature functions region:


def f1(open: pd.Series,
       high: pd.Series,
       low: pd.Series,
       close: pd.Series) -> np.array:
    """Extracts series of BOP indicator values

    :param open: series of bar open
    :param high: series of bar high
    :param low: series of bar low
    :param close: series of bar close
    :return: series of BOP values
    """
    return bop(open, high, low, close)


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, prefix='bop')
inputs = get_inputs(features_list)
