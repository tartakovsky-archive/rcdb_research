import numpy as np
import pandas as pd
from tulipindicators import ti

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def roc(series: pd.Series, period: int) -> np.array:
    """Calculates Rate Of Change indicator

    :param series: series of real
    :param period: ROC period
    :return: series of ROC values
    """
    return ti.roc(series, period)


# Feature functions region:


def f1(series: pd.Series, period: int) -> np.array:
    """Extracts series of ROC indicator values

    :param series: series of real
    :param period: ROC period
    :return: series of OBV values
    """
    return roc(series, period)


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, prefix='roc')
inputs = get_inputs(features_list)
