from typing import NamedTuple

import numpy as np
from tulipindicators import ti

from commons.features.tulip._utils import cache, calc_all_helper
from commons.features.utils import get_inputs


@cache
def bbands(close: np.array, period: int, stddev: float) -> NamedTuple:
    """Calculates Bollinger Bands indicator

    :param close: bar close series
    :param period: BBANDS period
    :param stddev: BBANDS stddev
    :return: BBANDS outputs: bbands_upper, bbands_middle, bbands_lower
    """
    return ti.bbands(close, period, stddev)


# Feature functions region:


def f1(close: np.array, period: int, stddev: float) -> np.array:
    """Extracts Upper Band value series

    :param close: series of bar close
    :param period: BBANDS period
    :param stddev: BBANDS stddev
    :return: series of Upper Band values
    """
    return bbands(close, period, stddev).bbands_upper


def f2(close: np.array, period: int, stddev: float) -> np.array:
    """Extracts Middle Band value series

    :param close: series of bar close
    :param period: BBANDS period
    :param stddev: BBANDS stddev
    :return: series of Middle Band values
    """
    return bbands(close, period, stddev).bbands_middle


def f3(close: np.array, period: int, stddev: float) -> np.array:
    """Extracts Lower Band value series

    :param close: series of bar close
    :param period: BBANDS period
    :param stddev: BBANDS stddev
    :return: series of Lower Band values
    """
    return bbands(close, period, stddev).bbands_lower


def f4(close: np.array, period: int, stddev: float) -> np.array:
    """Extracts difference between Upper Band and Lower Band

    :param close: series of bar close
    :param period: BBANDS period
    :param stddev: BBANDS stddev
    :return: series of difference between Upper Band and Lower Band
    """
    outputs = bbands(close, period, stddev)
    return outputs.bbands_upper - outputs.bbands_lower


def f5(close: np.array, period: int, stddev: float) -> np.array:
    """Extracts difference between Upper Band and Middle Band

    :param close: series of bar close
    :param period: BBANDS period
    :param stddev: BBANDS stddev
    :return: series of difference between Upper Band and Middle Band
    """
    outputs = bbands(close, period, stddev)
    return outputs.bbands_upper - outputs.bbands_middle


def f6(close: np.array, period: int, stddev: float) -> np.array:
    """Extracts difference between Middle Band and Lower Band

    :param close: series of bar close
    :param period: BBANDS period
    :param stddev: BBANDS stddev
    :return: series of difference between Middle Band and Lower Band
    """
    outputs = bbands(close, period, stddev)
    return outputs.bbands_middle - outputs.bbands_lower


def f7(series: np.array, close: np.array, period: int, stddev: float) -> np.array:
    """Extracts difference between Upper Band values and series

    :param series: series of real
    :param close: series of bar close
    :param period: BBANDS period
    :param stddev: BBANDS stddev
    :return: series of difference between Upper Band values and series
    """
    outputs = bbands(close, period, stddev)
    return outputs.bbands_upper - series


def f8(series: np.array, close: np.array, period: int, stddev: float) -> np.array:
    """Extracts difference between Middle Band values and series

    :param series: series of real
    :param close: series of bar close
    :param period: BBANDS period
    :param stddev: BBANDS stddev
    :return: series of difference between Middle Band values and series
    """
    outputs = bbands(close, period, stddev)
    return outputs.bbands_middle - series


def f9(series: np.array, close: np.array, period: int, stddev: float) -> np.array:
    """Extracts difference between Lower Band values and series

    :param series: series of real
    :param close: series of bar close
    :param period: BBANDS period
    :param stddev: BBANDS stddev
    :return: series of difference between Lower Band values and series
    """
    outputs = bbands(close, period, stddev)
    return outputs.bbands_lower - series


# Calc all region:


features_list = [value for key, value in locals().items() if key[1:].isdigit()]
calc_all = calc_all_helper(features_list, prefix='bbands')
inputs = get_inputs(features_list)
