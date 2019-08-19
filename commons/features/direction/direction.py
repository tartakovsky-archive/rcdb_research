import numpy as np
from ..utils import rolling_window, feature_registrator_factory, generate_calc_all


PREFIX = "direction"
FEATURE_FUNCS = {}


@feature_registrator_factory(FEATURE_FUNCS)
def sum_of_direction(series: np.array, window: int) -> np.array:
    """
    Calculate sum of last bars directions
    :param series: input series
    :param window: rolling window size
    :return:
    """
    shifted = np.hstack(([series[0]], series[:-1]))
    direction = np.zeros(series.size)

    is_up = (series > shifted)
    is_down = (series < shifted)

    direction[is_up] = 1
    direction[is_down] = -1

    return np.sum(rolling_window(direction, window), axis=1)


__all__ = ("FEATURE_FUNCS", "PREFIX", "calc_all", *FEATURE_FUNCS.keys())

calc_all = generate_calc_all(PREFIX, FEATURE_FUNCS)
