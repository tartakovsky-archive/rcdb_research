import numpy as np

from .utils import _n_consecutive, _cond_after_n_bars


def higher_after_n_bars(series: np.array, n: int) -> np.array:
    """
    If value of current bar lower than value of bar after n bars
    :param series: input series
    :param n: bars between current and next
    :return:
    """
    return _cond_after_n_bars(series, n, lambda current, after_n_bars: current < after_n_bars)


def lower_after_n_bars(series: np.array, n: int) -> np.array:
    """
    If value of current bar higher than value of bar after n bars
    :param series: input series
    :param n: bars between current and next
    :return:
    """
    return _cond_after_n_bars(series, n, lambda current, after_n_bars: current > after_n_bars)


def n_consecutive_up(series: np.array, n: int) -> np.array:
    """
    1 if direction of next n bars is 1
    :param series: input series
    :param n: length of bars series
    :return:
    """
    return _n_consecutive(series, n, lambda arr: arr == 1)


def n_consecutive_down(series: np.array, n: int) -> np.array:
    """
    1 if direction of next n bars is -1
    :param series: input series
    :param n: length of bars series
    :return:
    """
    return _n_consecutive(series, n, lambda arr: arr == -1)
