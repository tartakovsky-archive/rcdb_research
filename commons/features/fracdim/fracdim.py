import numpy as np

from . import utils
from ..utils import generate_calc_all

PREFIX = "fracdim"

FEATURE_FUNCS = {}


def register_feature(features_dict):
    def inner(func):
        features_dict[func.__name__] = func
        return func
    return inner


register_feature_fracdim = lambda func: register_feature(FEATURE_FUNCS)(func)


@register_feature_fracdim
def higuchi_fd(series: np.array, window: int, kmax: int = 10) -> np.array:
    """
    Higuchi Fractal Dimension
    :param series: input data
    :param window: rolling window size
    :param kmax: (optional, default: 10) Maximum delay/offset (in number of samples)
    :return:
    """
    return utils.apply_to_window(
        func=utils.higuchi_fd,
        series=series,
        window=window,
        kmax=kmax
    )


@register_feature_fracdim
def katz_fd(series: np.array, window: int) -> np.array:
    """
    Katz Fractal Dimension
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return utils.apply_to_window(
        func=utils.katz_fd,
        series=series,
        window=window,
    )


@register_feature_fracdim
def petrosian_fd(series: np.array, window: int) -> np.array:
    """
    Petrosian fractal dimension
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return utils.apply_to_window(
        func=utils.petrosian_fd,
        series=series,
        window=window,
    )


__all__ = ("FEATURE_FUNCS", "PREFIX", "calc_all", *FEATURE_FUNCS.keys())

calc_all = generate_calc_all(PREFIX, FEATURE_FUNCS)
