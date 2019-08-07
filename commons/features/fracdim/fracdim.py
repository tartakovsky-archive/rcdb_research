import numpy as np

from . import utils
from ..utils import generate_calc_all, apply_to_window, feature_registrator_factory

PREFIX = "fracdim"
FEATURE_FUNCS = {}

register_feature_fracdim = feature_registrator_factory(FEATURE_FUNCS)


@register_feature_fracdim
def higuchi_fd(series: np.array, window: int, kmax: int = 10) -> np.array:
    """
    Higuchi Fractal Dimension
    :param series: input data
    :param window: rolling window size
    :param kmax: (optional, default: 10) Maximum delay/offset (in number of samples)
    :return:
    """
    return apply_to_window(
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
    return apply_to_window(
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
    return apply_to_window(
        func=utils.petrosian_fd,
        series=series,
        window=window,
    )


__all__ = ("FEATURE_FUNCS", "PREFIX", "calc_all", *FEATURE_FUNCS.keys())

calc_all = generate_calc_all(PREFIX, FEATURE_FUNCS)
