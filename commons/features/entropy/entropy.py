import numpy as np
from tsfresh.feature_extraction.feature_calculators import binned_entropy as tsf_binned_entropy

from . import utils
from ..utils import generate_calc_all, feature_registrator_factory, apply_to_window

PREFIX = "entropy"
FEATURE_FUNCS = {}

register_feature_entropy = feature_registrator_factory(FEATURE_FUNCS)


@register_feature_entropy
def app_entropy(series: np.array, window: int) -> np.array:
    """
    Approximate Entropy
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=utils.app_entropy,
        series=series,
        window=window
    )


@register_feature_entropy
def sample_entropy(series: np.array, window: int) -> np.array:
    """
    Sample Entropy
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=utils.sample_entropy,
        series=series,
        window=window
    )


@register_feature_entropy
def spectral_entropy(series: np.array, window: int, sf: float) -> np.array:
    """
    Spectral Entropy
    :param series: input data
    :param window: rolling window size
    :param sf:
    :return:
    """
    return apply_to_window(
        func=utils.spectral_entropy,
        series=series,
        window=window,
        sf=sf
    )


@register_feature_entropy
def svd_entropy(series: np.array, window: int) -> np.array:
    """
    Singular Value Decomposition entropy
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=utils.svd_entropy,
        series=series,
        window=window
    )


@register_feature_entropy
def perm_entropy(series: np.array, window: int) -> np.array:
    """
    Permutation Entropy
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=utils.perm_entropy,
        series=series,
        window=window
    )


@register_feature_entropy
def binned_entropy(series: np.array, window: int, max_bins: int) -> np.array:
    """
    First bins the values of x into max_bins equidistant bins in each window
    :param series: input data
    :param window: rolling window size
    :param max_bins: the maximal number of bins
    :return:
    """
    return apply_to_window(
        func=tsf_binned_entropy,
        series=series,
        window=window,
        max_bins=max_bins
    )


__all__ = ("calc_all", "PREFIX", "FEATURE_FUNCS", *FEATURE_FUNCS.keys())

calc_all = generate_calc_all(PREFIX, FEATURE_FUNCS)
