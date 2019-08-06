import numpy as np
from scipy import stats as st
from scipy import ndimage

from ..utils import rolling_window, feature_registrator_factory, generate_calc_all

PREFIX = "stats"
FEATURE_FUNCS = {}

register_feature_stats = feature_registrator_factory(FEATURE_FUNCS)


@register_feature_stats
def gmean(series: np.array, window: int) -> np.array:
    """
    Compute the geometric mean
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return st.gmean(rolling_window(series, window), axis=1)


@register_feature_stats
def hmean(series: np.array, window: int) -> np.array:
    """
    Calculate the harmonic mean
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return st.hmean(rolling_window(series, window), axis=1)


@register_feature_stats
def tmean(series: np.array, window: int) -> np.array:
    """
    Compute the trimmed mean
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return np.array([st.tmean(x) for x in rolling_window(series, window)])


@register_feature_stats
def cmean(series: np.array, window: int) -> np.array:
    """
    Compute the circular mean
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return st.circmean(rolling_window(series, window), axis=1)


@register_feature_stats
def hdmedian(series: np.array, window: int) -> np.array:
    """
    Compute the Harrell-Davis estimate of the median
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return st.mstats.hdmedian(rolling_window(series, window), axis=1)


@register_feature_stats
def median(series: np.array, window: int) -> np.array:
    """
    Compute median
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return np.array([ndimage.median(x) for x in rolling_window(series, window)])


@register_feature_stats
def fkurtosis(series: np.array, window: int) -> np.array:
    """
    Compute the Fisher kurtosis
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return st.kurtosis(rolling_window(series, window), axis=1)


@register_feature_stats
def pkurtosis(series: np.array, window: int) -> np.array:
    """
    Compute the Pearson kurtosis
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return st.kurtosis(rolling_window(series, window), fisher=False, axis=1)


@register_feature_stats
def skewness(series: np.array, window: int) -> np.array:
    """

    :param series: input data
    :param window: rolling window size
    :return:
    """
    return st.skew(rolling_window(series, window), axis=1)


@register_feature_stats
def variance(series: np.array, window: int) -> np.array:
    """
    Compute variance
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return np.array([ndimage.variance(x) for x in rolling_window(series, window)])


@register_feature_stats
def stddev(series: np.array, window: int) -> np.array:
    """
    Compute standard deviation
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return np.array([ndimage.standard_deviation(x) for x in rolling_window(series, window)])


__all__ = ("calc_all", "PREFIX", "FEATURE_FUNCS", *FEATURE_FUNCS.keys())

calc_all = generate_calc_all(PREFIX, FEATURE_FUNCS)
