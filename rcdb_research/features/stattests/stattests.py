from . import utils
from ..utils import apply_to_window
import numpy as np


def kstest(series: np.array, offset: int, window: int) -> np.array:
    return apply_to_window(
        func=utils.kstest_1,
        series=series,
        offset=offset,
        window=window + offset
    )


def ks_2samp(series: np.array, offset: int, window: int) -> np.array:
    return apply_to_window(
        func=utils.ks_2samp_1,
        series=series,
        offset=offset,
        window=window + offset
    )


def epps_singleton_2samp(series: np.array, offset: int, window: int):
    return apply_to_window(
        func=utils.epps_singleton_2samp_1,
        series=series,
        offset=offset,
        window=window + offset
    )


def mannwhitneyu(series: np.array, offset: int, window: int) -> np.array:
    return apply_to_window(
        func=utils.mannwhitneyu_1,
        series=series,
        offset=offset,
        window=window + offset
    )


def wilcoxon(series: np.array, offset: int, window: int) -> np.array:
    return apply_to_window(
        func=utils.wilcoxon_1,
        series=series,
        offset=offset,
        window=window + offset
    )


def anderson_ksamp(series: np.array, offset: int, window: int) -> np.array:
    return apply_to_window(
        func=utils.anderson_ksamp_1,
        series=series,
        offset=offset,
        window=window + offset
    )


def adfuller(series, window):
    return apply_to_window(
        func=utils.adfuller_1,
        series=series,
        window=window
    )


def chisquare(series, offset, window, bins):
    return apply_to_window(
        func=utils.chisquare_1,
        series=series,
        offset=offset,
        bins=bins,
        window=window + offset
    )


def power_divergence(series, offset, window, bins):
    return apply_to_window(
        func=utils.power_divergence_1,
        series=series,
        offset=offset,
        bins=bins,
        window=window + offset
    )


def runs_test(series, offset, window):
    return apply_to_window(
        func=utils.runs_test_1,
        series=series,
        offset=offset,
        window=window + offset
    )
