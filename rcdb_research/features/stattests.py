import os
import contextlib

import numpy as np
import scipy.stats
import statsmodels.tsa.stattools
import statsmodels.sandbox.stats.runs
from statsmodels.distributions.empirical_distribution import ECDF
from numpy_ext import rolling_apply

###########
# Helpers
###########


def kstest_1(series, offset):
    a = series[offset:]
    b = series[:-offset]
    return scipy.stats.kstest(
        rvs=a,
        cdf=ECDF(b),
    ).pvalue


def ks_2samp_1(series, offset):
    a = series[offset:]
    b = series[:-offset]
    return scipy.stats.ks_2samp(
        data1=a,
        data2=b,
    ).pvalue


def epps_singleton_2samp_1(series, offset):
    a = series[offset:]
    b = series[:-offset]
    if (~np.isfinite(a)).sum() + (~np.isfinite(b)).sum() > 0:
        return np.nan
    else:
        return scipy.stats.epps_singleton_2samp(
            x=a,
            y=b,
        ).pvalue


def mannwhitneyu_1(series, offset):
    a = series[offset:]
    b = series[:-offset]
    return scipy.stats.mannwhitneyu(
        x=a,
        y=b,
    ).pvalue


def wilcoxon_1(series, offset):
    a = series[offset:]
    b = series[:-offset]
    return scipy.stats.wilcoxon(
        x=a,
        y=b,
    ).pvalue


def anderson_ksamp_1(series, offset):
    a = series[offset:]
    b = series[:-offset]
    return scipy.stats.anderson_ksamp(
        samples=[a, b]
    ).significance_level


def runs_test_1samp_1(series, cutoff='median'):
    if (~np.isfinite(series)).sum() > 0:
        return np.nan
    a = series

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            return statsmodels.sandbox.stats.runs.runstest_1samp(a, cutoff=cutoff)[1]


def runs_test_2samp_1(series, offset):
    if (~np.isfinite(series)).sum() > 0:
        return np.nan
    a = series[offset:]
    b = series[:-offset]

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            return statsmodels.sandbox.stats.runs.runstest_2samp(a, b)[1]


def power_divergence_1(series, offset, bins):
    if (~np.isfinite(series)).sum() > 0:
        return np.nan

    a = series[offset:]
    b = series[:-offset]
    conc = np.concatenate((a, b))
    m = conc.min()
    M = conc.max()

    return scipy.stats.power_divergence(
        f_obs=np.histogram(a, bins=bins, range=(m, M))[0],
        f_exp=np.histogram(b, bins=bins, range=(m, M))[0]
    ).pvalue


def chisquare_1(series, offset, bins):
    if (~np.isfinite(series)).sum() > 0:
        return np.nan

    a = series[offset:]
    b = series[:-offset]
    conc = np.concatenate((a, b))
    m = conc.min()
    M = conc.max()

    return scipy.stats.chisquare(
        f_obs=np.histogram(a, bins=bins, range=(m, M))[0],
        f_exp=np.histogram(b, bins=bins, range=(m, M))[0]
    ).pvalue


def adfuller_1(series):
    if (~np.isfinite(series)).sum() > 0:
        return np.nan
    else:
        return statsmodels.tsa.stattools.adfuller(series, maxlag=1, regression='c', autolag=None)[1]


###########
# Features
###########
def kstest(series: np.ndarray, offset: int, window: int) -> np.ndarray:
    """
    Calculates kstest feature

    Parameters
    ----------
    series : np.ndarray
        Input series
    offset : int
        Index of the input series split
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        kstest series
    """
    return rolling_apply(
        kstest_1,
        window + offset,
        series,
        offset=offset
    )


def ks_2samp(series: np.ndarray, offset: int, window: int) -> np.ndarray:
    """
    Calculates ks_2samp feature

    Parameters
    ----------
    series : np.ndarray
        Input series
    offset : int
        Index of the input series split
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        ks_2samp series
    """
    return rolling_apply(
        ks_2samp_1,
        window + offset,
        series,
        offset=offset
    )


def epps_singleton_2samp(series: np.ndarray, offset: int, window: int):
    """
    Calculates epps_singleton_2samp feature

    Parameters
    ----------
    series : np.ndarray
        Input series
    offset : int
        Index of the input series split
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        epps_singleton_2samp series
    """
    return rolling_apply(
        epps_singleton_2samp_1,
        window + offset,
        series,
        offset=offset
    )


def mannwhitneyu(series: np.ndarray, offset: int, window: int) -> np.ndarray:
    """
    Calculates mannwhitneyufeature

    Parameters
    ----------
    series : np.ndarray
        Input series
    offset : int
        Index of the input series split
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        mannwhitneyu series
    """
    return rolling_apply(
        mannwhitneyu_1,
        window + offset,
        series,
        offset=offset
    )


def wilcoxon(series: np.ndarray, offset: int, window: int) -> np.ndarray:
    """
    Calculates wilcoxon feature

    Parameters
    ----------
    series : np.ndarray
        Input series
    offset : int
        Index of the input series split
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        wilcoxon series
    """
    return rolling_apply(
        wilcoxon_1,
        window + offset,
        series,
        offset=offset,
    )


def anderson_ksamp(series: np.ndarray, offset: int, window: int) -> np.ndarray:
    """
    Calculates anderson_ksamp feature

    Parameters
    ----------
    series : np.ndarray
        Input series
    offset : int
        Index of the input series split
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        anderson_ksamp series
    """
    return rolling_apply(
        anderson_ksamp_1,
        window + offset,
        series,
        offset=offset
    )


def adfuller(series, window):
    """
    Calculates adfuller feature

    Parameters
    ----------
    series : np.ndarray
        Input series
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        adfuller series
    """
    return rolling_apply(
        adfuller_1,
        window,
        series
    )


def chisquare(series, offset, window, bins):
    return rolling_apply(
        chisquare_1,
        window + offset,
        series,
        offset=offset,
        bins=bins
    )


def power_divergence(series, offset, window, bins):
    return rolling_apply(
        power_divergence_1,
        window + offset,
        series,
        offset=offset,
        bins=bins
    )


def runs_test_1samp(series, cutoff, window):
    return rolling_apply(
        runs_test_1samp_1,
        window,
        series,
        cutoff=cutoff
    )


def runs_test_2samp(series, offset, window):
    return rolling_apply(
        runs_test_2samp_1,
        window + offset,
        series,
        offset=offset
    )
