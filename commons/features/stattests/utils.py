from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats
import statsmodels.sandbox.stats.runs
import os
import contextlib
import statsmodels.tsa.stattools
import numpy as np


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


def runs_test_1(series, offset):
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
