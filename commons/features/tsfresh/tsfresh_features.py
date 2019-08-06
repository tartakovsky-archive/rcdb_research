from typing import Union

import numpy as np
from tsfresh.feature_extraction import feature_calculators as fc

from ..utils import rolling_window, get_feature_funcs, generate_calc_all


PREFIX = "tsfresh"


def apply_to_window(func, series, window, to_np=True, *args, **kwargs):
    arr = [func(x, *args, **kwargs) for x in rolling_window(series, window)]
    return np.array(arr) if to_np else arr


def apply_parametric_to_window(func, series, window, *args, **kwargs):
    res = apply_to_window(
        func=func,
        series=series,
        window=window,
        to_np=False,
        param=[{**kwargs}]
    )
    if isinstance(res[-1], zip):
        res = [list(x) for x in res]
    return np.array([x[0][1] for x in res])


def f1(series: np.array, window: int) -> np.array:
    """
    Returns the absolute energy of the time series which is the sum over the squared values
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(fc.abs_energy, series, window)


def f2(series: np.array, window: int) -> np.array:
    """
    Returns the sum over the absolute value of consecutive changes in each window of series
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(fc.absolute_sum_of_changes, series, window)


def f3(series: np.array, window: int, maxlag: int, f_agg: str) -> np.array:
    """
    Calculates the value of an aggregation function (one of "mean", "var", "std", "median")
    :param series: input data
    :param window: rolling window size
    :param maxlag: is an int and the maximal number of lags to consider
    :param f_agg: agg function
    :return:
    """
    return apply_parametric_to_window(
        func=fc.agg_autocorrelation,
        series=series,
        window=window,
        maxlag=maxlag, f_agg=f_agg
    )


def f4(series: np.array, window: int, chunk_len: int, attr: str, f_agg: str) -> np.array:
    """
    Calculates a linear least-squares regression for values of the time series that
    were aggregated over chunks versus the sequence from 0 up to the number of chunks
    minus one.
    This feature assumes the signal to be uniformly sampled. It will not use the time
    stamps to fit the model.
    The parameters attr controls which of the characteristics are returned.Possible
    extracted attributes are “pvalue”, “rvalue”, “intercept”, “slope”, “stderr”, see
    the documentation of linregress for more information.
    The chunksize is regulated by “chunk_len”. It specifies how many time series values
    are in each chunk.
    Further, the aggregation function is controlled by “f_agg”, which can use “max”,
    “min” or , “mean”, “median”

    :param series: input data
    :param window: rolling window size
    :param chunk_len:
    :param attr:
    :param f_agg:
    :return:
    """
    return apply_parametric_to_window(
        func=fc.agg_linear_trend,
        series=series,
        window=window,
        attr=attr, chunk_len=chunk_len, f_agg=f_agg
    )


def f6(series: np.array, window: int, coeff: float, k: int) -> np.array:
    """
    This feature calculator fits the unconditional maximum likelihoodof an autoregressive
    AR(k) process. The k parameter is the maximum lag of the process
    For the configurations from param which should contain the maxlag “k” and such an AR
    process is calculated. Then the coefficients \varphi_{i} whose index i contained from “coeff” are returned.
    :param series: input data
    :param window: rolling window size
    :param coeff:
    :param k:
    :return:
    """
    return apply_parametric_to_window(
        func=fc.ar_coefficient,
        series=series,
        window=window,
        coeff=coeff, k=k
    )


def f7(series: np.array, window: int, attr: str) -> np.array:
    """
    The Augmented Dickey-Fuller test is a hypothesis test which checks whether a unit
    root is present in a time series sample. This feature calculator returns the value
    of the respective test statistic.

    :param series: input data
    :param window: rolling window size
    :param attr: one of “teststat”, “pvalue” or “usedlag”
    :return:
    """
    return apply_parametric_to_window(
        func=fc.augmented_dickey_fuller,
        series=series,
        window=window,
        attr=attr
    )


def f8(series: np.array, window: int, lag: int) -> np.array:
    """
    Calculates the autocorrelation of the specified lag

    :param series: input data
    :param window: rolling window size
    :param lag:  the lag
    :return:
    """
    return apply_to_window(
        func=fc.autocorrelation,
        series=series,
        window=window,
        lag=lag
    )


def f10(series: np.array, window: int, lag: int) -> np.array:
    """
    c3
    :param series: input data
    :param window: rolling window size
    :param lag: the lag
    :return:
    """
    return apply_to_window(
        func=fc.c3,
        series=series,
        window=window,
        lag=lag
    )


def f11(series: np.array, window: int, ql: float, qh: float, isabs: bool, f_agg: str) -> np.array:
    """
    First fixes a corridor given by the quantiles ql and qh of the distribution of x. Then calculates the average,
    absolute value of consecutive changes of the series x inside this corridor.
    Think about selecting a corridor on the y-Axis and only calculating the mean of the absolute change of the time
    series inside this corridor.

    :param series: input data
    :param window: rolling window size
    :param ql: the lower quantile of the corridor
    :param qh: the higher quantile of the corridor
    :param isabs: should the absolute differences be taken?
    :param f_agg:  name of a numpy function (e.g. mean, var, std, median)  the aggregator function
        that is applied to the differences in the bin
    :return:
    """
    return apply_to_window(
        func=fc.change_quantiles,
        series=series,
        window=window,
        ql=ql,
        qh=qh,
        isabs=isabs,
        f_agg=f_agg
    )


def f12(series: np.array, window: int, normalize: bool) -> np.array:
    """
    This function calculator is an estimate for a time series complexity [1]
    (A more complex time series has more peaks, valleys etc.).
    [1] Batista, Gustavo EAPA, et al (2014).
    CID: an efficient complexity-invariant distance for time series.
    Data Mining and Knowledge Discovery 28.3 (2014): 634-669.

    :param series: input data
    :param window: rolling window size
    :param normalize: should the time series be z-transformed?
    :return:
    """
    return apply_to_window(
        func=fc.cid_ce,
        series=series,
        window=window,
        normalize=normalize
    )


def f13(series: np.array, window: int) -> np.array:
    """
    Returns the number of values in x that are higher than the mean of window
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.count_above_mean,
        series=series,
        window=window,
    )


def f14(series: np.array, window: int) -> np.array:
    """
    Returns the number of values in x that are lower than the mean of window
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.count_below_mean,
        series=series,
        window=window,
    )


def f15(series: np.array, window: int, num_segments: int, segment_focus: int) -> np.array:
    """
    Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the
    sum of squares over the whole series.
    Takes as input parameters the number num_segments of segments to divide the series into
    and segment_focus which is the segment number (starting at zero) to return a feature on.
    If the length of the time series is not a multiple of the number of segments, the remaining
    data points are distributed on the bins starting from the first. For example, if your time
    series consists of 8 entries, the first two bins will contain 3 and the last two values,
    e.g. [ 0., 1., 2.], [ 3., 4., 5.] and [ 6., 7.].
    Note that the answer for num_segments = 1 is a trivial “1” but we handle this scenario in
    case somebody calls it. Sum of the ratios should be 1.0.

    :param series: input data
    :param window: rolling window size
    :param num_segments:
    :param segment_focus:
    :return:
    """
    return apply_parametric_to_window(
        func=fc.energy_ratio_by_chunks,
        series=series,
        window=window,
        num_segments=num_segments,
        segment_focus=segment_focus
    )


def f16(series: np.array, window: int) -> np.array:
    """
    Returns the spectral centroid (mean) of the absolute fourier transform spectrum.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_parametric_to_window(
        func=fc.fft_aggregated,
        series=series,
        window=window,
        aggtype="centroid"
    )


def f17(series: np.array, window: int) -> np.array:
    """
    Returns the spectral variance of the absolute fourier transform spectrum.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_parametric_to_window(
        func=fc.fft_aggregated,
        series=series,
        window=window,
        aggtype="variance"
    )


def f18(series: np.array, window: int) -> np.array:
    """
    Returns the spectral skew of the absolute fourier transform spectrum.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_parametric_to_window(
        func=fc.fft_aggregated,
        series=series,
        window=window,
        aggtype="skew"
    )


def f19(series: np.array, window: int) -> np.array:
    """
    Returns the spectral kurtosis of the absolute fourier transform spectrum.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_parametric_to_window(
        func=fc.fft_aggregated,
        series=series,
        window=window,
        aggtype="kurtosis"
    )


def f20(series: np.array, window: int, coeff: int, attr: str) -> np.array:
    """
    Calculates the fourier coefficients of the one-dimensional discrete
    Fourier Transform for real input by fast fourier transformation algorithm.

    The resulting coefficients will be complex, this feature calculator can return
    the real part (attr==”real”), the imaginary part (attr==”imag), the absolute
    value (attr=”“abs) and the angle in degrees (attr==”angle).

    :param series: input data
    :param window: rolling window size
    :param coeff:
    :param attr: in [“real”, “imag”, “abs”, “angle”]
    :return:
    """
    return apply_parametric_to_window(
        func=fc.fft_coefficient,
        series=series,
        window=window,
        coeff=coeff,
        attr=attr
    )


def f21(series: np.array, window: int) -> np.array:
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.first_location_of_maximum,
        series=series,
        window=window,
    )


def f22(series: np.array, window: int) -> np.array:
    """
    Returns the first location of the minimum value of x.
    The position is calculated relatively to the length of x.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.first_location_of_minimum,
        series=series,
        window=window,
    )


# Todo: raised TypeError: len() of unsized object
# def f23(series: np.array, window: int, m: int, r: float, coeff: int) -> np.array:
#     """
#     Coefficients of polynomial h(x), which has been fitted
#     to the deterministic dynamics of Langevin model as described by [1].
#
#     For short time-series this method is highly dependent on the parameters.
#
#     References
#
#     [1] Friedrich et al. (2000): Physics Letters A 271, p. 217-222
#     Extracting model equations from experimental data
#
#     :param series: input data
#     :param window: rolling window size
#     :param m: the order of polynom to fit for estimating fixed points of dynamics
#     :param r: the number of quantils to use for averaging
#     :param coeff: a positive integer corresponding to the returned coefficient
#     :return:
#     """
#     return apply_parametric_to_window(
#         func=fc.friedrich_coefficients,
#         series=series,
#         window=window,
#         coeff=coeff,
#         m=m, r=r
#     )


def f24(series: np.array, window: int) -> np.array:
    """
    Checks if any value in window occurs more than once
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.has_duplicate,
        series=series,
        window=window,
    )


def f25(series: np.array, window: int) -> np.array:
    """
    Checks if the maximum value in window occurs more than once
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.has_duplicate_max,
        series=series,
        window=window,
    )


def f26(series: np.array, window: int) -> np.array:
    """
    Checks if the minimum value in window occurs more than once
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.has_duplicate_max,
        series=series,
        window=window,
    )


def f27(series: np.array, window: int, q: float) -> np.array:
    """
    Those apply features calculate the relative index i where q%
    of the mass of the time series x lie left of i. For example
    for q = 50% this feature calculator will return the mass
    center of the time series
    :param series: input data
    :param window: rolling window size
    :param q:
    :return:
    """
    return apply_parametric_to_window(
        func=fc.index_mass_quantile,
        series=series,
        window=window,
        q=q,
    )


def f28(series: np.array, window: int) -> np.array:
    """
    Returns the kurtosis for each window
    (calculated with the adjusted Fisher-Pearson standardized moment coefficient G2).
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.kurtosis,
        series=series,
        window=window,
    )


def f29(series: np.array, window: int, r: float) -> np.array:
    """
    Variable denoting if the standard dev of x is higher than
    ‘r’ times the range = difference between max and min of x
    :param series: input data
    :param window: rolling window size
    :param r: the percentage of the range to compare with
    :return:
    """
    return apply_to_window(
        func=fc.large_standard_deviation,
        series=series,
        window=window,
        r=r
    ) * 1


def f30(series: np.array, window: int) -> np.array:
    """
    Returns the relative last location of the maximum value of window
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.last_location_of_maximum,
        series=series,
        window=window,
    )


def f31(series: np.array, window: int) -> np.array:
    """
    Returns the relative last location of the minimum value of window
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.last_location_of_minimum,
        series=series,
        window=window,
    )


def f32(series: np.array, window: int, attr: str) -> np.array:
    """
    Calculate a linear least-squares regression for the values of
    the window versus the sequence from 0 to length of the window minus one.
    This feature assumes the signal to be uniformly sampled. It will not use
    the time stamps to fit the model. The parameters control which of the
    characteristics are returned.

    Possible extracted attributes are “pvalue”, “rvalue”, “intercept”, “slope”, “stderr”
    :param series: input data
    :param window: rolling window size
    :param attr: the attribute name of the regression model
    :return:
    """
    return apply_parametric_to_window(
        func=fc.linear_trend,
        series=series,
        window=window,
        attr=attr,
    )


def f33(series: np.array, window: int) -> np.array:
    """
    Returns the length of the longest consecutive subsequence
    in x that is bigger than the mean of each window series
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.longest_strike_above_mean,
        series=series,
        window=window,
    )


def f34(series: np.array, window: int) -> np.array:
    """
    Returns the length of the longest consecutive subsequence
    in x that is smaller than the mean of each window series
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.longest_strike_below_mean,
        series=series,
        window=window,
    )

# Todo: raised TypeError: len() of unsized object
# def f35(series: np.array, window: int, m: int, r: int) -> np.array:
#     """
#     Largest fixed point of dynamics :math:argmax_x {h(x)=0}` estimated from polynomial h(x),
#     which has been fitted to the deterministic dynamics of Langevin model
#     :param series: input data
#     :param window: rolling window size
#     :param m: order of polynom to fit for estimating fixed points of dynamics
#     :param r: number of quantils to use for averaging
#     :return:
#     """
#     return apply_to_window(
#         func=fc.max_langevin_fixed_point,
#         series=series,
#         window=window,
#         m=m, r=r
#     )


def f36(series: np.array, window: int) -> np.array:
    """
    Calculates the highest value of each window time series.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.maximum,
        series=series,
        window=window,
    )


def f37(series: np.array, window: int) -> np.array:
    """
    Calculates the mean value of each window time series.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.mean,
        series=series,
        window=window,
    )


def f38(series: np.array, window: int) -> np.array:
    """
    Returns the mean over the absolute differences
    between subsequent time series values of each window time series
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.mean_abs_change,
        series=series,
        window=window,
    )


def f39(series: np.array, window: int) -> np.array:
    """
    Returns the mean over the differences
    between subsequent time series values of each window time series
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.mean_change,
        series=series,
        window=window,
    )


def f40(series: np.array, window: int) -> np.array:
    """
    Returns the mean value of a central approximation of the second derivative
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.mean_second_derivative_central,
        series=series,
        window=window,
    )


def f41(series: np.array, window: int) -> np.array:
    """
    Returns the median of each window series
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.median,
        series=series,
        window=window,
    )


def f42(series: np.array, window: int) -> np.array:
    """
    Calculates the lowest value of each window series
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.median,
        series=series,
        window=window,
    )


def f43(series: np.array, window: int, m: float) -> np.array:
    """
    Calculates the number of crossings of each window series on m
    :param series: input data
    :param window: rolling window size
    :param m: the threshold for the crossing
    :return:
    """
    return apply_to_window(
        func=fc.number_crossing_m,
        series=series,
        window=window,
        m=m
    )


def f44(series: np.array, window: int, n: int) -> np.array:
    """
    This feature calculator searches for different peaks in x.
    To do so, window series is smoothed by a ricker wavelet and for widths
    ranging from 1 to n. This feature calculator returns the number
    of peaks that occur at enough width scales and with sufficiently
    high Signal-to-Noise-Ratio (SNR)

    :param series: input data
    :param window: rolling window size
    :param n: maximum width to consider
    :return:
    """
    return apply_to_window(
        func=fc.number_cwt_peaks,
        series=series,
        window=window,
        n=n
    )


def f45(series: np.array, window: int, lag: int) -> np.array:
    """
    Calculates the value of the partial autocorrelation function at the given lag.
    The lag k partial autocorrelation of a time series
    Following [2]

    [1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
    Time series analysis: forecasting and control. John Wiley & Sons.
    [2] https://onlinecourses.science.psu.edu/stat510/node/62
    :param series: input data
    :param window: rolling window size
    :param lag: indicating the lag to be returned
    :return:
    """
    return apply_parametric_to_window(
        func=fc.partial_autocorrelation,
        series=series,
        window=window,
        lag=lag,
    )


def f46(series: np.array, window: int) -> np.array:
    """
    Returns the percentage of unique values, that are present in the window time series more than once.
        len(different values occurring more than once) / len(different values)
    This means the percentage is normalized to the number of unique values,
    in contrast to the percentage_of_reoccurring_values_to_all_values.

    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.percentage_of_reoccurring_datapoints_to_all_datapoints,
        series=series,
        window=window,
    )


def f47(series: np.array, window: int) -> np.array:
    """
    Returns the ratio of unique values, that are present in the window time series more than once.
        # of data points occurring more than once / # of all data points
    This means the ratio is normalized to the number of data points in the time series,
    in contrast to the percentage_of_reoccurring_datapoints_to_all_datapoints.

    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.percentage_of_reoccurring_values_to_all_values,
        series=series,
        window=window,
    )


def f48(series: np.array, window: int, q: float) -> np.array:
    """
    Calculates the q quantile of each window series.
    This is the value of x greater than q% of the ordered values from window series.
    :param series: input data
    :param window: rolling window size
    :param q: the quantile to calculate
    :return:
    """
    return apply_to_window(
        func=fc.quantile,
        series=series,
        window=window,
        q=q
    )


def f49(series: np.array, window: int, min: Union[int, float], max: Union[int, float]) -> np.array:
    """
    Count observed values within the interval [min, max).
    :param series: input data
    :param window: rolling window size
    :param min: the inclusive lower bound of the range
    :param max: the exclusive upper bound of the range
    :return:
    """
    return apply_to_window(
        func=fc.range_count,
        series=series,
        window=window,
        min=min, max=max
    )


def f50(series: np.array, window: int, r: float) -> np.array:
    """
    Ratio of values that are more than r*std(x) (so r sigma) away from the mean of each window series.
    :param series: input data
    :param window: rolling window size
    :param r:
    :return:
    """
    return apply_to_window(
        func=fc.ratio_beyond_r_sigma,
        series=series,
        window=window,
        r=r
    )


def f51(series: np.array, window: int) -> np.array:
    """
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case. In principle, it just returns
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.ratio_value_number_to_time_series_length,
        series=series,
        window=window,
    )


def f53(series: np.array, window: int) -> np.array:
    """
    Returns the sample skewness of each window series
    (calculated with the adjusted Fisher-Pearson standardized moment coefficient G1).
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.skewness,
        series=series,
        window=window,
    )


def f54(series: np.array, window: int, coeff: int) -> np.array:
    """
    This feature calculator estimates the cross power spectral density of each window series at different frequencies.
    To do so, the time series is first shifted from the time domain to the frequency domain.

    The feature calculators returns the power spectrum of the different frequencies.
    :param series: input data
    :param window: rolling window size
    :param coeff:
    :return:
    """
    return apply_parametric_to_window(
        func=fc.spkt_welch_density,
        series=series,
        window=window,
        coeff=coeff,
    )


def f55(series: np.array, window: int) -> np.array:
    """
    Returns the standard deviation of each window series
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.standard_deviation,
        series=series,
        window=window,
    )


def f56(series: np.array, window: int) -> np.array:
    """
    Returns the sum of all data points, that are present in the time series more than once.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.sum_of_reoccurring_data_points,
        series=series,
        window=window,
    )


def f57(series: np.array, window: int) -> np.array:
    """
    Returns the sum of all values, that are present in the time series more than once.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.sum_of_reoccurring_data_points,
        series=series,
        window=window,
    )


def f58(series: np.array, window: int) -> np.array:
    """
    Returns the sum of all values, that are present in the time series more than once.
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.sum_of_reoccurring_values,
        series=series,
        window=window,
    )


def f59(series: np.array, window: int) -> np.array:
    """
    Calculates the sum over the time series values
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.sum_values,
        series=series,
        window=window,
    )


def f60(series: np.array, window: int, r: float) -> np.array:
    """
    Variable denoting if the distribution of x looks symmetric
    :param series: input data
    :param window: rolling window size
    :param r: the percentage of the range to compare with
    :return:
    """
    return apply_parametric_to_window(
        func=fc.symmetry_looking,
        series=series,
        window=window,
        r=r,
    ) * 1


def f61(series: np.array, window: int, lag: int) -> np.array:
    """
    Times reversal asymmetry statistic
    :param series: input data
    :param window: rolling window size
    :param lag: the lag that should be used in the calculation of the feature
    :return:
    """
    return apply_to_window(
        func=fc.time_reversal_asymmetry_statistic,
        series=series,
        window=window,
        lag=lag
    )


def f62(series: np.array, window: int, value: Union[int, float]) -> np.array:
    """
    Count occurrences of value in each window series
    :param series: input data
    :param window: rolling window size
    :param value: the value to be counted
    :return:
    """
    return apply_to_window(
        func=fc.value_count,
        series=series,
        window=window,
        value=value
    )


def f63(series: np.array, window: int) -> np.array:
    """
    Returns the variance of each window series
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.variance,
        series=series,
        window=window,
    )


def f64(series: np.array, window: int) -> np.array:
    """
    Boolean (0 or 1) variable denoting if the variance of each window series
    is greater than its standard deviation.
    Is equal to variance of x being larger than 1
    :param series: input data
    :param window: rolling window size
    :return:
    """
    return apply_to_window(
        func=fc.variance_larger_than_standard_deviation,
        series=series,
        window=window,
    )


FEATURE_FUNCS = get_feature_funcs(__name__)

__all__ = ("FEATURE_FUNCS", "PREFIX", "calc_all", *FEATURE_FUNCS.keys())

calc_all = generate_calc_all(PREFIX, FEATURE_FUNCS)
