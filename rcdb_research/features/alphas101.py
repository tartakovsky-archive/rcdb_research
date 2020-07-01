# flake8: noqa
# Good docstring here:
# https://github.com/trajanfan/101alpha/blob/master/AlphaFactory.py
# Reference:
# https://poseidon01.ssrn.com/delivery.php?ID=076096124100087113078104090095098089121045061078028062023071090075002114064107095110039054098101105044027112114119086101080084010025046038052091022126119126096108127062069039091006123020007111026126030094087010112122006001031096016074026023005066123088&EXT=pdf # noqa
import numpy as np
import pandas as pd
import numpy_ext as npext
from numpy import abs, log, sign
from tulipindicators import ti
from scipy.stats import rankdata



###########
# Helpers
###########


def correlation(x: np.ndarray, y: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Wrapper function to estimate rolling correlations.
    :param x: corr param.
    :param y: corr param
    :param period: the rolling window period.
    :return: a pandas DataFrame with the time-series min over
    the past 'window' days.
    """
    # return pd.Series(x).rolling(period).corr(pd.Series(y)).values
    res = [
        np.corrcoef(x[idxs], y[idxs])[0][1]
        for idxs in npext.rolling(np.arange(len(x)), period, as_array=True)[period - 1:].astype(np.int)
    ]
    return npext.prepend_na(np.array(res), period - 1)


def covariance(x: np.ndarray, y: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Wrapper function to estimate rolling covariance.
    :param x: corr param 1
    :param y: corr param 2
    :param period: the rolling window period.
    :return: a pandas DataFrame with the time-series min over the past
    'window' days.
    """
    # return pd.Series(x).rolling(period).cov(pd.Series(y)).values
    res = [
        np.cov(x[idxs], y[idxs])[0][1]
        for idxs in npext.rolling(np.arange(len(x)), period, as_array=True)[period - 1:].astype(np.int)
    ]
    return npext.prepend_na(np.array(res), period - 1)


def ts_rank(series: np.ndarray, period: int = 10) -> np.ndarray:
    return npext.rolling_apply((lambda x: rankdata(x)[-1]), period, series)


def delta(series: np.ndarray, period: int = 1) -> np.ndarray:
    return npext.prepend_na(series[period:] - series[:-period], period)


def delay(series: np.ndarray, period: int = 1) -> np.ndarray:
    return npext.prepend_na(series[:len(series) - period], period)


def rank(series: np.ndarray) -> np.ndarray:
    # return pd.core.algorithms.rank(series, pct=True)
    r_data = rankdata(series)
    return r_data / r_data.max()


def scale(series: np.ndarray, k: int = 1) -> np.ndarray:
    return series * k / np.sum(np.abs(np.nan_to_num(series)))


def ts_argmax(series: np.ndarray, period: int = 10) -> np.ndarray:
    return npext.rolling_apply(np.argmax, period, series)


def ts_argmin(series: np.ndarray, period: int = 10) -> np.ndarray:
    return npext.rolling_apply(np.argmin , period, series)


def decay_linear(series: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Linear weighted moving average implementation.
    :param series: input series.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    df = pd.DataFrame(series)
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.values

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return np.hstack(na_lwma)


###########
# Features
###########


def f1(close: np.ndarray, returns: np.ndarray, std_window: int = 20, argmax_window: int = 5) -> np.ndarray:
    """
    alpha001

    Formula: rank(
                ts_argmax(
                    (returns < 0 ? stddev(returns, 20) : close) ** 2,
                    5
                )
             ) - 0.5
    Parameter: std_window = 20, argmax_window = 5
    Explanation: Try to long the most recently volatile stock

    :param close: input array
    :param returns: input array
    :param std_window:
    :param argmax_window:
    :return:
    """
    close = close.copy()

    zero_less = (returns < 0)

    close[zero_less] = ti.stddev(returns, std_window)[zero_less]

    last_nan = np.hstack(np.argwhere(np.isnan(close)))[-1]

    argmax = ts_argmax(close[last_nan + 1:] ** 2, argmax_window)
    ranks = rank(argmax[argmax_window - 1:])
    return npext.prepend_na(
        ranks,
        len(close) - len(ranks)
    )


def f2(
    open: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    delta_step: int = 2,
    corr_window: int = 6,
    alpha_sign: int = -1
) -> np.ndarray:
    """
    alpha002
    Formula: -1 * correlation(
                rank(delta(log(volume), 2)),
                rank(((close - open) / open)),
                6
             )
    Parameter: delta_step = 2, corr_window = 6, alpha_sign = -1
    Explanation: Long the stock with different change
    directions in ranks of volume and price pct change daily

    :param open: input array
    :param close: input array
    :param volume: input array
    :param delta_step:
    :param corr_window:
    :param alpha_sign:
    :return:
    """
    res = alpha_sign * correlation(
        rank(delta(log(volume), delta_step)),
        rank((close - open) / open),
        period=corr_window
    )
    res[np.isnan(res) | np.isinf(res)] = 0
    return res


def f3(
    open: np.ndarray,
    volume: np.ndarray,
    corr_window: int = 10,
    alpha_sign: int = -1
) -> np.ndarray:
    """
    alpha003
    :param open: input array
    :param volume: input array
    :param corr_window:
    :param alpha_sign:
    :return:
    """
    # (-1 * correlation(rank(open), rank(volume), 10))
    res = alpha_sign * correlation(
        rank(open),
        rank(volume),
        period=corr_window)
    res[np.isnan(res) | np.isinf(res)] = 0
    return res


def f4(low: np.ndarray, alpha_sign: int = -1, period: int = 9) -> np.ndarray:
    """
    alpha004
    :param low: input array
    :param alpha_sign:
    :param period:
    :return:
    """
    # (-1 * Ts_Rank(rank(low), 9))
    return alpha_sign * ts_rank(rank(low), period)


def f5(open: np.ndarray, close: np.ndarray, vwap: np.ndarray, period: int = 10) -> np.ndarray:
    """
    alpha005
    :param open: input array
    :param close: input array
    :param vwap: input array
    :param period:
    :return:
    """
    # rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))
    return (rank((open - (ti.sum(vwap, period) / period))) * (
            -1 * abs(rank((close - vwap)))))


def f6(open: np.ndarray, volume: np.ndarray, corr_window: int = 10) -> np.ndarray:
    """
    alpha006
    :param open: input array
    :param volume: input array
    :param corr_window:
    :return:
    """
    # (-1 * correlation(open, volume, 10))
    res = -1 * correlation(open, volume, corr_window)
    res[np.isnan(res) | np.isinf(res)] = 0
    return res


def f7(
        close: np.ndarray,
        volume: np.ndarray,
        sma_window: int = 20,
        delta_window: int = 7,
        ts_rank_window: int = 60) -> np.ndarray:
    """
    alpha007
    :param close: input array
    :param volume: input array
    :param sma_window:
    :param delta_window:
    :param ts_rank_window:
    :return:
    """
    # (adv20 < volume) ?
    #   (-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7)) :
    #   (-1* 1)
    adv20 = ti.sma(volume, sma_window)
    alpha = -1 * ts_rank(abs(delta(close, delta_window)), ts_rank_window) * sign(
        delta(close, delta_window))

    alpha[adv20 >= volume] = -1
    return alpha


def f8(open: np.ndarray, returns: np.ndarray, ts_sum_window: int = 5, delay_window: int = 10) -> np.ndarray:
    """
    alpha008
    :param open: input array
    :param returns: input array
    :param ts_sum_window:
    :param delay_window:
    :return:
    """
    # -1 * rank(((sum(open, 5) * sum(returns, 5)) -
    #   delay((sum(open, 5) * sum(returns, 5)),10)))
    sum_open = ti.sum(open, ts_sum_window)
    sum_return = npext.prepend_na(ti.sum(returns[1:], ts_sum_window), 1)
    gen_change = sum_open * sum_return
    return -1 * (rank(gen_change - delay(gen_change, delay_window)))


def f9(close: np.ndarray, delta_window: int = 1, ts_min_window: int = 5, ts_max_window: int = 5) -> np.ndarray:
    """
    alpha009
    :param close: input array
    :param ts_min_window:
    :param ts_max_window:
    :return:
    """
    # ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
    #   ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) :
    #       (-1 * delta(close, 1))))
    delta_close = delta(close, delta_window)
    cond = (ti.min(delta_close, ts_min_window) > 0) | (ti.max(delta_close, ts_max_window) < 0)
    alpha = -1 * delta_close
    alpha[cond] = delta_close[cond]
    return alpha


def f10(close: np.ndarray, delta_window: int = 1, ts_min_window: int = 4, ts_max_window: int = 4) -> np.ndarray:
    """
    alpha010
    :param close: input array
    :param delta_window:
    :param ts_min_window:
    :param ts_max_window:
    :return:
    """
    # rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
    #   ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) :
    #       (-1 * delta(close, 1)))))
    delta_close = delta(close, delta_window)
    cond = (ti.min(delta_close, ts_min_window) > 0) | (ti.max(delta_close, ts_max_window) < 0)
    alpha = -1 * delta_close
    alpha[cond] = delta_close[cond]
    return alpha

def f11(
        close: np.ndarray,
        vwap: np.ndarray,
        volume: np.ndarray,
        delta_window: int = 3,
        ts_min_window: int = 3,
        ts_max_window: int = 3) -> np.ndarray:
    """
    alpha011
    :param close: input array
    :param vwap: input array
    :param delta_window:
    :param ts_min_window:
    :param ts_max_window:
    :return:
    """
    # (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3)))
    # * rank(delta(volume, 3))
    vwap_close_diff = vwap - close
    tmp = rank(ti.max(vwap_close_diff,ts_max_window )) \
        + rank(ti.min(vwap_close_diff, ts_min_window))
    return tmp * rank(delta(volume, delta_window))


def f12(close: np.ndarray, volume: np.ndarray, delta_window: int = 1) -> np.ndarray:
    """
    alpha012
    :param close: input array
    :param volume: input array
    :param delta_window:
    :return:
    """
    # (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    return sign(delta(volume, delta_window)) * (-1 * delta(close, delta_window))


def f13(close: np.ndarray, volume: np.ndarray, cov_window: int = 5) -> np.ndarray:
    """
    alpha013
    :param close: input array
    :param volume: input array
    :param cov_window:
    :return:
    """
    # (-1 * rank(covariance(rank(close), rank(volume), 5)))
    return -1 * rank(covariance(rank(close), rank(volume), cov_window))


def f14(open: np.ndarray, volume: np.ndarray, returns: np.ndarray, corr_window: int = 10, delta_window: int = 3) -> np.ndarray:
    """
    alpha014
    :param open: input array
    :param volume: input array
    :param returns: input array
    :return:
    """
    # ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    corr_res = correlation(open, volume, corr_window)
    corr_res[np.isnan(corr_res) | np.isinf(corr_res)] = 0
    return -1 * rank(delta(returns, delta_window)) * corr_res


def f15(high: np.ndarray, volume: np.ndarray, corr_window: int = 3, ts_sum_window: int = 3) -> np.ndarray:
    """
    alpha015
    :param high: input array
    :param volume: input array
    :param corr_window:
    :param ts_sum_window:
    :return:
    """
    # (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    corr_res = correlation(rank(high), rank(volume), corr_window)
    corr_res[~np.isfinite(corr_res)] = 0
    return -1 * ti.sum(rank(corr_res), ts_sum_window)


def f16(high: np.ndarray, volume: np.ndarray, cov_window: int = 5) -> np.ndarray:
    """
    alpha016
    :param high: input array
    :param volume: input array
    :param cov_window:
    :return:
    """
    # (-1 * rank(covariance(rank(high), rank(volume), 5)))
    return -1 * rank(covariance(rank(high), rank(volume), cov_window))


def f17(
        close: np.ndarray,
        volume: np.ndarray,
        sma_window: int = 20,
        first_ts_rank_window: int = 10,
        delta_window: int = 1,
        last_ts_rank_window: int = 5) -> np.ndarray:
    """
    alpha017
    :param close: input array
    :param volume: input array
    :param sma_window:
    :param first_ts_rank_window:
    :param delta_window:
    :param last_ts_rank_window:
    :return:
    """
    # ((-1 * rank(ts_rank(close, 10))) *
    # rank(delta(delta(close, 1), 1))) *
    # rank(ts_rank((volume / adv20), 5))
    adv20 = ti.sma(volume, sma_window)
    last_ts_rank = ts_rank((volume / adv20), last_ts_rank_window)
    last_ts_rank[:sma_window + last_ts_rank_window - 2] = np.nan

    a = rank(ts_rank(close, first_ts_rank_window))
    b = rank(delta(delta(close, delta_window), delta_window))
    c = rank(last_ts_rank)
    return -1 * (a * b * c)


def f18(open: np.ndarray, close: np.ndarray, corr_window: int = 10, stddev_window: int = 5) -> np.ndarray:
    """
    alpha018
    :param open: input array
    :param close: input array
    :param corr_window:
    :param stddev_window:
    :return:
    """
    # -1 * rank(((stddev(abs((close - open)), 5) +
    # (close - open)) + correlation(close, open,10)))
    corr_res = correlation(close, open, corr_window)
    corr_res = npext.fill_not_finite(corr_res, 0)
    return -1 * (rank(ti.stddev(abs(close - open), stddev_window) + close - open + corr_res))


def f19(
        close: np.ndarray,
        returns: np.ndarray,
        delay_window: int = 7,
        delta_window: int = 7,
        ts_sum_window: int = 250) -> np.ndarray:
    """
    alpha019
    :param close: input array
    :param returns: input array
    :param delay_window:
    :param delta_window:
    :param ts_sum_window:
    :return:
    """
    # (-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) *
    # (1 + rank((1 + sum(returns,250))))
    return (
        -1 * sign(close - delay(close, delay_window) + delta(close, delta_window)) * (
            1 + rank(1 + npext.prepend_na(ti.sum(returns[1:], ts_sum_window), 1))
        )
    )


def f20(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    alpha020
    :param open: input array
    :param high: input array
    :param low: input array
    :param close: input array
    :return:
    """
    # (((-1 * rank((open - delay(high, 1)))) *
    # rank((open - delay(close, 1)))) *
    # rank((open -delay(low, 1))))
    return -1 * (rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1)))


def f21(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha021
    :param close: input array
    :param volume: input array
    :return:
    """
    # ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2))
    # ? (-1 * 1)
    # : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8)))
    # ? 1
    # : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1
    # : (-1 * 1))))
    cond_1 = ti.sma(close, 8) + ti.stddev(close, 8) < ti.sma(close, 2)
    cond_2 = ti.sma(close, 2) < ti.sma(close, 8) - ti.stddev(close, 8)
    cond_3 = ti.sma(volume, 20) / volume < 1
    alpha = np.zeros_like(close)
    alpha[cond_1 | cond_3] = -1
    alpha[cond_2] = 1
    return alpha


def f22(high: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha022
    :param high: input array
    :param close: input array
    :param volume: input array
    :return:
    """
    # (-1 * (delta(correlation(high, volume, 5), 5) *
    # rank(stddev(close, 20))))
    corr_res = correlation(high, volume, 5)
    corr_res = npext.fill_not_finite(corr_res, 0)
    return -1 * delta(corr_res, 5) * rank(ti.stddev(close, 20))


def f23(high: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    alpha023
    :param high: input array
    :param close: input array
    :return:
    """
    # (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    cond = ti.sma(high, 20) < high
    alpha = np.zeros_like(close)
    alpha[cond] = -1 * np.nan_to_num(delta(high, 2))[cond]
    return alpha


def f24(close: np.ndarray) -> np.ndarray:
    """
    alpha024
    :param close: input array
    :return:
    """
    # ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05)
    # ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05))
    # ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
    cond = delta(ti.sma(close, 100), 100) / delay(close, 100) <= 0.05
    alpha = -1 * delta(close, 3)
    alpha[cond] = -1 * (close - ti.min(close, 100))[cond]
    return alpha


def f25(high: np.ndarray, close: np.ndarray, volume: np.ndarray, returns: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha025
    :param high: input array
    :param close: input array
    :param volume: input array
    :param returns: input array
    :param vwap: input array
    :return:
    """
    # rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    adv20 = ti.sma(volume, 20)
    return rank(-1 * returns * adv20 * vwap * (high - close))


def f26(high: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha026
    :param high: input array
    :param volume: input array
    :return:
    """
    # (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    corr_res = correlation(ts_rank(volume, 5), ts_rank(high, 5), 5)
    corr_res = npext.fill_not_finite(corr_res, 0)
    return -1 * ti.max(corr_res, 3)


def f27(volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha027
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # 0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0) ?
    # (-1 * 1) : 1)
    corr = correlation(rank(volume), rank(vwap), 6)[5:]
    a = rank(ti.sma(corr, 2))
    a = npext.prepend_na(a, 5)
    cond = 0.5 < a
    a[cond] = -1
    a[~cond] = 1
    return a


def f28(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, returns: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha028
    :param high: input array
    :param low: input array
    :param close: input array
    :param volume: input array
    :param returns: input array
    :param vwap: input array
    :return:
    """
    # scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    adv20 = ti.sma(volume, 20)
    corr_res = correlation(adv20, low, 5)
    corr_res = npext.fill_not_finite(corr_res, 0)
    return scale(((corr_res + ((high + low) / 2)) - close))


def f29(close: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """
    alpha029
    :param close: input array
    :param returns: input array
    :return:
    """
    # (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 *
    # rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(
    # delay((-1 * returns), 6), 5))
    a = ts_rank(delay((-1 * returns), 6), 5)
    a[:6 + 5] = np.nan
    expr = ti.min(rank(rank(-1 * rank(delta(close - 1, 5)))), 2)
    expr = npext.prepend_na(ti.sum(expr[6:], 1), 6)
    expr = rank(rank(scale(log(expr))))
    expr = npext.rolling_apply(np.prod, 1, expr)
    expr = ti.min(expr, 5)
    return expr + a


def f30(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha030
    :param close: input array
    :param volume: input array
    :return:
    """
    # (((1.0 - rank(((sign((close - delay(close, 1))) +
    # sign((delay(close, 1) - delay(close, 2)))) +
    # sign((delay(close, 2) - delay(close, 3)))))) *
    # sum(volume, 5)) / sum(volume, 20))
    delta_close = delta(close, 1)
    inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(
        delay(delta_close, 2))
    return ((1.0 - rank(inner)) * ti.sum(volume, 5)) / ti.sum(
        volume, 20)


def f31(low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha031
    :param low: input array
    :param close: input array
    :param volume: input array
    :return:
    """
    # ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))),
    # 10)))) + rank((-1 *delta(close, 3)))) +
    # sign(scale(correlation(adv20, low, 12))))
    adv20 = ti.sma(volume, 20)
    corr_res = correlation(adv20, low, 12)
    corr_res = npext.fill_not_finite(corr_res, 0)

    p1 = rank(rank(rank(
        decay_linear((-1 * rank(rank(delta(close, 10)))),
                     10))))
    p2 = rank((-1 * delta(close, 3)))
    p3 = sign(scale(corr_res))
    return p1 + p2 + p3


def f32(close: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha032
    :param close: input array
    :param vwap: input array
    :return:
    """
    # (scale(((sum(close, 7) / 7) - close)) +
    # (20 * scale(correlation(vwap, delay(close, 5),230))))
    return scale(((ti.sma(close, 7) / 7) - close)) + (20 * scale(
        correlation(vwap, delay(close, 5), 230)))


def f33(open: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    alpha033
    :param open: input array
    :param close: input array
    :return:
    """
    # rank((-1 * ((1 - (open / close))^1)))
    return rank(-1 + (open / close))


def f34(close: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """
    alpha034
    :param close: input array
    :param returns: input array
    :return:
    """
    # rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5))))
    # + (1 - rank(delta(close, 1)))))
    inner = ti.stddev(returns, 2) / ti.stddev(returns, 5)
    inner = npext.fill_not_finite(inner, 1)
    return rank(1 - rank(inner) + (1 - rank(delta(close, 1))))


def f35(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """
    alpha035
    :param high: input array
    :param low: input array
    :param close: input array
    :param volume: input array
    :param returns: input array
    :return:
    """
    # ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16)))
    # * (1 -Ts_Rank(returns, 32)))
    return ((ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16))) * (1 - ts_rank(returns, 32)))


def f36(open: np.ndarray, close: np.ndarray, volume: np.ndarray, returns: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha036
    :param open: input array
    :param close: input array
    :param volume: input array
    :param returns: input array
    :param vwap: input array
    :return:
    """
    # (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15)))
    # + (0.7 * rank((open- close)))) +
    # (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) +
    # rank(abs(correlation(vwap,adv20, 6)))) +
    # (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    adv20 = ti.sma(volume, 20)
    corr = correlation((close - open), delay(volume, 1), 15)
    tmp = (((((2.21 * rank(corr)) + (0.7 * rank((open - close))))
             + (0.73 * rank(ts_rank(delay((-1 * returns), 6), 5))))
            + rank(abs(correlation(vwap, adv20, 6))))
           + (0.6 * rank((((ti.sma(close, 200) / 200) - open)
                          * (close - open)))))
    return tmp


def f37(open: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    alpha037
    :param open: input array
    :param close: input array
    :return:
    """
    # (rank(correlation(delay((open - close), 1), close, 200)) +
    # rank((open - close)))
    tmp = rank(correlation(
        delay(open - close, 1),
        close,
        200)
    )
    return tmp + rank(open - close)


def f38(open: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    alpha038
    :param open: input array
    :param close: input array
    :return:
    """
    # ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    inner = close / open
    inner = npext.fill_not_finite(inner, 1)
    return -1 * rank(ts_rank(close, 10)) * rank(inner)


def f39(close: np.ndarray, volume: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """
    alpha039
    :param close: input array
    :param volume: input array
    :param returns: input array
    :return:
    """
    # ((-1 * rank((delta(close, 7) *
    # (1 - rank(decay_linear((volume / adv20), 9)))))) *
    # (1 +rank(sum(returns, 250))))
    adv20 = ti.sma(volume, 20)
    tmp = (1 - rank(decay_linear((volume / adv20), 9)))
    tmp = (-1 * rank(delta(close, 7) * tmp))
    returns = npext.fill_na(returns, 0)
    return tmp * (1 + rank(ti.sma(returns, 250)))


def f40(high: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha040
    :param high: input array
    :param volume: input array
    :return:
    """
    # ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    return -1 * rank(ti.stddev(high, 10)) * correlation(high, volume, 10)


def f41(high: np.ndarray, low: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha041
    :param high: input array
    :param low: input array
    :param vwap: input array
    :return:
    """
    # (((high * low)^0.5) - vwap)
    return pow((high * low), 0.5) - vwap


def f42(close: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha042
    :param close: input array
    :param vwap: input array
    :return:
    """
    # (rank((vwap - close)) / rank((vwap + close)))
    return rank((vwap - close)) / rank((vwap + close))


def f43(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha043
    :param close: input array
    :param volume: input array
    :return:
    """
    # (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    adv20 = ti.sma(volume, 20)
    tmp = ts_rank(volume / adv20, 20) * ts_rank((-1 * delta(close, 7)), 8)
    tmp[:38] = np.nan
    return tmp


def f44(high: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha044
    :param high: input array
    :param volume: input array
    :return:
    """
    # (-1 * correlation(high, rank(volume), 5))
    corr_res = correlation(high, rank(volume), 5)
    corr_res = npext.fill_not_finite(corr_res, 0)
    return -1 * corr_res


def f45(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha045
    :param close: input array
    :param volume: input array
    :return:
    """
    # (-1 * ((rank((sum(delay(close, 5), 20) / 20)) *
    # correlation(close, volume, 2)) *rank(correlation(sum(close, 5),
    # sum(close, 20), 2))))
    corr_res = correlation(close, volume, 2)
    corr_res = npext.fill_not_finite(corr_res, 0)

    delay_res = delay(close, 5)[5:]
    sma_res = npext.prepend_na(ti.sma(delay_res, 20), 5)

    return -1 * (rank(sma_res) * corr_res *
                 rank(correlation(ti.sum(close, 5),
                                  ti.sum(close, 20),
                                  2)))


def f46(close: np.ndarray) -> np.ndarray:
    """
    alpha046
    :param close: input array
    :return:
    """
    # ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) -
    # ((delay(close, 10) - close) / 10))) ?(-1 * 1) :
    # (((((delay(close, 20) - delay(close, 10)) / 10) -
    # ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) *
    # (close - delay(close, 1)))))

    tmp = (delay(close, 20) - delay(close, 10)) / 10 \
        - (delay(close, 10) - close) / 10

    replaced = np.empty(close.size)
    replaced.fill(False)

    alpha = -1 * (close - delay(close, 1))

    cond1 = (tmp < 0)
    alpha[cond1] = 1
    replaced[cond1] = True

    alpha[(tmp > 0.25) & (replaced == False)] = -1  # noqa
    return alpha


def f47(high: np.ndarray, close: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha047
    :param high: input array
    :param close: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((((rank((1 / close)) * volume) / adv20) *
    # ((high * rank((high - close))) /
    # (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
    adv20 = ti.sma(volume, 20)
    return ((((rank((1 / close)) * volume) / adv20)
             * ((high * rank((high - close)))
                / (ti.sma(high, 5))))
            - rank((vwap - delay(vwap, 5))))


def f49(close: np.ndarray) -> np.ndarray:
    """
    alpha049
    :param close: input array
    :return:
    """
    # (((((delay(close, 20) - delay(close, 10)) / 10) -
    # ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ?
    # 1 : ((-1 * 1) * (close - delay(close, 1))))
    inner = (delay(close, 20) - delay(close, 10)) / 10 \
        - (delay(close, 10) - close) / 10
    alpha = (-1 * delta(close))
    alpha[inner < -0.1] = 1
    return alpha


def f50(volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha050
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    return (-1 * ti.max(
        rank(correlation(rank(volume), rank(vwap), 5)), 5))


def f51(close: np.ndarray) -> np.ndarray:
    """
    alpha051
    :param close: input array
    :return:
    """
    # (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10)
    # - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) *
    # (close - delay(close, 1))))
    inner = (delay(close, 20) - delay(close, 10)) / 10 \
        - (delay(close, 10) - close) / 10
    alpha = (-1 * delta(close))
    alpha[inner < -0.05] = 1
    return alpha


def f52(low: np.ndarray, volume: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """
    alpha052
    :param low: input array
    :param volume: input array
    :param returns: input array
    :return:
    """
    # ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
    # rank(((sum(returns, 240) -sum(returns, 20)) / 220))) *
    # ts_rank(volume, 5))
    returns = returns[1:]
    return (((-1 * ti.min(low, 5) + delay(ti.min(low, 5), 5)) *
             rank(((npext.prepend_na(ti.sum(returns, 240), 1) - npext.prepend_na(ti.sum(returns, 20), 1)) /
                   220))) * ts_rank(volume, 5))


def f53(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    alpha053
    :param high: input array
    :param low: input array
    :param close: input array
    :return:
    """
    # (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    close_low_diff = (close - low)
    high_close_diff = (high - close)
    tmp = (close_low_diff - high_close_diff) / close_low_diff
    tmp = npext.fill_not_finite(tmp, 0)
    return -1 * delta(tmp, 9)


def f54(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    alpha054
    :param high: input array
    :param low: input array
    :param close: input array
    :return:
    """
    # ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    low_high_diff = low - high
    low_close_diff = low - close
    return ((-1 * low_close_diff * (open ** 5))
            / (low_high_diff * (close ** 5)))


def f55(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha055
    :param high: input array
    :param low: input array
    :param close: input array
    :param volume: input array
    :return:
    """
    # (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12)
    # - ts_min(low,12)))), rank(volume), 6))
    divisor = ti.max(high, 12) - ti.min(low, 12)
    inner = (close - ti.min(low, 12)) / divisor
    res = correlation(rank(inner), rank(volume), 6)
    res = npext.fill_not_finite(res, 0)
    return -1 * res


def f57(close: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha057
    :param close: input array
    :param vwap: input array
    :return:
    """
    # (0 - (1 * ((close - vwap) /
    # decay_linear(rank(ts_argmax(close, 30)), 2))))
    return (0 - (1 * ((close - vwap) / decay_linear(
        rank(ts_argmax(close, 30)), 2))))


def f60(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha060
    :param high: input array
    :param low: input array
    :param close: input array
    :param volume: input array
    :return:
    """
    # (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) /
    # (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
    high_low_diff = high - low
    inner = ((close - low)
             - (high - close)) * volume / high_low_diff
    return 0 - ((2 * scale(rank(inner))) - scale(
        rank(ts_argmax(close, 10))))


def f61(volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha061
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # (rank((vwap - ts_min(vwap, 16.1219))) <
    # rank(correlation(vwap, adv180, 17.9282)))
    adv180 = ti.sma(volume, 180)
    return (rank((vwap - ti.min(vwap, 16))) < rank(
        correlation(vwap, adv180, 18))) * 1


def f62(open: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha062
    :param open: input array
    :param high: input array
    :param low: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) <
    # rank(((rank(open) +rank(open)) < (rank(((high + low) / 2))
    # + rank(high))))) * -1)
    adv20 = ti.sma(volume, 20)[20:]
    return ((rank(correlation(vwap, npext.prepend_na(ti.sma(adv20, 22), 20), 10))
             < rank((rank(open) + rank(open))
                    < (rank((high + low) / 2)
                       + rank(high)))) * -1)


def f64(open: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha064
    :param open: input array
    :param high: input array
    :param low: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((rank(correlation(sum(((open * 0.178404) +
    # (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) <
    # rank(delta(((((high + low) / 2) * 0.178404) +
    # (vwap * (1 -0.178404))), 3.69741))) * -1)
    adv120 = ti.sma(volume, 120)[120:]
    return ((rank(correlation(
        ti.sma(((open * 0.178404) + (low * (1 - 0.178404))), 13),
        npext.prepend_na(ti.sma(adv120, 13), 120), 17)) < rank(
            delta(((((high + low) / 2) * 0.178404)
                   + (vwap * (1 - 0.178404))), 4))) * -1)


def f65(open: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha065
    :param open: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))),
    # sum(adv60,8.6911), 6.40374)) <
    # rank((open - ts_min(open, 13.635)))) * -1)
    adv60 = ti.sma(volume, 60)[60:]
    return ((rank(correlation(
        ((open * 0.00817205) + (vwap * (1 - 0.00817205))),
        npext.prepend_na(ti.sma(adv60, 9), 60), 6)) < rank(
        (open - ti.min(open, 14)))) * -1)


def f66(open: np.ndarray, high: np.ndarray, low: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha066
    :param open: input array
    :param high: input array
    :param low: input array
    :param vwap: input array
    :return:
    """
    # ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) +
    # Ts_Rank(decay_linear(((((low* 0.96633) +
    # (low * (1 - 0.96633))) - vwap) /
    # (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    return ((rank(
        decay_linear(delta(vwap, 4), 7)) + ts_rank(
        decay_linear(((((low * 0.96633) + (low * (1 - 0.96633)))
                       - vwap)
                      / (open - ((high + low) / 2))), 11), 7)) * -1)


def f71(open: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha071
    :param open: input array
    :param low: input array
    :param close: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976),
    # Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948),
    # Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2),
    # 16.4662), 4.4388))
    adv180 = ti.sma(volume, 180)
    p1 = ts_rank(decay_linear(
        correlation(ts_rank(close, 3), ts_rank(adv180, 12),
                    18), 4), 16)
    p2 = ts_rank(decay_linear((rank(
        ((low + open) - (vwap + vwap))) ** 2), 16), 4)

    alpha = npext.nans(p1.size)

    cond = p1 >= p2
    alpha[cond] = p1[cond]

    cond = p2 >= p1
    alpha[cond] = p2[cond]

    return alpha


def f72(high: np.ndarray, low: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha072
    :param high: input array
    :param low: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # (rank(decay_linear(correlation(((high + low) / 2),
    # adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(
    # vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
    adv40 = ti.sma(volume, 40)
    return (rank(decay_linear(
        correlation(((high + low) / 2), adv40, 9),
        10)) / rank(decay_linear(correlation(
            ts_rank(vwap, 4),
            ts_rank(volume, 19), 7),
            3)))


def f73(open: np.ndarray, low: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha073
    :param open: input array
    :param low: input array
    :param vwap: input array
    :return:
    """
    # (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
    # Ts_Rank(decay_linear(((delta(((open * 0.147155) +
    # (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) +
    # (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    p1 = rank(decay_linear(delta(vwap, 5), 3))
    p2 = ts_rank(decay_linear(((delta(
        (open * 0.147155) + (low * (1 - 0.147155)),
        2
    ) / ((open * 0.147155)
         + (low * (1 - 0.147155)))) * -1), 3), 17)

    alpha = npext.nans(open.size)

    cond = p1 >= p2
    alpha[cond] = p1[cond]

    cond = p2 >= p1
    alpha[cond] = p2[cond]

    return -1 * alpha


def f74(high: np.ndarray, close: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha074
    :param high: input array
    :param close: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <
    # rank(correlation(rank(((high * 0.0261661) +
    # (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
    adv30 = ti.sma(volume, 30)[30:]
    return ((rank(correlation(close, npext.prepend_na(ti.sma(adv30, 37), 30), 15)) < rank(
        correlation(
            rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))),
            rank(volume), 11))) * -1)


def f75(low: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha075
    :param low: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # (rank(correlation(vwap, volume, 4.24304)) <
    # rank(correlation(rank(low), rank(adv50),12.4413)))
    adv50 = ti.sma(volume, 50)
    return (rank(correlation(vwap, volume, 4)) < rank(
        correlation(rank(low), rank(adv50), 12))) * 1


def f77(high: np.ndarray, low: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha077
    :param high: input array
    :param low: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # min(rank(decay_linear(((((high + low) / 2) + high) -
    # (vwap + high)), 20.0451)), rank(decay_linear(correlation(
    # ((high + low) / 2), adv40, 3.1614), 5.64125)))
    adv40 = ti.sma(volume, 40)
    p1 = rank(decay_linear(((((high + low) / 2) + high) - (
        vwap + high)), 20))
    p2 = rank(decay_linear(
        correlation(((high + low) / 2), adv40, 3),
        6))
    alpha = npext.nans(high.size)

    cond = p1 >= p2
    alpha[cond] = p2[cond]

    cond = p2 >= p1
    alpha[cond] = p1[cond]

    return alpha


def f78(low: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha078
    :param low: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # (rank(correlation(sum(((low * 0.352233) +
    # (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428),
    # 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    adv40 = ti.sma(volume, 40)
    return (rank(correlation(
        ti.sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 20),
        npext.prepend_na(ti.sum(adv40[39:], 20), 39), 7)) ** rank(correlation(rank(vwap), rank(volume), 6)))


def f81(volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha081
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),
    # 8.47743))^4)), 14.9655))) <
    # rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    adv10 = ti.sma(volume, 10)
    return ((rank(log(npext.rolling_apply(
            np.prod, 1, rank((rank(correlation(vwap, npext.prepend_na(ti.sum(adv10[9:], 50), 9), 8)) ** 4))
        ))) < rank(
        correlation(rank(vwap), rank(volume), 5))) * -1)


def f83(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha083
    :param high: input array
    :param low: input array
    :param close: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) *
    # rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) /
    # (vwap - close)))
    return ((rank(delay((
        high - low) / (ti.sum(close, 5) / 5),
        2)
    ) * rank(rank(volume))) / (((high - low) / (ti.sum(close, 5) / 5)) / (vwap - close)))


def f84(close: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha084
    :param close: input array
    :param vwap: input array
    :return:
    """
    # SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127),
    # delta(close,4.96796))
    return pow(ts_rank((vwap - ti.max(vwap, 15)), 21),
               delta(close, 5))


def f85(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha085
    :param high: input array
    :param low: input array
    :param close: input array
    :param volume: input array
    :return:
    """
    # (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))),
    # adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596),
    # Ts_Rank(volume, 10.1595),7.11408)))
    adv30 = ti.sma(volume, 30)
    return (rank(correlation(
        ((high * 0.876703) + (close * (1 - 0.876703))), adv30,
        10)) ** (rank(correlation(ts_rank(((high + low) / 2), 4),
                                  ts_rank(volume, 10), 7))))


def f88(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha088
    :param open: input array
    :param high: input array
    :param low: input array
    :param close: input array
    :param volume: input array
    :return:
    """
    # min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) +
    # rank(close))),8.06882)), Ts_Rank(decay_linear(correlation(
    # Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266),
    # 6.65053), 2.61957))
    adv60 = ti.sma(volume, 60)
    p1 = rank(decay_linear(
        (rank(open)
         + rank(low)
         - rank(high)
         + rank(close)),
        8)
    )
    p2 = ts_rank(decay_linear(
        correlation(ts_rank(close, 8), ts_rank(adv60, 21),
                    8), 7), 3)

    alpha = npext.nans(high.size)

    cond = p1 >= p2
    alpha[cond] = p2[cond]

    cond = p2 >= p1
    alpha[cond] = p1[cond]

    return alpha


def f92(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha092
    :param open: input array
    :param high: input array
    :param low: input array
    :param close: input array
    :param volume: input array
    :return:
    """
    # min(Ts_Rank(decay_linear(((((high + low) / 2) + close) <
    # (low + open)), 14.7221),18.8683), Ts_Rank(decay_linear(
    # correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
    adv30 = ti.sma(volume, 30)
    p1 = ts_rank(
        decay_linear(
            ((((high + low) / 2) + close)
             < (low + open)), 15),
        19
    )
    p2 = ts_rank(
        decay_linear(correlation(rank(low), rank(adv30), 8),
                     7), 7)
    alpha = npext.nans(high.size)

    cond = p1 >= p2
    alpha[cond] = p2[cond]

    cond = p2 >= p1
    alpha[cond] = p1[cond]

    return alpha


def f94(volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha094
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(
    # vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    adv60 = ti.sma(volume, 60)
    r = ((rank((vwap - ti.min(vwap, 12))) ** (
        ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18),
                3)) * -1))

    r[:81] = np.nan
    return r


def f96(close: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha096
    :param close: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume),
    # 3.83878),4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(
    # correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242),
    # 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    adv60 = ti.sma(volume, 60)
    p1 = ts_rank(decay_linear(
        correlation(rank(vwap), rank(volume), 4), 4), 8)
    p2 = ts_rank(decay_linear(
        ts_argmax(correlation(ts_rank(close, 7), ts_rank(adv60, 4), 4),
                  13), 14), 13)

    alpha = npext.nans(p1.size)

    cond = p1 >= p2
    alpha[cond] = p1[cond]

    cond = p2 >= p1
    alpha[cond] = p2[cond]

    return -1 * alpha


def f98(open: np.ndarray, volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """
    alpha098
    :param open: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418),
    # 7.18088)) -rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open),
    # rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
    adv5 = ti.sma(volume, 5)[5:]
    adv15 = ti.sma(volume, 15)
    return (rank(
        decay_linear(
            correlation(vwap, npext.prepend_na(ti.sma(adv5, 26), 5), 5),
            7
        )) - rank(decay_linear(ts_rank(ts_argmin(
            correlation(rank(open), rank(adv15), 21),
            9), 7), 8)))


def f99(high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    alpha099
    :param high: input array
    :param low: input array
    :param volume: input array
    :return:
    """
    # ((rank(correlation(sum(((high + low) / 2), 19.8975),
    # sum(adv60, 19.8975), 8.8136)) <
    # rank(correlation(low, volume, 6.28259))) * -1)
    adv60 = ti.sma(volume, 60)
    return ((rank(correlation(ti.sum(((high + low) / 2), 20),
                              npext.prepend_na(ti.sum(adv60[59:], 20), 59), 9)) < rank(
        correlation(low, volume, 6))) * -1)


def f101(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    alpha101
    :param open: input array
    :param high: input array
    :param low: input array
    :param close: input array
    :return:
    """
    # ((close - open) / ((high - low) + .001))
    return (close - open) / ((high - low) + 0.001)
