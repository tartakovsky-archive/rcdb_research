# flake8: noqa
# Good docstring here:
# https://github.com/trajanfan/101alpha/blob/master/AlphaFactory.py
# Reference:
# https://poseidon01.ssrn.com/delivery.php?ID=076096124100087113078104090095098089121045061078028062023071090075002114064107095110039054098101105044027112114119086101080084010025046038052091022126119126096108127062069039091006123020007111026126030094087010112122006001031096016074026023005066123088&EXT=pdf # noqa
import sys
import inspect
from typing import List, Dict

import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign

from ..utils import feature_filter
from .utils import nan_to_value, nans_array, \
    prepand_nans, ts_sum, sma, stddev, correlation, covariance, \
    ts_rank, product, ts_min, ts_max, delta, delay, rank, scale, \
    ts_argmax, ts_argmin, decay_linear


def f1(close: np.array, returns: np.array, std_window: int = 20, argmax_window: int = 5) -> np.array:
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

    close[zero_less] = stddev(returns, std_window)[zero_less]

    last_nan = np.hstack(np.argwhere(np.isnan(close)))[-1]

    argmax = ts_argmax(close[last_nan + 1:] ** 2, argmax_window)
    ranks = rank(argmax[argmax_window - 1:])
    return prepand_nans(
        ranks,
        len(close) - len(ranks)
    )


def f2(
    open: np.array,
    close: np.array,
    volume: np.array,
    delta_step: int = 2,
    corr_window: int = 6,
    alpha_sign: int = -1
) -> np.array:
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
    open: np.array,
    volume: np.array,
    corr_window: int = 10,
    alpha_sign: int = -1
) -> np.array:
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


def f4(low: np.array, alpha_sign: int = -1, period: int = 9) -> np.array:
    """
    alpha004
    :param low: input array
    :param alpha_sign:
    :param period:
    :return:
    """
    # (-1 * Ts_Rank(rank(low), 9))
    return alpha_sign * ts_rank(rank(low), period)


def f5(open: np.array, close: np.array, vwap: np.array, period: int = 10) -> np.array:
    """
    alpha005
    :param open: input array
    :param close: input array
    :param vwap: input array
    :param period:
    :return:
    """
    # rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))
    return (rank((open - (ts_sum(vwap, period) / period))) * (
            -1 * abs(rank((close - vwap)))))


def f6(open: np.array, volume: np.array, corr_window: int = 10) -> np.array:
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
        close: np.array,
        volume: np.array,
        sma_window: int = 20,
        delta_window: int = 7,
        ts_rank_window: int = 60) -> np.array:
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
    adv20 = sma(volume, sma_window)
    alpha = -1 * ts_rank(abs(delta(close, delta_window)), ts_rank_window) * sign(
        delta(close, delta_window))

    alpha[adv20 >= volume] = -1
    return alpha


def f8(open: np.array, returns: np.array, ts_sum_window: int = 5, delay_window: int = 10) -> np.array:
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
    sum_open = ts_sum(open, ts_sum_window)
    sum_return = prepand_nans(ts_sum(returns[1:], ts_sum_window), 1)
    gen_change = sum_open * sum_return
    return -1 * (rank(gen_change - delay(gen_change, delay_window)))


def f9(close: np.array, delta_window: int = 1, ts_min_window: int = 5, ts_max_window: int = 5) -> np.array:
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
    cond = (ts_min(delta_close, ts_min_window) > 0) | (ts_max(delta_close, ts_max_window) < 0)
    alpha = -1 * delta_close
    alpha[cond] = delta_close[cond]
    return alpha


def f10(close: np.array, delta_window: int = 1, ts_min_window: int = 4, ts_max_window: int = 4) -> np.array:
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
    cond = (ts_min(delta_close, ts_min_window) > 0) | (ts_max(delta_close, ts_max_window) < 0)
    alpha = -1 * delta_close
    alpha[cond] = delta_close[cond]
    return alpha

def f11(
        close: np.array,
        vwap: np.array,
        volume: np.array,
        delta_window: int = 3,
        ts_min_window: int = 3,
        ts_max_window: int = 3) -> np.array:
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
    tmp = rank(ts_max(vwap_close_diff, ts_min_window)) \
        + rank(ts_min(vwap_close_diff, ts_max_window))
    return tmp * rank(delta(volume, delta_window))


def f12(close: np.array, volume: np.array, delta_window: int = 1) -> np.array:
    """
    alpha012
    :param close: input array
    :param volume: input array
    :param delta_window:
    :return:
    """
    # (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    return sign(delta(volume, delta_window)) * (-1 * delta(close, delta_window))


def f13(close: np.array, volume: np.array, cov_window: int = 5) -> np.array:
    """
    alpha013
    :param close: input array
    :param volume: input array
    :param cov_window:
    :return:
    """
    # (-1 * rank(covariance(rank(close), rank(volume), 5)))
    return -1 * rank(covariance(rank(close), rank(volume), cov_window))


def f14(open: np.array, volume: np.array, returns: np.array, corr_window: int = 10, delta_window: int = 3) -> np.array:
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


def f15(high: np.array, volume: np.array, corr_window: int = 3, ts_sum_window: int = 3) -> np.array:
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
    return -1 * ts_sum(rank(corr_res), ts_sum_window)


def f16(high: np.array, volume: np.array, cov_window: int = 5) -> np.array:
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
        close: np.array,
        volume: np.array,
        sma_window: int = 20,
        first_ts_rank_window: int = 10,
        delta_window: int = 1,
        last_ts_rank_window: int = 5) -> np.array:
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
    adv20 = sma(volume, sma_window)
    last_ts_rank = ts_rank((volume / adv20), last_ts_rank_window)
    last_ts_rank[:sma_window + last_ts_rank_window - 2] = np.nan

    a = rank(ts_rank(close, first_ts_rank_window))
    b = rank(delta(delta(close, delta_window), delta_window))
    c = rank(last_ts_rank)
    return -1 * (a * b * c)


def f18(open: np.array, close: np.array, corr_window: int = 10, stddev_window: int = 5) -> np.array:
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
    nan_to_value(corr_res)
    return -1 * (rank(stddev(abs(close - open), stddev_window) + close - open + corr_res))


def f19(
        close: np.array,
        returns: np.array,
        delay_window: int = 7,
        delta_window: int = 7,
        ts_sum_window: int = 250) -> np.array:
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
            1 + rank(1 + prepand_nans(ts_sum(returns[1:], ts_sum_window), 1))
        )
    )


def f20(open: np.array, high: np.array, low: np.array, close: np.array) -> np.array:
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


def f21(close: np.array, volume: np.array) -> np.array:
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
    cond_1 = sma(close, 8) + stddev(close, 8) < sma(close, 2)
    cond_2 = sma(close, 2) < sma(close, 8) - stddev(close, 8)
    cond_3 = sma(volume, 20) / volume < 1
    alpha = np.zeros_like(close)
    alpha[cond_1 | cond_3] = -1
    alpha[cond_2] = 1
    return alpha


def f22(high: np.array, close: np.array, volume: np.array) -> np.array:
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
    nan_to_value(corr_res)
    return -1 * delta(corr_res, 5) * rank(stddev(close, 20))


def f23(high: np.array, close: np.array) -> np.array:
    """
    alpha023
    :param high: input array
    :param close: input array
    :return:
    """
    # (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    cond = sma(high, 20) < high
    alpha = np.zeros_like(close)
    alpha[cond] = -1 * np.nan_to_num(delta(high, 2))[cond]
    return alpha


def f24(close: np.array) -> np.array:
    """
    alpha024
    :param close: input array
    :return:
    """
    # ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05)
    # ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05))
    # ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
    cond = delta(sma(close, 100), 100) / delay(close, 100) <= 0.05
    alpha = -1 * delta(close, 3)
    alpha[cond] = -1 * (close - ts_min(close, 100))[cond]
    return alpha


def f25(high: np.array, close: np.array, volume: np.array, returns: np.array, vwap: np.array) -> np.array:
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
    adv20 = sma(volume, 20)
    return rank(-1 * returns * adv20 * vwap * (high - close))


def f26(high: np.array, volume: np.array) -> np.array:
    """
    alpha026
    :param high: input array
    :param volume: input array
    :return:
    """
    # (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    corr_res = correlation(ts_rank(volume, 5), ts_rank(high, 5), 5)
    nan_to_value(corr_res)
    return -1 * ts_max(corr_res, 3)


def f27(volume: np.array, vwap: np.array) -> np.array:
    """
    alpha027
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # 0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0) ?
    # (-1 * 1) : 1)
    corr = correlation(rank(volume), rank(vwap), 6)
    a = rank(sma(corr, 2))
    cond = 0.5 < a
    a[cond] = -1
    a[~cond] = 1
    return a


def f28(high: np.array, low: np.array, close: np.array, volume: np.array, returns: np.array, vwap: np.array) -> np.array:
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
    adv20 = sma(volume, 20)
    corr_res = correlation(adv20, low, 5)
    nan_to_value(corr_res)
    return scale(((corr_res + ((high + low) / 2)) - close))


def f29(close: np.array, returns: np.array) -> np.array:
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
    expr = ts_min(rank(rank(-1 * rank(delta(close - 1, 5)))), 2)
    expr = prepand_nans(ts_sum(expr[6:], 1), 6)
    expr = rank(rank(scale(log(expr))))
    expr = product(expr, 1)
    expr = ts_min(expr, 5)
    return expr + a


def f30(close: np.array, volume: np.array) -> np.array:
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
    return ((1.0 - rank(inner)) * ts_sum(volume, 5)) / ts_sum(
        volume, 20)


def f31(low: np.array, close: np.array, volume: np.array) -> np.array:
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
    adv20 = sma(volume, 20)
    corr_res = correlation(adv20, low, 12)
    nan_to_value(corr_res)

    p1 = rank(rank(rank(
        decay_linear((-1 * rank(rank(delta(close, 10)))),
                     10))))
    p2 = rank((-1 * delta(close, 3)))
    p3 = sign(scale(corr_res))
    return p1 + p2 + p3


def f32(close: np.array, vwap: np.array) -> np.array:
    """
    alpha032
    :param close: input array
    :param vwap: input array
    :return:
    """
    # (scale(((sum(close, 7) / 7) - close)) +
    # (20 * scale(correlation(vwap, delay(close, 5),230))))
    return scale(((sma(close, 7) / 7) - close)) + (20 * scale(
        correlation(vwap, delay(close, 5), 230)))


def f33(open: np.array, close: np.array) -> np.array:
    """
    alpha033
    :param open: input array
    :param close: input array
    :return:
    """
    # rank((-1 * ((1 - (open / close))^1)))
    return rank(-1 + (open / close))


def f34(close: np.array, returns: np.array) -> np.array:
    """
    alpha034
    :param close: input array
    :param returns: input array
    :return:
    """
    # rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5))))
    # + (1 - rank(delta(close, 1)))))
    inner = stddev(returns, 2) / stddev(returns, 5)
    nan_to_value(inner, 1)
    return rank(1 - rank(inner) + (1 - rank(delta(close, 1))))


def f35(high: np.array, low: np.array, close: np.array, volume: np.array, returns: np.array) -> np.array:
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


def f36(open: np.array, close: np.array, volume: np.array, returns: np.array, vwap: np.array) -> np.array:
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
    adv20 = sma(volume, 20)
    corr = correlation((close - open), delay(volume, 1), 15)
    tmp = (((((2.21 * rank(corr)) + (0.7 * rank((open - close))))
             + (0.73 * rank(ts_rank(delay((-1 * returns), 6), 5))))
            + rank(abs(correlation(vwap, adv20, 6))))
           + (0.6 * rank((((sma(close, 200) / 200) - open)
                          * (close - open)))))
    return tmp


def f37(open: np.array, close: np.array) -> np.array:
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


def f38(open: np.array, close: np.array) -> np.array:
    """
    alpha038
    :param open: input array
    :param close: input array
    :return:
    """
    # ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    inner = close / open
    nan_to_value(inner, 1)
    return -1 * rank(ts_rank(close, 10)) * rank(inner)


def f39(close: np.array, volume: np.array, returns: np.array) -> np.array:
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
    adv20 = sma(volume, 20)
    tmp = (1 - rank(decay_linear((volume / adv20), 9)))
    tmp = (-1 * rank(delta(close, 7) * tmp))
    return tmp * (1 + rank(sma(returns, 250)))


def f40(high: np.array, volume: np.array) -> np.array:
    """
    alpha040
    :param high: input array
    :param volume: input array
    :return:
    """
    # ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    return -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)


def f41(high: np.array, low: np.array, vwap: np.array) -> np.array:
    """
    alpha041
    :param high: input array
    :param low: input array
    :param vwap: input array
    :return:
    """
    # (((high * low)^0.5) - vwap)
    return pow((high * low), 0.5) - vwap


def f42(close: np.array, vwap: np.array) -> np.array:
    """
    alpha042
    :param close: input array
    :param vwap: input array
    :return:
    """
    # (rank((vwap - close)) / rank((vwap + close)))
    return rank((vwap - close)) / rank((vwap + close))


def f43(close: np.array, volume: np.array) -> np.array:
    """
    alpha043
    :param close: input array
    :param volume: input array
    :return:
    """
    # (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    adv20 = sma(volume, 20)
    tmp = ts_rank(volume / adv20, 20) * ts_rank((-1 * delta(close, 7)), 8)
    tmp[:38] = np.nan
    return tmp


def f44(high: np.array, volume: np.array) -> np.array:
    """
    alpha044
    :param high: input array
    :param volume: input array
    :return:
    """
    # (-1 * correlation(high, rank(volume), 5))
    corr_res = correlation(high, rank(volume), 5)
    nan_to_value(corr_res, 0)
    return -1 * corr_res


def f45(close: np.array, volume: np.array) -> np.array:
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
    nan_to_value(corr_res, 0)
    return -1 * (rank(sma(delay(close, 5), 20)) * corr_res *
                 rank(correlation(ts_sum(close, 5),
                                  ts_sum(close, 20),
                                  2)))


def f46(close: np.array) -> np.array:
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


def f47(high: np.array, close: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv20 = sma(volume, 20)
    return ((((rank((1 / close)) * volume) / adv20)
             * ((high * rank((high - close)))
                / (sma(high, 5))))
            - rank((vwap - delay(vwap, 5))))


def f49(close: np.array) -> np.array:
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


def f50(volume: np.array, vwap: np.array) -> np.array:
    """
    alpha050
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    return (-1 * ts_max(
        rank(correlation(rank(volume), rank(vwap), 5)), 5))


def f51(close: np.array) -> np.array:
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


def f52(low: np.array, volume: np.array, returns: np.array) -> np.array:
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
    return (((-1 * ts_min(low, 5) + delay(ts_min(low, 5), 5)) *
             rank(((prepand_nans(ts_sum(returns, 240), 1) - prepand_nans(ts_sum(returns, 20), 1)) /
                   220))) * ts_rank(volume, 5))


def f53(high: np.array, low: np.array, close: np.array) -> np.array:
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
    nan_to_value(tmp, 0)
    return -1 * delta(tmp, 9)


def f54(open: np.array, high: np.array, low: np.array, close: np.array) -> np.array:
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


def f55(high: np.array, low: np.array, close: np.array, volume: np.array) -> np.array:
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
    divisor = ts_max(high, 12) - ts_min(low, 12)
    inner = (close - ts_min(low, 12)) / divisor
    res = correlation(rank(inner), rank(volume), 6)
    nan_to_value(res, 0)
    return -1 * res


def f57(close: np.array, vwap: np.array) -> np.array:
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


def f60(high: np.array, low: np.array, close: np.array, volume: np.array) -> np.array:
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


def f61(volume: np.array, vwap: np.array) -> np.array:
    """
    alpha061
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # (rank((vwap - ts_min(vwap, 16.1219))) <
    # rank(correlation(vwap, adv180, 17.9282)))
    adv180 = sma(volume, 180)
    return (rank((vwap - ts_min(vwap, 16))) < rank(
        correlation(vwap, adv180, 18)))


def f62(open: np.array, high: np.array, low: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv20 = sma(volume, 20)
    return ((rank(correlation(vwap, sma(adv20, 22), 10))
             < rank((rank(open) + rank(open))
                    < (rank((high + low) / 2)
                       + rank(high)))) * -1)


def f64(open: np.array, high: np.array, low: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv120 = sma(volume, 120)
    return ((rank(correlation(
        sma(((open * 0.178404) + (low * (1 - 0.178404))), 13),
        sma(adv120, 13), 17)) < rank(
            delta(((((high + low) / 2) * 0.178404)
                   + (vwap * (1 - 0.178404))), 4))) * -1)


def f65(open: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv60 = sma(volume, 60)
    return ((rank(correlation(
        ((open * 0.00817205) + (vwap * (1 - 0.00817205))),
        sma(adv60, 9), 6)) < rank(
        (open - ts_min(open, 14)))) * -1)


def f66(open: np.array, high: np.array, low: np.array, vwap: np.array) -> np.array:
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


def f68(high: np.array, low: np.array, close: np.array, volume: np.array) -> np.array:
    """
    alpha068
    :param high: input array
    :param low: input array
    :param close: input array
    :param volume: input array
    :return:
    """
    # ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333)
    # < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157)))
    # * -1)
    adv15 = sma(volume, 15)
    return -1 * ((ts_rank(
        correlation(rank(high), rank(adv15), 9),
        14
    ) < rank(
        delta((close * 0.518371 + low * (1 - 0.518371)), 1)))
    )


def f71(open: np.array, low: np.array, close: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv180 = sma(volume, 180)
    p1 = ts_rank(decay_linear(
        correlation(ts_rank(close, 3), ts_rank(adv180, 12),
                    18), 4), 16)
    p2 = ts_rank(decay_linear((rank(
        ((low + open) - (vwap + vwap))) ** 2), 16), 4)

    alpha = nans_array(p1.size)

    cond = p1 >= p2
    alpha[cond] = p1[cond]

    cond = p2 >= p1
    alpha[cond] = p2[cond]

    return alpha


def f72(high: np.array, low: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv40 = sma(volume, 40)
    return (rank(decay_linear(
        correlation(((high + low) / 2), adv40, 9),
        10)) / rank(decay_linear(correlation(
            ts_rank(vwap, 4),
            ts_rank(volume, 19), 7),
            3)))


def f73(open: np.array, low: np.array, vwap: np.array) -> np.array:
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

    alpha = nans_array(open.size)

    cond = p1 >= p2
    alpha[cond] = p1[cond]

    cond = p2 >= p1
    alpha[cond] = p2[cond]

    return -1 * alpha


def f74(high: np.array, close: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv30 = sma(volume, 30)
    return ((rank(correlation(close, sma(adv30, 37), 15)) < rank(
        correlation(
            rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))),
            rank(volume), 11))) * -1)


def f75(low: np.array, volume: np.array, vwap: np.array) -> np.array:
    """
    alpha075
    :param low: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # (rank(correlation(vwap, volume, 4.24304)) <
    # rank(correlation(rank(low), rank(adv50),12.4413)))
    adv50 = sma(volume, 50)
    return (rank(correlation(vwap, volume, 4)) < rank(
        correlation(rank(low), rank(adv50), 12)))


def f77(high: np.array, low: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv40 = sma(volume, 40)
    p1 = rank(decay_linear(((((high + low) / 2) + high) - (
        vwap + high)), 20))
    p2 = rank(decay_linear(
        correlation(((high + low) / 2), adv40, 3),
        6))
    alpha = nans_array(high.size)

    cond = p1 >= p2
    alpha[cond] = p2[cond]

    cond = p2 >= p1
    alpha[cond] = p1[cond]

    return alpha


def f78(low: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv40 = sma(volume, 40)
    return (rank(correlation(
        ts_sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 20),
        prepand_nans(ts_sum(adv40[39:], 20), 39), 7)) ** rank(correlation(rank(vwap), rank(volume), 6)))


def f81(volume: np.array, vwap: np.array) -> np.array:
    """
    alpha081
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),
    # 8.47743))^4)), 14.9655))) <
    # rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    adv10 = sma(volume, 10)
    return ((rank(log(product(
        rank((rank(correlation(vwap, prepand_nans(ts_sum(adv10[9:], 50), 9), 8)) ** 4)),
        15))) < rank(
        correlation(rank(vwap), rank(volume), 5))) * -1)


def f83(high: np.array, low: np.array, close: np.array, volume: np.array, vwap: np.array) -> np.array:
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
        high - low) / (ts_sum(close, 5) / 5),
        2)
    ) * rank(rank(volume))) / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close)))


def f84(close: np.array, vwap: np.array) -> np.array:
    """
    alpha084
    :param close: input array
    :param vwap: input array
    :return:
    """
    # SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127),
    # delta(close,4.96796))
    return pow(ts_rank((vwap - ts_max(vwap, 15)), 21),
               delta(close, 5))


def f85(high: np.array, low: np.array, close: np.array, volume: np.array) -> np.array:
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
    adv30 = sma(volume, 30)
    return (rank(correlation(
        ((high * 0.876703) + (close * (1 - 0.876703))), adv30,
        10)) ** (rank(correlation(ts_rank(((high + low) / 2), 4),
                                  ts_rank(volume, 10), 7))))


def f86(open: np.array, close: np.array, volume: np.array, vwap: np.array) -> np.array:
    """
    alpha086
    :param open: input array
    :param close: input array
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) <
    # rank(((open+ close) - (vwap + open)))) * -1)
    adv20 = sma(volume, 20)
    return -1 * (ts_rank(correlation(close, sma(adv20, 15), 6), 20) <
                 rank(((open + close) - (vwap + open))))


def f88(open: np.array, high: np.array, low: np.array, close: np.array, volume: np.array) -> np.array:
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
    adv60 = sma(volume, 60)
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

    alpha = nans_array(high.size)

    cond = p1 >= p2
    alpha[cond] = p2[cond]

    cond = p2 >= p1
    alpha[cond] = p1[cond]

    return alpha


def f92(open: np.array, high: np.array, low: np.array, close: np.array, volume: np.array) -> np.array:
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
    adv30 = sma(volume, 30)
    p1 = ts_rank(
        decay_linear(
            ((((high + low) / 2) + close)
             < (low + open)), 15),
        19
    )
    p2 = ts_rank(
        decay_linear(correlation(rank(low), rank(adv30), 8),
                     7), 7)
    alpha = nans_array(high.size)

    cond = p1 >= p2
    alpha[cond] = p2[cond]

    cond = p2 >= p1
    alpha[cond] = p1[cond]

    return alpha


def f94(volume: np.array, vwap: np.array) -> np.array:
    """
    alpha094
    :param volume: input array
    :param vwap: input array
    :return:
    """
    # ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(
    # vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    adv60 = sma(volume, 60)
    r = ((rank((vwap - ts_min(vwap, 12))) ** (
        ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18),
                3)) * -1))

    r[:81] = np.nan
    return r


def f95(open: np.array, high: np.array, low: np.array, volume: np.array) -> np.array:
    """
    alpha095
    :param open: input array
    :param high: input array
    :param low: input array
    :param volume: input array
    :return:
    """
    # (rank((open - ts_min(open, 12.4105))) < Ts_Rank((
    # rank(correlation(sum(((high + low)/ 2), 19.1351),
    # sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    adv40 = sma(volume, 40)

    tmp = (rank(
        correlation(
            ts_sum(((high + low) / 2), 19),
            prepand_nans(ts_sum(adv40[39:], 19), 39),
            13)) ** 5)
    tmp = ts_rank(tmp, 12)

    return rank((open - ts_min(open, 12))) < tmp


def f96(close: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv60 = sma(volume, 60)
    p1 = ts_rank(decay_linear(
        correlation(rank(vwap), rank(volume), 4), 4), 8)
    p2 = ts_rank(decay_linear(
        ts_argmax(correlation(ts_rank(close, 7), ts_rank(adv60, 4), 4),
                  13), 14), 13)

    alpha = nans_array(p1.size)

    cond = p1 >= p2
    alpha[cond] = p1[cond]

    cond = p2 >= p1
    alpha[cond] = p2[cond]

    return -1 * alpha


def f98(open: np.array, volume: np.array, vwap: np.array) -> np.array:
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
    adv5 = sma(volume, 5)
    adv15 = sma(volume, 15)
    return (rank(
        decay_linear(
            correlation(vwap, sma(adv5, 26), 5),
            7
        )) - rank(decay_linear(ts_rank(ts_argmin(
            correlation(rank(open), rank(adv15), 21),
            9), 7), 8)))


def f99(high: np.array, low: np.array, volume: np.array) -> np.array:
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
    adv60 = sma(volume, 60)
    return ((rank(correlation(ts_sum(((high + low) / 2), 20),
                              prepand_nans(ts_sum(adv60[59:], 20), 59), 9)) < rank(
        correlation(low, volume, 6))) * -1)


def f101(open: np.array, high: np.array, low: np.array, close: np.array) -> np.array:
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


PREFIX = "alphas101"
FEATURE_FUNCS = dict(inspect.getmembers(sys.modules[__name__], feature_filter))

__all__ = ("FEATURE_FUNCS", "PREFIX", "calc_all", *FEATURE_FUNCS.keys())


def calc_all(
    data: pd.DataFrame,
    param_set: List[Dict] = None,
    column_names: dict = dict(
        open="open",
        high = "high",
        low = "low",
        close = "close",
        volume_buy="volume_buy",
        volume_sell="volume_sell",
        volume_quote_buy="volume_quote_buy",
        volume_quote_sell="volume_quote_sell"
    )
) -> pd.DataFrame:

    input_data  = {}
    input_data["open"] = data[column_names["open"]].values
    input_data["high"] = data[column_names["high"]].values
    input_data["low"] = data[column_names["low"]].values
    input_data["close"] = data[column_names["close"]].values
    input_data["volume"] = (data[column_names["volume_buy"]] + data[column_names["volume_sell"]]).values
    input_data["returns"] = data[column_names["close"]].pct_change().values
    input_data["vwap"] = ((data[column_names["volume_quote_buy"]] + data[column_names["volume_quote_sell"]])
            / (data[column_names["volume_buy"]] + data[column_names["volume_sell"]])).values

    df = pd.DataFrame([])
    for name, func in FEATURE_FUNCS.items():
        params = {k: input_data[k] for k in inspect.getfullargspec(func).args if k in input_data}
        df[f"{PREFIX}_{name}"] = func(**params)

    return df
