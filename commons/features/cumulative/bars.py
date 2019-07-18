import numpy as np
import pandas as pd


def f1(close: np.array, threshold: float) -> np.array:
    """Cusum fixed bars feature.

    :param close: series of bar close
    :param timeframe: cusum threshold.
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = [0]
    s_pos, s_neg = 0, 0
    for value in np.diff(np.log(close)):
        s_pos = max(0.0, s_pos + value)
        s_neg = min(0.0, s_neg + value)
        if s_neg < -threshold:
            s_neg = 0
            bars.append(1)
        elif s_pos > threshold:
            s_pos = 0
            bars.append(1)
        else:
            bars.append(0)
    feature = np.array(bars)
    assert feature.shape == close.shape
    return feature


def f2(ticks: np.array, threshold: float) -> np.array:
    """Tick fixed bars feature.

    :param ticks: series of tick amounts
    :param threshold: tick threshold
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = []
    ticks_sum = 0
    for value in ticks:
        ticks_sum += value
        if ticks_sum >= threshold:
            bars.append(1)
            ticks_sum = 0
        else:
            bars.append(0)
        
    feature = np.array(bars)
    assert feature.shape == ticks.shape
    return feature


def f3(volume: np.array, threshold: float) -> np.array:
    """Base volume fixed bars feature.

    :param volume: series of bar volume in base currency
    :param threshold: volume threshold
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = []
    volume_sum = 0
    for value in volume:
        volume_sum += value
        if volume_sum >= threshold:
            bars.append(1)
            volume_sum = 0
        else:
            bars.append(0)
    feature = np.array(bars)
    assert feature.shape == volume.shape
    return feature


def f4(quote_volume: np.array, quote_threshold: float) -> np.array:
    """Quote volume fixed bars feature.

    :param volume: series of bar volume in quote currency
    :param quote_threshold: quote volume threshold
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = []
    quote_volume_sum = 0
    for value in quote_volume:
        quote_volume_sum += value
        if quote_volume_sum >= quote_threshold:
            bars.append(1)
            quote_volume_sum = 0
        else:
            bars.append(0)
    feature = np.array(bars)
    assert feature.shape == quote_volume.shape
    return feature


def f5(open: np.array, close: np.array, pct_threshold: float) -> np.array:
    """Range percentage fixed bars feature.

    :param open: series of bar open
    :param close: series of bar close
    :param pct_threshold: percentage threshold
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = []
    get_limits = lambda x: (x * (1 + pct_threshold), x * (1 - pct_threshold))
    upper_limit, lower_limit = get_limits(open[0])
    for value in close:
        if value > upper_limit or value < lower_limit:
            bars.append(1)
            upper_limit, lower_limit = get_limits(value)
        else:
            bars.append(0)
    feature = np.array(bars)
    assert feature.shape == close.shape
    return feature


def f6(open: np.array, close: np.array, abs_threshold: float) -> np.array:
    """Range absolute fixed bars feature.

    :param open: series of bar open
    :param close: series of bar close
    :param abs_threshold: absolute threshold
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = []
    get_limits = lambda x: (x + abs_threshold, x - abs_threshold)
    upper_limit, lower_limit = get_limits(open[0])
    for value in close:
        if value > upper_limit or value < lower_limit:
            bars.append(1)
            upper_limit, lower_limit = get_limits(value)
        else:
            bars.append(0)
    feature = np.array(bars)
    assert feature.shape == close.shape
    return feature
