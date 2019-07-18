import numpy as np

from commons.features.utils import get_inputs

# Feature functions fixed bars region


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


def f2(ticks_buy: np.array, ticks_sell: np.array,
       threshold: float) -> np.array:
    """Tick fixed bars feature.

    :param ticks_buy: series of ticks buy
    :param ticks_sell: series of ticks sell
    :param threshold: tick threshold
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = []
    ticks_sum = 0
    for value in ticks_buy + ticks_sell:
        ticks_sum += value
        if ticks_sum >= threshold:
            bars.append(1)
            ticks_sum = 0
        else:
            bars.append(0)

    feature = np.array(bars)
    assert feature.shape == ticks_buy.shape
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
            upper_limit, lower_limit = get_limits(value)
            bars.append(1)
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
            upper_limit, lower_limit = get_limits(value)
            bars.append(1)
        else:
            bars.append(0)
    feature = np.array(bars)
    assert feature.shape == close.shape
    return feature


# Feature functions adaptive bars region

# def f7(datetimes: np.array, volume_buy: np.array, volume_sell: np.array,
#        avg_per: int, window: int) -> np.array:
#     """Base volume adaptive bars feature.

#     :param datetimes: series of datetame index that match volume series
#     :param volume_buy: series of volume buy
#     :param volume_sell: series of volume sell
#     :param avg_per: average per time in seconds
#     :param window: rolling window for avg_per calculation in seconds
#     :return: binary series where 1 means bar generation event, 0 means bar
#     doesn't exist yet
#     """
#     # TODO: Make decision how to realize rolling window by timestamp
#     pass

# Feature functions hybrid bars region


def f8(open: np.array, close: np.array, ticks_buy: np.array,
       ticks_sell: np.array, range_pct_threshold: float,
       ticks_threshold: int) -> np.array:
    """Range percentage fixed and tick fixed bars bars feature.

    :param open: series of bar open
    :param close: series of bar close
    :param ticks_buy: series of ticks buy
    :param ticks_sell: series of ticks sell
    :param range_pct_threshold: range percentage threshold
    :param ticks_threshold: ticks threshold
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = []
    ticks_sum = 0
    get_limits = lambda x: (x * (1 + range_pct_threshold),
                            x * (1 - range_pct_threshold))  # yapf: disable
    upper_limit, lower_limit = get_limits(open[0])
    for idx in range(len(close)):
        ticks_sum += ticks_buy[idx] + ticks_sell[idx]
        if ticks_sum >= ticks_threshold \
                and (close[idx] > upper_limit or close[idx] < lower_limit):
            upper_limit, lower_limit = get_limits(close[idx])
            ticks_sum = 0
            bars.append(1)
        else:
            bars.append(0)
    feature = np.array(bars)
    assert feature.shape == close.shape
    return feature


def f9(open: np.array, close: np.array, ticks_buy: np.array,
       ticks_sell: np.array, range_abs_threshold: float,
       ticks_threshold: int) -> np.array:
    """Range percentage fixed and tick fixed bars bars feature.

    :param open: series of bar open
    :param close: series of bar close
    :param ticks_buy: series of ticks buy
    :param ticks_sell: series of ticks sell
    :param range_abs_threshold: range absolute threshold
    :param ticks_threshold: ticks threshold
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = []
    ticks_sum = 0
    get_limits = lambda x: (x + range_abs_threshold, x - range_abs_threshold)
    upper_limit, lower_limit = get_limits(open[0])
    for idx in range(len(close)):
        ticks_sum += ticks_buy[idx] + ticks_sell[idx]
        if ticks_sum >= ticks_threshold \
                and (close[idx] > upper_limit or close[idx] < lower_limit):
            upper_limit, lower_limit = get_limits(close[idx])
            ticks_sum = 0
            bars.append(1)
        else:
            bars.append(0)
    feature = np.array(bars)
    assert feature.shape == close.shape
    return feature


# Helpers region

features_list = [value for key, value in locals().items() if key[1:].isdigit()]
inputs = get_inputs(features_list)
