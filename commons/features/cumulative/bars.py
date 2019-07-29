import numpy as np

from commons.features.utils import get_inputs


def ft(ticks: np.array, threshold: float) -> np.array:
    """Fixed Ticks

    Tick accumulation feature. Fixed threshold.

    :param ticks: Series of ticks
    :param threshold: Event is generated after cumulative number of ticks reaches this threshold
    :return: Binary series. 1 signals firing of accumulation event.
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


def fv(volume: np.array, threshold: float) -> np.array:
    """Fixed Volume

    Volume accumulation feature. Fixed threshold.

    :param volume: Series of trading volume
    :param threshold: Event is generated after cumulative volume reaches this threshold
    :return: Binary series. 1 signals firing of accumulation event.
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

def fr(open: np.array, close: np.array, pct_threshold: float) -> np.array:
    """Fixed Range

    Price move (range) accumulation feature. Fixed % range.

    :param open: Series of open prices
    :param close: Series of close prices
    :param pct_threshold: Event is generated after price moves by more percent than this threshold
    :return: Binary series. 1 signals firing of accumulation event.
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


def frft(open: np.array, close: np.array, ticks: np.array,
         range_pct_threshold: float, ticks_threshold: int) -> np.array:
    """Fixed Range Fixed Ticks

    Price move (range) and ticks accumulation feature. Fixed % range, fixed n ticks.

    :param open: Series of open prices
    :param close: Series of close prices
    :param ticks: Series of ticks
    :param range_pct_threshold: Range condition satisfied is  after price moves by more percent than this threshold
    :param ticks_threshold: Ticks condition is satisfied after cumulative number of ticks reaches this threshold
    :return: Binary series. 1 signals firing of accumulation event when both conditions are satisfied.
    """
    bars = []
    ticks_sum = 0
    get_limits = lambda x: (x * (1 + range_pct_threshold),
                            x * (1 - range_pct_threshold))  # yapf: disable
    upper_limit, lower_limit = get_limits(open[0])
    for idx in range(len(close)):
        ticks_sum += ticks[idx]
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
