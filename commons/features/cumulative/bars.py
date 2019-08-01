import numpy as np

from commons.features.utils import get_inputs


def pct_threshold(series: np.array, threshold: float) -> np.array:
    """Fixed Volume

    Volume accumulation feature. Fixed threshold.

    :param series: Series of trading volume
    :param threshold: Event is generated after cumulative volume reaches this threshold
    :return: Binary series. 1 signals firing of accumulation event.
    """
    bars = []
    agg_sum = 0
    curr_threshold = None

    for v in series:
        agg_sum += v

        if not curr_threshold:
            curr_threshold = v * threshold

        if agg_sum >= curr_threshold:
            bars.append(1)
            agg_sum = 0
            curr_threshold = None
        else:
            bars.append(0)

    feature = np.array([0] + bars[:-1])
    assert feature.shape == series.shape
    return feature


def price_pct_threshold(open: np.array, close: np.array,
                        threshold_up: float, threshold_down: float = None) -> np.array:
    """Fixed Range

    Price move (range) accumulation feature. Fixed % range.

    :param open: Series of open prices
    :param close: Series of close prices
    :param threshold_up: Event UP is generated after price moves by more percent than this threshold
    :param threshold_down: (if None than equal to threshold_up) Event DOWN is generated after price
                            moves by more percent than this threshold
    :return: Binary series. 1 signals firing of accumulation event.
    """
    if threshold_down is None:
        threshold_down = threshold_up

    bars = []
    upper_limit, lower_limit = None, None
    for v_close, v_open in np.c_[close, open]:
        if upper_limit is None:
            upper_limit, lower_limit = (v_open * (1 + threshold_up), v_open * (1 - threshold_down))

        if v_close >= upper_limit or v_close <= lower_limit:
            upper_limit, lower_limit = None, None
            bars.append(1)
        else:
            bars.append(0)
    feature = np.array([0] + bars[:-1])
    assert feature.shape == close.shape
    return feature


def fixed_threshold(series: np.array, threshold: float) -> np.array:
    """Fixed Threshold

    Fixed threshold accumulating feature.

    :param series: Series of trading volume
    :param threshold: Event is generated after cumulative volume reaches this threshold
    :return: Binary series. 1 signals firing of accumulation event.
    """
    bars = []
    agg_sum = 0
    for v in series:
        agg_sum += v
        if agg_sum >= threshold:
            bars.append(1)
            agg_sum = 0
        else:
            bars.append(0)
    feature = np.array([0] + bars[:-1])
    assert feature.shape == series.shape
    return feature


def rollin_average(seq, window, per=1):
    s = np.insert(np.cumsum(seq), 0, [0])
    r = (s[window:] - s[:-window]) / (window / per)
    return np.append(np.repeat(np.nan, window - 1), r)


def adaptive_threshold(series: np.array, avg_per: int, window: int) -> np.array:
    """Fixed Threshold

    Adaptive accumulating feature. Create new bar when threshold reaches "weekly average for year".

    :param series: Series of trading volume
    :param avg_per: Get rolling avg_per count series avg
    :param window: Series should aggregate window amount of averaged (by avg_per) series
    :return: Binary series. 1 signals firing of accumulation event.
    """

    series_threshold = rollin_average(series, window, avg_per)
    bars = []
    agg_sum = 0
    print(series.shape, series_threshold.shape)

    for [v_series, v_threshold] in np.column_stack([series, series_threshold]):
        if np.isnan(v_threshold):
            bars.append(v_threshold)
            continue

        agg_sum += v_series
        if agg_sum >= v_threshold:
            bars.append(1)
            agg_sum = 0
        else:
            bars.append(0)

    feature = np.array([bars[0]] + bars[:-1])
    print(feature.shape, series.shape)
    assert feature.shape == series.shape
    return feature


def price_pct__series_fixed(open: np.array, close: np.array, price_threshold: float,
                            series: np.array, series_threshold: float) -> np.array:
    """Percent price threshold combined with any fixed threshold series feature

    Price move (range) and ticks accumulation feature. Fixed % range, fixed n ticks.

    :param open: Series of open prices
    :param close: Series of close prices
    :param price_threshold: Range condition satisfied is  after price moves by more percent than this threshold
    :param series: Series of ticks
    :param series_threshold: Ticks condition is satisfied after cumulative number of ticks reaches this threshold
    :return: Binary series. 1 signals firing of accumulation event when both conditions are satisfied.
    """

    bars = []
    upper_limit, lower_limit = None, None
    series_sum = 0
    for [v_open, v_close, v_series] in np.column_stack([open, close, series]):
        series_sum += v_series

        if upper_limit is None:
            upper_limit, lower_limit = (v_open * (1 + price_threshold), v_open * (1 - price_threshold))

        is_price_pct = v_close >= upper_limit or v_close <= lower_limit
        is_fixed = series_sum > series_threshold

        if is_price_pct and is_fixed:
            upper_limit, lower_limit = None, None
            series_sum = 0
            bars.append(1)
        else:
            bars.append(0)

    feature = np.array([0] + bars[:-1])
    assert feature.shape == close.shape
    return feature


def f_combine(f_exact=[], f_continues=[]):
    if not f_exact and not f_continues:
        return []

    f_exact_len = len(f_exact)
    features = np.column_stack(f_exact + f_continues)
    features_count = f_exact_len + len(f_continues)

    bars = []
    f_is_active = np.zeros(features_count, dtype=bool)
    print(0, f_exact_len, features_count)

    for f_values in features:
        # print("exact")
        # exact features
        for f_num in range(0, f_exact_len):
            # print(f_num)
            f_is_active[f_num] = f_values[f_num]

        # print("cont")
        # continues features
        for f_num in range(f_exact_len, features_count):
            # print(f_num)
            f_is_active[f_num] = f_is_active[f_num] or f_values[f_num] == 1

        if f_is_active.all():
            bars.append(1)
            f_is_active = np.zeros(features_count, dtype=bool)
        else:
            bars.append(0)

        print(f_is_active, bars[-1])

    feature = np.array([0] + bars[:-1])
    print(feature.size, features.size)
    assert feature.shape[0] == features.shape[0]
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


features_list = [pct_threshold, price_pct_threshold, fixed_threshold, price_pct__series_fixed, f_combine]
inputs = get_inputs(features_list)
