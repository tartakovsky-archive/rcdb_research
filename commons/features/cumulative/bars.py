import numpy as np
import pandas as pd


def f1(datetimes: np.array, timeframe: int) -> pd.Series:
    """Time fixed bars feature.

    :param datetimes: series of datetime
    :param timeframe: timeframe in sec.
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    # TODO: Find out could we change source series shape
    index = pd.date_range(datetimes[0], datetimes[-1], freq='s')
    feature = pd.Series(np.zeros(index.size), index=index)
    feature.iloc[timeframe - 1::timeframe] = 1
    return feature[(feature == 1) | (feature.index.isin(datetimes))]


def f2(ticks: np.array, threshold: float) -> np.array:
    """Tick fixed bars feature.

    :param ticks: series of tick amounts
    :param threshold: tick threshold
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = []
    ticks_sum = 0
    for amount in ticks:
        ticks_sum += amount
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
    for amount in volume:
        volume_sum += amount
        if volume_sum >= threshold:
            bars.append(1)
            volume_sum = 0
        else:
            bars.append(0)
    feature = np.array(bars)
    assert feature.shape == volume.shape
    return feature


# def f4(quote_volume: np.array, threshold: float) -> np.array:
#     """Quote volume fixed bars feature.

#     :param volume: series of bar volume in quote currency
#     :param threshold: quote volume threshold
#     :return: binary series where 1 means bar generation event, 0 means bar
#     doesn't exist yet
#     """
#     bars = []
#     quote_volume_sum = 0
#     for amount in quote_volume:
#         quote_volume_sum += amount
#         if quote_volume_sum >= threshold:
#             bars.append(1)
#             quote_volume_sum = 0
#         else:
#             bars.append(0)
#     feature = np.array(bars)
#     assert feature.shape == quote_volume.shape
#     return feature
