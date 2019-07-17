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


def f2(ticks: np.array, threshold: int) -> np.array:
    """Tick fixed bars feature.

    :param ticks: series of tick amounts
    :param threshold: tick threshold
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet
    """
    bars = []
    ticks_sum = 0
    for tick in ticks:
        ticks_sum += tick
        if ticks_sum >= threshold:
            bars.append(1)
            ticks_sum = 0
        else:
            bars.append(0)
    feature = np.array(bars)
    assert feature.shape == ticks.shape
    return feature
