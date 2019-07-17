import numpy as np
import pandas as pd


def f1(datetimes: np.array, timeframe: int) -> pd.Series:
    """Time fixed bars feature.

    :param datetimes: series of datetime
    :param timeframe: timeframe in sec.
    :return: binary series where 1 means bar generation event, 0 means bar
    doesn't exist yet.
    """
    index = pd.date_range(datetimes[0], datetimes[-1], freq='s')
    feature = pd.Series(np.zeros(index.size), index=index)
    feature.iloc[timeframe - 1::timeframe] = 1
    return feature[(feature == 1) | (feature.index.isin(datetimes))]


def f2() -> pd.Series:
    pass
