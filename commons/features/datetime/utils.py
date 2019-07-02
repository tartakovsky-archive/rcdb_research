from datetime import timedelta

import pandas as pd
import numpy as np
from workalendar.registry import registry


def is_date_in_timestamp(timestamp: np.array, timestamp_check: np.array):
    df = pd.DataFrame(index=timestamp)
    df["checked"] = 0
    for dt in timestamp_check:
        dt = str(dt)
        if dt in df.index:
            df.loc[dt, "checked"] = 1
    return df.checked.values


def is_hour_away(timestamp: np.array, search_timestamp: np.array):
    idxs = search_timestamp.searchsorted(
        timestamp,
        side="left"
    )

    idxs[idxs == len(idxs)] = len(idxs) - 1
    timediff = np.vectorize(timedelta.total_seconds)(search_timestamp[idxs] - timestamp)
    return (0. <= timediff) & (timediff <= 3600.) * 1


def supported_countries():
    return registry.region_registry.keys()
