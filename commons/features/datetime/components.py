from typing import List, Dict

import pandas as pd

from .utils import add_datepart

PREFIX = "dt_components"


def calc_all(data: pd.DataFrame, param_sets: List[Dict] = None, column_names=None) -> pd.DataFrame:
    """
    Extract datetime components features from datetime by fastai module
    Features:
        'year', 'month', 'week', 'day', 'dayofweek', 'dayofyear',
        'is_month_end', 'is_month_start', 'is_quarter_end', 'is_quarter_start',
        'is_year_end', 'is_year_start', 'hour', 'minute', 'second', 'elapsed', 'timediff'

    :param pd.DataFrame data: df with DatetimeIndex. No required columns
    :param param_sets: unused template parameter
    :param column_names: unused template parameter
    :return: df with features
    """
    ts_col = "ts"
    df = data.copy()
    df[ts_col] = df.index

    add_datepart(df, ts_col, drop=False, time=True)

    df.columns = [
        (col[len(ts_col):] if col != ts_col and col.startswith(ts_col) else col).lower() for col in df.columns
    ]

    df["timediff"] = (df[ts_col] - df[ts_col].shift()).fillna(0).apply(lambda x: x.total_seconds())
    df[df.dtypes[(df.dtypes == bool)].index] *= 1

    df = df.drop(ts_col, axis=1)
    df.columns = [f"{PREFIX}_{cname}" for cname in df.columns]
    return df
