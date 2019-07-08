import numpy as np
import pandas as pd


def group_period(prefix_period, dt_component_name):
    def group(n):
        return lambda x: x.to_period(prefix_period).astype(str) + (getattr(x, dt_component_name) // n).astype(str)

    return group


FREQ_GROUP = {
    "s": group_period("T", "second"),
    "m": group_period("H", "minute"),
    "h": group_period("D", "hour"),
    "D": group_period("M", "day"),
    "W": group_period("Y", "week"),
    "M": group_period("Y", "month"),
    "Q": group_period("Y", "quarter"),
    "Y": lambda n: lambda x: x.year // n
}


def get_group_func(tf):
    if tf == "ALL":
        return lambda x: 1

    return FREQ_GROUP[tf[-1]](
        int(tf[:-1])
    )


def is_extremum(series: pd.Series, period: str, maximum: bool) -> np.array:
    """
    Find extremums in the period
    :param pd.Series series: series with DatetimeIndex
    :param str period: tf in format `<int>(s|m|h|D|W|M|Q|Y)` Example: 35s, 1h, 3D, 15m, etc.
    :param bool maximum: if maximum is True then find highest
    :return: array of 0 and 1 for each series row
    """
    df = pd.DataFrame([])
    df["series"] = series

    df["group"] = get_group_func(period)(df.index)
    df["is_extremum"] = 0

    if maximum:
        extremum_find = lambda x: df[x].series.idxmax()
    else:
        extremum_find = lambda x: df[x].series.idxmin()

    for group in df.group.unique():
        selected_group = df.group == group
        idx = extremum_find(selected_group)

        df_selected = df[selected_group]

        df.loc[df_selected[df_selected.series == df_selected.loc[idx, "series"]].index, "is_extremum"] = 1
        # df.loc[idx, "is_extremum"] = 1

    return df.is_extremum.values


def time_since_extremum(series: pd.Series, period: str, maximum: bool) -> np.array:
    """
    Calculate time since extremums
    :param pd.Series series: series with DatetimeIndex
    :param str period: tf in format `<int>(s|m|h|D|W|M|Q|Y)` Example: 35s, 1h, 3D, 15m, etc.
    :param bool maximum: if maximum is True then find highest
    :return: array with seconds since extremums
    """
    df = pd.DataFrame(
        is_extremum(series, period, maximum),
        index=series.index,
        columns=["extremum"]
    )

    extremum_idxs = df.extremum[df.extremum == 1].index

    df["timediff"] = 0
    for i in range(1, len(extremum_idxs)):
        start, end = extremum_idxs[i - 1], extremum_idxs[i]

        df.loc[start:end, "timediff"] = (df.loc[start:end].index - start) / 1e9

    last_idx = extremum_idxs[-1]
    df.loc[last_idx:, "timediff"] = (df.loc[last_idx:].index - last_idx) / 1e9

    return df.timediff.values


def time_in(series: pd.Series, drawdown: bool) -> np.array:
    """
    Calculate time in drawdon/roll up
    :param pd.Series series: series with DatetimeIndex
    :param bool drawdown: if drawdown is True calculate for drawdown
    :return: array with seconds
    """
    df = pd.DataFrame([])
    df["series"] = series
    df["change"] = df.series.pct_change().fillna(0)

    if drawdown:
        df["mark"] = (df.change < 0) * 1
    else:
        df["mark"] = (df.change > 0) * 1

    df["diff_1"] = df.mark.diff().fillna(0)
    df["diff_2"] = df.diff_1.shift(-1).fillna(0)

    dd_start = df[df.diff_2 == 1.].index
    dd_second = df[df.diff_1 == 1.].index
    dd_end = df[df.diff_2 == -1.].index

    df["time_in"] = 0
    for start, second, end in zip(dd_start, dd_second, dd_end):
        df.loc[second:end, "time_in"] = (df.loc[second:end].index - start) / 1e9

    return df.time_in.values


def change_since_period(series: pd.Series, period: str, maximum: bool) -> np.array:
    """
    Calculate change % since extremum in period
    :param pd.Series series: series with DatetimeIndex
    :param str period: tf in format `<int>(s|m|h|D|W|M|Q|Y)` Example: 35s, 1h, 3D, 15m, etc.
    :param bool maximum: if maximum is True then calculate since highest
    :return: array with seconds
    """
    df = pd.DataFrame([])
    df["series"] = series
    df["extremum"] = is_extremum(series, period, maximum)

    extremum_idxs = df.extremum[df.extremum == 1].index.append(df.index[-1:])

    df["change_since_extremum"] = 0
    for i in range(1, len(extremum_idxs)):
        start, end = extremum_idxs[i - 1], extremum_idxs[i]
        extrm_in_period = df.loc[start, "series"]
        df.loc[start:end, "change_since_extremum"] = (df.loc[start:end].series - extrm_in_period) / extrm_in_period

    return df.change_since_extremum.values
