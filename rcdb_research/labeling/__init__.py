import numpy as np
import pandas as pd

from .triple_barrier import triple_barrier

from .utils import _n_consecutive, _cond_after_n_bars, \
    calculate_daily_volatility


__all__ = (
    "higher_after_n_bars", "lower_after_n_bars",
    "n_consecutive_up", "n_consecutive_down",
    "triple_barrier_labeling", "triple_barrier"
)


def higher_after_n_bars(series: np.array, n: int) -> np.array:
    """
    If value of current bar lower than value of bar after n bars
    :param series: input series
    :param n: bars between current and next
    :return:
    """
    return _cond_after_n_bars(series, n, lambda current, after_n_bars: current < after_n_bars)


def lower_after_n_bars(series: np.array, n: int) -> np.array:
    """
    If value of current bar higher than value of bar after n bars
    :param series: input series
    :param n: bars between current and next
    :return:
    """
    return _cond_after_n_bars(series, n, lambda current, after_n_bars: current > after_n_bars)


def n_consecutive_up(series: np.array, n: int) -> np.array:
    """
    1 if direction of next n bars is 1
    :param series: input series
    :param n: length of bars series
    :return:
    """
    return _n_consecutive(series, n, lambda arr: arr == 1)


def n_consecutive_down(series: np.array, n: int) -> np.array:
    """
    1 if direction of next n bars is -1
    :param series: input series
    :param n: length of bars series
    :return:
    """
    return _n_consecutive(series, n, lambda arr: arr == -1)


def triple_barrier_labeling(
    close: pd.Series,
    pt_coef: float = 1.,
    sl_coef: float = 1.,
    window: int = 100,
    daily_volatility_span: int = 100
) -> np.array:
    """
    Triple Barrier labeling
    :param close: series with DatetimeIndex (needed for daily volatility calculation)
    :param pt_coef: top border multiplier (set np.nan for disabling border)
    :param sl_coef: bottom border multiplier (set np.nan for disabling border)
    :param window: count of bars to vertical border
    :param daily_volatility_span: (needed for daily volatility calculation)
    :return:
    """
    close = close.copy()
    df = pd.DataFrame(close)
    df["daily_vol"] = calculate_daily_volatility(close, daily_volatility_span)
    df = df.dropna()

    res = triple_barrier(
        close=df.close.values,
        daily_volatility=df.daily_vol.values,
        pt_coef=pt_coef,
        sl_coef=sl_coef,
        window=window
    )

    df_r = pd.concat([df.reset_index(), pd.DataFrame(res, columns=["crossed"])], axis=1)
    df_r = df_r.set_index("timestamp")
    df_r = pd.concat([pd.DataFrame(close), df_r], axis=1)
    return df_r.crossed.values
