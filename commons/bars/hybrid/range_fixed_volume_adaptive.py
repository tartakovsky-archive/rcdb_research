import pandas as pd
import numpy as np

from commons import bars

UPPER_LIMIT = 11
LOWER_LIMIT = 12
VOLUME_SUM = 13
ROLLING_VOLUME = 14


def _is_above_pct_threshold(current_bar):
    return (current_bar[bars.CLOSE] >= current_bar[UPPER_LIMIT] or
            current_bar[bars.CLOSE] <= current_bar[LOWER_LIMIT])


def _is_above_volume_threshold(current_bar):
    return current_bar[VOLUME_SUM] > current_bar[ROLLING_VOLUME]


def range_fixed_volume_adaptive(ohlc, pct_threshold, avg_per, window):
    bars.base.idx_to_column(ohlc)
    bars.base.validate_columns(ohlc)

    if avg_per >= window:
        raise AttributeError(
            f"Average per source bar period must be greater than window period")
    if not 0 < pct_threshold:
        raise AttributeError(f"Percentage threshold must be greater than zero.")

    ohlc = ohlc[bars.COLUMNS]
    ohlc.loc[:, 'upper_limit'] = np.nan
    ohlc.loc[:, 'lower_limit'] = np.nan
    ohlc.loc[:, 'volume_sum'] = (ohlc.iloc[:, bars.VOLUME_BUY] +
                                 ohlc.iloc[:, bars.VOLUME_SELL])
    ohlc.loc[:, 'rolling_volume'] = ohlc.volume_sum.rolling(
        window, min_periods=window).sum() / (window / avg_per)
    consolidated_bars = []
    current_bar = None

    for bar in ohlc.to_numpy():
        if current_bar is None:
            current_bar = list(bar)
            close_prev = consolidated_bars[-1][bars.CLOSE] \
                if consolidated_bars else current_bar[bars.OPEN]

            current_bar[UPPER_LIMIT] = close_prev * (1 + pct_threshold)
            current_bar[LOWER_LIMIT] = close_prev * (1 - pct_threshold)
        else:
            bars.base.update(current_bar, bar)
            current_bar[VOLUME_SUM] += bar[VOLUME_SUM]
            current_bar[ROLLING_VOLUME] = bar[ROLLING_VOLUME]

        if (not np.isnan(current_bar[ROLLING_VOLUME])
                and _is_above_volume_threshold(current_bar)
                and _is_above_pct_threshold(current_bar)):
            consolidated_bars.append(current_bar)
            current_bar = None

    return bars.base.output_format(consolidated_bars)
