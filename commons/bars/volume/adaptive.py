import pandas as pd
import numpy as np

from commons import bars

VOLUME_SUM = 11
ROLLING_VOLUME = 12


def adaptive(ohlc, avg_per, window):
    bars.base.idx_to_column(ohlc)
    bars.base.validate_columns(ohlc)

    if avg_per >= window:
        raise AttributeError(
            f"Average per source bar period must be greater than window period")

    ohlc = ohlc[bars.COLUMNS]
    ohlc.loc[:, 'volume_sum'] = (ohlc.iloc[:, bars.VOLUME_BUY] +
                                 ohlc.iloc[:, bars.VOLUME_SELL])
    ohlc.loc[:, 'rolling_volume'] = ohlc.volume.rolling(
        window, min_periods=window).sum() / (window / avg_per)
    consolidated_bars = []
    current_bar = None

    for bar in ohlc.to_numpy():
        if current_bar is None:
            current_bar = list(bar)
        else:
            bars.base.update(current_bar, bar)
            current_bar[ROLLING_VOLUME] = bar[ROLLING_VOLUME]

        if (current_bar[VOLUME_SUM] > current_bar[ROLLING_VOLUME]
                and not np.isnan(current_bar[ROLLING_VOLUME])):
            consolidated_bars.append(current_bar)
            current_bar = None

    return bars.base.output_format(consolidated_bars)
