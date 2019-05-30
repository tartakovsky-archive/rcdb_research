import pandas as pd
import numpy as np

from commons import bars

UPPER_LIMIT = 11
LOWER_LIMIT = 12


def _is_above_pct_threshold(current_bar):
    return (current_bar[bars.CLOSE] >= current_bar[UPPER_LIMIT] or
            current_bar[bars.CLOSE] <= current_bar[LOWER_LIMIT])


def range_fixed_ticks_fixed(ohlc, pct_threshold, ticks_threshold):
    bars.base.idx_to_column(ohlc)
    bars.base.validate_columns(ohlc)

    if not 0 < pct_threshold:
        raise AttributeError(f"Percentage threshold must be greater than zero.")
    if not 0 < ticks_threshold:
        raise AttributeError(f"Ticks threshold must be greater than zero.")

    ohlc = ohlc[bars.COLUMNS]
    consolidated_bars = []
    current_bar = None

    for bar in ohlc.to_numpy():
        if current_bar is None:
            current_bar = list(bar)
            close_prev = consolidated_bars[-1][bars.CLOSE] \
                if consolidated_bars else current_bar[bars.OPEN]

            current_bar += [
                close_prev * (1 + pct_threshold),
                close_prev * (1 - pct_threshold)
            ]
        else:
            bars.base.update(current_bar, bar)

        ticks_sum = current_bar[bars.TICKS_BUY] + current_bar[bars.TICKS_SELL]
        if (ticks_sum >= ticks_threshold and
                _is_above_pct_threshold(current_bar)):
            consolidated_bars.append(current_bar)
            current_bar = None

    return bars.base.output_format(consolidated_bars)
