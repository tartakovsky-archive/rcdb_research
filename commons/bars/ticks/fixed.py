import pandas as pd

from commons import bars


def fixed(ohlc, threshold):
    bars.base.idx_to_column(ohlc)
    bars.base.validate_columns(ohlc)

    if not 0 < threshold:
        raise AttributeError(f'Ticks threshold must be greater than zero.')

    ohlc = ohlc[bars.COLUMNS]
    consolidated_bars = []
    current_bar = None

    for bar in ohlc.values:
        if current_bar is None:
            current_bar = list(bar)
        else:
            bars.base.update(current_bar, bar)

        if current_bar[bars.TICKS_SELL] + current_bar[bars.TICKS_BUY] >= threshold:
            consolidated_bars.append(current_bar)
            current_bar = None

    return bars.base.output_format(consolidated_bars)