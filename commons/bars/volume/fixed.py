import pandas as pd

from commons import bars


def fixed(ohlc, threshold, by_quote=True):
    bars.base.idx_to_column(ohlc)
    bars.base.validate_columns(ohlc)

    if not 0 < threshold:
        raise AttributeError(f'Volume threshold must be greater than zero.')

    ohlc = ohlc[bars.COLUMNS]
    consolidated_bars = []
    current_bar = None

    for bar in ohlc.values:
        if current_bar is None:
            current_bar = list(bar)
        else:
            bars.base.update(current_bar, bar)

        volume_sum = (
            current_bar[bars.VOLUME_QUOTE_SELL] + current_bar[bars.VOLUME_QUOTE_BUY]
            if by_quote else
            current_bar[bars.VOLUME_SELL] + current_bar[bars.VOLUME_BUY])

        if volume_sum >= threshold:
            consolidated_bars.append(current_bar)
            current_bar = None

    return bars.base.output_format(consolidated_bars)
