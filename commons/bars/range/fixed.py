import pandas as pd

from commons import bars

UPPER_LIMIT = 11
LOWER_LIMIT = 12


def fixed(ohlc, threshold, absolute=False):
    bars.base.idx_to_column(ohlc)
    bars.base.validate_columns(ohlc)

    if not 0 < threshold:
        raise AttributeError(
            f"{'Price' if absolute else 'Percentage'} "
            f"threshold must be greater than zero.")

    ohlc = ohlc[bars.COLUMNS]
    consolidated_bars = []
    current_bar = None

    for bar in ohlc.to_numpy():
        if current_bar is None:
            current_bar = list(bar)
            close_prev = consolidated_bars[-1][bars.CLOSE] \
                if consolidated_bars else current_bar[bars.OPEN]

            # Setting upper and lower limit
            if absolute:
                current_bar += [
                    close_prev + threshold,
                    close_prev - threshold,
                ]
            else:
                current_bar += [
                    close_prev * (1 + threshold),
                    close_prev * (1 - threshold),
                ]
        else:
            bars.base.update(current_bar, bar)

        if (current_bar[bars.CLOSE] >= current_bar[UPPER_LIMIT]
                or current_bar[bars.CLOSE] <= current_bar[LOWER_LIMIT]):
            consolidated_bars.append(current_bar)
            current_bar = None

    return bars.base.output_format(consolidated_bars)
