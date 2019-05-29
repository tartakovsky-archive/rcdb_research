import pandas as pd
import numpy as np  # noqa


TIME = 0
OPEN = 1
HIGH = 2
LOW = 3
CLOSE = 4
VOLUME = 5
VOLUME_SELL = 6
TICKS = 7
TICKS_SELL = 8
UPPER_LIMIT = 9
LOWER_LIMIT = 10


BAR_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'volume',
               'volume_sell', 'ticks', 'ticks_sell']


def update(current_bar, bar):
    current_bar[VOLUME] += bar[VOLUME]
    current_bar[VOLUME_SELL] += bar[VOLUME_SELL]
    current_bar[TICKS] += bar[TICKS]
    current_bar[TICKS_SELL] += bar[TICKS_SELL]
    current_bar[CLOSE] = bar[CLOSE]
    if bar[LOW] < current_bar[LOW]:
        current_bar[LOW] = bar[LOW]
    if bar[HIGH] > current_bar[HIGH]:
        current_bar[HIGH] = bar[HIGH]


def min_pct_bars(ohlc, pct):
    if 'time' not in ohlc:
        ohlc['time'] = ohlc.index.copy()

    if len(ohlc.columns) < len(BAR_COLUMNS):
        raise AttributeError(
            f"Input DataFrame must have "
            f"{', '.join([f'`{c}`' for c in BAR_COLUMNS])} columns.")

    ohlc = ohlc[BAR_COLUMNS]

    if not 0 < pct:
        raise AttributeError(f'Percentage must be greater than zero.')

    consolidated_bars = []
    current_bar = None

    for bar in ohlc.values:
        if current_bar is None:
            current_bar = list(bar)

            close_prev = current_bar[OPEN]
            if consolidated_bars:
                close_prev = consolidated_bars[-1][CLOSE]

            current_bar += [
                close_prev * (1 + pct),  # UPPER_LIMIT
                close_prev * (1 - pct),  # LOWER_LIMIT
            ]

        else:
            update(current_bar, bar)

        if current_bar[CLOSE] >= current_bar[UPPER_LIMIT] \
                or current_bar[CLOSE] <= current_bar[LOWER_LIMIT]:
            consolidated_bars.append(current_bar)
            current_bar = None


    df = pd.DataFrame(consolidated_bars, columns=BAR_COLUMNS + ["UPPER_LIMIT", "LOWER_LIMIT"])
    df = df[BAR_COLUMNS]
    df = df.set_index("time")

    return df
