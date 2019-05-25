import pandas as pd

OPEN = 0
HIGH = 1
LOW = 2
CLOSE = 3
VOLUME = 4
VOLUME_SELL = 5
TICKS = 6
TICKS_SELL = 7
BAR_COLUMNS = ['open', 'high', 'low', 'close', 'volume',
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
    if len(ohlc.columns) < len(BAR_COLUMNS):
        raise AttributeError(
            f"Input DataFrame must have "
            f"{', '.join([f'`{c}`' for c in BAR_COLUMNS])} columns.")

    if not 0 < pct:
        raise AttributeError(f'Percentage must be greater than zero.')

    consolidated_bars = pd.DataFrame(columns=BAR_COLUMNS)
    consolidated_bars.index.name = 'time'
    current_bar = None

    for time, bar in ohlc.iterrows():
        if current_bar is None:
            current_bar = bar
            current_bar.open_time = time
        else:
            update(current_bar, bar)

        delta = current_bar[OPEN] * pct
        upper_limit = current_bar[OPEN] + delta
        lower_limit = current_bar[OPEN] - delta
        if (current_bar[CLOSE] >= upper_limit
                or current_bar[CLOSE] <= lower_limit):
            consolidated_bars.loc[current_bar.open_time] = current_bar
            current_bar = None

    return consolidated_bars
