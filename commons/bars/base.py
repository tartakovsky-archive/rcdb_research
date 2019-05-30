import pandas as pd

from commons import bars


def idx_to_column(ohlc):
    if 'timestamp' not in ohlc:
        ohlc.index.name = 'timestamp'
        ohlc.reset_index(level='timestamp', inplace=True)


def validate_columns(ohlc):
    if len(ohlc.columns) < len(bars.COLUMNS):
        raise AttributeError(
            f"Input DataFrame must have "
            f"{', '.join([f'`{c}`' for c in bars.COLUMNS])} columns.")


def output_format(consolidated_bars):
    df = pd.DataFrame(
        [bar[:len(bars.COLUMNS)] for bar in consolidated_bars],
        columns=bars.COLUMNS
    )
    df.set_index(bars.COLUMNS[0], inplace=True)
    return df


def update(current_bar, bar):
    current_bar[bars.VOLUME_BUY] += bar[bars.VOLUME_BUY]
    current_bar[bars.VOLUME_SELL] += bar[bars.VOLUME_SELL]
    current_bar[bars.VOLUME_QUOTE_BUY] += bar[bars.VOLUME_QUOTE_BUY]
    current_bar[bars.VOLUME_QUOTE_SELL] += bar[bars.VOLUME_QUOTE_SELL]
    current_bar[bars.TICKS_BUY] += bar[bars.TICKS_BUY]
    current_bar[bars.TICKS_SELL] += bar[bars.TICKS_SELL]
    current_bar[bars.CLOSE] = bar[bars.CLOSE]
    if bar[bars.LOW] < current_bar[bars.LOW]:
        current_bar[bars.LOW] = bar[bars.LOW]
    if bar[bars.HIGH] > current_bar[bars.HIGH]:
        current_bar[bars.HIGH] = bar[bars.HIGH]
