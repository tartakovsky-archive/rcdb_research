import numpy as np

from commons import bars


class TimeFixedConsolidator(bars.base.BaseConsolidator):
    def __init__(self, ohlc, period, timestamp_close):
        if period <= 0:
            raise AttributeError('Time period must be greater than zero.')
        self.period = period
        super().__init__(ohlc, timestamp_close)

    def prepare(self):
        self.ohlc = self.ohlc.resample(f"{self.period}{self.FREQUENCY}").agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume_buy': 'sum',
            'volume_sell': 'sum',
            'volume_quote_buy': 'sum',
            'volume_quote_sell': 'sum',
            'ticks_buy': 'sum',
            'ticks_sell': 'sum'
        }).dropna()

    def get(self):
        if self.timestamp_close:
            self.ohlc.loc[:, 'timestamp_close'] = np.asarray(
                self.ohlc.index.copy().shift(self.period, freq=self.FREQUENCY),
                dtype=object
            )
            return self.ohlc[self.COLUMNS[1:]]
        return self.ohlc[self.COLUMNS[2:]]

    def bar_is_close_condition(self, bar):
        pass


def fixed(ohlc, period, timestamp_close=False):
    return TimeFixedConsolidator(ohlc, period, timestamp_close).get()
