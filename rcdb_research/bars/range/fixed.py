from rcdb_research import bars
import numpy as np


class RangeFixedConsolidator(bars.base.BaseConsolidator):
    UPPER_LIMIT = 12
    LOWER_LIMIT = 13

    def __init__(self, ohlc, threshold, threshold_is_absolute, timestamp_close):
        if not 0 < threshold:
            raise AttributeError(
                f"{'Price' if threshold_is_absolute else 'Percentage'} "
                f"threshold must be greater than zero.")
        self.threshold = threshold
        if threshold_is_absolute:
            self.upper_limit = lambda x: x + self.threshold
            self.lower_limit = lambda x: x - self.threshold
        else:
            self.upper_limit = lambda x: x * (1 + self.threshold)
            self.lower_limit = lambda x: x * (1 - self.threshold)
        super().__init__(ohlc, timestamp_close)

    def prepare(self):
        super().prepare()
        self.ohlc.loc[:, 'upper_limit'] = np.nan
        self.ohlc.loc[:, 'lower_limit'] = np.nan

    def bar_create(self, bar):
        super().bar_create(bar)
        close_prev = self.consolidated_bars[-1][self.CLOSE] \
            if self.consolidated_bars else self.current_bar[self.OPEN]

        # Setting upper and lower limit
        self.current_bar[self.UPPER_LIMIT] = self.upper_limit(close_prev)
        self.current_bar[self.LOWER_LIMIT] = self.lower_limit(close_prev)

    def bar_is_close_condition(self, bar):
        return (self.current_bar[self.CLOSE] >= self.current_bar[self.UPPER_LIMIT] or
                self.current_bar[self.CLOSE] <= self.current_bar[self.LOWER_LIMIT])


def fixed(ohlc, threshold, threshold_is_absolute=False, timestamp_close=False):
    return RangeFixedConsolidator(
        ohlc, threshold, threshold_is_absolute, timestamp_close).get()
