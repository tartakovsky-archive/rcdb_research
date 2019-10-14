from rcdb_research import bars


class TickFixedConsolidator(bars.base.BaseConsolidator):
    def __init__(self, ohlc, threshold, timestamp_close):
        if not 0 < threshold:
            raise AttributeError('Ticks threshold must be greater than zero.')
        self.threshold = threshold
        super().__init__(ohlc, timestamp_close)

    def bar_is_close_condition(self, bar):
        ticks_sum = (self.current_bar[self.TICKS_SELL] +
                     self.current_bar[self.TICKS_BUY])
        return ticks_sum >= self.threshold


def fixed(ohlc, threshold, timestamp_close=False):
    return TickFixedConsolidator(ohlc, threshold, timestamp_close).get()
