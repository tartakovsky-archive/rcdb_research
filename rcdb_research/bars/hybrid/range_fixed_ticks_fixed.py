from rcdb_research import bars


class RangeFixedTicksFixedConsolidator(bars.range.RangeFixedConsolidator):
    def __init__(self, ohlc, range_threshold, ticks_threshold,
                 absolute, timestamp_close):
        if not 0 < ticks_threshold:
            raise AttributeError(f"Ticks threshold must be greater than zero.")
        self.ticks_threshold = ticks_threshold
        super().__init__(ohlc, range_threshold, absolute, timestamp_close)

    def bar_is_close_condition(self, bar):
        ticks_sum = (self.current_bar[self.TICKS_BUY] +
                     self.current_bar[self.TICKS_SELL])
        return (ticks_sum >= self.ticks_threshold and
                super().bar_is_close_condition(bar))


def range_fixed_ticks_fixed(
        ohlc, range_threshold, ticks_threshold,
        absolute=False, timestamp_close=False):
    return RangeFixedTicksFixedConsolidator(
        ohlc, range_threshold, ticks_threshold,
        absolute, timestamp_close).get()
