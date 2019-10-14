from rcdb_research import bars


class RangeFixedVolumeAdaptiveConsolidator(bars.range.RangeFixedConsolidator):
    VOLUME_SUM = 14
    VOLUME_AVG = 15

    def __init__(self, ohlc, range_threshold, avg_per, window,
                 range_threshold_is_absolute, timestamp_close):
        if avg_per >= window:
            raise AttributeError(
                f"Average per source bar period must be greater than window period")
        self.avg_per = avg_per
        self.window = window
        super().__init__(ohlc, range_threshold,
                         range_threshold_is_absolute, timestamp_close)

    def prepare(self):
        super().prepare()
        self.ohlc.loc[:, 'volume_sum'] = (
                self.ohlc.iloc[:, self.VOLUME_BUY] +
                self.ohlc.iloc[:, self.VOLUME_SELL])
        timeframe = (self.window / self.avg_per)
        self.ohlc.loc[:, 'volume_avg'] = self.ohlc.volume_sum.asfreq(
            self.FREQUENCY).fillna(0).rolling(self.window).sum() / timeframe
        self.ohlc.dropna(subset=['volume_avg'], inplace=True)

    def bar_update(self, bar):
        if super().bar_update(bar):
            self.current_bar[self.VOLUME_SUM] += bar[self.VOLUME_SUM]
            self.current_bar[self.VOLUME_AVG] = bar[self.VOLUME_AVG]

    def bar_is_close_condition(self, bar):
        greater_avg = (self.current_bar[self.VOLUME_SUM] >
                       self.current_bar[self.VOLUME_AVG])
        return greater_avg and super().bar_is_close_condition(bar)


def range_fixed_volume_adaptive(
        ohlc, range_threshold, avg_per, window,
        range_threshold_is_absolute=False, timestamp_close=False):
    return RangeFixedVolumeAdaptiveConsolidator(
        ohlc, range_threshold, avg_per,
        window, range_threshold_is_absolute, timestamp_close).get()
