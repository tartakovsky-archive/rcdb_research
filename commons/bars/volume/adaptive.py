import numpy as np

from commons import bars


class AdaptiveVolumeConsolidator(bars.base.BaseConsolidator):
    VOLUME_SUM = 12
    VOLUME_AVG = 13

    def __init__(self, ohlc, avg_per, window, timestamp_close):
        if avg_per >= window:
            raise AttributeError(
                f"Average per source bar period must be "
                f"greater than window period")
        self.avg_per = avg_per
        self.window = window
        super().__init__(ohlc, timestamp_close)

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
        return (not np.isnan(self.current_bar[self.VOLUME_AVG]) and
                self.current_bar[self.VOLUME_SUM] >=
                self.current_bar[self.VOLUME_AVG])


def adaptive(ohlc, avg_per, window, timestamp_close=False):
    return AdaptiveVolumeConsolidator(
        ohlc, avg_per, window, timestamp_close).get()
