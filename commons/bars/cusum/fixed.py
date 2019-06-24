import numpy as np

from commons import bars


class CusumFixedConsolidator(bars.base.BaseConsolidator):
    LOG_CLOSE_DIFF = 12

    def __init__(self, ohlc, threshold, timestamp_close):
        if threshold <= 0:
            raise AttributeError('Threshold must be greater than zero.')
        self.threshold = threshold
        self.s_pos = 0
        self.s_neg = 0
        super().__init__(ohlc, timestamp_close)

    def prepare(self):
        super().prepare()
        self.ohlc.loc[:, 'log_close_diff'] = np.log(
            self.ohlc.loc[:, 'close']
        ).diff().fillna(0)

    def bar_update(self, bar):
        super().bar_update(bar)
        pos = self.s_pos + bar[self.LOG_CLOSE_DIFF]
        neg = self.s_neg + bar[self.LOG_CLOSE_DIFF]
        self.s_pos = max(0.0, pos)
        self.s_neg = min(0.0, neg)

    def bar_is_close_condition(self, bar):
        if self.s_neg < -1 * self.threshold:
            self.s_neg = 0
            return True
        elif self.s_pos > self.threshold:
            self.s_pos = 0
            return True


def fixed(ohlc, threshold, timestamp_close=False):
    """
    The CUSUM filter is a quality-control method, designed to detect a shift
    in the mean value of a measured quantity away from a target value.
    The filter is set up to identify a sequence of upside or downside
    divergences from any reset level zero.

    We sample a bar t if and only if S_t >= threshold, at which point S_t is
    reset to 0.

    One practical aspect that makes CUSUM filters appealing is that multiple
    events are not triggered by raw_time_series hovering around a threshold
    level, which is a flaw suffered by popular market signals such as
    Bollinger Bands. It will require a full run of length threshold for
    raw_time_series to trigger an event.

    Once we have obtained this subset of event-driven bars, we will let the ML
    algorithm determine whether the occurrence of such events constitutes
    actionable intelligence.

    :param raw_time_series: (series) of close prices (or other time series,
    e.g. volatility).
    :param threshold: (float) when the abs(change) is larger than the
    threshold, the function captures it as an event.
    :param time_stamps: (bool) default is to return a DateTimeIndex, change
    to false to have it return a list.
    :return: (datetime index vector) vector of datetimes when the events
    occurred. This is used later to sample.
    """
    return CusumFixedConsolidator(ohlc, threshold, timestamp_close).get()
