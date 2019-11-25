from ... import bars


class VolumeFixedConsolidator(bars.base.BaseConsolidator):
    def __init__(self, ohlc, threshold, by_quote, timestamp_close):
        if not 0 < threshold:
            raise AttributeError('Volume threshold must be greater than zero.')
        self.threshold = threshold
        if by_quote:
            self.volume_sell_id = self.VOLUME_QUOTE_SELL
            self.volume_buy_id = self.VOLUME_QUOTE_BUY
        else:
            self.volume_sell_id = self.VOLUME_SELL
            self.volume_buy_id = self.VOLUME_BUY
        super().__init__(ohlc, timestamp_close)

    def bar_is_close_condition(self, bar):
        volume_sum = (self.current_bar[self.volume_buy_id] +
                      self.current_bar[self.volume_sell_id])
        return volume_sum >= self.threshold


def fixed(ohlc, threshold, by_quote=True, timestamp_close=False):
    return VolumeFixedConsolidator(
        ohlc, threshold, by_quote, timestamp_close).get()
