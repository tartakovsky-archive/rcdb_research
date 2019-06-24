from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseConsolidator(ABC):
    RESOLUTION = 1
    FREQUENCY = 's'
    COLUMNS = [
        'timestamp', 'timestamp_close', 'open', 'high', 'low', 'close',
        'volume_buy', 'volume_sell', 'volume_quote_buy', 'volume_quote_sell',
        'ticks_buy', 'ticks_sell']

    TIMESTAMP = 0
    TIMESTAMP_CLOSE = 1
    OPEN = 2
    HIGH = 3
    LOW = 4
    CLOSE = 5
    VOLUME_BUY = 6
    VOLUME_SELL = 7
    VOLUME_QUOTE_BUY = 8
    VOLUME_QUOTE_SELL = 9
    TICKS_BUY = 10
    TICKS_SELL = 11

    def __init__(self, ohlc, timestamp_close):
        self.validate(ohlc)
        self.ohlc = ohlc.copy()
        self.consolidated_bars = []
        self.current_bar = None
        self.timestamp_close = timestamp_close
        self.prepare()

    def __get_timeframe(self):
        timeframe = {
            's': {'seconds': self.RESOLUTION},
            'min': {'minutes': self.RESOLUTION},
            'h': {'hours': self.RESOLUTION},
            'd': {'days': self.RESOLUTION},
        }[self.FREQUENCY.lower()]
        return timeframe

    def prepare(self):
        if 'timestamp' not in self.ohlc:
            self.ohlc.loc[:, 'timestamp'] = self.ohlc.index.copy()

        self.ohlc.loc[:, 'timestamp_close'] = np.nan
        self.ohlc = self.ohlc[self.COLUMNS]

    def validate(self, ohlc):
        if len(ohlc.columns) < len(self.COLUMNS[2:]):
            raise AttributeError(
                f"Input DataFrame must have "
                f"{', '.join([f'`{c}`' for c in self.COLUMNS[2:]])} columns.")

    @staticmethod
    def format(consolidated, timestamp_close):
        df = pd.DataFrame(
            [bar[:len(BaseConsolidator.COLUMNS)] for bar in consolidated],
            columns=BaseConsolidator.COLUMNS
        )
        df.set_index(BaseConsolidator.COLUMNS[0], inplace=True)
        if timestamp_close:
            return df[BaseConsolidator.COLUMNS[1:]]
        else:
            df.drop('timestamp_close', axis=1)
            return df[BaseConsolidator.COLUMNS[2:]]

    def clean(self):
        del self.consolidated_bars[:]
        self.current_bar = None
        self.ohlc = None

    def get(self):
        for bar in self.ohlc.values:
            self.bar_update(bar)

            if self.bar_is_close_condition(bar):
                if self.timestamp_close:
                    self.bar_set_close_time(bar)
                self.bar_close()

        output = self.format(self.consolidated_bars, self.timestamp_close)
        self.clean()
        return output

    # Below bar methods

    def bar_update(self, bar):
        if self.current_bar is None:
            self.bar_create(bar)
            return False

        to_sum = [
            self.VOLUME_BUY, self.VOLUME_SELL,
            self.VOLUME_QUOTE_BUY, self.VOLUME_QUOTE_SELL,
            self.TICKS_BUY, self.TICKS_SELL]
        for column in to_sum:
            self.current_bar[column] += bar[column]

        self.current_bar[self.CLOSE] = bar[self.CLOSE]
        if bar[self.LOW] < self.current_bar[self.LOW]:
            self.current_bar[self.LOW] = bar[self.LOW]
        if bar[self.HIGH] > self.current_bar[self.HIGH]:
            self.current_bar[self.HIGH] = bar[self.HIGH]
        return True

    def bar_create(self, bar):
        self.current_bar = list(bar)

    def bar_close(self):
        self.consolidated_bars.append(self.current_bar)
        self.current_bar = None

    def bar_set_close_time(self, bar):
        self.current_bar[self.TIMESTAMP_CLOSE] = (
                pd.Timedelta(**self.__get_timeframe()) +
                bar[self.TIMESTAMP])

    @abstractmethod
    def bar_is_close_condition(self, bar):
        pass
