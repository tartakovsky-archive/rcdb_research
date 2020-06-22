import math
import backtrader as bt

from dataclasses import dataclass


@dataclass
class OrderFake:
    size: float
    price: float


@dataclass
class StoryPoint:
    balance: float
    exposure: float
    unrealized_pnl: float
    index: float


class DataFeedMissingFieldsException(Exception):
    pass


def bt_data_feed_factory(df):
    cols = list(df.columns) + ['datetime']

    required_columns = ["open", "high", "low", "close", "volume", "signal"]
    if set(required_columns) - set(cols):
        raise DataFeedMissingFieldsException(f"Missing required columns {set(required_columns) - set(cols)}")

    fields = [('datetime', None)]
    for col in cols[:-1]:
        fields.append((col, -1))

    class TmpPandasDataFeed(bt.feeds.PandasData):
        linesoverride = True  # discard usual OHLC structure
        lines = tuple(cols)
        datafields = tuple(cols)
        params = tuple(fields)

    return TmpPandasDataFeed


class BtRcdbStrategy(bt.Strategy):
    def __init__(self, sizing, use_worst_pnl=False, verbose=False, risk_management_pre_trade=None):
        self.story = []
        self.sizing = sizing
        self.use_worst_pnl = use_worst_pnl
        self.verbose = verbose

        self.risk_management_pre_trade = risk_management_pre_trade if risk_management_pre_trade else []

    def get_risk_adjusted_exposure(self, desired_exposure):
        exposure_arr = [desired_exposure]
        for cb in self.risk_management_pre_trade:
            exposure_arr.append(cb(self, exposure_arr[-1]))
        return exposure_arr[-1]

    def get_size(self):
        return self.sizing.size(self.data.signal[0])

    def has_signal(self):
        signal = self.data.signal[0]
        return not math.isnan(signal)

    def next(self):
        size_info = self.calc_size()
        self.story.append(size_info)

        if not self.has_signal():
            return

        size_to_execute = size_info['size_to_execute']

        if size_to_execute != 0:
            if size_to_execute > 0:
                self.buy(size=size_to_execute)
            else:
                self.sell(size=size_to_execute)

    @property
    def max_long_lev(self):
        return self.data.max_long_lev[0]

    @property
    def max_short_lev(self):
        return self.data.max_short_lev[0]

    @property
    def close(self):
        return self.data.close[0]

    @property
    def high(self):
        return self.data.high[0]

    @property
    def low(self):
        return self.data.low[0]

    @property
    def open(self):
        return self.data.open[0]

    @property
    def open_next(self):
        try:
            return self.data.open[1]
        except IndexError:
            return self.data.open[0]

    def get_position_pnl(self, at_price):
        # for the default data (aka self.data0 and aka self.datas[0])
        pos = self.getposition(self.data)
        comminfo = self.broker.getcommissioninfo(self.data)

        #         if self.use_worst_pnl:
        #             price = self.low if pos.size > 0 else self.high
        #         else:
        #             price = self.close

        pnl = comminfo.profitandloss(pos.size, pos.price, at_price)
        pnl -= comminfo.getcommission(pos.size, at_price)
        # pnl -= comminfo.getcommission(pos.size, pos.price) \

        return pnl

    def calc_size(self):
        position_pnl_close = self.get_position_pnl(self.close)
        position_pnl_worst = None

        if self.use_worst_pnl:
            position_pnl_worst = self.get_position_pnl(self.low if self.position.size > 0 else self.high)

        portfolio_value = self.broker.get_value()
        exposure_curr = (self.position.size * self.position.price) / portfolio_value
        exposure_desired = self.get_size()

        exposure_desired = self.get_risk_adjusted_exposure(exposure_desired)

        size_desired = portfolio_value / self.close * exposure_desired
        size_to_execute = size_desired - self.position.size

        return dict(
            datetime=self.data.num2date(self.data.datetime[0]),
            balance=portfolio_value - position_pnl_close,
            unrealized_pnl=position_pnl_worst if self.use_worst_pnl else position_pnl_close,
            exposure_curr=exposure_curr,
            exposure_desired=exposure_desired,
            size_desired=size_desired,
            size_to_execute=size_to_execute
        )


class CommInfoFractional(bt.CommissionInfo):
    def getsize(self, price, cash):
        """
        Returns fractional size for cash operation @price
        :param price:
        :param cash:
        :return:
        """
        return self.p.leverage * (cash / price)
