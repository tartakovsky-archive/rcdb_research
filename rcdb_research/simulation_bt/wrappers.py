import math
import backtrader as bt
import datetime
from dataclasses import dataclass


@dataclass
class SizeInfo:
    datetime: datetime.datetime
    balance: float
    unrealized_pnl: float
    exposure_current: float
    exposure_desired: float
    size_desired: float
    size_to_execute: float


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
    def __init__(self,
                 sizing,
                 use_worst_pnl=False,
                 verbose=False,
                 risk_management_pre_trade=None,
                 entry_limit=False):
        self.story = []
        self.sizing = sizing
        self.use_worst_pnl = use_worst_pnl
        self.verbose = verbose
        self.entry_limit = entry_limit
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

    def is_position_increase(self, size_info: SizeInfo) -> (bool, bool):
        """

        :param size_info:
        :return: is_position_increase, is_sign_change
        """
        if size_info.exposure_desired * size_info.exposure_current < 0:
            # sign is changed
            return False, True

        if size_info.exposure_desired > 0 and size_info.exposure_desired > size_info.exposure_current:
            # position increase (long)
            return True, False
        elif size_info.exposure_desired < 0 and size_info.exposure_desired < size_info.exposure_current:
            # position increase (short)
            return True, False

        return False, False

    def next(self):
        size_info = self.calc_size()
        self.story.append(size_info.__dict__)
        if not self.has_signal():
            return

        is_position_increase, is_sign_change = self.is_position_increase(size_info)

        if is_sign_change:
            self.order_target_size(target=0)
            size_to_execute = size_info.size_desired
        else:
            size_to_execute = size_info.size_to_execute

        exec_type = bt.Order.Limit if self.entry_limit else bt.Order.Market
        if size_to_execute != 0:
            order_kwargs = dict(
                size=size_to_execute,
                exectype=exec_type
            )
            if exec_type == bt.Order.Limit:
                order_kwargs['exectype'] = exec_type
                order_kwargs['price'] = self._close

            if size_to_execute > 0:
                o = self.buy(**order_kwargs)
            else:
                o = self.sell(**order_kwargs)

            return o

    @property
    def max_long_lev(self):
        return self.data.max_long_lev[0]

    @property
    def max_short_lev(self):
        return self.data.max_short_lev[0]

    @property
    def _open(self):
        return self.data.open[0]

    @property
    def _close(self):
        return self.data.close[0]

    @property
    def _high(self):
        return self.data.high[0]

    @property
    def _low(self):
        return self.data.low[0]

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
        #             price = self.low if pos.size > 0 else self.h
        #         else:
        #             price = self._close

        pnl = comminfo.profitandloss(pos.size, pos.price, at_price)
        pnl -= comminfo.getcommission(pos.size, at_price)
        # pnl -= comminfo.getcommission(pos.size, pos.price) \

        return pnl

    def calc_size(self):
        position_pnl_close = self.get_position_pnl(self._close)
        position_pnl_worst = None

        if self.use_worst_pnl:
            position_pnl_worst = self.get_position_pnl(self._low if self.position.size > 0 else self._high)

        portfolio_value = self.broker.get_value()
        exposure_curr = (self.position.size * self.position.price) / portfolio_value
        exposure_desired = self.get_size()

        exposure_desired = self.get_risk_adjusted_exposure(exposure_desired)

        size_desired = portfolio_value / self._close * exposure_desired
        size_to_execute = size_desired - self.position.size

        return SizeInfo(
            datetime=self.data.num2date(self.data.datetime[0]),
            balance=portfolio_value - position_pnl_close,
            unrealized_pnl=position_pnl_worst if self.use_worst_pnl else position_pnl_close,
            exposure_current=exposure_curr,
            exposure_desired=exposure_desired,
            size_desired=size_desired,
            size_to_execute=size_to_execute
        )

    # def notify_trade(self, trade):
    #     print(trade)
    #     return
    #
    # def notify_order(self, order):
    #     print(order)
    #     return


class CommInfoFractional(bt.CommissionInfo):
    def getsize(self, price, cash):
        """
        Returns fractional size for cash operation @price
        :param price:
        :param cash:
        :return:
        """
        return self.p.leverage * (cash / price)
