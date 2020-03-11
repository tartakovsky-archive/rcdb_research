from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class Trades:
    balance: np.ndarray
    exposure: np.ndarray
    unrealized_pnl: np.ndarray
    context: list
    index: np.ndarray

    def __post_init__(self):
        self.metrics = TradingMetrics(self)

    def head(self, n: int) -> Trades:
        """
        Returns copy of Trades with first n items
        :param n: number of first items
        :return:
        """

        new_trades = Trades(
            balance=self.balance[:n],
            exposure=self.exposure[:n],
            unrealized_pnl=self.unrealized_pnl[:n],
            context=self.context[:n],
            index=self.index[:n]
        )
        return new_trades

    def tail(self, n: int) -> Trades:
        """
        Returns copy of Trades with last n items
        :param n: number of last items
        :return:
        """
        new_trades = Trades(
            balance=self.balance[-n:],
            exposure=self.exposure[-n:],
            unrealized_pnl=self.unrealized_pnl[-n:],
            context=self.context[-n:],
            index=self.index[-n:]
        )
        return new_trades

    def in_date_range(self, date_start: Optional[str] = None, date_end: Optional[str] = None) -> Trades:
        """
        Returns copy of Trades with items with indexes between date_start and date_end
        :param date_start: Date to drop observations before
        :param date_end: Date to drop observations after
        :return:
        """

        if not isinstance(self.index, pd.DatetimeIndex):
            raise ValueError(f'index should be an instance of pd.DatetimeIndex to use in_date_range method')

        sub_index = self.index
        if date_start is not None:
            sub_index = sub_index[sub_index >= date_start]
        if date_end is not None:
            sub_index = sub_index[sub_index < date_end]

        new_trades = Trades(
            balance=self.balance[np.isin(self.index, sub_index)],
            exposure=self.exposure[np.isin(self.index, sub_index)],
            unrealized_pnl=self.unrealized_pnl[np.isin(self.index, sub_index)],
            context=self.context[np.isin(self.index, sub_index)],
            index=sub_index,
        )
        return new_trades


class TradingMetrics:
    def __init__(self, trades: Trades):
        self.trades = weakref.proxy(trades)

    def equity(self, include_unrealized: bool = True, raw: bool = False) -> Union[pd.Series, tuple]:
        equity = self.trades.balance + self.trades.unrealized_pnl if include_unrealized else self.trades.balance
        index = self.trades.index

        return (equity, index) if raw else pd.Series(equity, index=index)

    def equity_change_abs(self, include_unrealized: bool = True, raw: bool = False) -> Union[pd.Series, tuple]:
        equity, index = self.equity(include_unrealized=include_unrealized, raw=True)
        change = np.diff(equity)
        change = np.insert(change, 0, 0)

        return (change, index) if raw else pd.Series(change, index=index)

    def equity_change_pct(self, include_unrealized: bool = True, raw: bool = False) -> Union[pd.Series, tuple]:
        equity, index = self.equity(include_unrealized=include_unrealized, raw=True)
        change = np.diff(equity) / equity[:-1]
        change = np.insert(change, 0, 0)

        return (change, index) if raw else pd.Series(change, index=index)

    def cum_equity_change_pct(self, include_unrealized: bool = True, raw: bool = False) -> Union[pd.Series, tuple]:
        equity, index = self.equity(include_unrealized=include_unrealized, raw=True)
        cum_return = (equity / equity[0]) - 1

        return (cum_return, index) if raw else pd.Series(cum_return, index=index)

    def drawdown(self, include_unrealized: bool = True, raw: bool = False) -> Union[pd.Series, tuple]:
        equity, index = self.equity(include_unrealized=include_unrealized, raw=True)
        drawdown = equity / np.maximum.accumulate(equity) - 1

        return (drawdown, index) if raw else pd.Series(drawdown, index=index)
