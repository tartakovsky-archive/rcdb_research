from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class TradesOld:
    """
    Class for storing trades of quantitative strategies
    """
    ############
    # Initialization
    ############
    """
    Params
    :param directions: pd.Series, direction of the trade, one of [1, 0, -1]  for long, flat or short trade
    :param changes: pd.Series, price change inside the bar, close / open -1
    :param fees: pd.Series, fee for each trade, positive for rebates, negative for fees
    """

    def __init__(self,
                 directions: np.array,
                 changes: np.array,
                 fees: np.array,
                 index: np.array):

        if not directions.size == index.size:
            raise ValueError('directions.size is not equal to index.size')
        if not changes.size == index.size:
            raise ValueError('changes.size is not equal to index.size')
        if not fees.size == index.size:
            raise ValueError('fees.size is not equal to index.size')

        self.directions = directions
        self.changes = changes
        self.fees = fees
        self.index = index

        self.metrics = None
        self.metric_params = None

    ############
    # Public methods
    ############
    def init_metrics(self, initial_capital: float = 100,
                     position_size: float = 0.5, compounded: bool = False) -> TradesOld:
        # Located here to avoid circular import: TradingMetrics -> Trades -> TradingMetrics -> ...
        from .trading_metrics import TradingMetrics

        self.metric_params = dict(initial_capital=initial_capital, position_size=position_size, compounded=compounded)
        self.metrics = TradingMetrics(self, **self.metric_params)
        return self

    def head(self, n: int) -> TradesOld:
        """
        Returns copy of Trades with first n items
        :param n: number of first items
        :return:
        """

        new_trades = TradesOld(
            directions=self.directions[:n],
            changes=self.changes[:n],
            fees=self.fees[:n],
            index=self.index[:n]
        )

        if self.metric_params is not None:
            new_trades.init_metrics(**self.metric_params)

        return new_trades

    def tail(self, n: int) -> TradesOld:
        """
        Returns copy of Trades with last n items
        :param n: number of last items
        :return:
        """
        new_trades = TradesOld(
            directions=self.directions[-n:],
            changes=self.changes[-n:],
            fees=self.fees[-n:],
            index=self.index[-n:]
        )

        if self.metric_params is not None:
            new_trades.init_metrics(**self.metric_params)

        return new_trades

    def in_date_range(self, date_start: Optional[str] = None, date_end: Optional[str] = None) -> TradesOld:
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

        sub_directions = self.directions[np.isin(self.index, sub_index)]
        sub_changes = self.changes[np.isin(self.index, sub_index)]
        sub_fees = self.fees[np.isin(self.index, sub_index)]

        new_trades = TradesOld(
            directions=sub_directions,
            changes=sub_changes,
            fees=sub_fees,
            index=sub_index,
        )

        if self.metric_params is not None:
            new_trades.init_metrics(**self.metric_params)

        return new_trades
