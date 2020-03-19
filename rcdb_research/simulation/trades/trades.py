from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy_ext as npext
import pandas as pd


@dataclass
class Trades:
    balance: np.ndarray
    exposure: np.ndarray
    unrealized_pnl: np.ndarray
    context: list
    index: np.ndarray

    def __post_init__(self):
        # TODO: validate that all arrays are of equal size
        self.metrics = TradingMetrics(self)

    @property
    def size(self) -> int:
        return self.balance.size

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
    def __init__(self, trades: Trades, include_unrealized: bool = True):
        self.trades = weakref.proxy(trades)
        self.include_unrealized = include_unrealized

    #########
    # Equity-related series
    #########
    def equity(self, raw: bool = False) -> Union[pd.Series, tuple]:
        equity = self.trades.balance + self.trades.unrealized_pnl if self.include_unrealized else self.trades.balance
        index = self.trades.index

        return (equity, index) if raw else pd.Series(equity, index=index)

    def cum_return(self, raw: bool = False) -> Union[pd.Series, tuple]:
        equity, index = self.equity(raw=True)
        cum_return = (equity / equity[0]) - 1

        return (cum_return, index) if raw else pd.Series(cum_return, index=index)

    def drawdown(self, raw: bool = False) -> Union[pd.Series, tuple]:
        equity, index = self.equity(raw=True)
        drawdown = equity / np.maximum.accumulate(equity) - 1

        return (drawdown, index) if raw else pd.Series(drawdown, index=index)

    #########
    # Bar-wise series
    #########
    def returns_abs(self, raw: bool = False) -> Union[pd.Series, tuple]:
        equity, index = self.equity(raw=True)
        change = np.diff(equity)
        change = npext.fill_na(np.insert(change, 0, 0), 0)

        return (change, index) if raw else pd.Series(change, index=index)

    def returns(self, raw: bool = False) -> Union[pd.Series, tuple]:
        equity, index = self.equity(raw=True)
        change = np.diff(equity) / equity[:-1]
        change = npext.fill_na(np.insert(change, 0, 0), 0)

        return (change, index) if raw else pd.Series(change, index=index)

    def profits(self, raw: bool = False) -> Union[pd.Series, tuple]:
        wins, index = self.returns(raw=True)
        wins[wins <= 0] = 0.0
        return (wins, index) if raw else pd.Series(wins, index=index)

    def losses(self, raw: bool = False) -> Union[pd.Series, tuple]:
        losses, index = self.returns(raw=True)
        losses[losses >= 0] = 0.0
        return (losses, index) if raw else pd.Series(losses, index=index)

    #########
    # Scalar equity-related metrics
    #########
    def total_return(self) -> float:
        cum_return, _ = self.cum_return(raw=True)
        return cum_return[-1]

    def cagr(self) -> float:
        cum_equity_change_pct, index = self.cum_return(raw=True)

        if type(index) != pd.DatetimeIndex:
            raise ValueError('trades.index should be of type pd.DatetimeIndex to calculate CAGR')

        abs_return = cum_equity_change_pct[-1]
        years_passed = (index[-1] - index[0]).total_seconds() / 60 / 60 / 24 / 365.25

        return (1 + abs_return) ** (1 / years_passed) - 1

    def max_drawdown(self) -> float:
        return self.drawdown().min()

    def mar(self) -> float:
        cagr = self.cagr()
        max_drawdown = self.max_drawdown()
        return cagr / np.abs(max_drawdown)

    #########
    # Scalar bar-related metrics
    #########
    def mean_profit(self) -> float:
        profits, _ = self.profits(raw=True)
        return profits[profits > 0].mean()

    def max_profit(self) -> float:
        profits, _ = self.profits(raw=True)
        return profits.max()

    def mean_loss(self) -> float:
        losses, _ = self.losses(raw=True)
        return losses[losses < 0].mean()

    def max_loss(self) -> float:
        losses, _ = self.losses(raw=True)
        return losses.min()

    def n_wins(self) -> int:
        profits, _ = self.profits(raw=True)
        return np.count_nonzero(profits)

    def n_losses(self) -> int:
        losses, _ = self.losses(raw=True)
        return np.count_nonzero(losses)

    def n_bars_active(self) -> int:
        returns, _ = self.returns(raw=True)
        return np.count_nonzero(returns)

    def n_bars(self) -> int:
        return self.trades.size

    def pct_wins(self) -> float:
        return self.n_wins() / self.n_bars_active()

    def pct_losses(self) -> float:
        return self.n_losses() / self.n_bars_active()

    def activity(self) -> float:
        return self.n_bars_active() / self.n_bars()

    def expectancy(self) -> float:
        return self.pct_wins() * self.mean_profit() + self.pct_losses() * self.mean_loss()

    def tail_ratio(self, tail: int = 10) -> float:
        returns, _ = self.returns(raw=True)

        positive_tail = np.percentile(returns, range(100 - tail, 100 + 1)).mean()
        negative_tail = np.abs(np.percentile(returns, range(0, tail + 1)).mean())

        return positive_tail / negative_tail

    def dataframe(self, tail: int = 10) -> pd.DataFrame:
        metrics_dict = dict(
            total_return=self.total_return(),
            cagr=self.cagr(),
            max_drawdown=self.max_drawdown(),
            mar=self.mar(),
            pct_wins=self.pct_wins(),
            pct_losses=self.pct_losses(),
            n_bars_active=self.n_bars_active(),
            expectancy=self.expectancy(),
            mean_profit=self.mean_profit(),
            max_profit=self.max_profit(),
            mean_loss=self.mean_loss(),
            max_loss=self.max_loss(),
            tail_ratio=self.tail_ratio(tail=tail),
        )
        metrics_df = pd.DataFrame({**metrics_dict}, index=[0])
        return metrics_df
