from typing import Union
import weakref

import numpy as np
import pandas as pd
import numba

from ..entities import TradesOld


class TradingMetrics:
    """
    Class for analyzing trades of quantitative strategies
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
                 trades: TradesOld,
                 initial_capital: float = 100,
                 position_size: float = 0.5,  # TODO: replace with PositionSizing function
                 compounded: bool = False):  # TODO: remove, controlled by position sizing strat
        self.trades = weakref.proxy(trades)
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.compounded = compounded

    ############
    # Trading metrics
    ############

    def returns(self, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:
        index = self.trades.index
        returns = self.trades.changes + self.trades.fees

        if dense:
            ids = (returns != 0)
            index = index[ids]
            returns = returns[ids]

        return (returns, index) if raw else pd.Series(returns, index=index)

    def profits(self, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:
        profits, index = self.returns(dense=dense, raw=True)

        if dense:
            ids = (profits > 0)
            index = index[ids]
            profits = profits[ids]
        else:
            profits[profits <= 0] = 0.0

        return (profits, index) if raw else pd.Series(profits, index=index)

    def losses(self, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:
        losses, index = self.returns(dense=dense, raw=True)

        if dense:
            ids = (losses < 0)
            index = index[ids]
            losses = losses[ids]
        else:
            losses[losses >= 0] = 0.0

        return (losses, index) if raw else pd.Series(losses, index=index)

    def mean_profit(self) -> float:
        profits, _ = self.profits(dense=True, raw=True)
        return profits.mean()

    def max_profit(self) -> float:
        profits, _ = self.profits(dense=True, raw=True)
        return profits.max()

    def mean_loss(self) -> float:
        losses, _ = self.losses(dense=True, raw=True)
        return losses.mean()

    def max_loss(self) -> float:
        losses, _ = self.losses(dense=True, raw=True)
        return losses.min()

    def n_wins(self) -> int:
        profits, _ = self.profits(dense=True, raw=True)
        return profits.size

    def n_losses(self) -> int:
        losses, _ = self.losses(dense=True, raw=True)
        return losses.size

    def n_trades(self) -> int:
        return np.count_nonzero(self.returns(dense=True))

    def pct_wins(self) -> float:
        return self.n_wins() / self.n_trades()

    def pct_losses(self) -> float:
        return self.n_losses() / self.n_trades()

    def n_bars(self) -> int:
        return self.returns(dense=False).size

    def activity(self) -> float:
        return self.n_trades() / self.n_bars()

    def expectancy(self) -> float:
        return self.pct_wins() * self.mean_profit() + self.pct_losses() * self.mean_loss()

    def tail_ratio(self, tail: int = 10) -> float:
        returns, _ = self.returns(dense=True, raw=True)

        positive_tail = np.percentile(returns, range(100-tail, 100+1)).mean()
        negative_tail = np.abs(np.percentile(returns, range(0, tail+1)).mean())

        return positive_tail / negative_tail

    ############
    # Equity metrics
    ############
    def expected_equity(self, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:

        index = self.trades.index
        expectancy = self.expectancy()

        if dense:
            index = index[self.trades.directions != 0]
        else:
            expectancy = expectancy*self.activity()

        returns = np.full(index.size, expectancy)

        equity = TradingMetrics._calculate_equity(returns, self.initial_capital, self.position_size, self.compounded)

        return (equity, index) if raw else pd.Series(equity, index=index)

    def expected_cum_return(self, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:

        equity, index = self.expected_equity(dense=dense, raw=True)

        cum_return = (equity / self.initial_capital) - 1

        return (cum_return, index) if raw else pd.Series(cum_return, index=index)

    def equity(self, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:

        returns, index = self.returns(dense=dense, raw=True)

        equity = TradingMetrics._calculate_equity(returns, self.initial_capital, self.position_size, self.compounded)

        return (equity, index) if raw else pd.Series(equity, index=index)

    def cum_return(self, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:

        equity, index = self.equity(dense=dense, raw=True)

        cum_return = (equity / self.initial_capital) - 1

        return (cum_return, index) if raw else pd.Series(cum_return, index=index)

    def abs_return(self) -> float:

        cum_return, _ = self.cum_return(dense=True, raw=True)
        return cum_return[-1]

    def drawdown(self, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:

        equity, index = self.equity(dense=dense, raw=True)

        drawdown = equity / np.maximum.accumulate(equity) - 1

        return (drawdown, index) if raw else pd.Series(drawdown, index=index)

    def max_drawdown(self) -> float:
        return self.drawdown().min()

    def annual_return(self, compounded: bool = False) -> float:

        cum_return, index = self.cum_return(dense=True, raw=True)

        if type(index) != pd.DatetimeIndex:
            raise ValueError('trades.index should be of type pd.DatetimeIndex to calculate annualized returns')

        abs_return = cum_return[-1]
        years_passed = (index[-1] - index[0]).total_seconds()/60/60/24/365.25

        if compounded:
            return (1+abs_return)**(1/years_passed) - 1
        else:
            return abs_return / years_passed

    def mar(self) -> float:
        annual_return = self.annual_return()
        max_drawdown = self.max_drawdown()
        return annual_return / np.abs(max_drawdown)

    def dataframe(self, tail: int = 10) -> pd.DataFrame:

        abs_return = self.abs_return()
        annual_return = self.annual_return()
        max_drawdown = self.max_drawdown()
        mar = self.mar()

        metrics_dict = dict(
            abs_return=abs_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            mar=mar,
            pct_wins=self.pct_wins(),
            pct_losses=self.pct_losses(),
            n_trades=self.n_trades(),
            expectancy=self.expectancy() * self.position_size,
            mean_profit=self.mean_profit(),
            max_profit=self.max_profit(),
            mean_loss=self.mean_loss(),
            max_loss=self.max_loss(),
            tail_ratio=self.tail_ratio(tail=tail),
        )
        metrics_df = pd.DataFrame({**metrics_dict}, index=[0])
        return metrics_df

    ############
    # Private functions
    ############
    @staticmethod
    @numba.njit
    def _calculate_equity(returns, initial_equity, position_size, compounded):

        equity = np.empty(returns.size)
        equity[0] = initial_equity
        if compounded:
            for i in range(1, equity.size):
                equity[i] = equity[i - 1] * (1 + returns[i] * position_size)
        else:
            for i in range(1, equity.size):
                _position_size = position_size * initial_equity
                if equity[i - 1] >= _position_size:
                    equity[i] = equity[i - 1] + _position_size * returns[i]
                else:
                    # Stop trading if there is not enough money left for a full position
                    equity[i] = equity[i - 1]
        return equity
